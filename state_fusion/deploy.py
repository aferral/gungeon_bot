import pickle
from multiprocessing import Process
import os
import threading
import time
from main import TemplateMatching, Tflite_clf
from state_fusion.deploy_utils import random_img_from_folder, video_feed, do_send_loop, video_feed_realtime, now_string
from state_fusion.msg_sockets import socket_read_msg
import cv2
from utils_read_imgs import try_to_get_latest

all_colors = {'bullets': (255, 0, 255),  # RGB
              'hp': (0, 0, 255),
              'scope': (51, 204, 51),
              'blanks': (51, 204, 204),
              'enemys': (255, 0, 0)
              }

if __name__ == '__main__':

    debug = False
    show = False
    record_output=True

    clean_state_socket = './clean_state'

    if debug:
        feed_function = random_img_from_folder('./samples')
        feed_function = video_feed("./screen_capture/video_data/out.mp4",frame_start=4400)

        skip_if_delay = False
        refresh_time = 1
        n_msgs = 200
    else:
        feed_function = try_to_get_latest
        feed_function = video_feed_realtime("./screen_capture/video_data/out.mp4", frame_start=4400)
        skip_if_delay = True # False
        refresh_time = 1
        n_msgs = 100 # run forever


    launched = []
    launched_threads = []
    recorded_output = []

    try:
        folder_model = './clf_images/saved_models/clf_enemy/12_Oct_2019__20_51_58'
        deployed_clf_folder = './deployed_clf' # tflite version of folder_model

        #enemy_clf = ClfEnemy(folder_model,feed_fun=feed_function) # use tensorflow version
        enemy_clf = Tflite_clf(folder_model,deployed_clf_folder,build_model=False,feed_fun=feed_function,send_imgs=True)# use tflite version

        all_features = {
            'bullets' : TemplateMatching('./templates/r2/bala.png',resize_factor=0.5,feed_fun=feed_function,send_imgs=True),
            'hp' : TemplateMatching('./templates/r2/hearth.png',resize_factor=0.5,feed_fun=feed_function,send_imgs=True),
            # 'scope' : TemplateMatching('./templates/r2/mira.png',resize_factor=0.5,feed_fun=feed_function,send_imgs=True),
            'blanks' : TemplateMatching('./templates/r2/blank.png',resize_factor=0.5,feed_fun=feed_function,send_imgs=True),
            'enemys' : enemy_clf
        }

        sleep_times = {
            'bullets': -1,  # RGB
            'hp': -1,
            'scope': -1,
            'blanks': -1,
            'enemys': -1
        }

        # cv2 uses BGR instead of RGB
        all_colors = {k : tuple([list(v)[i] for i in [2,1,0]]) for k,v in all_colors.items()}

        n_features = len(all_features)


        # iniciar captura de features en otros procesos
        for k,feature_instance in all_features.items():
            address = k
            p = Process(target=do_send_loop, args=(address, feature_instance, n_msgs),kwargs={'sleep_time' : sleep_times[k]})
            p.start()
            launched.append(p)


        print('INIT READER')

        # start,join
        lock = threading.Lock()

        barrier_next_msg = threading.Barrier(n_features+1)
        all_timestamps ={}
        all_bbs = {}



        def read_socket_and_update(key,s_addres,lock,barrier_next_msg):
            for msg in socket_read_msg(s_addres,verbose=False,skip_if_delay=skip_if_delay):

                if 'END' in msg:
                    with lock:
                        all_timestamps['END'] = True
                    break

                with lock:
                    all_timestamps['t_{0}'.format(key)] = msg['t_proc']
                    all_bbs['bb_{0}'.format(key)] = msg['coords']
                    if record_output:
                        recorded_output.append({key : msg})

                if debug:
                    barrier_next_msg.wait()

            if debug:
                barrier_next_msg.abort() # final barrier to threads waiting
                # (low-pri) final img has race condition


        for k,feature_instance in all_features.items():
            address = k
            name = k
            tx = threading.Thread(target=read_socket_and_update, args=(name,address,lock,barrier_next_msg))
            tx.start()
            launched_threads.append(tx)

        time.sleep(0.4)

        #with socket_sender(clean_state_socket, verbose=False) as send_clean_state: # TODO this blocks till comunication is stablished
        #ipdb.set_trace()

        updates=0
        st=time.time()
        should_cont = True
        try:
            while should_cont:
                with lock:
                    current_state = all_timestamps.copy()
                    current_bbs = all_bbs.copy()
                    updates +=1
                    if 'END' in all_timestamps:
                        should_cont = False

                # read current img
                timestamp, img_c = feed_function()
                #print('Vis: {0}'.format(timestamp))
                current_state['time_img'] = timestamp

                # Show result
                if show:
                    # show img and show overlay
                    img_cloned = img_c.copy()

                    # draw all the timestamps
                    x_cord = 400
                    for k, v in current_state.items():
                        cv2.putText(img_cloned, 'c_{1:15s}: {0}'.format(v, k), (10, x_cord), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 2)
                        x_cord += 30

                    # draw all bbs
                    for ind, (k, bb_list) in enumerate(current_bbs.items()):
                        for t in bb_list:
                            x2 = cv2.rectangle(img_cloned, t[0], t[1], tuple(all_colors[k.split('_')[1]]), 2)
                    cv2.imshow('t2',img_cloned)
                    k = cv2.waitKey(refresh_time)

                if debug:
                    barrier_next_msg.wait()  # wait all threads before vis

                # send clean state
                #send_clean_state({**current_state,**current_bbs}) # TODO fix no comm

            # TODO comparar timestamp enviado a timestamp recivido
            # plotear update en el tiempo

        except threading.BrokenBarrierError as e:
            print('Closing state fusion')

    finally:
        en=time.time()
        if record_output:
            output_folder = 'saved_state_updates'
            os.makedirs(output_folder,exist_ok=True)
            output_path = os.path.join(output_folder,'states_{0}.pkl'.format(now_string()))

            print('Saving state_updates in {0}'.format(output_path))
            with open(output_path,'wb') as f:
                pickle.dump(recorded_output,f)

        # espera cerrar procesos abiertos
        for th in launched_threads:
            th.join()

        for p in launched:
            p.join()

        for k in all_features:
            os.unlink(k)
            if os.path.exists(k):
                os.remove(k)
        print('Elapsed {1} FPS: {0}'.format(updates/(en-st),(en-st)))