import random
from multiprocessing import Process
import os
import threading
import time
from main import TemplateMatching, ClfEnemy
from state_fusion.msg_sockets import socket_sender, socket_read_msg
import cv2
from utils_read_imgs import try_to_get_latest

def process_img_fun(server_address,feature_instance,n_msgs=1000,raise_excp=False):
    print('INIT SENDER')

    def loop_send():
        with socket_sender(server_address,verbose=False) as send_fun:
            if n_msgs == -1:
                while True:
                    try:
                        out_dict = feature_instance.process()
                        send_fun(out_dict)
                    except Exception as e:
                        if raise_excp:
                            raise e
            else:
                for i in range(n_msgs):
                    try:
                        out_dict = feature_instance.process()
                        send_fun(out_dict)
                    except Exception as e:
                        if raise_excp:
                            raise e

    print('fi: {0}'.format(feature_instance.use_context))
    if feature_instance.use_context:
        with feature_instance.model_context():
            loop_send()
    else:
        loop_send()

    print('END SENDER')
    return



def cicle_list(l):
    ind=0
    n=len(l)
    def temp():
        nonlocal ind
        val = l[ind]
        ind = (ind+1)%n
        return val
    return temp

def open_img(folder,path):
    format_name = lambda x: x.split('.')[0]
    path_full = os.path.join(folder,path)
    out = cv2.imread(path_full,cv2.IMREAD_COLOR)
    return format_name(path),out



if __name__ == '__main__':

    debug = False
    show = False

    clean_state_socket = './clean_state'

    if debug:
        fun_iter_folder = cicle_list(list(reversed(sorted(os.listdir('./samples')))))

        lt = os.listdir('./samples')
        random.shuffle(lt)
        fun_iter_folder = cicle_list(list(lt))

        feed_function = lambda  : open_img('./samples',fun_iter_folder())
        skip_if_delay = False
        refresh_time = 0
        n_msgs = 10
    else:
        feed_function = try_to_get_latest
        skip_if_delay = True # False
        refresh_time = 2
        n_msgs = -1 # run forever


    launched = []
    launched_threads = []

    try:
        folder_model = './clf_images/saved_models/clf_enemy/12_Oct_2019__20_51_58'
        all_features = {'bullets' : TemplateMatching('./templates/bala_enemy.png', feed_fun=feed_function),
                        'hp' : TemplateMatching('./templates/full_hp.png', feed_fun=feed_function),
                        'scope' : TemplateMatching('./templates/t_mira.png', feed_fun=feed_function),
                        'blanks' : TemplateMatching('./templates/blank.png', feed_fun=feed_function),
                        'enemys' : ClfEnemy(folder_model,feed_fun=feed_function)
        }
        n_features = len(all_features)


        # iniciar captura de features en otros procesos
        for k,feature_instance in all_features.items():
            address = k

            p = Process(target=process_img_fun,args=(address,feature_instance,n_msgs))
            p.start()
            launched.append(p)


        print('INIT READER')

        # start,join
        lock = threading.Lock()

        barrier_next_msg = threading.Barrier(n_features+1)
        all_timestamps ={}
        all_bbs = {}

        # TODO filtrar segun historia

        def read_socket_and_update(key,s_addres,lock,barrier_next_msg):
            for msg in socket_read_msg(s_addres,verbose=False,skip_if_delay=skip_if_delay):


                with lock:
                    all_timestamps['t_{0}'.format(key)] = msg['t_proc']
                    all_bbs['bb_{0}'.format(key)] = msg['coords']
                print('New message ',msg)

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

        st_color = (102,255,51)
        en_color = (255,0,255)
        dif_color = tuple([a-b for a,b in zip(en_color,st_color) ])

        with socket_sender(clean_state_socket, verbose=False) as send_clean_state:

            try:
                while True:
                    with lock:
                        current_state = all_timestamps
                        current_bbs = all_bbs

                    # read current img
                    timestamp, img_c = feed_function()
                    print('Vis: {0}'.format(timestamp))
                    current_state['time_img'] = timestamp

                    # show img and show overlay
                    img_cloned = img_c.copy()

                    # draw all the timestamps
                    x_cord=400
                    for k,v in current_state.items():
                        cv2.putText(img_cloned, 'c_{1:15s}: {0}'.format(v,k), (10, x_cord), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                        x_cord+=30

                    # draw all bbs
                    n_c = len(current_bbs.items())
                    for ind,(k,bb_list) in enumerate(current_bbs.items()):
                        x_c = ind*1.0/n_c
                        res_color = list([base+delta*x_c for base,delta in zip(st_color,dif_color)])
                        for t in bb_list:
                            x2 = cv2.rectangle(img_cloned, t[0], t[1], tuple(res_color), 2)

                    # Show result
                    if show:
                        cv2.imshow('t2',img_cloned)
                        k = cv2.waitKey(refresh_time)

                    if debug:
                        barrier_next_msg.wait()  # wait all threads before vis

                    # send clean state
                    send_clean_state({**current_state,**current_bbs})

                # TODO comparar timestamp enviado a timestamp recivido
                # plotear update en el tiempo

            except threading.BrokenBarrierError as e:
                print('Closing state fusion')

    finally:
        # espera cerrar procesos abiertos
        for th in launched_threads:
            th.join()

        for p in launched:
            p.join()

        for k in all_features:
            os.unlink(k)
            if os.path.exists(k):
                os.remove(k)
