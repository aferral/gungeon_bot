import os
import random
import threading
import time
import cv2

from state_fusion.msg_sockets import socket_sender
from datetime import datetime

def now_string():
    now=datetime.now()
    return now.strftime('%Y_%b_%d__%H:%M')

def cicle_list(l):
    """
    Function that return the next element in list. At the end return the first element and then repeat
    :param l:
    :return:
    """
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


def random_img_from_folder(folder_path):
    lt = os.listdir(folder_path)
    random.shuffle(lt)
    fun_iter_folder = cicle_list(list(lt))
    feed_function = lambda: open_img(folder_path, fun_iter_folder())
    return feed_function

def video_feed_realtime(video_path,frame_start=0):

    cap = None
    st=time.time()
    fps=40
    def retrieve_frame():
        nonlocal cap
        if cap is None:
            cap = cv2.VideoCapture(video_path)
            if frame_start != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        now = frame_start+(time.time()-st)*fps
        success_set = cap.set(cv2.CAP_PROP_POS_FRAMES, int(now))
        success_grab = cap.grab()
        if success_set and success_grab:
            flag, frame = cap.retrieve()
            cf = cap.get(cv2.CAP_PROP_POS_FRAMES)
            out_name = '{0}_-1'.format(cf)
            return out_name,frame

        else:
            return None,None


    return retrieve_frame

def video_feed(video_path,frame_start=0):

    cap = None

    def retrieve_frame(): 
        nonlocal cap
        if cap is None:
            cap = cv2.VideoCapture(video_path)
            if frame_start != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        success = cap.grab()

        if success:
            flag, frame = cap.retrieve()
            cf = cap.get(cv2.CAP_PROP_POS_FRAMES)
            out_name = '{0}_-1'.format(cf)
            return out_name,frame

        else:
            return None,None


    return retrieve_frame


def calc_port():
    """
    Calc a port for each thread (for remote debugging)
    :return:
    """
    id_th = threading.get_ident()
    return 8000+id_th%100


def do_send_loop(server_address, feature_instance, n_msgs=1000, raise_excp=False,sleep_time=-1):
    print('INIT SENDER')

    def loop_send():
        with socket_sender(server_address,verbose=False) as send_fun:
            if n_msgs == -1:
                while True:
                    try:
                        if sleep_time != -1:
                            time.sleep(sleep_time)
                        out_dict = feature_instance.process()
                        send_fun(out_dict)
                    except Exception as e:
                        print('EXCEPTION {0}'.format(e))
                        if raise_excp:
                            raise e
            else:
                for i in range(n_msgs):
                    try:
                        if sleep_time != -1:
                            time.sleep(sleep_time)
                        out_dict = feature_instance.process()
                        send_fun(out_dict)
                    except Exception as e:
                        print('EXCEPTION {0}'.format(e))
                        if raise_excp:
                            raise e
                print('EXITING SENDER')
                send_fun({'END':True}) # END signal

    if feature_instance.use_context:
        with feature_instance.model_context():
            loop_send()
    else:
        loop_send()

    print('END SENDER')
    return


try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    raise ImportError
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner
