from datetime import datetime
import cv2
import os
path_imgs='./images_buffer'

current=[0]
flag_color = cv2.IMREAD_COLOR
flag_gray = cv2.IMREAD_GRAYSCALE

def in_order(folder):
    all_files = os.listdir(folder)
    times = sorted(all_files)
    path_out = os.path.join(folder,times[current[0]])
    current[0] = (current[0]+1)%len(all_files)
    out=cv2.imread(path_out)
    return current[0],out

def try_to_get_latest(folder=path_imgs,open_flag=flag_color):
    max_int = 3
    intentos = max_int
   
    format_name = lambda x : '.'.join(x.split('_')[1].split('.')[:-1])
    
    while intentos > 0:
        all_files = os.listdir(folder)
        times = sorted([(format_name(x),x) for x in all_files])
        path_out = os.path.join(folder,times[-1][1])
        if os.path.exists(path_out):
            out=cv2.imread(path_out,flag_color)
            if (out is None) or (out.shape[0] == 0):
                os.remove(path_out)
                continue
            return times[-1][0],out
        else:
            intentos -= 1
    raise Exception("Maximos intentos alcanzados ({0} intentos) aumentar cantidad de archivos del buffer circular o aumentar intentos".format(max_int)) 

if __name__ == '__main__':
    delay = 200
    k = -1
    while k != 27:
        timestamp,img_now = try_to_get_latest(path_imgs)
        #timestamp,img_now = in_order(path_imgs)
        cv2.imshow('t',img_now)
        tf = "%H:%M:%S:%f"
        now = datetime.now().timestamp()
        readed = float(timestamp)
        print("Leido {0} Actual {1}".format(readed,now))
        k=cv2.waitKey(delay)
    


