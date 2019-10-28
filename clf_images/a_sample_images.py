"""
toma las imagenes en carpeta del otro proceso
captura diversas caracteristicas del juego
"""
import cv2
import time
import matplotlib.pyplot as plt 
import numpy as np 
from utils_read_imgs import try_to_get_latest
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    out_folder = './samples'
    os.makedirs(out_folder,exist_ok=True)
    current_t = -1
    wait_time = 1
    while True:
        next_t,current_img = try_to_get_latest()

        if current_t == next_t:
            print("Imagen {0} no ha cambiado se salta".format(next_t))
        else:
            print("Imagen {0} AGREGADA".format(next_t))
            out_path = os.path.join(out_folder,'{0}.png'.format(next_t))
            cv2.imwrite(out_path,current_img)
            
        current_t = next_t
        time.sleep(wait_time)
