"""
toma las imagenes en carpeta del otro proceso
captura diversas caracteristicas del juego
"""
import cv2
import time
import matplotlib.pyplot as plt 
import numpy as np

from clf_images.clasify_enemy import model_load_from_folder
from utils_read_imgs import try_to_get_latest
import matplotlib.pyplot as plt

# Idea importante necesito poder crear rapidamente y robustamente. Detectores de diversos patrones

# Idea dar varios patrones y dependiendo de los parametros iniciales que patron busca.
# Idea postproceso busca mantener continuidad de valore

def match_find_multiple(img,template,th=0.89):
    t0=time.time()
    h,w=template.shape[0:2]
    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= th)
    out = [((x,y),(x+h,y+w)) for x,y in zip(*loc[::-1]) ]
    tf=time.time()
    print('Elapsed {0}'.format(tf-t0))
    return out
def match_find_max(img,template):
    t0 =time.time()
    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    minv,maxv,minl,maxl=cv2.minMaxLoc(res)
    bt=(maxl[0]+template.shape[0],maxl[1]+template.shape[1])
    tf=time.time()
    print('Elapsed {0}'.format(tf-t0))
    return maxl,bt

def show_two_images(a,b,wait=0):
    c=np.hstack(a,b)
    cv2.imshow('t',c)
    cv2.waitKey(wait)
    return

def one_show(a,wait=0):
    cv2.imshow('t',a)
    cv2.waitKey(wait)
    #plt.imshow(a)
    #plt.show()



def pipeline_one_img(img_c):
    flag_color = cv2.IMREAD_COLOR
    flag_gray = cv2.IMREAD_GRAYSCALE


    img = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)


    # calcula mira
    t = cv2.imread('./templates/t_mira.png',flag_gray)
    cords = match_find_max(img,t)
    x2=cv2.rectangle(img,cords[0],cords[1],255,2)
    #one_show(x2)

    # calcula ubicacion personaje
    t = cv2.imread('./templates/body_character.png',flag_color)
    #cords = match_find_max(img_c,t)
    x0=(int(img.shape[1]*0.5),int(img.shape[0]*0.5))
    cords = (x0,(x0[0]+t.shape[1],x0[1]+t.shape[0]))
    print(cords)
    x2=cv2.rectangle(img,cords[0],cords[1],255,2)
    #one_show(x2)

    # calcula HP
    t = cv2.imread('./templates/full_hp.png',flag_gray)
    cords = match_find_multiple(img,t,th=0.7)
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],255,2)
    #one_show(x2)


    # calcula blanks
    t = cv2.imread('./templates/blank.png',flag_color)
    cords = match_find_multiple(img_c,t,th=0.7)
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],255,2)
    #one_show(x2)

    # calcula balas enemigos
    t = cv2.imread('./templates/bala_enemy.png',flag_color)
    cords = match_find_multiple(img_c,t,th=0.7)
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],255,2)


    folder_model = './clf_images/saved_models/clf_enemy/12_Oct_2019__20_51_58'
    # eval_from_folder(folder_model)

    # calcular ubicacion enemigos

    with  model_load_from_folder(folder_model) as x:
        t0=time.time()
        bbs=x.predict(img_c,1,0.5)
        tf=time.time()
        print('Elapsed (enemy_clf): {0}'.format(tf-t0))
        for bd in bbs:
            xmin, xmax, ymin, ymax = [bd[x] for x in ['xmin', 'xmax', 'ymin', 'ymax']]
            x2=cv2.rectangle(img, (ymin, xmin), (ymax, xmax), 255, 2)


    one_show(x2)


    # TODO calcular terreno inapasable


    # idea de herramienta dado un path generar un filtro que genere alta activacion
    # idea extrare varias features como morfologia en diversas escalas y elegui cuales son mejroes


if __name__ == "__main__":
    timestamp,img_c = try_to_get_latest()
    pipeline_one_img(img_c)
