"""
toma las imagenes en carpeta del otro proceso
captura diversas caracteristicas del juego
"""

import cv2
import time
import numpy as np
from contextlib import contextmanager

from clf_images.clasify_enemy import model_load_from_folder
from utils_read_imgs import try_to_get_latest

# Idea importante necesito poder crear rapidamente y robustamente. Detectores de diversos patrones

# Idea dar varios patrones y dependiendo de los parametros iniciales que patron busca.
# Idea postproceso busca mantener continuidad de valore

def match_find_multiple(img,template,th=0.89,verbose=False):
    t0=time.time()
    h,w=template.shape[0:2]
    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= th)
    out = [((x,y),(x+h,y+w)) for x,y in zip(*loc[::-1]) ]
    tf=time.time()
    if verbose:
        print('Elapsed {0}'.format(tf-t0))
    return out
def match_find_max(img,template,verbose=False):
    t0 =time.time()
    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    minv,maxv,minl,maxl=cv2.minMaxLoc(res)
    bt=(maxl[0]+template.shape[0],maxl[1]+template.shape[1])
    tf=time.time()
    if verbose:
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

class feature_function:
    def __init__(self,feed_fun=None):
        self.feed_fun = try_to_get_latest if feed_fun is None else feed_fun
        self.use_context=False

    def process(self,img_c = None):
        if img_c is None:
            timestamp, img_c = self.feed_fun()
        else:
            timestamp = -1


        return self.base_process(img_c,timestamp)

    def base_process(self,img_c,timestamp):
        raise NotImplementedError()


class TemplateMatching(feature_function):
    def __init__(self,template_path,**kwargs):
        flag_color = cv2.IMREAD_COLOR
        self.template = cv2.imread(template_path, flag_color)
        super().__init__(**kwargs)


    def base_process(self,img_c,timestamp):
        # calcula balas enemigos
        print('Proc: {0}'.format(timestamp))
        cords = match_find_multiple(img_c, self.template, th=0.7)
        out_dict = {'t_proc': timestamp, 'coords': cords}
        print(out_dict)
        return out_dict

class ClfEnemy(feature_function):
    def __init__(self,model_path,**kwargs):
        self.model_path = model_path
        self.model = None
        super().__init__(**kwargs)
        self.use_context = True

    @contextmanager
    def model_context(self):
        with  model_load_from_folder(self.model_path) as x:
            self.model = x
            yield

    def base_process(self,img_c,timestamp):
        assert(self.model is not None)

        # calcula balas enemigos
        print('Proc: {0}'.format(timestamp))
        t0 = time.time()
        bbs = self.model.predict(img_c, 1, 0.5)
        tf = time.time()
        print('Elapsed (enemy_clf): {0}'.format(tf - t0))
        cords = []
        for bd in bbs:
            xmin, xmax, ymin, ymax = [bd[x] for x in ['xmin', 'xmax', 'ymin', 'ymax']]
            cords.append( ((ymin, xmin), (ymax, xmax)) )

        out_dict = {'t_proc': timestamp, 'coords': cords}
        print(out_dict)
        return out_dict



def pipeline_one_img(img_c):
    flag_color = cv2.IMREAD_COLOR
    flag_gray = cv2.IMREAD_GRAYSCALE

    folder_model = './clf_images/saved_models/clf_enemy/12_Oct_2019__20_51_58'

    calc_bullet_coords = TemplateMatching('./templates/bala_enemy.png')
    calc_HP_coords = TemplateMatching('./templates/full_hp.png')
    calc_mira = TemplateMatching('./templates/t_mira.png')
    calc_blanks = TemplateMatching('./templates/blank.png')
    calc_enemy_pos = ClfEnemy(folder_model)

    img = cv2.cvtColor(img_c,flag_color)


    # calcula mira
    cords = calc_mira.process(img_c)['coords']
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],255,2)


    # calcula HP
    cords = calc_HP_coords.process(img_c)['coords']
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],255,2)


    # calcula blanks
    cords = calc_blanks.process(img_c)['coords']
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],255,2)

    # calcula balas enemigos
    cords = calc_bullet_coords.process(img_c)['coords']
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],255,2)


    # calcular ubicacion enemigos
    with calc_enemy_pos.model_context():
        cords = calc_enemy_pos.process(img_c)['coords']
        for t in cords:
            x2 = cv2.rectangle(img, t[0], t[1], 255, 2)

    one_show(img)


    # TODO calcular terreno inapasable


    # idea de herramienta dado un path generar un filtro que genere alta activacion
    # idea extrare varias features como morfologia en diversas escalas y elegui cuales son mejroes


if __name__ == "__main__":
    #timestamp,img_c = try_to_get_latest()
    timestamp,img_c = '123',cv2.imread('./samples/1569689066_76304414.png',cv2.IMREAD_COLOR)
    pipeline_one_img(img_c)
