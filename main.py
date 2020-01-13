"""
toma las imagenes en carpeta del otro proceso
captura diversas caracteristicas del juego
"""
import cv2
import time
import numpy as np
from contextlib import contextmanager

from clf_images.clasify_enemy import model_load_from_folder
from clf_images.tflite_utils import export_model_to_tflite, load_tflite
from state_fusion.deploy_utils import do_profile
from utils_read_imgs import try_to_get_latest

def match_find_multiple(img,template,th=0.89):
    h, w = template.shape[0:2]

    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)

    loc = np.where( res >= th)
    out = [((x,y),(x+h,y+w)) for x,y in zip(*loc[::-1]) ]
    return out
def match_find_max(img,template):
    res=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) # cv2.matchTemplate(img_c, img_t, cv2.TM_CCORR)
    minv,maxv,minl,maxl=cv2.minMaxLoc(res)
    bt=(maxl[0]+template.shape[0],maxl[1]+template.shape[1])
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
    def __init__(self,feed_fun=None,send_imgs=False):
        self.feed_fun = try_to_get_latest if feed_fun is None else feed_fun
        self.use_context=False
        self.send_imgs=send_imgs

    def process(self,img_c = None):
        if img_c is None:
            timestamp, img_c = self.feed_fun()
        else:
            timestamp = -1

        out= self.base_process(img_c,timestamp)
        if self.send_imgs and 'coords' in out:
            #(ymin, xmin), (ymax, xmax)
            out['imgs'] = [img_c[p0[1]:p1[1],p0[0]:p1[0]] for p0,p1 in out['coords']]
        return out


    def base_process(self,img_c,timestamp):
        raise NotImplementedError()


class TemplateMatching(feature_function):
    def __init__(self,template_path,resize_factor=-1,**kwargs):
        flag_color = cv2.IMREAD_COLOR
        self.template = cv2.imread(template_path, flag_color)
        self.resize_factor = resize_factor
        super().__init__(**kwargs)


    def base_process(self,img_c,timestamp):
        if self.resize_factor != -1:
            h,w = img_c.shape[0:2]
            new_shape = (int(w*self.resize_factor),int(h*self.resize_factor))
            img_to_use = cv2.resize(img_c,new_shape)
        else:
            img_to_use = img_c
        # calcula balas enemigos
        cords = match_find_multiple(img_to_use, self.template, th=0.9)
        if self.resize_factor != -1:
            r= 1.0/self.resize_factor
            out_dict = {'t_proc': timestamp, 'coords': [tuple([tuple([int(z*r) for z in list(y)]) for y in list(x)]) for x in cords]}
            return out_dict
        else:
            out_dict = {'t_proc': timestamp, 'coords': cords}
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

        bbs = self.model.predict(img_c, 1, 0.5)
        cords = []
        for bd in bbs:
            xmin, xmax, ymin, ymax = [bd[x] for x in ['xmin', 'xmax', 'ymin', 'ymax']]
            cords.append( ((ymin, xmin), (ymax, xmax)) )

        out_dict = {'t_proc': timestamp, 'coords': cords}
        return out_dict

class Tflite_clf(feature_function):
    def __init__(self,model_path,export_folder,build_model=True,**kwargs):
        """
        Use a tensorflow model in tflite format (faster than ClfEnemy).
        :param model_path: Tensorflow model folder used (Same format of ClfEnemy)
        :param export_folder: Folder where to export or load the tflite model
        :param build_model: If True Load (model_path) and save tflite in export_folder. Else just load export_folder
        """
        self.model_path = model_path
        self.exported_folder = export_folder
        super().__init__(**kwargs)

        if build_model:
            export_model_to_tflite(model_path, export_folder)

        self.tflite_predict_fun = load_tflite(export_folder)


    def base_process(self,img_c,timestamp):
        bbs_b, heatmap_b = self.tflite_predict_fun(img_c, 1, 0.5)
        cords = []
        for bd in bbs_b:
            xmin, xmax, ymin, ymax = [bd[x] for x in ['xmin', 'xmax', 'ymin', 'ymax']]
            cords.append( ((ymin, xmin), (ymax, xmax)) )
        out_dict = {'t_proc': timestamp, 'coords': cords}
        return out_dict

def pipeline_one_img(img_c):
    flag_color = cv2.IMREAD_COLOR
    flag_gray = cv2.IMREAD_GRAYSCALE

    folder_model = './clf_images/saved_models/clf_enemy/12_Oct_2019__20_51_58'

    calc_bullet_coords = TemplateMatching('./templates/r2/bala.png',resize_factor=0.5)
    calc_HP_coords = TemplateMatching('./templates/r2/hearth.png',resize_factor=0.5)
    calc_mira = TemplateMatching('./templates/r2/mira.png',resize_factor=0.5)
    calc_blanks = TemplateMatching('./templates/r2/blank.png',resize_factor=0.5)
    calc_enemy_pos = Tflite_clf(folder_model, './deployed_clf', build_model=False)  # use tflite version

    img = cv2.cvtColor(img_c,flag_color)


    #calcula mira 0.07 [s]
    cords = calc_mira.process(img_c)['coords']
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],(0,255,255),2)


    # calcula HP
    cords = calc_HP_coords.process(img_c)['coords']
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],(0,255,0),2)

    # calcula blanks
    cords = calc_blanks.process(img_c)['coords']
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],(255,0,0),2)


    # calcula balas enemigos
    cords = calc_bullet_coords.process(img_c)['coords']
    for t in cords:
        x2=cv2.rectangle(img,t[0],t[1],(255,0,255),2)

    #calcular ubicacion enemigos 0.051[s]
    cords = calc_enemy_pos.process(img_c)['coords']
    for t in cords:
        x2 = cv2.rectangle(img, t[0], t[1], (0,0,255), 2)


    one_show(img)


    # idea de herramienta dado un path generar un filtro que genere alta activacion
    # idea extrare varias features como morfologia en diversas escalas y elegui cuales son mejroes


if __name__ == "__main__":
    #timestamp,img_c = try_to_get_latest()
    timestamp,img_c = '123',cv2.imread('./samples/1569689066_76304414.png',cv2.IMREAD_COLOR)
    pipeline_one_img(img_c)
