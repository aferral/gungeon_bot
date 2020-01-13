import pickle
import cv2
import numpy as np
from typing import List
from state_fusion.deploy import all_colors
from state_fusion.deploy_utils import video_feed

path_bag = './saved_state_updates/states_2019_Nov_15__16:13.pkl'

# Paso C
# Agente dado estado
# Disparo:
# 	get_or_set_target (elige de los enemigos en vista o usa el que estaba)
# 	calcula vector unitario a tarjet
# 	Enviar mouse a esa coordenada
# 	Si esta dentro de cierto angulo dispara
#
#
# Movimiento:
# 	(Metodo basado en gradientes)
# 	Coloca peaks en cada posicion de enemigo
# 	Dado trayectorias y velocidades de balas. Calcula funcion de influencia.
#	(Ignora a proposito las caidas)
#	Evalua el gradiente local y da paso segun delta
#	Calcula vector de movimiento segun flechas
#
# Dodge:
	# Si ves caida adelante o gradientes de dano muy cerca activa dodge
	# Calcula mov de forma que llege a no caida o no golpe


def get_specific_frame(frame_n,cap):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)

    success_grab = cap.grab()
    if success_grab:
        flag, frame = cap.retrieve()
        return frame
    return None,


"""
- siempre muestra el maximo t_proc
- para cada key muestra el ultimo mensaje

(Sea la lista de entidades)
(Sea una lista historica de tiempos,bbs y trozos)
(Puede ser util trayectoria de disparos)

Meta modelo  (posicion?,area?)
Consigue N bbs que calcula features
Genera match probabilistico de cada entidad con vector
Genera probabilidad de nueva entidad
Genera Actualizacion y desgaste de entidades antiguas

Grafica entidades en modelo.

Algoritmo rectificacion historico

(Se genera una entidad limpia. Posee posicion,velocidad) Si no es continuamente detectada desaparece.

(Que pasa con nueva entidaad) (Crear entidad con posicion, velocidad)
(Que pasa con vieja entidad detectada) (Actualizar entidad,posicion,velocidad)
(Que pasa con vieja entidad no detectada) (Mantener entidad limpia en posicion antigua o desplazar segun velocidad?)
(Que pasa con falso positivo) (vector colores no hace sentido se bota)

Para cada bb calcular histograma de colores
- Buscar match dado posicion, histograma color, tiempo.
- Si no hay match dejar en historico solamente
- Si hay match calcular desplazamiento dado centro de caja
"""

cap = cv2.VideoCapture("./screen_capture/video_data/out.mp4")

keep_for_n_frame = 4
instancias_lista = []

from collections import namedtuple
from scipy.spatial.distance import cdist

class Measure:
    def __init__(self,img,vector,coord,t_proc,timestamp):
        self.img,self.vector,self.coord,self.t_proc,self.timestamp = img,vector,coord,t_proc,timestamp
class Entity:
    def __init__(self,id,frames_till_update,last_t_updated,n_med,measures):
        self.id,self.frames_till_update,self.last_t_updated,self.n_med,self.measures = id,frames_till_update,last_t_updated,n_med,measures



def calc_histogram_vector(img):
    # TODO take center
    chans = cv2.split(img)
    bins = 8
    c_vectors = []
    for c_img in chans:
        channel_hist = cv2.calcHist([c_img], [0], None, [bins], [0, 256])
        c_vectors.append(channel_hist/channel_hist.sum())

    vector_out = np.concatenate(c_vectors,axis=0).T
    return vector_out

class instance_updater:
    def __init__(self,wait_till_delete=4):
        self.all_entitys = [] # Type List[Entity]
        self.current_id = 0
        self.wait_till_delete = wait_till_delete

        self.old_entitys = []

        # Sea una lista de instancias (que se va limpiando segun keep_for_n_frames)
        # sea instancia (id_inst,frames_without_update,t_last_updated, n_med, lista de Medicion(img,vector,posx,posy,tiempo) )

    def calc_features(self,imgs):
        new_vectors = []
        for img in imgs:
            vector = calc_histogram_vector(img)
            new_vectors.append(vector)
        new_matrix = np.concatenate(new_vectors,axis=0)
        return new_matrix

    def calc_feature_dist(self,feature_matrix):

        features_entitys = []
        for ent in self.all_entitys:
            features_entitys.append(ent.measures[-1].vector)
        matrix_entitys = np.array(features_entitys)

        dists = cdist(feature_matrix,matrix_entitys)

        # norm to sum 1 per row
        norm_dist = dists/dists.sum(axis=1).reshape(-1,1)
        return norm_dist


        pass
    def calc_pos_time_dist(self,points,time):
        v0 = 4000 # 4 000 pixels / [s]

        positions = np.array(points)
        times = np.expand_dims(np.repeat(time,positions.shape[0]),-1)
        position_z = times*v0 # TODO posible overflow ??

        pos_matrix = np.concatenate([positions,position_z],axis=1)

        # entitys positions
        old_times = np.array([entity.measures[-1].timestamp for entity in self.all_entitys]).reshape(-1,1)
        old_z = old_times*v0 # TODO posible overflow ??
        old_bbs = [entity.measures[-1].coord for entity in self.all_entitys]
        old_points = np.array([((p0[0] + pf[0]) * 0.5, (p0[1] + pf[1]) * 0.5) for p0, pf in old_bbs])
        pos_olds = np.concatenate([old_points,old_z],axis=1)

        dists = cdist(pos_matrix,pos_olds)

        # norm to sum 1 per row
        norm_dist = dists / dists.sum(axis=1).reshape(-1, 1)
        return norm_dist

    def update_with_measures(self, points, imgs, t_now, t_proc):


        features = self.calc_features(imgs)

        has_elements = len(self.all_entitys) > 0

        # transform bbs to points (take mean)
        if has_elements:
            mean_points = [((p0[0]+pf[0])*0.5,(p0[1]+pf[1])*0.5) for p0,pf in data['coords']]


            dists_feature = self.calc_feature_dist(features)
            dists_point_time = self.calc_pos_time_dist(mean_points, t_now)

            th_dist = 0.15 # TODO th

            # import ipdb;ipdb.set_trace()

            # TODO how to ponderate

            weighted_distance = (dists_feature*0.5+dists_point_time*0.5)

            # calc min per row (per each new measure)
            reduce_min = weighted_distance.min(axis=1)

            # if the closest is closest than th is a match
            matches = weighted_distance.argmin(axis=1)
            lower_than_th = (reduce_min <= th_dist)
        else:
            lower_than_th = [False for x in points]

        # Si la medicion cumple eso se agrega a la instancias actualizando valores
        # Else crear nueva instancia con (t_last=noe, n_med=1, [Instancia]

        for ind,(point,img) in enumerate(zip(points,imgs)):
            newMeasure = Measure(img, features[ind], point,t_proc, t_now)

            if has_elements:
                args = (point,lower_than_th[ind],reduce_min[ind],matches[ind],th_dist)
                print('Measure: {0} is_match: {1} min_dist: {2} min_id: {3} th: {4} '.format(*args))

            if lower_than_th[ind]:    # if measure has match add measure to respective entity
                selected_entity = self.all_entitys[matches[ind]]

                selected_entity.frames_till_update = 0
                selected_entity.n_med += 1
                selected_entity.measures.append(newMeasure)

            else: # else create new entity
                new_id = self.current_id
                self.current_id += 1

                new_entity = Entity(new_id, 0, t_now, 0, [])
                new_entity.n_med += 1
                new_entity.measures.append(newMeasure)

                self.all_entitys.append(new_entity)

        for entity in self.all_entitys:
            entity.frames_till_update += 1

        self.delete_old()


    def delete_old(self): # delete entitys with frames_till_update >= wait_till_delete

        to_delete = []
        for ind,entity in enumerate(self.all_entitys):
            if entity.frames_till_update >= self.wait_till_delete:
                print('Deleting {0}'.format(entity.id))
                to_delete.append(ind)

        entitys_to_keep = []
        for ind in range(len(self.all_entitys)):
            if ind in to_delete:
                self.old_entitys.append(self.all_entitys[ind])
            else:
                entitys_to_keep.append(self.all_entitys[ind])

        self.all_entitys = entitys_to_keep


    def get_last_instances(self,infer_non_updated=False) -> List[Measure]: # TODO infer olds
        out={}
        for ind,entity in enumerate(self.all_entitys): # TODO GIVE ID OF ENTITY AND DRAW
            if entity.frames_till_update == 1:
                out[entity.id] = entity.measures[-1]
        return out

with open(path_bag,'rb') as f:
    data = pickle.load(f)

img_cloned = np.zeros((600,800,3))
wait_time = 0
last_t = -1

updater = instance_updater(wait_till_delete=4)



for msg in data:
    key,data = msg.popitem()

    time_now = data['timestamp']
    frame_str = data['t_proc']
    frame_n = int(float(frame_str.split('_')[0]))

    if frame_n > last_t:
        img_cloned = get_specific_frame(frame_n,cap).copy()
        last_t = frame_n

    if key == 'enemys':
        # update in
        try:
            updater.update_with_measures(data['coords'],data['imgs'],time_now,frame_str)
            measures = updater.get_last_instances()
        except Exception as e:
            raise e
            import ipdb
            ipdb.pm()


        # draw msg bbs TODO USE THINGY ???
        img_cloned_measure = get_specific_frame(frame_n, cap).copy()
        for t in data['coords']:
            x2 = cv2.rectangle(img_cloned_measure, t[0], t[1], (0,0,255), 2)
        cv2.imshow('t2',img_cloned_measure)


        # draw msg bbs
        for k,mt in measures.items():
            points = mt.coord
            cv2.rectangle(img_cloned, points[0], points[1], tuple(all_colors[key]), 2)
            cv2.putText(img_cloned, str(k), points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('t3',img_cloned)
        cv2.waitKey(wait_time)