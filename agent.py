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
from state_fusion.msg_sockets import socket_read_msg
import random
import logging
import sys
import os
import time

logging.basicConfig(level=logging.DEBUG)

addres = './clean_state'
display=':1'

# set display for guicontrol
os.environ['DISPLAY'] = display
import pyautogui


logging.info('Starting agent in DISPLAY: {0}'.format(os.environ['DISPLAY']))

def bb_to_points(bb_list):
    out=[]
    for p1,p2 in bb_list:
        out.append( ((p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5 ) )
    return out

def point_dist(x,y):
    return (x[0]-y[0])**2+(x[1]-y[1])**2

closest_to_target = None

# read stream of states

for msg in socket_read_msg(addres, verbose=False, skip_if_delay=False):
    print(msg.keys())

    enemys_key = 'bb_bullets'
    # shoot rutine
    if enemys_key in msg:

        bb_enemys = msg[enemys_key]
        if len(bb_enemys) == 0:
            continue

        mean_points = bb_to_points(bb_enemys)
        if closest_to_target is None:
            closest_to_target = random.choice(mean_points)
        else:
            closest_to_target = min(mean_points,key=lambda v : point_dist(closest_to_target,v))


        logging.info('Moving mouse to target: {0}'.format(closest_to_target))
        norm_target = [closest_to_target[1],closest_to_target[0]] # mouse cords are inverted

        x,y = pyautogui.position()

        # move mouse to target
        pyautogui.moveTo(*norm_target)

        # send click if close
        if point_dist((x,y),norm_target):
            pyautogui.click()

        time.sleep(0.1)



    # shoot a random target
