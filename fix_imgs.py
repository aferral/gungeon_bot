import cv2
import os
import sys

folder = sys.argv[1]
print('Arreglando archivos en {0}')

a_files = os.listdir(folder)

for e in a_files:
    if e.split('.')[-1] == 'png':
        fp = os.path.join(folder,e)
        x=cv2.imread(fp)
        cv2.imwrite(fp,x)
