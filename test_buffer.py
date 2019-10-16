import os
import cv2
from main import pipeline_one_img


img_f = './samples'
all_f = os.listdir(img_f)
for ff in all_f:
    print(ff)
    x=cv2.imread(os.path.join(img_f,ff))
    pipeline_one_img(x)
