import os
import cv2
from main import pipeline_one_img
from utils_read_imgs import try_to_get_latest
from utils_read_imgs import suppress_stdout_stderr

img_f = './samples'
all_f = os.listdir(img_f)
for ff in all_f:
    print(ff)
    with suppress_stdout_stderr():
        x=cv2.imread(os.path.join(img_f,ff))
        timestamp,img_c = try_to_get_latest()
        #timestamp,img_c = '123',cv2.imread('./samples/1569689066_76304414.png',cv2.IMREAD_COLOR)
        cv2.imshow('t1',img_c)
        cv2.waitKey(1)
