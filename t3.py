import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time

templates_files = os.listdir('./templates/r2')

img = cv2.imread('./samples/1569689066_76304414.png', cv2.IMREAD_COLOR)

r_factor = 0.5
threshold=0.9
h, w = img.shape[0:2]
new_shape = (int(w * r_factor), int(h * r_factor))  # cv2 require rotated cords for resize
rimg = cv2.resize(img, new_shape)

for t_path in templates_files:
    print(t_path)
    t_full_path = os.path.join('./templates/r2', t_path)

    template = cv2.imread(t_full_path)
    th, tw = template.shape[0:2]

    t0 = time.time()
    res = cv2.matchTemplate(rimg, template, cv2.TM_CCOEFF_NORMED)
    tf = time.time()
    elapsed_v0 = tf - t0
    print('ELAPSED: {0}'.format(elapsed_v0))

    print(res.max())
    tloc = list(zip(*(np.where(res >= threshold)[::-1])))[0]
    sel = (tloc[0], tloc[1]), (tloc[0] + th, tloc[1] + tw)


    print(sel)
    img_t = rimg.copy()
    cv2.rectangle(rimg,sel[1],sel[0],255,2)
    plt.imshow(img_t)
    plt.show()


