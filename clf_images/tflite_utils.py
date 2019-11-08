import random
import os
import numpy as np
import tensorflow as tf
import time
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from clf_images.clf_utils import from_act_to_bb
from clf_images.clasify_enemy import model_load_from_folder
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import shutil

from state_fusion.deploy_utils import do_profile


def load_tflite(exported_model):
    mean_img = np.load(os.path.join(exported_model,'mean_img.npy'))

    # Load TFLite model and allocate tensors.
    conv=tf.lite.TFLiteConverter.from_saved_model(exported_model,tag_set=['exported'],signature_key='predict')
    temp = conv.convert()
    interpreter = tf.lite.Interpreter(model_content=temp)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_data = [x for x in input_details if x['name'] == 'model_input'][0]
    mean_data = [x for x in input_details if x['name'] == 'mean'][0]

    # Test model on random input data.
    input_shape = img_data['shape']

    interpreter.set_tensor(mean_data['index'],mean_img) # set mean img
    w, h = input_shape[1:3]

    input_pointer = interpreter.tensor(img_data['index'])

    def predict(img,i_label,pred_th):
        interp_method = cv2.INTER_AREA
        img_res = cv2.resize(img, (h, w),interpolation=interp_method)
        input_pointer()[0] = img_res.astype(np.float32)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        high_res = cv2.resize(output_data[0, :, :, i_label], tuple(reversed(img.shape[0:2])),interpolation=interp_method)
        bbs = from_act_to_bb(high_res, pred_th)
        return bbs,high_res

    return predict

def export_model_to_tflite(saved_model_folder,out_folder='deployed_clf'):
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    builder = tf.saved_model.builder.SavedModelBuilder(out_folder)
    mean_img = None
    with  model_load_from_folder(saved_model_folder) as x:
        mean_img = np.expand_dims(x.mean_img,axis=0).astype(np.float32)
        signature = predict_signature_def(inputs={'in': x.input_layer,'mean':x.mean_value},
                                          outputs={'out': x.pred})
        builder.add_meta_graph_and_variables(x.sess, ['exported'], signature_def_map={'predict': signature})
        builder.save()

    out_path_mean = os.path.join(out_folder,'mean_img.npy')
    np.save(out_path_mean,mean_img)
    return out_folder


def cmp_with_full(model_folder,tflite_folder):
    # open random imgs
    img_folder = './clf_images/data/images'
    n_to_test = 10


    file_list = os.listdir(img_folder)
    file_list = random.choices(file_list,k=n_to_test)

    tf_slim_predict = load_tflite(tflite_folder)

    def draw_bbs(img,bbs):
        img_t = np.copy(img)
        for bd in bbs:
            xmin, xmax, ymin, ymax = [bd[x] for x in ['xmin', 'xmax', 'ymin', 'ymax']]
            cv2.rectangle(img_t, (ymin, xmin), (ymax, xmax), 255, 2)
        return img_t

    with  model_load_from_folder(model_folder) as x:

        for img_path in file_list:
            f_path = os.path.join(img_folder,img_path)
            img_raw=cv2.imread(f_path)

            # predict with full model
            bbs_a,heatmap_a = x.predict_with_heatmap(img_raw, 1, 0.5)

            # predict with slim
            bbs_b, heatmap_b = tf_slim_predict(img_raw,1,0.5)

            print('BBs_full: {0} BBs_lite: {1}'.format(bbs_a,bbs_b))

            f,axs=plt.subplots(2,2,figsize=(20,20))
            axs[0,0].imshow(heatmap_a)
            axs[0,0].set_title('FULL')
            axs[0,1].imshow(heatmap_b)
            axs[0,1].set_title('LITE')

            axs[1,0].imshow(draw_bbs(img_raw,bbs_a))
            axs[1,0].set_title('FULL_bbs')
            axs[1,1].imshow(draw_bbs(img_raw,bbs_b))
            axs[1,1].set_title('LITE_bbs')

            plt.show()


def test_speed(tflite_folder):
    n_to_test = 10
    st=time.time()
    tf_slim_predict = load_tflite(tflite_folder)
    en=time.time()
    print('Init time: {0}'.format(en-st))

    t0=time.time()
    raw_img = np.random.rand(600,800,3).astype(np.uint8)
    for i in range(n_to_test):
        # predict with slim
        bbs_b, heatmap_b = tf_slim_predict(raw_img,1,0.5)
    tf=time.time()
    elap = tf-t0
    mean_t = elap*1.0/n_to_test
    print('Elapsed time: {0} MEAN TIME: {1}'.format(elap,mean_t))


if __name__ == '__main__':
    folder_model = './clf_images/saved_models/clf_enemy/12_Oct_2019__20_51_58'
    mode=2

    if mode == 1:
        out_folder = 'tmp_tflite'
        tf_lite_folder = export_model_to_tflite(folder_model,out_folder)
        cmp_with_full(folder_model,tf_lite_folder)
        shutil.rmtree(tf_lite_folder)

    elif mode == 2:
        out_folder = 'tmp_tflite'
        tf_lite_folder = export_model_to_tflite(folder_model,out_folder)
        test_speed(tf_lite_folder)
        shutil.rmtree(tf_lite_folder)

