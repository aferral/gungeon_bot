import inspect
from contextlib import contextmanager
import shutil
from tensorflow.python.client import device_lib
import numpy as np
import cv2
import time
import json
from contextlib import ExitStack
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.transform import resize

from skimage.morphology import dilation
from skimage.morphology import square

import matplotlib.pyplot as plt
from clf_images.clf_utils import parse_all_xmls, bb_to_binary, from_act_to_bb, calc_relev, optimistic_restore, \
    show_graph, now_string

# STATIC PARAMS
pred_th = 0.5
IOU_th = 0.3
split_params = {'random_state' : 42,'test_size' : 0.1}
original_img_shape = [600,800,3]

# define lista de tags a utiliar
labels_to_use = ['bullet_boss', 'b0', 'bullet_adv', 'speed_bullet', 'grenate', 'book', 'aly_dog', 'wizard_0', 'cube', 'ghost', 'big_red', 'spider', 'player_0']
#labels_to_use = ['book','b0','player_0','aly_dog','ghost','grenate','spider']
#labels_to_use = ['player_0','aly_dog']

# define map de tags a labels target numerico
label_to_target = {x : i for i,x in enumerate(labels_to_use)}
label_to_target = {x : int(x == 'player_0') for i,x in enumerate(labels_to_use)}

clasf_model_folder = 'clf_images/saved_models'



def plot_predict_class(img, pred, loss_term, i_class, pred_th, y_data, y_rel, out_name=None):
    # img es float de 0 a 255. Algunas funciones requieren valores 0-1
    img_float = img / 255

    w, h = img.shape[0:2]
    cmap = lambda x, y: (cv2.applyColorMap(((1 - x) * 255).astype(np.uint8), y))
    high_res = resize(pred[:, :, i_class], (w, h))


    n_cols = 2 if ((y_data is None) and (y_rel is None)) else 3

    f0, axs = plt.subplots(1, n_cols)

    # SHOW LOW RES
    pred_low_res = pred[:, :, i_class]
    bin_low_res = pred_low_res > pred_th

    axs[0].set_title('low_res_map')
    axs[0].imshow(pred_low_res)
    axs[1].set_title('low_res_bin')
    axs[1].imshow(bin_low_res)

    if (y_data is not None) and (y_rel is not None):
        rel_and_target = y_data[:, :, i_class]
        axs[2].set_title('low_res_target_and_rel {0}'.format(loss_term))
        axs[2].imshow(rel_and_target)

    # SHOW low res overlay
    f1, axs = plt.subplots(1, n_cols)
    img_low_overlay = (resize(img_float, pred_low_res.shape) * 255).astype(np.uint8)

    w0 = cv2.addWeighted(img_low_overlay, 0.5, cmap(pred_low_res, cv2.COLORMAP_JET), 0.5, 0.0)
    w1 = cv2.addWeighted(img_low_overlay, 0.5, cmap(bin_low_res, cv2.COLORMAP_JET), 0.5, 0.0)
    axs[0].set_title("LOW_res_map_overlay")
    axs[0].imshow(w0)
    axs[1].set_title("LOW_res_bin_overlay")
    axs[1].imshow(w1)
    # if y_data show
    if (y_data is not None) and (y_rel is not None):
        w0 = cv2.addWeighted(img_low_overlay, 0.5, cmap(y_data[:, :, i_class], cv2.COLORMAP_JET), 0.5, 0.0)
        w0 = cv2.addWeighted(w0, 0.5, cmap(y_rel[:, :, i_class], cv2.COLORMAP_PINK), 0.5, 0.0)
        axs[2].set_title('LOW_res_target_and_rel')
        axs[2].imshow(w0)

    # SHOW HIGH RES OVERLAY
    f2, axs = plt.subplots(1, n_cols)
    w0 = cv2.addWeighted(img.astype(np.uint8), 0.5, cmap(high_res, cv2.COLORMAP_JET), 0.5, 0.0)
    high_res_bin = high_res > pred_th
    w1 = cv2.addWeighted(img.astype(np.uint8), 0.5, cmap(high_res_bin, cv2.COLORMAP_JET), 0.5, 0.0)
    axs[0].set_title("high_res_map_overlay")
    axs[0].imshow(w0)
    axs[1].set_title("high_res_bin_overlay")
    axs[1].imshow(w1)

    # if y_data show
    if (y_data is not None) and (y_rel is not None):
        y_data_resized = resize(y_data[:, :, i_class], (w, h))
        rel_resized = resize(y_rel[:, :, i_class], (w, h))
        w0 = cv2.addWeighted(img.astype(np.uint8), 0.5, cmap(rel_resized, cv2.COLORMAP_PINK), 0.5, 0.0)
        w0 = cv2.addWeighted(w0, 0.5, cmap(y_data_resized, cv2.COLORMAP_JET), 0.5, 0.0)
        axs[2].set_title('high_res_target_and_rel')
        axs[2].imshow(w0)

    # show resulting bbs in GREEN pred
    bbs = from_act_to_bb(high_res_bin, pred_th)
    img_cloned = np.copy(img)
    for bd in bbs:
        xmin, xmax, ymin, ymax = [bd[x] for x in ['xmin', 'xmax', 'ymin', 'ymax']]
        cv2.rectangle(img_cloned, (ymin, xmin), (ymax, xmax), (0, 255, 0), 2)

    # add target bbs
    if (y_data is not None) and (y_rel is not None):
        y_data_resized = resize(y_data[:, :, i_class], (w, h))
        bbs = from_act_to_bb(y_data_resized, pred_th)
        for bd in bbs:
            xmin, xmax, ymin, ymax = [bd[x] for x in ['xmin', 'xmax', 'ymin', 'ymax']]
            cv2.rectangle(img_cloned, (ymin, xmin), (ymax, xmax), (0, 0, 255), 2)
    f3=plt.figure()
    plt.title('Bounding boxes')
    plt.imshow(img_cloned/255)

    if out_name is None:
        plt.show()
    else:
        f0.savefig("{0}_c_{1}_low_res.png".format(out_name,i_class),bbox_inches='tight')
        f1.savefig("{0}_c_{1}_low_res_overlay.png".format(out_name,i_class),bbox_inches='tight')
        f2.savefig("{0}_c_{1}_high_res_overlay.png".format(out_name,i_class),bbox_inches='tight')
        f3.savefig("{0}_c_{1}_bbs.png".format(out_name,i_class),bbox_inches='tight')

    plt.close('all')

class clf_base(ExitStack):
    def __init__(self,model_prefix='clf_enemy',tf_config=None,
                 max_pool_layers=2,stack_layers=2,input_factor = 0.5,extra_porc=2,lr=0.01,
                 st_filter=40,inc_filter=50,loss_string='cross_entropy',replace_max_pool_with_stride=False):
        super().__init__()
        self.tf_config = tf_config

        self.model_prefix = model_prefix

        # guarda todos los parametros del init para describir el experimento
        temp=locals()
        self.arg_dict = {str(arg) : str(temp[arg]) for arg in inspect.signature(self.__init__).parameters }

        # MODEL PARAMS
        self.stack_layers = stack_layers

        self.extra_porc = extra_porc
        self.lr = lr
        self.st_filter = st_filter
        self.inc_filter = inc_filter
        self.loss_string = loss_string
        self.max_pool_layers = max_pool_layers
        self.replace_max_pool_with_stride = replace_max_pool_with_stride

        self.input_factor = input_factor
        self.bs = 32


        # parametros calculados
        self.n_labels = len(set(label_to_target.values()))
        assert(self.stack_layers == max_pool_layers or (max_pool_layers == 0))
        self.shape_input_img = [int(x*self.input_factor) for x in original_img_shape[0:2]]+[original_img_shape[2]]



        def calc_mask_shape(replace_max_pool_with_stride,max_pool_layers,shape_input):
            def same_conv_form(in_d,stride):
                return int(np.ceil(float(in_d) / float(stride)))

            w, h = shape_input
            output_factor = 0.5 ** max_pool_layers

            if replace_max_pool_with_stride is False:
                new_shape = [int(w*output_factor), int(h*output_factor)]
            else:
                for i in range(max_pool_layers):
                    w=same_conv_form(w,2)
                    h=same_conv_form(h,2)
                new_shape = [w,h]

            return output_factor,new_shape


        self.output_factor,self.mask_shape = calc_mask_shape(replace_max_pool_with_stride,max_pool_layers,self.shape_input_img[0:2])

        self.relevant_layer = tf.compat.v1.placeholder(tf.float32, shape=[None] + self.mask_shape + [self.n_labels],
                                             name='relevant_layer')

        self.input_layer = tf.compat.v1.placeholder(tf.float32, shape=[None] + self.shape_input_img, name='model_input')
        self.mean_value = tf.compat.v1.placeholder(tf.float32, shape=[1] + self.shape_input_img, name='mean')

        self.process_input = (self.input_layer - self.mean_value) / 255

        self.targets = tf.compat.v1.placeholder(tf.float32, shape=[None] + self.mask_shape + [self.n_labels], name='target')


    def load_data(self,data_folder):

        # Carga de datos
        xml_data,label_set= parse_all_xmls(data_folder)

        # abrir datos crearn matrices binarias de N_label canales
        all_x = []
        all_y = []
        file_names = []
        all_rel_areas=[]

        for (img_path,labels_tuples) in xml_data:
            img = cv2.imread(img_path)
            img = self.img_preprocess(img)
            target_matrix = np.zeros(list(img.shape[0:2])+[self.n_labels],dtype=np.bool)
            relevant_matrix = np.zeros(list(img.shape[0:2])+[self.n_labels],dtype=np.bool)

            # Los labels que no esten en labels_to_use se saltan
            filter_tuples = list(filter(lambda x : x[0] in labels_to_use,labels_tuples))
          
            for label_str,xmin,xmax,ymin,ymax in filter_tuples:
                target = label_to_target[label_str]
                target_matrix[ymin:ymax,xmin:xmax,target] = 1
            
                # calc rel area
                xmin_r,xmax_r,ymin_r,ymax_r = calc_relev(xmin, xmax, ymin, ymax, self.extra_porc)
                relevant_matrix[ymin_r:ymax_r,xmin_r:xmax_r,target] = 1


            # resize images
            if self.input_factor != 1:
                new_shape = (int(img.shape[0]*self.input_factor),int(img.shape[1]*self.input_factor))
                resized_img = resize(img,new_shape)

            # resize and process target
            if self.output_factor != -1:
                new_shape = self.mask_shape
                resized_target_matrix = np.zeros(new_shape+[self.n_labels])
                resized_rel_matrix = np.zeros(new_shape + [self.n_labels])
                for n_c in range(self.n_labels):

                    # dilate till values appear
                    dilate_n = 5
                    should_expand = target_matrix[:,:,n_c].any()
                    while should_expand and (not resized_target_matrix[:,:,n_c].any()):
                        dilated_target = dilation(target_matrix[:,:,n_c], square(dilate_n) )
                        resized_target_matrix[:,:,n_c]=resize(dilated_target,new_shape,anti_aliasing=False) 
                        dilate_n += 1

                    # dilate till values appear
                    dilate_n = 5
                    should_expand = relevant_matrix[:,:,n_c].any()
                    while should_expand and (not resized_rel_matrix[:,:,n_c].any()):
                        dilated_target = dilation(relevant_matrix[:,:,n_c], square(dilate_n) )
                        resized_rel_matrix[:,:,n_c]=resize(dilated_target,new_shape,anti_aliasing=False)
                        dilate_n += 1


            for i in range(self.n_labels):
                assert(resized_target_matrix[:,:,i].any() == target_matrix[:,:,i].any())
                assert(resized_rel_matrix[:,:,i].any() == relevant_matrix[:,:,i].any())

            all_x.append(resized_img)
            all_y.append(resized_target_matrix > 0)
            
            file_names.append(img_path)
            all_rel_areas.append(resized_rel_matrix > 0)
        
        all_x = np.stack(all_x,axis=0)
        all_y = np.stack(all_y,axis=0)
        file_names = np.array(file_names)
        all_rel_areas = np.array(all_rel_areas)

        # calcula mean image
        if (not hasattr(self, 'mean_img')) or (self.mean_img is None):
            # SOLAMENTE SI YA NO ESTABA CARGADA UNA MEAN IMAGE
            self.mean_img = np.mean(all_x,axis=0)


        # realiza split train,val,test
        x_trv, x_test, y_trv, y_test, files_trv, files_test, rel_ar_trv,rel_ar_test = train_test_split(all_x, all_y,file_names,all_rel_areas, **split_params)
        x_train,x_val,y_train,y_val, files_train, files_val, rel_ar_train,rel_ar_val = train_test_split(x_trv,y_trv, files_trv,rel_ar_trv, **split_params)
       
       
        self.train_data = (x_train,y_train,files_train,rel_ar_train)
        self.test_data = (x_test,y_test,files_test,rel_ar_test)
        self.val_data = (x_val,y_val,files_val,rel_ar_val)





    def prepare_feed_img(self,img):
        img_reshaped = img if len(img.shape) == 4 else np.expand_dims(img,0)
        assert(len(img_reshaped.shape) == 4)

        out_dict = {}
        out_dict[self.input_layer] = img_reshaped
        out_dict[self.mean_value] = np.expand_dims(self.mean_img,axis=0)
        return out_dict

    def whole_dataset_generator(self,data_tuple,batch_size):
        c = 0
        max_n  = data_tuple[0].shape[0]
        data_x,data_y,file_list,data_matrix_rel = data_tuple
        while c < max_n:
            selected = [data_x[c:c+batch_size],data_y[c:c+batch_size],file_list[c:c+batch_size],data_matrix_rel[c:c+batch_size]]
            c += batch_size
            yield self._format_to_feed(*selected)



    def _format_to_feed(self,data_x,data_y,file_list,data_matrix_rel):
        out_dict = {}
        out_dict['model_input:0'] = data_x
        out_dict[self.targets] = data_y
        out_dict[self.mean_value] = np.expand_dims(self.mean_img,axis=0)
        out_dict[self.relevant_layer] = data_matrix_rel
        
        index_list = file_list
        return out_dict,index_list

    def prepare_feed(self,batch_size,data='train'):
        if data == 'train':
            data_x,data_y,file_list,data_matrix_rel = self.train_data
        elif  data == 'val':
            data_x,data_y,file_list,data_matrix_rel = self.val_data
        elif data == 'test':
            data_x,data_y,file_list,data_matrix_rel = self.test_data
        else:
            raise Exception('Data no valida {0}'.format(data))

        sel=np.random.choice(data_x.shape[0], batch_size)
        return self._format_to_feed(data_x[sel],data_y[sel],file_list[sel],data_matrix_rel[sel])

    def img_preprocess(self,img):
        assert(img.dtype == np.uint8),'Img entrada debe ser uint8 rango 0 a 255'

        # quita arma y contadore de vida
        assert (img.shape == (600, 800, 3))
        img[450:520, 690:800] = 0
        img[80:150, 0:100] = 0

        return img.astype(np.float32)

    def predict(self,img,target_label,pred_th):
        return self.predict_with_heatmap(img,target_label,pred_th)[0]

    def predict_with_heatmap(self,img,target_label,pred_th):
        w,h = img.shape[0:2]
        img_f = self.img_preprocess(img)
        target_shape = [1]+self.mask_shape+[self.n_labels]

        if self.input_factor != 1:
            new_shape = (int(img_f.shape[0]*self.input_factor),int(img_f.shape[1]*self.input_factor))
            t = resize(img_f,new_shape)
            res_img=t
        else:
            res_img = img_f


        fd,index_list = self._format_to_feed(np.expand_dims(res_img,0), np.zeros(target_shape), [], np.zeros(target_shape))

        pred,loss_term = self.sess.run([self.pred,self.loss],fd)
        pred= pred[0,:,:,target_label]

        high_res = resize(pred, (w, h))
        bbs = from_act_to_bb(high_res, pred_th)
        return bbs,high_res



    def eval_with_img(self, img : np.ndarray, y_data=None, y_rel=None, out_name=None):

        # normaliza la imagen a float32
        img_f = self.img_preprocess(img)

        if self.input_factor != 1:
            new_shape = (int(img_f.shape[0]*self.input_factor),int(img_f.shape[1]*self.input_factor))
            t = resize(img_f,new_shape)
            res_img=t
        else:
            res_img = img_f

        data_x = np.expand_dims(res_img,0)
        shape_y = self.mask_shape
        shape_y = tuple([1]+shape_y+[self.n_labels])
        data_y = np.zeros(shape_y) if y_data is None else np.expand_dims(y_data,0)
        file_list = []
        data_matrix_rel = np.zeros_like(data_y) if y_data is None else np.expand_dims(y_rel,0)
        fd,index_list = self._format_to_feed(data_x, data_y, file_list, data_matrix_rel)

        pred,loss_term = self.sess.run([self.pred,self.loss],fd)
        pred= pred[0]


        n_classes = pred.shape[2]

        # Visual comparision ()
        for i in range(n_classes):
            plot_predict_class(img_f, pred, loss_term, i, pred_th, y_data, y_rel,out_name=out_name)

        return 


    def val_eval(self):
        n_samples_val = 5
        samples_loss_val = []
        for i_val in range(n_samples_val):
            fd_val, file_list = self.prepare_feed(self.bs, 'val')
            loss_val = self.sess.run(self.loss, fd_val)
            samples_loss_val.append(loss_val)
        mean_val_loss = np.array(samples_loss_val).mean()
        print('Sampled loss in val ({0} batches) : {1}'.format(n_samples_val, mean_val_loss))
        return mean_val_loss

    def val_eval_and_save(self, saver, temp_folder, best_mean_val_loss):
        mean_val_loss = self.val_eval()

        if (best_mean_val_loss > mean_val_loss):  # keep best model given validation (save in temporal folder)
            self.save_model(saver,out_path=temp_folder,prefix='val_best')
            best_mean_val_loss = mean_val_loss

        return best_mean_val_loss,mean_val_loss


    def train(self,iterations,save=False):

        best_mean_val_loss = 99999
        iters_till_eval = 50
        iters_till_print = 10

        saver = tf.compat.v1.train.Saver()
        all_train_batchs_loss = []
        all_val_batchs_loss = []

        # create temp folder for checkpoints
        temp_folder = 'temp'
        os.makedirs(temp_folder,exist_ok=True)

        for iteration in range(iterations):

            # get batch
            fd,index_list = self.prepare_feed(self.bs) 
            l, _= self.sess.run([self.loss, self.train_step], fd)
            all_train_batchs_loss.append(l)

            if iteration % iters_till_print == 0:
                log ="It: {}, loss_batch: {:.3f}".format(iteration, l)
                print(log)

            if iteration % iters_till_eval == 0: # check eval performance, only save the model if improves in val set
                best_mean_val_loss, mean_val_loss  = self.val_eval_and_save(saver, temp_folder, best_mean_val_loss)
                all_val_batchs_loss.append(mean_val_loss)

        # last eval check
        best_mean_val_loss, mean_val_loss = self.val_eval_and_save(saver, temp_folder, best_mean_val_loss)
        all_val_batchs_loss.append(mean_val_loss)

        # LOAD best model acoording to val set
        self.load_weights(temp_folder)

        print('LAST VAL {0}'.format(self.val_eval()))
        mean_loss_test = self.eval_test_set()
        print("Train ended loss at test: {0}".format(mean_loss_test))

        # borra checkpoints temporales
        shutil.rmtree(temp_folder)

        if save:
            self.save_model(saver) # save model in corresponding folder
            # save loss plot
            out_path_train_loss = os.path.join(self.get_model_out_folder(),'train_loss.txt')
            out_path_val_loss = os.path.join(self.get_model_out_folder(), 'val_loss.txt')
            np.savetxt(out_path_train_loss, all_train_batchs_loss)
            np.savetxt(out_path_val_loss, all_val_batchs_loss)


    def eval_test_set(self):
        # itera el test set
        all_test_loss = []
        for fd,index_list in self.whole_dataset_generator(self.test_data,self.bs):
            loss_test = self.sess.run(self.loss,fd)
            all_test_loss.append(loss_test)
        all_test_loss = np.array(all_test_loss)
        return all_test_loss.mean()



    def __enter__(self):
        self.graph = tf.compat.v1.get_default_graph()

        self.sess = tf.compat.v1.Session(config=self.tf_config)

        self.enter_context(self.sess.as_default())

        super().__enter__()

        #Build network
        self.define_arch_base()

        return self


    def get_model_out_folder(self):
        if (not hasattr(self,'model_out_folder')) or (self.model_out_folder is None):
            now = now_string()
            path_model_checkpoint = os.path.join(clasf_model_folder, self.model_prefix, now)
            os.makedirs(path_model_checkpoint, exist_ok=True)
            self.model_out_folder = path_model_checkpoint
        return self.model_out_folder

    def save_model(self,saver,out_path=None,prefix='saved_model'):
        if out_path :
            out_folder = out_path
        else:
            out_folder = self.get_model_out_folder()

        # guarda label used,label map
        labels_and_names = "\n".join(["{0} : {1}".format(x,y) for x,y in label_to_target.items()])
        out_labels = os.path.join(out_folder,'labels_map.txt')
        with open(out_labels,'w') as f:
           f.write(labels_and_names)

        # guarda mean
        np.save(os.path.join(out_folder,'mean.npy'),self.mean_img)

        # guarda todos ls parametros utilizados en el init
        out_params = os.path.join(out_folder,'used_params.json')
        with open(out_params,'w') as f:
            f.write(json.dumps(self.arg_dict, indent=4, sort_keys=True))

        # guarda pesos y grafo
        print("Saving model at {0}".format(out_folder))
        saver.save(self.sess,os.path.join(out_folder, prefix))

        return out_folder


    def load_weights(self,model_folder):
        assert(os.path.isdir(model_folder))
        checkpoint_path = tf.train.latest_checkpoint(model_folder)
        print("Using Weights: {0}".format(checkpoint_path))
        # carga pesos y grafo
        optimistic_restore(self.sess, checkpoint_path )
        show_graph(tf.compat.v1.get_default_graph())

    def load_full(self, model_folder):
        assert(os.path.isdir(model_folder))
        self.model_out_folder = model_folder

        self.load_weights(model_folder)
        # carga mean
        self.mean_img = np.load(os.path.join(self.model_out_folder,'mean.npy'))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()
        super().__exit__(exc_type, exc_val, exc_tb)
    

    def define_arch_base(self):

        n_labels = self.n_labels 

        input_current_layer = self.process_input
        for i in range(self.stack_layers):

            n_filters = self.st_filter + self.inc_filter *i
            conv_out = tf.keras.layers.Conv2D(n_filters, (3, 3), padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())(input_current_layer)
            if self.max_pool_layers > 0:
                if self.replace_max_pool_with_stride:
                    pool = tf.keras.layers.Conv2D(n_filters,(3, 3),strides=(2,2),padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)(conv_out)
                else:
                    pool = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(conv_out)
                input_current_layer = pool
            else:
                input_current_layer = conv_out

        
        conv3 = tf.keras.layers.Conv2D(n_labels, (3, 3), padding='same', activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer()
                                 ,name='conv_clasf')(input_current_layer)


        def out_mse(last_layer,targets, relevant_layer):
            last_layer_act = tf.keras.layers.Activation(activation='sigmoid')(last_layer)
            elem_dif = tf.identity((last_layer_act - targets) ** 2, 'elem_dif')
            masked_dif = tf.identity(relevant_layer * elem_dif, 'masked_dif')
            masksum = tf.identity(tf.reduce_sum(masked_dif,axis=(1,2)),'masksum')
            nmask= tf.reduce_sum(relevant_layer, axis = (1,2)) + 0.001 # (evita problemas al existir 0)
            mean_mask = tf.identity(masksum / nmask,'mean_mask')
            loss_t =tf.reduce_mean(mean_mask)
            return loss_t
        def out_cross_entropy(last_layer,targets,relevant_layer):
            loss_t = tf.compat.v1.losses.sigmoid_cross_entropy(
                targets,
                last_layer,
                weights=relevant_layer,
                label_smoothing=0)
            return loss_t



        if self.loss_string == 'mse':
            loss_t = out_mse(conv3,self.targets,self.relevant_layer)
        elif self.loss_string == 'cross_entropy':
            loss_t = out_cross_entropy(conv3,self.targets, self.relevant_layer)
        else:
            raise Exception('LOSS STRING NOT VALID {0}'.format(self.loss_string))

        # Configure values for visualization
        self.pred = tf.keras.layers.Activation(activation='sigmoid')(conv3)
        self.loss = loss_t
        

        self.train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)

    def evaluate_model(self,save_out=False):

        # open test images 
        (x_test,y_test,files_test,relevant_area_test) = self.test_data
       
        full_dataset_tp = []
        full_dataset_predicted = []
        full_dataset_real = []
        full_dataset_ios = []

        for img,f_name,target in zip(x_test,files_test,y_test):
            n_c = target.shape[-1]
            
            # open original img
            original_img = cv2.imread(f_name).astype(np.float32)
            original_size = original_img.shape[0:2]
        
            # generar bounding box target
            per_class_target_bbs = {}
            per_class_target_binary = {}
            for i in range(n_c):
                resized_target = resize(target[:,:,i],original_size)
                per_class_target_bbs[i] = from_act_to_bb(resized_target, 0)
                per_class_target_binary[i] = [bb_to_binary(x, original_size) for x in per_class_target_bbs[i]]

            fd=self.prepare_feed_img(img)
            predicted_values = self.sess.run(self.pred, fd)[0]

            
            tp={i : 0 for i in range(n_c)}
            all_predicted = {i : 0 for i in range(n_c)}
            all_real = {i : len(per_class_target_bbs[i]) for i in range(n_c)}
            all_ios = {i : [] for i in range(n_c)}

            # por cada clase calcula bounding boxes
            for i in range(n_c):
                resized_pred = resize(predicted_values[:,:,i],original_size)
                
                # generate bbs
                bbs= from_act_to_bb(resized_pred, pred_th)
                binary_bbs = [bb_to_binary(x, original_size) for x in bbs]

                # por cada bounding box calcula su mejor IOU
                all_matchs = []
                for bb_current,bbb in zip(bbs,binary_bbs):

                    # calc IOU iterando las bounging box de target
                    best_iou = 0
                    tp_bool = False
                    match_data = (best_iou, 0,0, bbb,None,tp_bool)

                    if len(per_class_target_binary[i]) > 0:
                        all_intersections = np.array([np.sum(np.bitwise_and(bbb,binary_bb_real )) for binary_bb_real in per_class_target_binary[i] ])
                        all_unions = ([np.sum(bbb)+np.sum(binary_bb_real) for binary_bb_real in per_class_target_binary[i] ])
                        all_ious = all_intersections/all_unions
                    
                    
                        best_iou_index = np.argmax(all_ious)
                        best_iou = all_ious[best_iou_index] 
                        best_inter = all_intersections[best_iou_index] 
                        best_union = all_unions[best_iou_index] 
                        best_bb_target = per_class_target_bbs[i][best_iou_index] 

                        # define si fue tp o fp
                        if best_iou >= IOU_th: 
                            tp[i] += 1
                            tp_bool = True

                        match_data = (best_iou, best_inter,best_union, bb_current, best_bb_target,tp_bool)

                    all_matchs.append(match_data)
                    all_ios[i].append(best_iou)
                    all_predicted[i] += 1 
            
                total_tp = sum([x[-1] for x in all_matchs])
                #print("Total tp: {0}".format(total_tp))
                #print('Max IOU : {0}'.format(max(all_matchs,key=lambda x : x[0])[0]))
                ltargets = len(per_class_target_binary[i])
                lpreds = len(all_matchs)

                local_recall = total_tp/ltargets if ltargets > 0 else 0
                local_precission = total_tp / lpreds if lpreds > 0 else 0

                #print('local recall: {0} local precission: {1} '.format(local_recall,local_precission))


            full_dataset_tp.append(tp)
            full_dataset_predicted.append(all_predicted)
            full_dataset_real.append(all_real)
            full_dataset_ios.append(all_ios)


        # calcula_prec, calcula recall , lista IOU
        out_dict = {} 

        all_tp = 0 
        all_predicted = 0
        all_real = 0
        all_iou = []
        for c in range(n_c):
            all_tp_this_c = sum([x[c] for x in full_dataset_tp ]) 
            all_tp += all_tp_this_c
            
            all_predicted_this_c = sum([x[c] for x in full_dataset_predicted])
            all_predicted += all_predicted_this_c

            all_real_this_c = sum([x[c] for x in full_dataset_real])
            all_real += all_real_this_c

            all_iou_this_c = sum([x[c] for x in full_dataset_ios],[])
            all_iou += all_iou_this_c

            out_dict['{0}_precision'.format(c)] = all_tp_this_c / all_predicted_this_c if all_predicted_this_c != 0 else 0
            out_dict['{0}_recall'.format(c)] = all_tp_this_c / all_real_this_c if all_real_this_c != 0 else 0
            out_dict['{0}_IOU'.format(c)] = np.mean(np.array(all_iou_this_c)) 


        gb_prec =  all_tp / all_predicted if all_predicted != 0 else 0
        gb_recall =  all_tp / all_real if all_real != 0 else 0
        out_dict['global_precision'] =gb_prec
        out_dict['global_recall'] = gb_recall
        out_dict['global_F1'] = 2*(gb_prec*gb_recall)/(gb_prec+gb_recall) if (gb_prec+gb_recall) != 0 else 0
        out_dict['global_IOU'] = np.mean(np.array(all_iou))

        # evaluacion tiempo ejecucion
        n_trials = 3
        t0 = time.time()
        for i in range(n_trials):
            fd, index_list = self.prepare_feed(10)
            pred = self.sess.run(self.pred,fd)
        tf=time.time()
        out_dict['eval_time_bs10'] = (tf-t0)*1.0/n_trials

            # evaluacion de test set
        mean_loss_test = self.eval_test_set()
        out_dict['global_loss_test'] = mean_loss_test

        for k in sorted(out_dict.keys()):
            print("{0}---{1}".format(k,out_dict[k]))

        if save_out:
            with open(os.path.join(self.get_model_out_folder(),'result_dict.json'),'w') as f:
                f.write(json.dumps({k : float(v) for k,v in out_dict.items()}, indent=4, sort_keys=True))

        return out_dict

def get_device_str():
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    if len(get_available_gpus()) > 0:
        device_str = '/GPU:0'
        print('======= USANDO GPU ========')
    else:
        device_str = '/CPU:0'
        print('======= USANDO CPU ========')
    return device_str

def parse_params_from_model_folder(folder):
    assert (os.path.isdir(folder))
    params_path = os.path.join(folder,'used_params.json')
    with open(params_path,'r') as f:
        params_dict = json.load(f)

    def check_int(y):
        try:
            z=int(y)
            return True
        except:
            return False
    def check_bool(y):
        if y == 'True' or y == 'False':
            return True
        else:
            return False

    def infer_type(x):
        if '.' in x:
            return float(x)
        if check_int(x):
            return int(x)
        if check_bool(x):
            return (x == 'True')
        else:
            return x

    params_dict = {k : infer_type(v) for k,v in params_dict.items()}
    params_dict["tf_config"] = tf.compat.v1.ConfigProto(allow_soft_placement = True)
    return folder, params_dict


def eval_from_folder(folder):
    interactive_plot = False
    model_folder,params_dict = parse_params_from_model_folder(folder)
    train_run(0, save_model=False, load_saved=model_folder, interactive_plot=interactive_plot, **params_dict)


@contextmanager
def simple_model_load(exp_params,load_saved=None):

    data_folder = './clf_images/data'
    device_str = get_device_str()

    with tf.device(device_str):
        with tf.Graph().as_default() as g:
            with clf_base(**exp_params) as x:

                x.load_data(data_folder)

                if load_saved is not None:
                    x.load_full(load_saved)

                yield x

@contextmanager
def model_load_from_folder(folder):
    model_folder, params_dict = parse_params_from_model_folder(folder)
    with simple_model_load(params_dict,load_saved=model_folder) as x:
        yield x

def eval_model(model,save_model,interactive_plot):
    # evaluacion modelo
    res_dict = model.evaluate_model(save_out=save_model)

    # visualizacion de imagenes ejemplares
    t = cv2.imread('./clf_images/data/images/1569689579_93958518.png')
    out_name = os.path.join(model.get_model_out_folder(), '1569689579_93958518_img') if not (
        interactive_plot) else None
    model.eval_with_img(t, out_name=out_name)

    sample_img = './clf_images/data/images/1569688965_886240415.png'
    out_name = os.path.join(model.get_model_out_folder(), '1569688965_886240415_img') if not (
        interactive_plot) else None
    t = cv2.imread(sample_img)
    model.eval_with_img(t, out_name=out_name)
    return res_dict


def train_run(train_it, save_model=True, load_saved=None, interactive_plot=False, **exp_params):


    if interactive_plot:
        plt.switch_backend('TkAgg')


    out_path = None
    res_dict = {}

    #import ipdb;ipdb.set_trace()

    with simple_model_load(exp_params,load_saved=load_saved) as x:

        x.train(train_it, save=save_model)

        res_dict = eval_model(x, save_model, interactive_plot)

        out_path=x.get_model_out_folder()

    return res_dict,out_path

if __name__ == '__main__':
    #import ipdb; ipdb.set_trace()

    """

    'max_pool_layers': 4, 'stack_layers': 4, 'input_factor': 0.5, 'extra_porc': 4, 'lr': 0.0001, 'st_filter': 40, 'inc_filter': 50, 'loss_string': 'mse', 'replace_max_pool_with_stride': False

    'max_pool_layers': 4, 'stack_layers': 4, 'input_factor': 0.5, 'extra_porc': 2, 'lr': 0.001,
        'st_filter': 30, 'inc_filter': 40, 'loss_string': 'cross_entropy', 'replace_max_pool_with_stride': True
    """

    # train_it = 1
    # model_folder = 'clf_images/saved_models/clf_enemy/10_Oct_2019__12_50'
    # exp_params = {
    #     'tf_config' : tf.compat.v1.ConfigProto(allow_soft_placement = True),
    # }
    #
    # train_run(train_it, save_model=False, load_saved=model_folder, interactive_plot=False, **exp_params)


    folder_model = './clf_images/saved_models/clf_enemy/12_Oct_2019__20_51_58'
    #eval_from_folder(folder_model)

    with  model_load_from_folder(folder_model) as x:
        eval_model(x,False,True)