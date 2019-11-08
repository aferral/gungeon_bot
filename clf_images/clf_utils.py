import os
from xml.dom import minidom

import numpy as np
import tensorflow as tf
from datetime import datetime

from scipy import ndimage
from state_fusion.deploy_utils import do_profile


def now_string():
    a = datetime.now()
    return a.strftime("%d_%b_%Y__%H_%M_%S")

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.compat.v1.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))

    with open("test.html",'w') as f:
        f.write(iframe)

def optimistic_restore(session, save_file):
    """
    Restore all the variables in saved file. Usefull for restore a model and add a
    new layers. From https://gist.github.com/iganichev/d2d8a0b1abc6b15d4a07de83171163d4
    :param session:
    :param save_file:
    :return:
    """
    reader = tf.compat.v1.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for
                        var in tf.compat.v1.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                            tf.compat.v1.global_variables()),
                        tf.compat.v1.global_variables()))
    with tf.compat.v1.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)

    saver = tf.compat.v1.train.Saver(restore_vars)
    saver.restore(session, save_file)


def parse_all_xmls(data_folder):
    xml_folder = os.path.join(data_folder,'xmls')
    xml_files = os.listdir(xml_folder)

    out = []
    label_set = set()

    def parse_bb_xml(node):
        parse_data = lambda x,y : x.getElementsByTagName(y)[0].childNodes[0].data
        out = {}
        out['xmin'] = int(parse_data(node,'xmin'))
        out['xmax'] = int(parse_data(node,'xmax'))
        out['ymin'] = int(parse_data(node,'ymin'))
        out['ymax'] = int(parse_data(node,'ymax'))
        return out

    for elem in xml_files:
        ext = elem.split('.')[-1]
        if ext != 'xml':
            continue

        full_path = os.path.join(xml_folder,elem)
        xml=minidom.parse(full_path)

        path_to_img = xml.getElementsByTagName('path')[0].childNodes[0].data

        objects = xml.getElementsByTagName('object')

        label_list = []
        for y in objects:

            tag=y.getElementsByTagName('name')[0].childNodes[0].data
            bb_node = y.getElementsByTagName('bndbox')[0]
            bb= parse_bb_xml(bb_node)
            bb_args = ['xmin','xmax','ymin','ymax']

            label_set.add(tag)
            label_list.append((tag,*[bb[t] for t in bb_args]))

        # file_path,label,bb
        tuple_data = (path_to_img,label_list)
        out.append(tuple_data)

    return out,label_set



def bb_to_binary(bb,img_shape):
    temp=np.zeros(img_shape,dtype=np.bool)
    temp[bb['xmin']:bb['xmax'],bb['ymin']:bb['ymax']] = 1
    return temp

def bbox2(img):
    """
    From https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    :param img:
    :return:
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def from_act_to_bb(act_matrix,th):
    bin_img = act_matrix > th
    res, n_labels = ndimage.label(bin_img)

    bbs=[]
    for i in range(1,n_labels+1):
        xmin,xmax,ymin,ymax = bbox2(res==i)
        bbs.append({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax})

    return bbs


def calc_relev(xmin,xmax,ymin,ymax,extra_porc):
    # calcula area relevante
    ydelta = ymax-ymin
    xdelta = xmax-xmin
    x_rel_val = int(np.sqrt(1+extra_porc)*xdelta)-xdelta
    y_rel_val = int(np.sqrt(1+extra_porc)*ydelta)-ydelta
    if x_rel_val % 2 == 0:
        xminr,xmaxr = xmin-(x_rel_val/2),xmax+(x_rel_val/2)
    else:
        temp = x_rel_val-1
        xminr,xmaxr = xmin-1-(temp/2),xmax+(temp/2)

    if y_rel_val % 2 == 0:
        yminr,ymaxr = ymin-(y_rel_val/2),ymax+(y_rel_val/2)
    else:
        temp = y_rel_val-1
        yminr,ymaxr = ymin-1-(temp/2),ymax+(temp/2)

    return int(xminr),int(xmaxr),int(yminr),int(ymaxr)