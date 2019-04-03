
import numpy as np
import tensorflow as tf
from PIL import Image
import time

import mocode

from utilss import get_boxes_and_inputs_pb,load_graph, letter_box_image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'log_file', './label.txt', 'LOG of labeled data')

def main(argv = None):

    data = open(FLAGS.log_file,'r')
    lines = data.readlines()
    labels = []
    #print(lines)

    t0 = time.time()
    frozenGraph = load_graph("./frozen_darknet_yolov3_model.pb")
    print("Loaded graph in {:.2f}s".format(time.time()-t0))
    bottleneck_data, inputs = get_boxes_and_inputs_pb(frozenGraph)
    No = 0
    with tf.Session(graph=frozenGraph) as sess:
        for  x in lines:
            image , label = x.split()
            #img = Image.open("./"+image)
            img = Image.open("./image/"+image+".jpg")
            img_resized = letter_box_image(img, 416, 416, 128)
            img_resized = np.reshape(img_resized,(1,416,416,3))
            
            labels.append(label)
            preprocess_data = sess.run(bottleneck_data,feed_dict = {inputs : img_resized})
            t0 = time.time()
            if No == 0 :
                preprocess_tensor = preprocess_data
            else:
                preprocess_tensor = np.concatenate((preprocess_tensor,preprocess_data),axis = 0)
            print("Executed in {:.2f}s".format(time.time()-t0))
            print(preprocess_data.shape)

            No += 1
        print(preprocess_tensor.shape)
        labels = np.array(labels)
        np.save("data",preprocess_tensor)
        np.save("label",labels)
        print(labels.shape)
        #np.load("data.npy")
if __name__ == '__main__':
    tf.app.run()
