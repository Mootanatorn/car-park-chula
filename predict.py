import numpy as np
import tensorflow as tf
from PIL import Image
import time

from utilss import get_boxes_and_inputs,letter_box_image             
import mocode

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '', 'Input image')
tf.app.flags.DEFINE_string(
    'bottle_ckpt', './saved_model/model.ckpt', 'Bottleneck checkpoint file')
tf.app.flags.DEFINE_string(
    'classify_ckpt', './finetune/NEW.ckpt', 'Classify Checkpoint file')

def main(argv=None):
    t0 = time.time()
    img = Image.open(FLAGS.input_img)
    img_resized = letter_box_image(img, 416, 416, 128)
    img_resized = img_resized.astype(np.float32)
    img_resized = np.reshape(img_resized,(1,416,416,3))

    g1 = tf.Graph()
    g2 = tf.Graph()
    with g1.as_default():
        t1 = time.time()
        inputs = tf.placeholder(tf.float32, [1,416, 416, 3])
        bottleneck = get_boxes_and_inputs(mocode.yolo_v3,inputs, 416, 'NHWC')
        saver1 = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
        print("Loaded Bottleneck graph in {:.2f}s".format(time.time()-t1))
         
    with g2.as_default():
        t2 = time.time()
        predata = tf.placeholder(tf.float32,[1,13,13,1024])
        with slim.arg_scope([slim.conv2d], padding='VALID',
                      weights_initializer=tf.contrib.layers.xavier_initializer(),weights_regularizer=slim.l2_regularizer(0.0005)):

            net = slim.conv2d(predata,1024,[3,3],scope='conv1')
            net = slim.batch_norm(net)
            net = slim.conv2d(net,512,[3,3],scope='conv2')
            net = slim.batch_norm(net)
            net = slim.conv2d(net,512,[3,3],stride = 2,scope='conv3')
            net = slim.batch_norm(net)
       
            net = tf.layers.flatten(net)
            net = slim.fully_connected(net,512,scope='FC1')
            net = slim.fully_connected(net,256,scope='FC2')
            net = slim.fully_connected(net,3,scope='FC3')
            pred = tf.nn.softmax(net)
        print("Loaded Classification graph in {:.2f}s".format(time.time()-t2))
        saver2 = tf.train.Saver()
    print("Loaded graph in {:.2f}s".format(time.time()-t0))    
    with tf.Session(graph=g1) as sess1:
        t3 = time.time()
        saver1.restore(sess1,FLAGS.bottle_ckpt)
        preprocess_data = sess1.run(bottleneck,feed_dict = {inputs : img_resized})
        print("Executed Bottleneck in {:.2f}s".format(time.time()-t3))

    with tf.Session(graph = g2) as sess2:
        t4 = time.time()
        saver2.restore(sess2,FLAGS.classify_ckpt)
        prediction = sess2.run(pred, feed_dict = {predata: preprocess_data})
        print("Predicts in {:.2f}s".format(time.time()-t4))
        #print(prediction)
    final = np.add(np.argmax(prediction),1)
    print(final)
    if final == 1:
        print("You can park here")
    elif final == 2:
        print("It's quite crowded")
    else:
        print("RUN!!!!!!!!!!!!!!!!")
    print("Total time :{:.2f}s".format(time.time()-t0))
if __name__ == '__main__':
    tf.app.run()
