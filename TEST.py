import numpy as np
import tensorflow as tf
from PIL import Image
import time

import mocode
import yolo_v3_tiny

from utilss import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', '', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', '', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', './frozen_darknet_yolov3_model.pb', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')

def main(argv=None):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    img = Image.open(FLAGS.input_img)
    img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
    img_resized = img_resized.astype(np.float32)
    print(img_resized.shape)
    
    classes = load_coco_names(FLAGS.class_names)

    if FLAGS.frozen_model:

        t0 = time.time()
        frozenGraph = load_graph(FLAGS.frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

        with tf.Session(graph=frozenGraph, config=config) as sess:
            t0 = time.time()
            Z = sess.run(boxes, feed_dict={inputs: [img_resized]})
            print(Z.shape)
           
    else:
        if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = mocode.yolo_v3

        X = get_boxes_and_inputs(model, len(classes), FLAGS.size, FLAGS.data_format)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        with tf.Session(config=config) as sess:
            t0 = time.time()
            saver.restore(sess, FLAGS.ckpt_file)
            print('Model restored in {:.2f}s'.format(time.time()-t0))

            t0 = time.time()
            output = sess.run(
                X, feed_dict={inputs: [img_resized]})



if __name__ == '__main__':
    tf.app.run()
