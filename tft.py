import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.add(x,3)

with tf.Session() as sess:
    for i in range(4):
        print(sess.run(y,feed_dict = {x : int(i)}))
        
