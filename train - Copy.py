import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

slim = tf.contrib.slim

data =  np.load("./data.npy")
label = np.load("./label.npy")

label = np.int32(label)

temp_data = tf.placeholder(tf.float32,shape = [None,13,13,1024],name = "data")
temp_label = tf.placeholder(tf.int32,shape = [None,],name = "label")

def finetune(data,label):

    with slim.arg_scope([slim.conv2d], padding='VALID',
                      weights_initializer=tf.contrib.layers.xavier_initializer(),weights_regularizer=slim.l2_regularizer(0.0005)):

        net = slim.conv2d(data,1024,[3,3])
        net = slim.batch_norm(net)
        net = slim.conv2d(net,512,[3,3])
        net = slim.batch_norm(net)
        net = slim.conv2d(net,512,[3,3])
        net = slim.batch_norm(net)
        net = slim.conv2d(net,512,[3,3])
        net = slim.batch_norm(net)

        net = tf.layers.flatten(net)
        net = slim.fully_connected(net,2048)
        net = slim.fully_connected(net,512)
        net = slim.fully_connected(net,3)
        logit = tf.nn.softmax(net)
        

    
    return logit

logit= finetune(temp_data,temp_label)
labels = tf.subtract(temp_label,1)
labelx = tf.one_hot(labels,3)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labelx,logit))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.00005).minimize(cost)

#init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    #sess.run(init)
    saver.restore(sess, "/finetune/model.ckpt")

    for epoch in range(10):
        t0 = time.time()
        #databatch = data[(epoch*10):((epoch+3)*10),:,:,:]
        databatch = data[0:100,:,:,:]
        #print(databatch.shape)
        labelbatch = label[0:100]
        #labelbatch = label[(epoch*10):((epoch+3)*10)]
        #print(type(labelbatch))

        X , logits ,labels,costs = sess.run([optimizer,logit,labelx,cost],feed_dict = {temp_label : labelbatch,temp_data : databatch})
        #X  = sess.run(optimizer,feed_dict = {temp_data : databatch})
        print(logits[54:60])
        print(labels[54:60])
        print ("Cost = "+str(costs)+"in time:"+str(time.time()-t0))

    save_path = saver.save(sess, "/finetune/model.ckpt")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
