import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

slim = tf.contrib.slim

data =  np.load("./shuffle_data.npy")
label = np.load("./shuffle_label.npy")

label = np.int32(label)

temp_data = tf.placeholder(tf.float32,shape = [None,13,13,1024],name = "data")
temp_label = tf.placeholder(tf.int32,shape = [None,],name = "label")

def finetune(data):

    with slim.arg_scope([slim.conv2d], padding='VALID',
                      weights_initializer=tf.contrib.layers.xavier_initializer(),weights_regularizer=slim.l2_regularizer(0.0005)):

        net = slim.conv2d(data,1024,[3,3],scope='conv1')
        net = slim.batch_norm(net)
        net = slim.conv2d(net,512,[3,3],scope='conv2')
        net = slim.batch_norm(net)
        net = slim.conv2d(net,512,[3,3],stride = 2,scope='conv3')
        net = slim.batch_norm(net)
        #net = slim.conv2d(net,512,[3,3])
        #net = slim.batch_norm(net)
        
        net = tf.layers.flatten(net)
        net = slim.fully_connected(net,512,scope='FC1')
        net = slim.fully_connected(net,256,scope='FC2')
        net = slim.dropout(net,0.5)
        net = slim.fully_connected(net,3,scope='FC3')
        logit = tf.nn.softmax(net)
        

    
    return logit

logit= finetune(temp_data,temp_label)
labels = tf.subtract(temp_label,1)
labelx = tf.one_hot(labels,3)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labelx,logit))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.00009).minimize(cost)


init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    sess.run(init_l)
    #saver.restore(sess, "./finetune/modelx.ckpt")

    for epoch in range(600):
        t0 = time.time()
        ts = epoch%6
        #databatch = data[1200:1300,:,:,:]
        databatch = data[(ts*200):((ts+1)*200),:,:,:]
        #print(databatch.shape)
        labelbatch = label[(ts*200):((ts+1)*200)]
        #labelbatch = label[1200:1300]
        #print(type(labelbatch))

        X , logits ,labels,costs  = sess.run([optimizer,logit,labelx,cost],feed_dict = {temp_label : labelbatch,temp_data : databatch})
        #X  = sess.run(optimizer,feed_dict = {temp_data : databatch})
        diff = np.sum(np.absolute(labels-logits))
        #print(np.absolute(labels-logits))
        total,classes = logits.shape
        pred_acc = (1-diff/total)*100
        #print(logits[30:40])
        #print(labels[30:40])
        print(epoch)
        print ("Cost = "+str(costs)+" in time:"+str(time.time()-t0))
        print("Accurary on data = "+str(pred_acc))
        if ts == 5 :
            save_path = saver.save(sess, "./finetune/NEW.ckpt")
    #save_path = saver.save(sess, "/finetune/model.ckpt")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
