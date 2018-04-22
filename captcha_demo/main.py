# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 18:58:43 2018

@author: icetong
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

class AL_CNN(object):
    
    def __init__(self):
        
        self.data_dir = r"./data"
        self.img_width = 160
        self.img_height = 60
        self.len_label = 4
        self.len_dict = 62
        self.max_steps = 100000
        self.keep_prob = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.float32, shape=
                                [None, self.img_height, self.img_width, 1])
        self.Y = tf.placeholder(tf.float32, shape=
                                [None, self.len_label*self.len_dict])
        self.data_dict = self.get_shuffle_data_dict()
        
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        
        self.learning_rate=0.01

    
    def get_shuffle_data_dict(self, scale=0.2):
        
        img_list = os.listdir(self.data_dir)
        
        random.shuffle(img_list)
        
        len_img_list = len(img_list)
        len_train_list = int(len_img_list*(1-0.2))
        
        train_list = img_list[:len_train_list+1]
        test_list = img_list[len_train_list+1:]
        
        return {"train":train_list, "test":test_list}
        
    def get_lable_vector(self, char):
        
        l_vector = np.zeros(self.len_label*self.len_dict)
        for i, ch in enumerate(char):
            k = ord(ch)
            if k in range(48, 58):
                X = k - 48
            if k in range(97, 123):
                X = k - 97 + 10
            if k in range(65, 92):
                X = k - 65 + 36
            index = i*self.len_dict + X
            l_vector[index] = 1
        return l_vector
    
    def get_vector_lable(self, vector):
        
        char_tmp = np.nonzero(vector)[0]
        char = ''
        for z in char_tmp:
            t = z%self.len_dict
            if t in range(0, 10):
                Y = chr(t+48)
            if t in range(10, 36):
                Y = chr(t+97-10)
            if t in range(36, 62):
                Y = chr(t+65-36)
            char += Y
        return char
    
    def get_random_batch(self, batch_size=64, trainOrTest="train"):
        
        imgs = random.sample(self.data_dict[trainOrTest], batch_size)
        
        x = np.zeros(shape=[batch_size, self.img_height, self.img_width])
        y = np.zeros(shape=[batch_size, self.len_label*self.len_dict])
        
        for k, img in enumerate(imgs):
            img_dir = self.data_dir + "/" + img
            img_label = img.split(".")[0]
            
            x[k, : , : ] = np.mean(plt.imread(img_dir), -1)
            y[k, : ] = self.get_lable_vector(img_label)
        x = np.reshape(x, newshape=[-1, self.img_height, self.img_width, 1])
        return x, y
            
    def dense_dropout_layer(self, IN, w_shape, b_shape, w, b):
        
        dense_w = tf.Variable(w*tf.random_normal(shape=w_shape))
        dense_b = tf.Variable(b*tf.random_normal(shape=b_shape))
        reshape_IN = tf.reshape(IN, shape=[-1, 
                                           dense_w.get_shape().as_list()[0]])
        #dense_layer
        #relu_layer
        #dropout_layer
        dense = tf.add(tf.matmul(reshape_IN, dense_w), dense_b)
        dense_relu = tf.nn.relu(dense)
        norm = tf.contrib.slim.batch_norm(dense_relu, is_training=True)
        dense_dropout = tf.nn.dropout(norm, self.keep_prob)
        return dense_dropout
        
    def conv_relu_pool_layer(self, IN, F_shape, B_shape, w, b):
        
        Filter = tf.random_normal(shape=F_shape)
        Bias = tf.random_normal(shape=B_shape)
        #conv_layer
        #relu_layer
        #pool_layer
        conv = tf.nn.bias_add(
                tf.nn.conv2d(IN, tf.Variable(w*Filter), 
                             strides=[1, 1, 1, 1], padding="SAME"),
                                    tf.Variable(b*Bias))
        conv_relu = tf.nn.relu(conv)
        pool = tf.nn.max_pool(conv_relu, ksize=[1, 2, 2 , 1], 
                                strides=[1, 2, 2, 1], padding="SAME")
        norm = tf.contrib.slim.batch_norm(pool, is_training=True)
        return tf.nn.dropout(norm, self.keep_prob)
    
    def cap_cnn(self, w=0.01, b=0.1):
        
        #self.X.shape: ?*60*160*1 (batch, width, height, channel)
        c_r_p_1 = self.conv_relu_pool_layer(self.X, [3, 3, 1, 32],
                                            [32], w, b)
        #c_r_p_1.shape: ?*30*80*32 (batch, width, height, channel)
        c_r_p_2 = self.conv_relu_pool_layer(c_r_p_1, [3, 3, 32, 64],
                                            [64], w, b)
        #c_r_p_2.shape: ?*15*40*64 (batch, width, height, channel)
        c_r_p_3 = self.conv_relu_pool_layer(c_r_p_2, [3, 3, 64, 64],
                                            [64], w, b)
        #c_r_p_3.shape: ?*8*20*64  (batch, width, height, channel)
        c_r_p_4 = self.conv_relu_pool_layer(c_r_p_3, [3, 3, 64, 128],
                                            [128], w, b)
        #c_r_p_4.shape: ?*4*10*128 (batch, width, height, channel)
        # 4*10*128=4960
        d_d = self.dense_dropout_layer(c_r_p_4, [4*10*128, 1024],
                                       [1024], w, b)
        #d_d.shape: ?*1024
        
        out_w = tf.Variable(w*tf.random_normal(
                [1024, self.len_label*self.len_dict]))
        out_b = tf.Variable(b*tf.random_normal([self.len_label*self.len_dict]))
        #out_w.shape: 1024*(4*62)
        #out_b.shape: (4*62)
        out = tf.add(tf.matmul(d_d, out_w), out_b)
        #out.shape: ?*(4*62)
        return out
        
    def run(self):
        
        out = self.cap_cnn()
        #out.shape: ?*(4*62)
        
        diff = tf.nn.sigmoid_cross_entropy_with_logits(logits=out, 
                                                       labels=self.Y)
        loss = tf.reduce_mean(diff)
        #self.Y: ?*(4*62)
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        
        y_1 = tf.reshape(out, shape=[-1, self.len_label, self.len_dict])
        y_2 = tf.reshape(self.Y, shape=[-1, self.len_label, self.len_dict])
        correct_pred = tf.equal(tf.argmax(y_1, 2), tf.argmax(y_2, 2))
        acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("acc", acc_op)
        summary_op = tf.summary.merge_all()
        
        saver = tf.train.Saver()
        
        init = tf.global_variables_initializer()
        with tf.Session(config=self.config) as sess:
            
            sess.run(init)    
            summery_writer = tf.summary.FileWriter("./log", sess.graph)
        
            for i in range(1, self.max_steps):
                
                x, y = self.get_random_batch()
                feed_dict = {self.X:x, self.Y:y, self.keep_prob:0.8}
                loss_v, t, summ = sess.run(
                    [loss, train_op, summary_op], feed_dict=feed_dict)
                summery_writer.add_summary(summ, i)
                if i % 10 == 0:
                  print("第{}步， loss:{}".format(i, loss_v))
                
                if i%50 == 0:
                    #feed_dict.keys(): self.X, self.Y, self.keep_prob
                    #   self.X.shape: batch*60*160*1
                    #   self.Y.shape: batch*(4*62)
                    #   self.keep_prob: (float32)
                    x, y = self.get_random_batch(trainOrTest="test")
                    feed_dict = {self.X:x, self.Y:y, self.keep_prob:0.8}
                    acc, summ = sess.run(
                        [acc_op, summary_op], feed_dict=feed_dict)
                    summery_writer.add_summary(summ, i)
                    print("-----当前准确率：{}".format(acc))
                    
                    if acc > 0.95:
                        print("模型保存成功！！结束训练")
                        saver.save(sess, save_path="./model/AlexNet_Capth.model")
                        break
                    
                if i%500 == 0:
                    self.data_dict = self.get_shuffle_data_dict()
                    self.learning_rate = self.learning_rate**0.5
            
            summery_writer.close()
                    
        
if __name__=="__main__":
    tf.reset_default_graph()
    al_cnn = AL_CNN()
    al_cnn.run()
