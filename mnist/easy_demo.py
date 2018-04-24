# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:39:02 2018

@author: icetong
"""

import tensorflow as tf
from input_data import read_data


def main():
    
    data = read_data()
    learning_rate = 0.001
    max_steps = 10000
    height=28
    width=28
    
    X = tf.placeholder(tf.float32, [None, height*width])
    Y = tf.placeholder(tf.float32, [None, 10])
    
    w = tf.Variable(tf.random_normal([height*width, 10]))
    b = tf.Variable(tf.random_normal([10]))
    
    out = tf.add(tf.matmul(X, w), b)
    norm = tf.contrib.slim.batch_norm(out, is_training=True)
    Y_out = tf.nn.softmax(norm)
    
    cross_entropy = -tf.reduce_sum(Y*tf.log(
            tf.clip_by_value(Y_out, 1e-8, tf.reduce_max(Y_out))))
    train_op = tf.train.AdamOptimizer(
            learning_rate).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        for i in range(1, max_steps):
           x, y = data['train_images'], data['train_labels']
           loss, _ = sess.run([cross_entropy, train_op], feed_dict={X:x, Y:y})
           
           if i%10==0:
               print('step:{} cross_enrropy:{}'.format(i, loss))
           
           if i%100 == 0:
               x, y = data['t10k_images'], data['t10k_labels']
               acc = sess.run(accuracy, feed_dict={X:x, Y:y})
               print('accuracy:{}'.format(acc))
               if acc>0.9:
                   break

if __name__=="__main__":
    
    main()