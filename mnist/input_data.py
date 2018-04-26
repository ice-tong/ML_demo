# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 22:09:56 2018

@author: icetong
"""

import struct
import os
import numpy as np
import matplotlib.pyplot as plt

def read_images(file_path):
    
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, np.uint8)
    images = np.reshape(images, [num, rows*cols])
#    print(rows, cols)
    return images

def read_labels(file_path):
    
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, np.uint8)
    re_labels = np.zeros([num, 10])
    for k, item in enumerate(labels):
        re_labels[k, item] = 1
    return re_labels

def show_img(img, height=28, width=28):
    
    img = np.reshape(img, [height, width])
    plt.imshow(img)
    
def get_label(vec):
    
    for k, i in enumerate(vec):
        if i:
            return k
    return None
    
def read_data():

    '''
    return:
        data: 
            data.keys(): train_images, train_labels, t10k_images, t10k_labels
                train_images, t10k_images: [num, cols*rows]
                train_labels, t10k_labels: [num, 10]
    '''
    data_dir = './data'
    data = {}
    
    data_file_name = os.listdir(data_dir)
#    print(data_file_name)

    for file_name in data_file_name:
        file_path = data_dir + '/' + file_name
        if 'images' in file_name:
            data['_'.join(file_name.split('-')[:2])] = read_images(file_path)
        else:
            data['_'.join(file_name.split('-')[:2])] = read_labels(file_path)
        
    return data


if __name__=="__main__":
    
    data = read_data()
    
    show_img(data['train_images'][0])
    print(data['train_labels'][0])