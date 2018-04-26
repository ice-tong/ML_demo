# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 19:47:46 2018

@author: icetong
"""

from input_data import read_data, show_img, get_label
import numpy as np
import random

class KNN(object):
    
    def __init__(self, img_data, img_labels, k=20, num_axis=0):
        
        self.data = img_data
        self.labels = img_labels
        self.k = k
        self.num_axis = num_axis
        
    def L2Classifier(self, x):
        
        X = np.zeros([self.data.shape[self.num_axis], x.shape[0]])
        for i in range(X.shape[0]):
            X[i] = x
        
        L2_distance = np.sum(((X-self.data)**2), 1)**0.5
        
        index_sort = L2_distance.argsort()
        k_index_sort = index_sort[:self.k]
        
        result = {}
        for item in k_index_sort:
            V = get_label(self.labels[item])
            if V not in result.keys():
                result[V] = 1
            else:
                result[V] += 1
        return result
        
        
def main():
    
    data = read_data()
    img_data = data['train_images']
    img_labels = data['train_labels']
    
    knn = KNN(img_data, img_labels)
    
    test_img_data = random.choice(data['t10k_images'])
    show_img(test_img_data) 
    result = knn.L2Classifier(test_img_data)
    print(result)
    pre_label = sorted(result,key=lambda x:result[x])[-1]
    print(pre_label)

def get_pre_acc(num=100):
    
    data = read_data()
    img_data = data['train_images']
    img_labels = data['train_labels']
    
    knn = KNN(img_data, img_labels)
    
    acc = 0
    for i in range(num):
        a = random.randint(0, len(data['t10k_labels']))
        test_img_data = data['t10k_images'][a]
        test_img_label = data['t10k_labels'][a]
        result = knn.L2Classifier(test_img_data)
        pre_label = sorted(result,key=lambda x:result[x])[-1]
        test_label = get_label(test_img_label)
        acc += (pre_label == test_label)
    print(acc/num)
    

if __name__=="__main__":

#    main()
    get_pre_acc()