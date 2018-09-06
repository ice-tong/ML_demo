# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:33:40 2018

@author: icetong
"""
import os, time, random
import numpy as np

s = time.time()

def load_data(rate=1):
    data_dir = './data'
    labels = os.listdir(data_dir)
    data = list()
    for label in labels:
        label_dir = data_dir+os.sep+label
        filenames = os.listdir(label_dir)
        for filename in filenames:
            data_path = label_dir + os.sep + filename
            with open(data_path, encoding='utf-8') as f:
                text = f.read()
            data.append([1 if label == 'SPAM' else 0, text.split(',')])
    random.shuffle(data)
    print('读取数据完毕， 用时：{}'.format(time.time()-s))
    return data[:int(len(data)*rate)]

def build_word_set(data):
    wordset = set()
    for label, text in data:
        wordset |= set(text)
    return list(wordset)

def word2vec(wordset, data):
    wordict = {word:k for k, word in enumerate(wordset)}
    print('词集长度：{}'.format(len(wordset)))
    vecs = np.zeros([len(data), len(wordset)])
    labels = np.zeros(len(data))
    for k, [label, text] in enumerate(data):
        for word in text:
            if word in wordict:
                vecs[k, wordict[word]] = 1
        labels[k] = label
    print('文档向量化完毕, 用时：{}'.format(time.time()-s))
    return vecs, labels

def trainNB(data, labels):
    print('training\n-----')
    Pspam = sum(labels) / len(labels)
    Pham = 1 - Pspam
    SN = np.ones(data.shape[1])
    HN = np.ones(data.shape[1])
    for k, d in enumerate(data):
        if labels[k]:
            SN += d
        else:
            HN += d
    PS = SN / sum(SN)
    PH = HN / sum(HN)
    print('训练完毕， 用时：{}'.format(time.time()-s))
    return Pspam, Pham, PS, PH

def predictNB(data, Pspam, Pham, PS, PH):
    print('testing\n------')
    PS = np.log(PS)
    PH = np.log(PH)
    Pspam = np.math.log(Pspam)
    Pham = np.math.log(Pham)
    predict_vec = [1 if (Pspam+sum(d*PS)) >= (Pham+sum(d*PH)) else 0 
                   for d in data]
    return predict_vec


if __name__=="__main__":
    data = load_data()
    wordset = build_word_set(data)
    vecs, labels = word2vec(wordset, data)
    train_data = vecs[:int(0.8*vecs.shape[0]), :]
    train_labels = labels[:int(0.8*labels.shape[0])]
    Pspam, Pham, PS, PH = trainNB(train_data, labels)
    predict = predictNB(vecs[int(0.8*vecs.shape[0]):, :], Pspam, Pham, PS, PH)
    accuary = np.mean(predict==labels[int(0.8*labels.shape[0]):])
    print('精度：{}'.format(accuary))
    print('总用时：{}'.format(time.time()-s))