# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:28:25 2018

@author: icetong
"""

import pandas as pd
import numpy as np

def load_data(rate=0.9):
    data_path = './watermelon3_0.csv'
    df = pd.read_csv(data_path)
    del df['编号']
    del df['密度']
    del df['含糖率']
    data = df.values
    return data

def trainNB(data):
    labels = data[:, -1]
    PGood = sum([1 for l in labels if l=='是']) / len(labels)
    PBad = 1 - PGood
    NBClassify = {'是': {}, '否': {}}
    for label in NBClassify.keys():
        sub_data = data[data[:, -1] == label]
        sub_data = np.array(sub_data)
        for k in range(sub_data.shape[1]):
            NBClassify[label][k] = dict()
            tags = list(set(data[:, k]))
            d = sub_data[:, k]
            for tag in tags:
                NBClassify[label][k][tag] = (sum([1 for i in d
                if i == tag])+1)/len(d)
    return PGood, PBad, NBClassify

def testNB(data, PG, PB, NBClassify):
    predict_vec = list()
    for sample in data:
        pg = np.math.log(PG, 2)
        pb = np.math.log(PB, 2)
        for label in NBClassify.keys():
            for k, tag in enumerate(sample):
                if label == '是':
                    pg += np.math.log(NBClassify[label][k][tag], 2)
                else:
                    pb += np.math.log(NBClassify[label][k][tag], 2)
        if pg >= pb:
            predict_vec.append('是')
        else:
            predict_vec.append('否')
    return np.array(predict_vec)

if __name__=="__main__":
    test = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'], ]
    data = load_data()
    PG, PB, NBClassify = trainNB(data)
    predict_vec = testNB(test, PG, PB, NBClassify)
    print(test)
    print(predict_vec)