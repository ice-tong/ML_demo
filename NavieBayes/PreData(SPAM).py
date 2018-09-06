# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:51:25 2018

@author: icetong
"""

import os, re, time
import threading, queue
import jieba
import random

stop_word = open('./stopWord.txt', encoding='utf-8').read().split('\n')

def process(label, path):
    data_dir = './data'
    with open(path, 'rb') as f:
            content = f.read()
            content = content.decode('gbk', 'ignore')
    new_dir = '{}/{}'.format(data_dir, label)
    new_path = '{}/{}_{}.txt'.format(new_dir, 
                path.split('/')[-2], path.split('/')[-1])
    text = content.encode('utf-8', 'ignore').decode('utf-8')
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    text = jieba.cut(text, cut_all=False)
    text = ','.join([word for word in text if word not in stop_word])
    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(text)

lock = threading.Lock()
flag = False
class Worker(threading.Thread):
    def __init__(self, q):
        threading.Thread.__init__(self)
        self.q = q
    def run(self):
        while 1:
            if not self.q.empty():
                lock.acquire()
                label, path = self.q.get()
                lock.release()
                process(label, path)
            else:
                time.sleep(1)
            if flag:
                break

def pre_data():
    new_dir1 = './data/SPAM'
    new_dir2 = './data/HAM'
    if not os.path.exists(new_dir1):
        os.makedirs(new_dir1)
        os.makedirs(new_dir2)
    else:
        print('pre data exists!')
        return
    index_path = './trec06c/full/index'
    with open(index_path) as f:
        lines = f.readlines()
    random.shuffle(lines)
    lines = lines[:int(0.1*len(lines))]
    data = list()
    for line in lines:
        label = line.split(' ')[0]
        path = './trec06c'+line.split(' ')[1].replace('\n', '')[2:]
        data.append([label.upper(), path])
    Q = queue.Queue()
    for label, path in data:
        Q.put([label, path])
    ws = list()
    for j in range(100):
        w = Worker(Q)
        w.start()
        ws.append(w)
    while not Q.empty():
        pass
    print('empty queue!')
    for w in ws:
        w.join()
    global flag
    flag = True
    print('work over!!!')
    
if __name__=="__main__":
    pre_data()