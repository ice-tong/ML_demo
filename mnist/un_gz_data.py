# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:27:12 2018

@author: icetong
"""


import gzip
import os

def un_gz(un_gz_dir, gz_dir, gz_name):
    
    if gz_name[-3:] != ".gz":
        return False
    
    un_gz_name = gz_name[:-3] 
    un_gz_path = un_gz_dir + '/' + un_gz_name
    gz_path = gz_dir + '/' + gz_name
    
    if os.path.exists(un_gz_path):
        return True
    gz_file = gzip.GzipFile(gz_path)
    
    with open(un_gz_path, 'wb') as f:
        f.write(gz_file.read())
        print('un_gz {} over!'.format(gz_name))
        return True
    
def main():
    
    un_gz_dir = './data'
    gz_dir = './gz_data'

    if not os.path.exists(un_gz_dir):
        os.mkdir(un_gz_dir)
    if not os.path.exists(gz_dir):
        print('error! no such dir name "{}"'.format(gz_dir))     
    
    gz_list = os.listdir(gz_dir)
    for gz_name in gz_list:
        un_gz(un_gz_dir, gz_dir, gz_name)
    
    pass

if __name__=="__main__":
    main()