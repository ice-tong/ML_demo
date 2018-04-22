# -*- coding: utf-8 -*-

from captcha.image import ImageCaptcha
import random
import os

def get_random_char(char_list, num=4):
    
    char_list_k = random.choices(char_list, k=num)
    char = ''
    for tmp in char_list_k:
        char += tmp
    return char

def get_captcha(char, file_path='./data'):
    
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    image = ImageCaptcha()
    image.write(char, './data/{}.png'.format(char))
    
if __name__=="__main__":
    number = [str(x) for x in range(0, 10)]
    low = [chr(x) for x in range(97, 123)]
    up = [chr(x) for x in range(65, 91)]
    char_list = number + low + up
    
    for i in range(0, 50000):
        char = get_random_char(char_list)
        get_captcha(char)
