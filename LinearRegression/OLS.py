# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 15:21:06 2018

@author: icetong
"""

import matplotlib.pyplot as plot
import numpy as np

def linear_ols(data):
    def sigma_squa_x(x):
        return sum([i**2 for i in x])
    def sigma_x(x):
        return sum(x)
    def sigma_xy(x, y):
        return sum([i*j for i, j in zip(x, y)])
    def sigma_y(y):
        return sum(y)
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    n = len(x)
    a = (n*sigma_xy(x, y) - sigma_x(x)*sigma_y(y)) / (
            n*sigma_squa_x(x) - sigma_x(x)**2)
    b = (sigma_squa_x(x)*sigma_y(y) - sigma_x(x)*sigma_xy(x, y)) / (
            n*sigma_squa_x(x) - sigma_x(x)**2)
    return a, b

def display_data(data, a, b):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plot.scatter(x, y)
    plot.plot(x, a*np.array(x)+b, 'g-')
    plot.show()

if __name__=="__main__":
    data = [[143, 3100], [295, 6500], [295, 6500], [171, 2330],
            [232, 5805], [109, 5350], [262, 6050], [422, 7750],
            [225, 2925], [126, 2890], [275, 5710], [179, 3490]]
    a, b = linear_ols(data)
    display_data(data, a, b)