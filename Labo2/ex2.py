#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 04:44:51 2019

@author: edin
"""


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc, cm
from scipy import optimize
import math


def mapFeature(x1,x2,degreeMax):
    out = np.empty((x1.shape[0],0))
    for i in range(0, degreeMax + 1):
        for j in range(i+1):
            out = np.concatenate((out, np.reshape(np.power(x1,i-j) * np.power(x2,j), (-1, 1))), axis=1)

    return out
        
#Functions
def sigmoid (z):
    return 1.0 / (1.0 + np.exp(-z))


def costFunction (theta, x, y, l=1):
    m = len(x)
    h = sigmoid(np.inner(x, theta))
    return ((-1/m) * (np.sum(y * np.log(h) + (1-y) * np.log(1-h)))) + ((l/2/m) * np.sum(np.square(thetas[1:])))
    
    

with open('ex2data2.txt') as f1:
    dataset1 = np.loadtxt(f1, delimiter = ',',
        dtype = 'float', usecols = None)
    
test1 = dataset1[:, :-2]
test2 = dataset1[:, 1:-1]
result = dataset1[:, 2]

KO = np.where(result == 0)[0]
OK = np.where(result)[0]

print("Executing mapFeature function...")
mf = mapFeature(test1,test2,6)

print("Generating the graph...")
plt.figure()

y1 = ""
y0= ""

for i in range(len(test1)):
    if i in KO:
        y0 = plt.scatter(test1[i],test2[i], c="yellow", label="<= 0", edgecolor='black')
    else:
        y1 = plt.scatter(test1[i],test2[i], c="black", label="y = 1", marker="+")

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend((y1, y0),('y = 1','y = 0'),scatterpoints=1,loc='upper right')
plt.show()

thetas = [0] * len(mp[0])


print("Testing costFunction... Returned value is:" , costFunction(thetas, mf, result))
print("Optimizing theta with optimize.minimize function...")
thetaOpt = optimize.minimize(costFunction,thetas,args=(mf, result)).x
print("Result : " , thetaOpt)
