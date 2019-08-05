#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import pandas
import matplotlib.pyplot as plt
from matplotlib import rc,cm
from mpl_toolkits.mplot3d import Axes3D
# loading and parse data
citytruck = np.loadtxt('ex1data1.txt', delimiter=',', unpack=True)


print("Exercise 4.2")
def costFunction (points, theta):
    x = points[0]
    y = points[1]
    h = (theta[0] + theta[1] * x) - y
    return np.sum(h * h) / (2.0 * len(x))
 
result = costFunction(citytruck,[0,0])
print("Result is " + str(result))
print("Expected result is 32.07")
print("\n\n")


print("Exercise 4.3")
def gradientDescent (points, theta, alpha, epsilon):
    x = points[0]
    y = points[1]
    m = len(x)
    diff = epsilon + 1.0
    oldtheta = theta[:]
    while diff > epsilon:
        sigma1 = sigma2 = 0
        for i in range(m):
            sigma1 += oldtheta[0] + oldtheta[1]*x[i] - y[i]
            sigma2 += (oldtheta[0] + oldtheta[1]*x[i] - y[i])*x[i]  
        theta[0] = oldtheta[0] - ((alpha * sigma1)/m)
        theta[1] = oldtheta[1] - ((alpha * sigma2)/m)
        diff = costFunction(points,oldtheta) - costFunction(points,theta) 
        oldtheta = theta[:]
    
    return theta

print("Calculating theta...")
result2 = gradientDescent(citytruck, [0,0], 0.005,0.0000000000000001)
print("Theta : " + str(result2))
print("Generating the graph...")
plt.figure()
plt.plot(*citytruck,'rx')
plt.title('Food truck profit according to city size')
plt.xlabel('Population in 10,000s')
plt.ylabel('Profit of City in 10,000s')
x = citytruck[0]
plt.plot(x,  x * result2[1] + result2[0])
