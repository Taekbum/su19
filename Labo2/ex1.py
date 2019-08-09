

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc, cm
from scipy import optimize
import math


#Functions
def sigmoid (z):
    return 1.0 / (1.0 + np.exp(-z))

#Exercice 1.3
def costFunction (theta, x, y):
    m = len(x)
    sigma = 0
    for i in range(m):
        h = sigmoid(np.inner(x, theta))
        sigma += (y[i] * math.log(sigmoid(np.sum(theta * x[i]))) + (1 - y[i]) * math.log(1-sigmoid(np.sum(x[i] * theta))))
    sigma = sigma/(-m)
    return sigma


print("Exercise 1.1")
print("Generating the graph...")
with open('ex2data1.txt') as f1, open('ex2data2.txt') as f2:
    dataset1 = np.loadtxt(f1, delimiter = ',',
        dtype = 'float', usecols = None)
    dataset2 = np.loadtxt(f2, delimiter = ',',
        dtype = 'float', usecols = None)
X = dataset1[:, :-1]
Y = dataset1[:, 2]
KO = np.where(Y == 0)[0]
OK = np.where(Y)[0]

#Create graph
plt.figure()

admitted = ""
notAdmitted = ""

for i in range(len(X)):
    if i in KO:
        admitted = plt.scatter(X[i][0],X[i][1], c="yellow", label="Not admitted", edgecolor='black')
    else:
        notAdmitted = plt.scatter(X[i][0],X[i][1], c="black", label="Admitted", marker="+")

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((admitted, notAdmitted),('Admitted','Not admitted'),scatterpoints=1,loc='upper right')
plt.show()


print("Exercise 1.4")
newX = np.concatenate((np.ones((100, 1)), X), axis=1)
theta = (0.1, 0.1, 0.1)
print("Optimizing theta...")
theta = optimize.minimize(costFunction,theta,args=(newX, Y)).x

print("Generating the graph...")
#Create the graph
plt.figure()
admitted = ""
notAdmitted = ""

#Add points to graph
for i in range(len(X)):
    if i in KO:
        admitted = plt.scatter(X[i][0],X[i][1], c="yellow", label="Not admitted", edgecolor='black')
    else:
        notAdmitted = plt.scatter(X[i][0],X[i][1], c="black", label="Admitted", marker="+")
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

for j in range(len(newX)):
    plt.scatter(newX[j][1], (theta[0] + theta[1] * newX[j][1])/(-theta[2]),c="blue")

plt.legend((admitted, notAdmitted),('Admitted','Not admitted'),scatterpoints=1,loc='upper right')
plt.show()