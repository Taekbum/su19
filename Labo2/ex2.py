# Homework 2 part 2
# Authors : Taekbum Lee and Edin Mujkanovic


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

#Functions
def mapFeature(x, y):
    ans = []
    for i in range(7):
        for j in range(7 - i):
            ans.append(1 * (x ** i) * (y ** j))
    return ans


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def costFunction(theta, x, y, l):
    m = len(x)
    h = sigmoid(np.inner(x, theta))
    return ((-1/m) * (np.sum(y * np.log(h) + (1-y) * np.log(1-h)))) + ((l/2/m) * np.sum(np.square(theta[1:])))



print("Homework2 - part 2")
print("Extracting data from datasource")

with open('ex2data2.txt') as f1:
    dataset1 = np.loadtxt(f1, delimiter=',',
                          dtype='float', usecols=None)


X = dataset1[:, :-1]
Y = dataset1[:, 2]
KO = np.where(Y == 0)[0]
OK = np.where(Y)[0]

plt.figure()
RejectedX = []
RejectedY = []
AcceptedX = []
AcceptedY = []
for i in KO:
    RejectedX.append(X[i][0])
    RejectedY.append(X[i][1])
for i in OK:
    AcceptedX.append(X[i][0])
    AcceptedY.append(X[i][1])

print("Creating the graph")
plt.plot(RejectedX, RejectedY, 'yo', label="y = 0")
plt.plot(AcceptedX, AcceptedY, 'kx', label="y = 1")
plt.xlabel("Test 1 Score")
plt.ylabel("Test 2 Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

mf = []

for i in X:
    mf.append(mapFeature(i[0], i[1]))

thetas = [0.01] * 28

print("Optimizing the costFunction function...")
thetasOpt = optimize.minimize(costFunction, thetas, args=(mf, Y, 1))

print("Result is :" , thetasOpt.x)
tmp = np.linspace(-1, 1.5, 75)
tmp2 = np.linspace(-1, 1.5, 75)
tmp3 = np.zeros(shape=(len(tmp), len(tmp2)))
for i in range(len(tmp)):
    for j in range(len(tmp2)):
        tmp3[i, j] =  np.array((mapFeature(np.array(tmp[i]), np.array(tmp2[j])))).dot(np.array(thetasOpt.x))

tmp3 = tmp3.T
print("Creating the second graph")
plt.plot(RejectedX, RejectedY, 'yo', label="y = 0")
plt.plot(AcceptedX, AcceptedY, 'kx', label="y = 1")
plt.xlabel("Test 1 Score")
plt.ylabel("Test 2 Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
contour = plt.contour(tmp,tmp2,tmp3,0)
plt.show()