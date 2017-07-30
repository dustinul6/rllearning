import numpy as np
import matplotlib.pyplot as plt
import ipdb

Nclass = 500

X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
# Y is a lable array, [0,...,0, 1,...,1, 2,...,2]

plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
plt.show()

# D is the input dimension, 2
D = 2
# M is the hidden layer size, 1 hidden layer, M neurons
M = 3
# K is the number of classes, also the number of output neurons.
K = 3

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def forward(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1)  # sigmoid activation function
    A = Z.dot(W2) + b2
    # ipdb.set_trace()
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y


if __name__ == '__main__':
    forward(X, W1, b1, W2, b2)
