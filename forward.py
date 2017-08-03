import numpy as np
import matplotlib.pyplot as plt
import ipdb
import unittest


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1)  # sigmoid activation function
    A = Z.dot(W2) + b2
    # ipdb.set_trace()
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Z, Y


def visualize(X, Y):
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()


def classification_rate(P, Y):
    # P is the probability of y being cat 1, 2, and 3
    sample_size = np.size(P, 0)
    assert(sample_size == len(Y))

    prediction = np.argmax(P, axis=1)
    error = Y - prediction
    correct = map(lambda e: e == 0, error)
    correct_count = sum(correct)
    return correct_count/sample_size


def main():
    # Preparing data
    Nclass = 1000
    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    # Y is a lable array, [0,...,0, 1,...,1, 2,...,2]
    # D is the input dimension, 2
    D = 2
    # M is the hidden layer size, 1 hidden layer, M neurons
    M = 3
    # K is the number of classes, also the number of output neurons.
    K = 3
    N = len(Y)
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)
    learning_rate = 10e-7
    T = np.zeros((N, K))
    costs = []
    for n in range(N):
        T[n, Y[n]] = 1

    for epoch in range(100000):
        hidden, output = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            rate = classification_rate(output, Y)
            print(f"cost: {c}, classification_rate: {rate}")
            costs.append(c)

        # gradient decent
        W2 += learning_rate * derivative_w2(output, T, hidden)
        b2 += learning_rate * derivative_b2(output, T)
        W1 += learning_rate * derivative_w1(output, T, hidden, X, W2)
        b1 += learning_rate * derivative_b1(output, T, hidden, W2)

    return


def cost(T, Y):
    tot = T * np.log(Y)
    # sum over all the elements
    return tot.sum()


def derivative_w2(Y, T, Z):
    # partial J /partial V
    return Z.T.dot(T - Y)


def derivative_b2(Y, T):
    return (T - Y).sum(axis=0)


def derivative_w1(Y, T, Z, X, W2):
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    # N * M
    return X.T.dot(dZ)


def derivative_b1(Y, T, Z, W2):
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    return dZ.sum(axis=0)


class TestFunctions(unittest.TestCase):

    def test_classification_rate(self):
        P = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
        Y1 = np.array([1, 0])
        rate1 = classification_rate(P, Y1)
        self.assertEqual(rate1, 0.5)
        Y2 = np.array([2, 0])
        rate2 = classification_rate(P, Y2)
        self.assertEqual(rate2, 1.0)
        Y3 = np.array([0, 2])
        rate3 = classification_rate(P, Y3)
        self.assertEqual(rate3, 0.0)


if __name__ == '__main__':
    main()
    # unittest.main()
