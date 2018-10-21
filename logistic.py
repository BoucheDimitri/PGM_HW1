import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def likelihood_gradient(x, y, w, b):
    d = x.shape[0]
    grad = np.zeros((d + 1, ))
    z = sigmoid(np.dot(w, x) + b)
    grad[0] = np.sum(y - z)
    grad[1:] = np.dot(x, y - z)
    return grad


def likelihood_hessian(x, w, b):
    d = x.shape[0]
    n = x.shape[1]
    hess = np.zeros((d + 1, d + 1))
    z = sigmoid(np.dot(w, x) + b)
    hess[0, 0] = - np.dot(z.T, 1 - z)
    u = - np.dot(x, z * (1 - z))
    hess[0, 1:] = u
    hess[1:, 0] = u
    D = np.zeros((n, n))
    np.fill_diagonal(D, z * (1 - z))
    print(x.shape)
    print(D.shape)
    hess[1:, 1:] = np.dot(np.dot(x, D), x.T)
    return hess

