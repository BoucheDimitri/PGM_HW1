import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    hess[1:, 1:] = np.dot(np.dot(x, D), x.T)
    return hess


def newton_mle(x, y, w0, b, pace, maxit, epsilon):
    wb = np.concatenate((w0.copy(), np.array([b])))
    for i in range(0, maxit):
        grad = likelihood_gradient(x, y, wb[1:], wb[0])
        hess = likelihood_hessian(x, wb[1:], wb[0])
        hess_inv = np.linalg.inv(hess)
        delta = np.dot(hess_inv, grad)
        lamb_sqr = np.dot(np.dot(grad.T, hess_inv), grad)
        if lamb_sqr / 2 <= epsilon:
            return wb[1:], wb[0]
        wb += pace * delta
        print(i)
    return wb[1:], wb[0]


def proba_level_line(x1, w, b, q):
    lq = np.log((1-q) / q)
    return (- 1 / w[1]) * (w[0] * x1 - b - lq)



