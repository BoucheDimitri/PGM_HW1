__author__ = "Dimitri Bouche - dimi.bouche@gmail.com"



import numpy as np


def get_bw(x, y):
    xtx = np.dot(x.T, x)
    xtx_inv = np.linalg.inv(xtx)
    xty = np.dot(x.T, y)
    return np.dot(xtx_inv, xty)


def classify(xtest_mat, bw):
    return np.dot(xtest_mat, bw)


def proba_line(x0, bw, q):
    return (1 / bw[2]) * (q - bw[0] - bw[1] * x0)