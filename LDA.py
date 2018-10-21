import numpy as np
import pandas as pd


def mle_pi(xy):
    return xy.y.sum() / xy.shape[0]


def mle_mu(xy, i):
    return xy[xy.y == i].mean().iloc[:2].values


def mle_mus(xy):
    return mle_mu(xy, 0), mle_mu(xy, 1)


def mle_sigma(xy):
    return (xy[xy.y == 0].iloc[:, :2].cov() + xy[xy.y == 1].iloc[:, :2].cov()).values


def get_w(sigma_inv, mu0, mu1):
    return np.dot(mu0 - mu1, sigma_inv)


def get_b(sigma_inv, mu0, mu1):
    return 0.5*np.dot(np.dot(mu1.T, sigma_inv), mu1) - 0.5*np.dot(np.dot(mu0.T, sigma_inv), mu0)


def proba_level_line(x1, pi, w, b, q):
    return (1 / w[1]) * (np.log((pi * (1 - q)) / (q * (1 - pi))) - b - w[0] * x1)


def posterior_proba(xtest, pi, w, b):
    xwtest = np.dot(xtest, w)
    pifrac = pi / (1 - pi)
    return 1 / (1 + pifrac * np.exp(xwtest) + b)