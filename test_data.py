import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import importlib

import logistic
importlib.reload(logistic)
import LDA

n1 = 50
n2 = 50
mu1 = (-1, -1)
mu2 = (4, 4)
sigma1 = np.eye(2)
sigma2 = np.eye(2)

gauss1 = np.random.multivariate_normal(mu1, sigma1, n1)
gauss2 = np.random.multivariate_normal(mu2, sigma2, n2)
gauss = np.concatenate((gauss1, gauss2))
y = np.array(n1 * [0] + n2 * [1]).reshape((n1 + n2, 1))
xy = np.concatenate((gauss, y), axis=1)
xypd = pd.DataFrame(data=xy, columns=["x1", "x2", "y"])

x = xypd.iloc[:, :2].values
y = xypd.iloc[:, 2].values



##################"LDA##################################"

pi = LDA.mle_pi(xypd)
mu0, mu1 = LDA.mle_mus(xypd)
sigma = LDA.mle_sigma(xypd)
sigma_inv = np.linalg.inv(sigma)
w = LDA.get_w(sigma_inv, mu0, mu1)
b = LDA.get_b(sigma_inv, mu0, mu1)

fig, ax = plt.subplots()
ax.scatter(x[0, :], x[1, :])
sep_x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])
sep_x2 = LDA.proba_level_line(sep_x1, pi, w, b, 0.5)
ax.plot(sep_x1, sep_x2)



##############LOGISTIC TEST#########################
xlogistic = logistic.add_intercept_col(x)

bw = np.array([1.0, 1.0, 1.0])
print(logistic.diag_p_1minusp(xlogistic, bw))
bwbis = logistic.irls_update(xlogistic, y, bw)



# grad = logistic.likelihood_gradient(x, y, w, b)
# hess = logistic.likelihood_hessian(x, w, b)
#
# w, b = logistic.newton_mle(x, y, w, b, 1, 1000, 0.0001)


fig, ax = plt.subplots()
ax.scatter(x[0, :], x[1, :])
sep_x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])
sep_x2 = logistic.proba_level_line(sep_x1, w, b, 0.5)
ax.plot(sep_x1, sep_x2)


