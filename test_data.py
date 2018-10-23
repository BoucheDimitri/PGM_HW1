import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import importlib

import logistic
importlib.reload(logistic)
import discriminant
importlib.reload(discriminant)

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

pi = discriminant.mle_pi(xypd)
mu0, mu1 = discriminant.mle_mus(xypd)
sigma = discriminant.mle_sigma_lda(xypd)
sigma_inv = np.linalg.inv(sigma)
w = discriminant.get_w_lda(sigma_inv, mu0, mu1)
b = discriminant.get_b_lda(sigma_inv, mu0, mu1)
ytest = discriminant.posterior_proba_lda(x, pi, w, b)

fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1])
sep_x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])
sep_x2 = discriminant.proba_level_line_lda(sep_x1, pi, w, b, 0.5)
ax.plot(sep_x1, sep_x2)


pi = discriminant.mle_pi(xypd)
mu0, mu1 = discriminant.mle_mus(xypd)
sigma0, sigma1 = discriminant.mle_sigmas_qda(xypd)
sigma0_inv = np.linalg.inv(sigma0)
sigma1_inv = np.linalg.inv(sigma1)
a, b, c, d, e, f = discriminant.conic_coefs(pi, mu0, mu1, sigma0, sigma1)
xx0, xx1, zz = discriminant.contours_qda(pi, mu0, mu1, sigma0_inv, sigma1_inv, (0, 10), (0, 10))
ytest = discriminant.classify_qda(x, pi, mu0, mu1, sigma0_inv, sigma1_inv)

fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1])
xx0, xx1, zz = discriminant.contours_qda(pi, mu0, mu1, sigma0_inv, sigma1_inv, ax.get_xlim(), ax.get_ylim())
ax.contour(xx0, xx1, zz, [0])



##############LOGISTIC TEST#########################
xlogistic = logistic.add_intercept_col(x)

bw = np.array([0, 0, 0])
bwbis = logistic.iter_irls(xlogistic, y, bw, 0.005, 15)



# grad = logistic.likelihood_gradient(x, y, w, b)
# hess = logistic.likelihood_hessian(x, w, b)
#
# w, b = logistic.newton_mle(x, y, w, b, 1, 1000, 0.0001)


fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1])
sep_x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])
sep_x2 = logistic.proba_level_line(sep_x1, bwbis, 0.5)
ax.plot(sep_x1, sep_x2)


