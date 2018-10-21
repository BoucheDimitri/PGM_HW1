import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import LDA

# Path to the data
path = os.getcwd() + "\\Data\\"


# Data loading and basic processing
abc = ("A", "B", "C")
tt = ("train", "test")
data = {}
for lab in abc:
    for t in tt:
        # Load the data
        data[(lab, t)] = pd.read_table(path + "classification" + lab + "." + t,
                                       header=None,
                                       names=["x1", "x2", "y"])
        # Set label to integer type
        data[(lab, t)].y = data[(lab, t)].y.astype(np.int)


# Compute MLE estimates of pi, mu0, mu1 and sigma for all three datasets and store them
estimates = {}
for lab in abc:
    pi = LDA.mle_pi(data[(lab, "train")])
    mu0, mu1 = LDA.mle_mus(data[(lab, "train")])
    sigma = LDA.mle_sigma(data[(lab, "train")])
    estimates[lab] = (pi, mu0, mu1, sigma)


# Compute the parameters w and b
params = {}
for lab in abc:
    print(estimates[lab][3])
    sigma_inv = np.linalg.inv(estimates[lab][3])
    w = LDA.get_w(sigma_inv, estimates[lab][1], estimates[lab][2])
    b = LDA.get_b(sigma_inv, estimates[lab][1], estimates[lab][2])
    params[lab] = (w, b)


# Plot the data and the separation lines
for lab in abc:
    fig, ax = plt.subplots()
    ax.scatter(data[(lab, "train")].x1, data[(lab, "train")].x2)
    sep_x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])
    sep_x2 = LDA.proba_level_line(sep_x1, estimates[lab][0], params[lab][0], params[lab][1], 0.5)
    ax.plot(sep_x1, sep_x2)


# fig, ax = plt.subplots()
# ax.scatter(data[("A", "train")].x1, data[("A", "train")].x2)
# sep_x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])
# sep_x2 = LDA.proba_level_line(sep_x1, estimates["A"][0], params["A"][0], params["A"][1], 0.5)
# ax.plot(sep_x1, sep_x2)
# import sklearn.discriminant_analysis as da
# test = da.LinearDiscriminantAnalysis()
# test.fit(data[("A", "train")].iloc[:, :2], data[("A", "train")].iloc[:, 2])
#
# def test_func(b, x, w):
#     return (1 / w[0][1]) * (0.5 - b - x * w[0][0])
#
#
# sepbis = test_func(test.intercept_, sep_x1, test.coef_)
# ax.plot(sep_x1, sepbis, c="r")