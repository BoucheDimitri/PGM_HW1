import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import importlib

import logistic
importlib.reload(logistic)

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

x = data[("A", "train")].iloc[:, :2].values.T
y = data[("A", "train")].iloc[:, 2].values.T
w = np.array([1, 1])
b = 1

grad = logistic.likelihood_gradient(x, y, w, b)
hess = logistic.likelihood_hessian(x, w, b)
