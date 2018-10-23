__author__ = "Dimitri Bouche - dimi.bouche@gmail.com"



import numpy as np


def add_intercept_col(x):
    """
    Perform one step of IRLS

    Params:
        x (np.ndarray): the data matrix (n_data, n_features)

    Returns:
        np.ndarray: the data matrix to which a columns of ones have been added for the intercept in first position

    """
    n = x.shape[0]
    ones = np.ones((n, 1))
    return np.concatenate((ones, x), axis=1)


def zero_one_score(ypred, ytrue):
    return round(np.mean(np.abs(ypred - ytrue)), 3)
