import numpy as np


def mle_pi(xy):
    """
    MLE for pi in LDA

    Params:
        xy (pandas.core.frame.DataFrame with columns): (first dim of x, second dim of x, label)

    Returns:
       float: MLE for pi in LDA model
    """
    return xy.y.sum() / xy.shape


def mle_mu(xy, i):
    """
    MLE for mu_i in LDA

    Params:
        xy (pandas.core.frame.DataFrame with columns): (first dim of x, second dim of x, label)

    Returns:
       np.ndarray: MLE for mu_i in LDA
    """
    return xy[xy.y == i].mean().iloc[:2].values


def mle_mus(xy):
    """
    MLE for mus in LDA

    Params:
        xy (pandas.core.frame.DataFrame with columns): (first dim of x, second dim of x, label)

    Returns:
       tuple: MLE for mu0, MLE for mu1
    """
    return mle_mu(xy, 0), mle_mu(xy, 1)


def mle_sigma(xy):
    """
    MLE for covariance matrix in LDA

    Params:
        xy (pandas.core.frame.DataFrame with columns): (first dim of x, second dim of x, label)

    Returns:
       np.ndarray: MLE for sigma
    """
    return (xy[xy.y == 0].iloc[:, :2].cov() + xy[xy.y == 1].iloc[:, :2].cov()).values


def get_w(sigma_inv, mu0, mu1):
    """
    Compute w vector of parameters from the MLE estimators

    Params:
        sigma_inv (np.ndarray): the inverse of MLE covariance matrix
        mu0 (np.ndarray): MLE for mu0
        mu1 (np.ndarray): MLE for mu1

    Returns:
        np.ndarray: w, vector of parameters

    """
    return np.dot(mu0 - mu1, sigma_inv)


def get_b(sigma_inv, mu0, mu1):
    """
    Compute intercept from the MLE estimators

    Params:
        sigma_inv (np.ndarray): the inverse of MLE covariance matrix
        mu0 (np.ndarray): MLE for mu0
        mu1 (np.ndarray): MLE for mu1

    Returns:
        float: Intercept
    """
    return 0.5*np.dot(np.dot(mu1.T, sigma_inv), mu1) - 0.5*np.dot(np.dot(mu0.T, sigma_inv), mu0)


def proba_level_line(x1, pi, w, b, q):
    """
    Linear function that characterizes the line p(y=1|x) = q

    Params:
        x1 (float): the point at which to take the function
        pi (float): MLE for pi
        w (np.ndarray): the vector of parameters
        b (float): the intercept
        q (float): q

    Returns:
        Image of x1 by the linear function
    """
    return (1 / w[1]) * (np.log((pi * (1 - q)) / (q * (1 - pi))) - b - w[0] * x1)


def posterior_proba(xtest, pi, w, b):
    """
    Compute the vector of posterior probabilities for classification

    Params:
        xtest (np.ndarray): matrix of datapoints to classify
        pi (float): MLE for pi
        w (np.ndarray): the vector of parameters
        b (float): the intercept

    Returns:
        np.ndarray: vector of posterior probabilities
    """
    xwtest = np.dot(xtest, w)
    pifrac = pi / (1 - pi)
    return 1 / (1 + pifrac * np.exp(xwtest) + b)