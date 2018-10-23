import numpy as np


def mle_pi(xy):
    """
    MLE for pi in LDA

    Params:
        xy (pandas.core.frame.DataFrame with columns): (first dim of x, second dim of x, label)

    Returns:
       float: MLE for pi in LDA model
    """
    return xy.y.sum() / xy.shape[0]


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


def mle_sigma(xy, i):
    """
    Empirical covariance matrix restricted to one class

    Params:
        xy (pandas.core.frame.DataFrame with columns): (first dim of x, second dim of x, label)

    Returns:
       np.ndarray
    """
    return xy[xy.y == i].iloc[:, :2].cov()


def mle_sigma_lda(xy):
    """
    MLE for covariance matrix in LDA

    Params:
        xy (pandas.core.frame.DataFrame with columns): (first dim of x, second dim of x, label)

    Returns:
       np.ndarray
    """
    return mle_sigma(xy, 0) + mle_sigma(xy, 1)


def mle_sigmas_qda(xy):
    """
    MLEs for the two covariance matrixes in QDA

    Params:
        xy (pandas.core.frame.DataFrame with columns): (first dim of x, second dim of x, label)

    Returns:
       np.ndarray
    """
    return mle_sigma(xy, 0), mle_sigma(xy, 1)


def get_w_lda(sigma_inv, mu0, mu1):
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


def get_b_lda(sigma_inv, mu0, mu1):
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


def proba_level_line_lda(x1, pi, w, b, q):
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


def posterior_proba_lda(xtest, pi, w, b):
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
    pifrac = (1 - pi) / pi
    return 1 / (1 + pifrac * np.exp(xwtest + b))


def classify_lda(xtest_mat, pi, w, b):
    """
    Classification using LDA

    Params:
        xtest_mat (np.ndarray): matrix of datapoints to classify
        pi (float): MLE for pi
        w (np.ndarray): the vector of parameters
        b (float): the intercept

    Returns:
        np.ndarray: vector of predicted labels
    """
    probas = posterior_proba_lda(xtest_mat, pi, w, b)
    ytest = np.sign(probas - 0.5)
    ytest[ytest == -1] = 0
    return ytest


def conic_coefs(pi, mu0, mu1, sigma0, sigma1):
    """
    Compute the coefficient of the conic section defining the decision boundary for QDA

    Params:
        pi (float): MLE for pi
        mu0 (np.ndarray): MLE for mu0
        mu1 (np.ndarray): MLE for mu1
        sigma0 (np.ndarray): MLE of sigma0
        sigma1 (np.ndarray): MLE of sigma1

    Returns:
        float: Intercept
    """
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    lamb0 = np.dot(mu0.T, sigma0_inv)
    lamb1 = np.dot(mu1.T, sigma1_inv)
    nu0 = - 0.5 * np.log(np.linalg.det(sigma0)) - 0.5 * np.dot(lamb0, mu0) + np.log(1 - pi)
    nu1 = - 0.5 * np.log(np.linalg.det(sigma1)) - 0.5 * np.dot(lamb1, mu1) + np.log(pi)
    diff_sig = sigma1_inv - sigma0_inv
    diff_lamb = lamb1 - lamb0
    return 0.5*diff_sig[1, 1], 0.5*diff_sig[0, 0], diff_sig[0, 1], - diff_lamb[1], - diff_lamb[0], nu0 - nu1


def log_probas_qda(xtest, pi, mu, sigma_inv):
    """
    Log of p(y=1|x) for QDA

    Params:
        xtest (np.ndarray): matrix of datapoints on which to compute the log probabilities
        pi (float): MLE for pi
        mu (np.ndarray): MLE for mu
        sigma_inv (np.ndarray): inverse of MLE for sigma

    Returns:
        np.ndarray: vector of predicted labels
    """
    return - 0.5 * np.log(1 / np.linalg.det(sigma_inv)) \
           - 0.5*np.dot((xtest - mu).T, np.dot(sigma_inv, xtest - mu)) \
           + np.log(pi)


def classify_qda(xtest_mat, pi, mu0, mu1, sigma0_inv, sigma1_inv):
    """
    Classify according to fitted QDA

    Params:
        xtest_mat(np.ndarray): matrix of datapoints on which to compute the log probabilities
        pi (float): MLE for pi
        mu0 (np.ndarray): MLE for mu0
        mu1 (np.ndarray): MLE for mu1
        sigma0_inv(np.ndarray): inverse of MLE for sigma0
        sigma1_inv (np.ndarray): inverse of MLE for sigma1

    Returns:
        np.ndarray: vector of predicted labels
    """
    n = xtest_mat.shape[0]
    ytest = np.zeros((n, ))
    for i in range(0, n):
        ytest[i] = np.sign(log_probas_qda(xtest_mat[i, :], pi, mu1, sigma1_inv)
                           - log_probas_qda(xtest_mat[i, :], 1 - pi, mu0, sigma0_inv))
    ytest[ytest == -1] = 0
    return ytest
