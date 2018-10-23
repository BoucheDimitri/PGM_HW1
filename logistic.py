import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def probas(x, bw):
    """
    Vector of p(y=1|x=xi) according to the model

    Params
        x (np.ndarray): the data matrix (n_data, n_features + 1) with first column filled with 1s for intercept
        y (np.ndarray): the vector of labels (n_data, )
        bw (np.ndarray): the vector of parameters (n_features + 1, ), first coordinate is the intercept

    Returns:
        np.ndarray: vectors of p(y=1|x=xi) according to the model
    """
    p = sigmoid(np.dot(x, bw))
    return p


def irls_update(x, y, bw):
    """
    Perform one step of IRLS

    Params:
        x (np.ndarray): the data matrix (n_data, n_features + 1) with first column filled with 1s for intercept
        y (np.ndarray): the vector of labels (n_data, )
        bw (np.ndarray): the vector of parameters (n_features + 1, ), first coordinate is the intercept

    Returns:
        tuple: the new parameters vector and the norm of the gradient at this point
    """
    p = probas(x, bw)
    v = p * (1 - p)
    diag = np.diag(v)
    diag_inv = np.linalg.inv(diag)
    minus_hess = np.dot(x.T, np.dot(diag, x))
    minus_hess_inv = np.linalg.inv(minus_hess)
    z = np.dot(x, bw) + np.dot(diag_inv, y - p)
    u = np.dot(np.dot(x.T, diag), z)
    grad_norm = np.linalg.norm(np.dot(x.T, y - p))
    return np.dot(minus_hess_inv, u), grad_norm


def iter_irls(x, y, bw, epsilon=0.005, maxit=20):
    """
    IRLS algorithm

    Params:
        x (np.ndarray): the data matrix (n_data, n_features + 1) with last column filled with 1s for intercept
        y (np.ndarray): the vector of labels (n_data, )
        bw (np.ndarray): the vector of parameters (n_features + 1, ), first coordinate is the intercept
        epsilon (float): the stopping criterion on the l2 norm of the gradient
        maxit (int): maximum number of iterations

    Returns:
        tuple: the "fitted" parameters vector.
    """
    prev_bw = bw.copy()
    for i in range(0, maxit):
        try :
            next_bw, grad_norm = irls_update(x, y, prev_bw)
            if grad_norm < epsilon:
                return next_bw
            prev_bw = next_bw
            print("Iteration no: " + str(i) + ";   l2 norm of the gradient: " + str(grad_norm))
        except np.linalg.linalg.LinAlgError:
            return prev_bw
    return prev_bw


def proba_line(x1, bw, q):
    """
    Linear function that characterizes the line p(y=1|x) = q

    Params:
        x1 (float): the point at which to take the function
        bw (np.ndarray): the vector of parameters (n_features + 1, ), first coordinate is the intercept
        q (float): q

    Returns:
        Image of x1 by the linear function

    """
    lq = np.log((1-q) / q)
    return (- 1 / bw[2]) * (bw[1] * x1 + bw[0] + lq)


def classify(xmat_test, bw):
    ytest = np.sign(probas(xmat_test, bw) - 0.5)
    ytest[ytest == -1] = 0
    return ytest



