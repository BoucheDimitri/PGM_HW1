import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import importlib

import discriminant
import miscs
import logistic
import linear
importlib.reload(discriminant)
importlib.reload(miscs)
importlib.reload(logistic)
importlib.reload(linear)



################### DATA LOADING ###########################################################
# Path to the data
path = os.getcwd() + "/Data/"

# Data loading and basic processing
abc = ("A", "B", "C")
traintest = ("train", "test")
data = {}
data_reg = {}
for lab in abc:
    for t in traintest:
        # Load the data
        data[(lab, t)] = pd.read_table(path + "classification" + lab + "." + t,
                                       header=None,
                                       names=["x1", "x2", "y"])
        # Set label to integer type
        data[(lab, t)].y = data[(lab, t)].y.astype(np.int)
        # Data as a tuple (data matrix, label vector) with ones added as first feature in the data matrix for intercept
        data_reg[(lab, t)] = (miscs.add_intercept_col(
            data[(lab, t)].iloc[:, :2].values),
                              data[(lab, t)].iloc[:, 2].values)




####################### LDA ###########################################################
est_lda = {}
params_lda = {}
pred_lda = {}
scores_lda = {}
for lab in abc:
    # Compute MLE estimates
    pi = discriminant.mle_pi(data[(lab, "train")])
    mu0, mu1 = discriminant.mle_mus(data[(lab, "train")])
    sigma = discriminant.mle_sigma_lda(data[(lab, "train")])
    est_lda[lab] = (pi, mu0, mu1, sigma)
    sigma_inv = np.linalg.inv(est_lda[lab][3])
    # Compute parameters from mle estimates
    w = discriminant.get_w_lda(sigma_inv, est_lda[lab][1], est_lda[lab][2])
    b = discriminant.get_b_lda(sigma_inv, est_lda[lab][1], est_lda[lab][2])
    params_lda[lab] = (w, b)
    # Evaluate performance of tuned model on train and test sets
    xtrain, ytrain = data[(lab, "train")].iloc[:, :2].values, data[(lab, "train")].iloc[:, 2].values
    xtest, ytest = data[(lab, "test")].iloc[:, :2].values, data[(lab, "test")].iloc[:, 2].values
    pred_lda[(lab, "test")] = discriminant.classify_lda(xtest, pi, w, b)
    scores_lda[(lab, "test")] = miscs.zero_one_score(pred_lda[(lab, "test")], ytest)
    pred_lda[(lab, "train")] = discriminant.classify_lda(xtrain, pi, w, b)
    scores_lda[(lab, "train")] = miscs.zero_one_score(pred_lda[(lab, "train")], ytrain)




###################### LOGISTIC ##########################################################
params_logistic = {}
pred_logistic = {}
scores_logistic = {}
bw_init = np.zeros((3, ))

# Tune logistic regression on the train databases, predict for each on the test set and compute zero one score.
for lab in abc:
    # Extract data from data dict
    xtrain, ytrain = data_reg[(lab, "train")]
    xtest, ytest = data_reg[(lab, "test")]
    # Run IRLS to tune model
    bw = logistic.iter_irls(xtrain, ytrain, bw_init)
    params_logistic[lab] = bw
    # Evaluate performance of tuned model on train and test sets
    pred_logistic[(lab, "test")] = logistic.classify(xtest, bw)
    scores_logistic[(lab, "test")] = miscs.zero_one_score(pred_logistic[(lab, "test")], ytest)
    pred_logistic[(lab, "train")] = logistic.classify(xtrain, bw)
    scores_logistic[(lab, "train")] = miscs.zero_one_score(pred_logistic[(lab, "train")], ytrain)




###################### LINEAR ##########################################################
params_linear= {}
pred_linear = {}
scores_linear = {}

# Tune linear regression on the train databases, predict for each on the test set and compute zero one score.
for lab in abc:
    # Extract data from data dict
    xtrain, ytrain = data_reg[(lab, "train")]
    xtest, ytest = data_reg[(lab, "test")]
    # Compute the tuned parameters
    bw = linear.get_bw(xtrain, ytrain)
    params_linear[lab] = bw
    # Evaluate performance of tuned model on train and test sets
    pred_linear[(lab, "test")] = linear.classify(xtest, bw)
    scores_linear[(lab, "test")] = miscs.zero_one_score(pred_linear[(lab, "test")], ytest)
    pred_linear[(lab, "train")] = linear.classify(xtrain, bw)
    scores_linear[(lab, "train")] = miscs.zero_one_score(pred_linear[(lab, "train")], ytrain)




####################### QDA ###########################################################
est_qda = {}
pred_qda = {}
scores_qda = {}
for lab in abc:
    # Compute MLE estimates
    pi = discriminant.mle_pi(data[(lab, "train")])
    mu0, mu1 = discriminant.mle_mus(data[(lab, "train")])
    sigma0, sigma1 = discriminant.mle_sigmas_qda(data[(lab, "train")])
    est_qda[lab] = (pi, mu0, mu1, sigma0, sigma1)
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    # Evaluate performance of tuned model on train and test sets
    xtrain, ytrain = data[(lab, "train")].iloc[:, :2].values, data[(lab, "train")].iloc[:, 2].values
    xtest, ytest = data[(lab, "test")].iloc[:, :2].values, data[(lab, "test")].iloc[:, 2].values
    pred_qda[(lab, "test")] = discriminant.classify_qda(xtest, pi, mu0, mu1, sigma0_inv, sigma1_inv)
    scores_qda[(lab, "test")] = miscs.zero_one_score(pred_qda[(lab, "test")], ytest)
    pred_qda[(lab, "train")] = discriminant.classify_qda(xtrain, pi, mu0, mu1, sigma0_inv, sigma1_inv)
    scores_qda[(lab, "train")] = miscs.zero_one_score(pred_qda[(lab, "train")], ytrain)
















# Plot the data and the separation lines
for lab in abc:
    fig, ax = plt.subplots()
    ax.scatter(data[(lab, "train")].x1, data[(lab, "train")].x2)
    sep_x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1])
    sep_x2 = discriminant.proba_level_line_lda(sep_x1, est_lda[lab][0], params_lda[lab][0], params_lda[lab][1], 0.5)
    ax.plot(sep_x1, sep_x2)
