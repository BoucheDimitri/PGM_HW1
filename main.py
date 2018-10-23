import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
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




####################### LDA ########################################################################################
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




###################### LOGISTIC ####################################################################################
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




###################### LINEAR ######################################################################################
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



####################### QDA ########################################################################################
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




################################PRINT SCORES########################################################################
print(scores_lda)
print(scores_logistic)
print(scores_linear)
print(scores_qda)



################################ SYNTHESIS PLOTS ###################################################################


# Plot the data and the separation lines
for lab in abc:
    fig, axes = plt.subplots(2, 2)
    xypd = data[(lab, "train")]
    # LDA
    axes[0, 0].scatter(xypd[xypd.y == 0].x1, xypd[xypd.y == 0].x2, label="0")
    axes[0, 0].scatter(xypd[xypd.y == 1].x1, xypd[xypd.y == 1].x2, label="1")
    xlim = axes[0, 0].get_xlim()
    ylim = axes[0, 0].get_ylim()
    x1 = np.linspace(axes[0, 0].get_xlim()[0], axes[0, 0].get_xlim()[1])
    x2_lda = discriminant.proba_level_line_lda(x1, est_lda[lab][0], params_lda[lab][0], params_lda[lab][1], 0.5)
    axes[0, 0].plot(x1, x2_lda, c="g", label="p(y=1|x) = 0.5")
    axes[0, 0].set_ylim(ylim)
    axes[0, 0].legend()
    axes[0, 0].set_title("LDA")
    # Logistic
    axes[0, 1].scatter(xypd[xypd.y == 0].x1, xypd[xypd.y == 0].x2, label="0")
    axes[0, 1].scatter(xypd[xypd.y == 1].x1, xypd[xypd.y == 1].x2, label="1")
    x2_logistic = logistic.proba_line(x1, params_logistic[lab], 0.5)
    axes[0, 1].plot(x1, x2_logistic, c="g", label="p(y=1|x) = 0.5")
    axes[0, 1].set_ylim(ylim)
    axes[0, 1].set_title("Logistic")
    # Linear
    axes[1, 0].scatter(xypd[xypd.y == 0].x1, xypd[xypd.y == 0].x2, label="0")
    axes[1, 0].scatter(xypd[xypd.y == 1].x1, xypd[xypd.y == 1].x2, label="1")
    x2_linear = linear.proba_line(x1, params_logistic[lab], 0.5)
    axes[1, 0].plot(x1, x2_linear, c="g", label="p(y=1|x) = 0.5")
    axes[1, 0].set_ylim(ylim)
    axes[1, 0].set_title("Linear")
    # QDA
    axes[1, 1].scatter(xypd[xypd.y == 0].x1, xypd[xypd.y == 0].x2, label="0")
    axes[1, 1].scatter(xypd[xypd.y == 1].x1, xypd[xypd.y == 1].x2, label="1")
    xx1, xx2 = np.meshgrid(np.linspace(xlim[0], xlim[1], 1000), np.linspace(ylim[0], ylim[1], 1000))
    a, b, c, d, e, f = discriminant.conic_coefs(est_qda[lab][0],
                                                est_qda[lab][1],
                                                est_qda[lab][2],
                                                est_qda[lab][3],
                                                est_qda[lab][4])
    zz = a * xx2 ** 2 + b * xx1 ** 2 + c * xx1 * xx2 + d * xx2 + e * xx1 + f
    axes[1, 1].contour(xx1, xx2, zz, [0], colors="g", label="p(y=1|x) = 0.5")
    axes[1, 1].set_title("QDA")
    plt.suptitle("Dataset " + lab)
