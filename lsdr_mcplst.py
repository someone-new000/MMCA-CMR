# Inputs:
#    Xa: (n x da) feature matrix of modality a
#    Xb: (n x db) feature matrix of modality b
#    Y: (n x c) label matrix
#    params.h: hidden dimemtion
# Return:
#    U: (n x h) code vector
#    V: (h x c) project matrix
# adapt from https://github.com/hsuantien/mlc_lsdr

import numpy as np

def lsdr_mcplst(Xa, Xb, Y, h, max_iter, alph, beta1):

    shift = np.zeros((1,Y.shape[1]))
    for m in range(Y.shape[1]):
        sum = 0
        for n in range(Y.shape[0]):
            sum = sum + Y[n,m]
        mean = sum / Y.shape[0]
        shift[0,m] = mean

    N, K = np.shape(Y)
    Yshift = Y - np.tile(shift, (N, 1))

    Xarh = ridgereg_hat(Xa, 0.001)
    Xbrh = ridgereg_hat(Xb, 0.001)
    X = np.dot(np.dot(Yshift.T, Xarh), Yshift) + np.dot(np.dot(Yshift.T, Xbrh),Yshift)

    U, sigma, VT = np.linalg.svd(X, full_matrices=True)
    V = VT.T
    Vm = V[:, 0:h]
    U = np.dot(Yshift, Vm)

    return U

def ridgereg_hat(X,la):

    H = np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X) + la * np.identity(X.shape[1]))), X.T)

    return H