# min_W ||WX - S||^2 + alph||X - W^T S||^2
# Inputs:
#    X: (d x n) data matrix.
#    S: (k x n) semantic matrix.
#    alph: regularisation parameter.
# Return:
#    W: (k x d) projection matrix.
# adapt from https://github.com/Elyorcv/SAE

import scipy as sp
import numpy as np

def SAE(X, S, alph):
    A = np.dot(alph * S, S.T)
    B = np.dot(X, X.T)
    D = (1 + alph) * S
    XT = X.T
    C = np.dot(D, XT)
    W = sp.linalg.solve_sylvester(A, B, C)

    return W