
# mySVD  Accelerated singular value decomposition.
# [U, S, V] = mySVD(X) produces a diagonal matrix S, of the
# dimension as the rank of X and with nonnegative diagonal elements in
# decreasing order, and unitary matrices U and V so that
# X = U * S * V'.
#
# [U, S, V] = mySVD(X, ReducedDim) produces a diagonal matrix S, of the
# dimension as ReducedDim and with nonnegative diagonal elements in
# decreasing order, and unitary matrices U and V so that
# Xhat = U * S * V' is the best approximation (with respect to F norm) of X
# among all the matrices with rank no larger than ReducedDim.
#
# Based on the size of X, mySVD computes the eigvectors of X * X ^ T or X ^ T * X
# first, and then convert them to the eigenvectors of the other.
#
# See also SVD.
#

import numpy as np
import scipy
from util.isspare import isSparse

def mySVD(X, ReducedDim):
    MAX_MATRIX_SIZE = 1600   # You can change this number according your machine computational power
    EIGVECTOR_RATIO = 0.1    # You can change this number according your machine computational power

    if ReducedDim is None:
        ReducedDim = 0

    nSmp, mFea = np.shape(X)
#    if (mFea/nSmp) > 1.0713:
    ddata = np.dot(X, X.T)
    ddata = np.maximum(ddata, ddata.T)

    dimMatrix = ddata.shape[1]
    if (ReducedDim > 0) and (dimMatrix > MAX_MATRIX_SIZE) and (ReducedDim < dimMatrix * EIGVECTOR_RATIO):
        option = {}
        option['disp'] = 0
        eigvalue, U = np.linalg.eigh(ddata)
        eigvalue = np.diag(eigvalue)
    else:
#        if isSparse(ddata):
#            ddata = np.full(ddata,0)

        eigvalueT, U = np.linalg.eig(ddata)
        idx = eigvalueT.argsort()[::-1]
        eigvalueT = eigvalueT[idx]
        U = U[:,idx]
        eigvalue = eigvalueT.reshape(eigvalueT.shape[0],1)

    maxEigValue = np.max(np.abs(eigvalue))

    eigIdx = []
    for a in range(eigvalue.shape[0]):
        if np.abs(eigvalue[a,0])/maxEigValue < 1e-10:
            eigIdx.append(a)
    eigIdx = np.array(eigIdx)

#    for b in range(eigIdx.shape[0]):
    eigvalue = np.delete(eigvalue, eigIdx, axis=0)
    U = np.delete(U, eigIdx, axis=1)

    if (ReducedDim > 0) and (ReducedDim < eigvalue.shape[0]):
        eigvalue = eigvalue[1:ReducedDim]
        U = U[:, 1:ReducedDim]

    eigvalue_Half = pow(eigvalue, 0.5)

    S = np.zeros((eigvalue_Half.shape[0], eigvalue_Half.shape[0]))
    for line in range(eigvalue_Half.shape[0]):
        S[line,line] = eigvalue_Half[line,0]

#    else:
#        ddata = np.dot(X, X.T)
#        ddata = np.maximum(ddata, ddata.T)

#        dimMatrix = ddata.shape[1]
#        if (ReducedDim > 0) and (dimMatrix > MAX_MATRIX_SIZE) and (ReducedDim < dimMatrix * EIGVECTOR_RATIO):
#            option = {}
#            option['disp'] = 0
#            eigvalue, V = np.linalg.eigh(ddata)
#            eigvalue = np.diag(eigvalue)
#        else:
#            if isSparse(ddata):
#                ddata = np.full(ddata)

#            eigvalueT, V = np.linalg.eig(ddata)
#            idx = eigvalueT.argsort()[::-1]
#            eigvalueT = eigvalueT[idx]
#            V = V[:, idx]
#            eigvalue = eigvalueT.reshape(eigvalueT.shape[0],1)

#        maxEigValue = np.max(np.abs(eigvalue))

#        eigIdx = []
#        for a in range(eigvalue.shape[0]):
#            if np.abs(eigvalue[a, 0]) / maxEigValue < 1e-10:
#                eigIdx.append(a)# = np.row_stack(eigIdx, [a])
#        eigIdx = np.array(eigIdx)

#        for b in range(eigIdx.shape[0]):
#            eigvalue = np.delete(eigvalue, eigIdx[b], axis=0)
#            V = np.delete(U, eigIdx[b], axis=1)

#        if (ReducedDim > 0) and (ReducedDim < eigvalue.shape[0]):
#            eigvalue = eigvalue[1:ReducedDim]
#            V = V[:, 1:ReducedDim]

#        eigvalue_Half = pow(eigvalue, 0.5)

#        S = np.zeros((eigvalue_Half.shape[0], eigvalue_Half.shape[0]))
#        for line in range(eigvalue_Half.shape[0]):
#            S[line, line] = eigvalue_Half[line, 0]

#        eigvalue_MinusHalf = pow(eigvalue_Half, -1)
#        A = np.tile(eigvalue_MinusHalf.T, (V.shape[0], 1))
 #       B = V * A
 #       U = np.dot(B, X)

    return U, S  #, V