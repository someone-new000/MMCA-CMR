#PCA	Principal Component Analysis
#
#	Usage:
#       [eigvector, eigvalue] = PCA(data, options)
#       [eigvector, eigvalue] = PCA(data)
#
#             Input:
#               data       - Data matrix. Each row vector of fea is a data point.
#
#     options.ReducedDim   - The dimensionality of the reduced subspace. If 0,
#                         all the dimensions will be kept.
#                         Default is 0.
#
#             Output:
#               eigvector - Each column is an embedding function, for a new
#                           data point (row vector) x,  y = x*eigvector
#                           will be the embedding result of x.
#               eigvalue  - The sorted eigvalue of PCA eigen-problem.
#
#	Examples:
# 			fea = rand(7,10);
#           options=[];
#           options.ReducedDim=4;
# 			[eigvector,eigvalue] = PCA(fea,4);
#           Y = fea*eigvector;
#
#


#coding=utf-8
import numpy as np
from util.mySVD import mySVD

def myPCA(data, options):
    if options is None:
        options = {}

    ReducedDim = 0
    if ReducedDim in options:
        ReducedDim = options.ReducedDim

    nSmp, nFea = np.shape(data)
    if (ReducedDim > nFea) or (ReducedDim <= 0):
        ReducedDim = nFea

    sampleMeanT = np.mean(data, axis=0)
    sampleMean= sampleMeanT.reshape(1,sampleMeanT.shape[0])
    Mean = np.tile(sampleMean, (nSmp, 1))
    data = data - Mean

#    cov_Mat = np.dot(np.transpose(data),data)
#    eigvalue, eigvector = np.linalg.eig(cov_Mat)
#    index = np.argsort(-eigvalue)
#    eigvalueSort = eigvalue[index]
#    eigvectorSort = eigvector[:, index]
#    maxEigValue = max(abs(eigvalue))
#    eigIdx =  abs(eigvalueSort)/maxEigValue >= 1e-10
#    eigvalue = eigvalueSort[eigIdx]
#    eigvector = eigvectorSort[:, eigIdx]

    eigvector, eigvalue = mySVD(data.T, ReducedDim)
    eigvalue = pow((np.diag(eigvalue)), 2)
    eigvalue = eigvalue.reshape(eigvalue.shape[0],1)

    sumEig = np.sum(eigvalue)
    sumEig = sumEig * options['PCARatio']
    sumNow = 0
    for idx in range(eigvalue.shape[0]):
        sumNow = sumNow + eigvalue[idx,0]
        if sumNow >= sumEig:
            break
    idx = idx + 1
    eigvector = eigvector[:, 0:idx]

    return eigvector, eigvalue