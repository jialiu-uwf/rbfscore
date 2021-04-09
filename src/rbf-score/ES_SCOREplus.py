from scipy.sparse.linalg import eigs, eigsh
import utils.utils as utils
import sys
import numpy as np
import time
import math
from sklearn import mixture

sys.path.append("..")

from utils.laplacian import calLaplacianMatrix
from sklearn.cluster import KMeans


def run_ES_SCOREplus(W, k, c=0.1):
    start = time.time()
    r = k + 1
    n = len(W)
    Degree = np.sum(W, axis=1)
    # D = np.diag(Degree)
    delta = c * max(Degree)
    # I = np.identity(len(Degree))
    d = 1. / np.sqrt(np.add(delta, Degree))
    # d = 1. / np.add(delta, Degree)
    # D^(-1/2) L D^(-1/2)
    sqrtMatrix = np.diag(d)
    L = np.dot(np.dot(sqrtMatrix, W), sqrtMatrix)
    eig_val, eig_vect = eigsh(L, r, which='LM')
    tao = 0.2
    ratio = eig_val[k] / eig_val[k - 1]

    F = np.dot(eig_vect[:, :r], np.diag(eig_val[: r]))
    if ratio < 1 + tao:
        F = F[:, :]
        for i in range(r - 1):
            F[:, i] = np.multiply(eig_vect[:, i], 1. / eig_vect[:, r - 1])
        temp = (eig_val[0] - eig_val[1]) / eig_val[1]
        # print(temp)
        if temp < c:
            F = F[:, 1:(r-1)]
        # sp_kmeans = KMeans(n_clusters=k).fit(F)
        sp_kmeans = mixture.BayesianGaussianMixture(n_components=k + 1, covariance_type='full').fit(F)
    else:
        F = F[:, :r - 1]
        for i in range(r - 1):
            F[:, i] = np.multiply(eig_vect[:, i], 1. / eig_vect[:, r - 1])
        temp = (eig_val[0] - eig_val[1]) / eig_val[1]
        # print(temp)
        if temp < c:
            F = F[:, 1:(r - 1)]
        # sp_kmeans = KMeans(n_clusters=k).fit(F)
        sp_kmeans = mixture.BayesianGaussianMixture(n_components=k + 1, covariance_type='full').fit(F)
    # print(ratio, 1 + tao)
    end = time.time()
    # print(p, max(l)-min(l)+1)
    # return sp_kmeans.labels_, end - start
    return sp_kmeans.predict(F), end - start
