from scipy.sparse.linalg import eigs, eigsh
import utils.utils as utils
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import mixture
import sys
from sklearn import mixture
import numpy as np
import time
import SCOREplus
import ES_SCOREplus
import math
import utils.higer_approximation as ha
sys.path.append("..")

from utils.laplacian import calLaplacianMatrix
from sklearn.cluster import KMeans


def run_hSCOREplus(W, k, c=0.1, beta=0.01):
    start = time.time()
    # W = ha.get_AA(W)
    # W = ha.get_CN(W)
    # W = ha.get_katz(W, beta)
    # W = ha.get_1stplus2nd(W, alpha=1)
    return ES_SCOREplus.run_ES_SCOREplus(W, k, c=c)
    # return SCOREplus.run_SCOREplus(W, k, c=c)
    # r = k + 1
    # Degree = np.sum(W, axis=1)
    # # D = np.diag(Degree)
    # delta = c * max(Degree)
    # # I = np.identity(len(Degree))
    # d = 1. / np.sqrt(np.add(delta, Degree))
    # # D^(-1/2) L D^(-1/2)
    # sqrtMatrix = np.diag(d)
    # L = np.dot(np.dot(sqrtMatrix, W), sqrtMatrix)
    # # print(L.shape)
    # eig_val, eig_vect1 = eigsh(L, r, which='LM')
    # # print(eig_val, eig_vect1)
    # eig_vect = np.dot(eig_vect1, np.diag(eig_val))
    # # print(eig_vect.shape)
    # # l_eig_val, l_eig_vect = eig_val[k-1], eig_vect[:, k-1]
    # F = eig_vect[:, :r - 1]
    # for i in range(r - 1):
    #     F[:, i] = np.multiply(eig_vect[:, i], 1. / eig_vect[:, r - 1])
    #
    # temp = (eig_val[1] - eig_val[0]) / eig_val[1]
    # # print(temp)
    # if temp > c:
    #     F = F[:, 1:(r - 1)]
    #
    # sp_kmeans = KMeans(n_clusters=k).fit(F)
    # # dpgmm = mixture.BayesianGaussianMixture(n_components=k, covariance_type='full').fit(F)
    # end = time.time()
    # return sp_kmeans.labels_, end - start

