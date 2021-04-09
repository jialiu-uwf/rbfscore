from scipy.sparse.linalg import eigs, eigsh
import utils.utils as utils
import sys
import numpy as np
import time
from sklearn import mixture
import math

sys.path.append("..")

from utils.laplacian import calLaplacianMatrix
from sklearn.cluster import KMeans
from src.conductivity import *

def run_SCOREplus(W, k, c=0.1):
    # W = cal_C(W)
    start = time.time()
    r = k + 1
    Degree = np.sum(W, axis=1)
    # D = np.diag(Degree)
    delta = c * max(Degree)
    I = np.identity(len(Degree))
    d = 1. / np.sqrt(np.add(delta, Degree))
    # D^(-1/2) L D^(-1/2)
    sqrtMatrix = np.diag(d)
    L = np.dot(np.dot(sqrtMatrix, W), sqrtMatrix)
    # L=I-L
    # print(L.shape)
    eig_val, eig_vect = eigsh(L, r, which='LM')
    # print(eig_val, eig_vect1)
    # print(eig_val[k-1], eig_val[k])
    # eig_vect = np.dot(eig_vect1, np.diag(eig_val/eig_val[k]))
    # print(eig_vect.shape)
    # l_eig_val, l_eig_vect = eig_val[k-1], eig_vect[:, k-1]
    F = eig_vect[:, :r - 1]
    for i in range(r - 1):
        F[:, i] = np.multiply(eig_vect[:, i], 1. / eig_vect[:, r - 1])

    temp = (eig_val[0] - eig_val[1]) / eig_val[1]
    # print(temp)
    if temp <= c:
        F = F[:, 1:(r - 1)]
    # print(temp)
    sp_kmeans = KMeans(n_clusters=k).fit(F)
    # sp_kmeans = mixture.BayesianGaussianMixture(n_components=k+1, covariance_type='full').fit(F)
    end = time.time()

    return sp_kmeans.labels_, end - start
    # return sp_kmeans.predict(F), end - start