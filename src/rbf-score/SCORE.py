from scipy.sparse.linalg import eigs, eigsh

import sys
import numpy as np
import time

sys.path.append("..")

from utils.laplacian import calLaplacianMatrix
from sklearn.cluster import KMeans


def run_SCORE(W, k):
    start = time.time()
    # L, _ = calLaplacianMatrix(np.array(W))
    eig_val, eig_vect = eigsh(W.astype(float), k, which='LM')
    l_eig_val, l_eig_vect = eig_val[k - 1], eig_vect[:, k - 1]
    # eig_vect = np.dot(eig_vect1, np.diag(eig_val / eig_val[k-1]))
    F = eig_vect[:, :k - 1]
    for i in range(k - 1):
        F[:, i] = np.multiply(eig_vect[:, i], 1. / l_eig_vect)
    F = np.dot(F, np.diag(eig_val[:k-1]/eig_val[k-1]))
    sp_kmeans = KMeans(n_clusters=k, init='random').fit(F)
    end = time.time()
    return sp_kmeans.labels_, end - start
