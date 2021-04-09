from scipy.sparse.linalg import eigs, eigsh
import sys
import time
sys.path.append("..")

from utils.laplacian import calLaplacianMatrix
from sklearn.cluster import KMeans


def run_SC(W, k):
    # W, y = utils.get_mat(filename)
    start = time.time()
    # k = len(set(y))
    L, _ = calLaplacianMatrix(W)
    eig_val, eig_vect = eigsh(L, k, which='LM')
    F = eig_vect
    sp_kmeans = KMeans(n_clusters=k).fit(F)
    # NMI = normalized_mutual_info_score(y, sp_kmeans.labels_)
    end = time.time()
    return sp_kmeans.labels_, end-start
