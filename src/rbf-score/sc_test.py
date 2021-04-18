import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from cond_test import rbf, multiquadric, inv_multiquadric, gaussian


def normalized_laplacian_matrix(adjacentMatrix):

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (np.sqrt(degreeMatrix)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)


def spectral_clustering(adjacency_matrix, k, c, rbf_fn, eig_which='LA'):
    """
    Spectral clustering
    """
    W = rbf(c, adjacency_matrix, rbf_fn)
    L = normalized_laplacian_matrix(W)
    eig_val, eig_vect = eigsh(L, k, which=eig_which)  # 'SA' in matlib code
    sp_kmeans = KMeans(n_clusters=k).fit(eig_vect)
    return sp_kmeans.labels_


def sc_test():
    """
    Sprectral clustering test run
    """
    c = 0.01
    rbf_fn = multiquadric
    k = 11

    ds_name = 'lesmis'
    fnn = "../../data/lesmis/"
    data = pd.read_table(fnn + 'lesmis.txt', header=None, sep='\t').values
    assert data.shape[0] == data.shape[1]
    sc = spectral_clustering(data, k, c, rbf_fn)
    print(sc)


if __name__ == "__main__":
    # main()
    sc_test()
