from scipy.sparse.linalg import eigs, eigsh
import utils.utils as utils
from sklearn.metrics.cluster import normalized_mutual_info_score
import sys
import numpy as np
import time
import math
import networkx as nx
from numpy.linalg import inv
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

sys.path.append("..")
from src.conductivity import *


def get_1stplus2nd(W, l, alpha=1):
    """
        Function to generate a neighbourhood overlap matrix (second-order proximity matrix).
        :param G: Graph object.
        :return laps: Overlap matrix.
        """

    # for j in range(l):
    #     A_sparse = sparse.csr_matrix(W)
    #     laps = cosine_similarity(A_sparse)
    #     # for i in range(len(W)):
    #     #     laps[i][i] = 0
    #     W = W + laps/(j+1)

    K = get_katz(W, 0.0026)
    A_sparse = sparse.csr_matrix(W)
    laps = cosine_similarity(A_sparse)
    # for i in range(len(W)):
    #     laps[i][i] = 0
    # W = W + laps

    # return laps+2*W+K
    return K+laps


def get_katz(W, beta):
    # G = nx.from_numpy_matrix(W)
    # Degree = np.sum(W, axis=1)
    # n = nx.number_of_nodes(G)
    # m = nx.number_of_edges(G)
    # l = m/n
    # t = nx.density(G)/(l)
    # # beta = t*t
    # beta = t
    # print(beta)
    # print(beta)
    Ml = beta * W
    Mg = inv(np.identity(len(W)) - Ml)
    W = np.dot(Mg, Ml)
    return W


def get_CN(W):
    return np.dot(W, W)


def get_AA(W):
    Degree = np.sum(W, axis=1)
    D = np.diag(Degree)
    return np.dot(np.dot(W, np.multiply(D, 1. / max(Degree))), W)
