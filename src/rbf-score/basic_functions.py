import math
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import pandas as pd
from pylab import *
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs, eigsh
import scipy.io
from sklearn.metrics.cluster import normalized_mutual_info_score
import networkx as nx

from utils.evaluations import *
plt.rcParams["font.family"] = "Times New Roman"

def condi(M):
    return LA.cond(M)


def MQ(data, condi):
    l, w = data.shape
    distance = np.zeros((l, w))
    for i in range(l):
        for j in range(w):
            r = DMatrix(data[i], data[j])
            distance[i][j] = math.sqrt(r * r + condi * condi)

    return distance

def networkG(W):
    # g = igraph.Graph.Adjacency((W > 0).tolist())
    G = nx.from_numpy_matrix(W)

    return G

def get_mat(filename):
    mat = scipy.io.loadmat(filename)
    label = mat['label']
    y = np.asarray(label).reshape(label.shape[0])
    return mat['A'], y

def iMQ(data, condi):
    l, w = data.shape
    distance = np.zeros((l, w))
    for i in range(l):
        for j in range(w):
            r = DMatrix(data[i], data[j])
            d = math.sqrt(r * r + condi * condi)
            distance[i][j] = 0 if d == 0 else 1. / d

    return distance


def gaussianBF(data, condi):
    l, w = data.shape
    distance = np.zeros((l, w))
    for i in range(l):
        for j in range(w):
            r = DMatrix(data[i], data[j])
            d = np.exp(-r * r / (condi * condi))
            distance[i][j] = 0 if d == 0 else d

    return distance


def DMatrix(x, y, norm=False):
    """ distance computation """

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos

def get_NMI(y, pre):
    return normalized_mutual_info_score(y, pre)

def get_Q(com, G):
    # print(com)
    comm = [[] for i in range(len(set(com))+1)]
    # print(set(com))
    # print(comm)
    if min(com) == 1:
        for m in range(len(com)):
            comm[com[m] - 1].append(m)
    else:
        for m in range(len(com)):
            comm[com[m]].append(m)
    # print(comm)
    # print(nx.algorithms.community.modularity(G, comm))
    return nx.algorithms.community.modularity(G, comm)

def plotNMIQ(condition, NMI, Q, fn):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
    # ax1.set_xlabel(r"$\log_2 (condition~number)$")  # Add an x-label to the axes.
    ax1.set_ylabel("NMI")  # Add a y-label to the axes.
    ax1.set_title("Comparisons on condition numbers of "+fn.split('/')[1])  # Add a title to the axes.ax1.xlabel(r"$\log_2 (condition~number)$", fontdict={'fontsize': 20})  # Add an x-label to the axes.
    ax2.set_xlabel(r"$\log_2 (condition~number)$")  # Add an x-label to the axes
    ax2.set_ylabel("Modularity")  # Add a y-label to the axes.
    # ax2.set_title("Modularity on condition numbers of an adjacency matrix")  # Add a title to the axes.

    ax1.plot(condition, NMI, '*b')
    ax2.plot(condition, Q, '^m')
    # plt.show()
    fig.savefig(fn+'.pdf')
    print(fn+' completed...')

def SCOREplus(W, k, c=0.1):
    start = time.time()
    r = k + 1
    Degree = np.sum(W, axis=1)
    delta = c * max(Degree)
    d = 1. / np.sqrt(np.add(delta, Degree))
    sqrtMatrix = np.diag(d)
    L = np.dot(np.dot(sqrtMatrix, W), sqrtMatrix)
    eig_val, eig_vect = eigsh(L, r, which='LM')
    F = eig_vect[:, :r - 1]
    for i in range(r - 1):
        F[:, i] = np.multiply(eig_vect[:, i], 1. / eig_vect[:, r - 1])
    temp = (eig_val[0] - eig_val[1]) / eig_val[1]
    if temp <= c:
        F = F[:, 1:(r - 1)]
    sp_kmeans = KMeans(n_clusters=k).fit(F)
    end = time.time()
    return sp_kmeans.labels_, end - start


def run_networks(fn='football'):

    # for lesmis
    fnn = "../data/lesmis/"

    if fn.strip() == "lesmis":
        data = pd.read_table(fnn + 'lesmis.txt', header=None, sep='\t').to_numpy()
        label = pd.read_table(fnn + 'lesmis_stan.txt', header=None, sep='\t')
        l = label.shape[0]
        y = [label[0][i] for i in range(l)]
        k = len(set(y))

    # for other data
    else:
        folder = '../data/datasets/'

        # filename = '../data/datasets/dolphins.mat'
        W, y = get_mat(folder+fn+'.mat')
        data = W
        # print(len(W))
        k = max(y) - min(y) + 1

    G = networkG(data)
    n = 50
    c = np.linspace(0, 5, n)
    N = data.shape[1]
    x = np.linspace(0, 1, N)

    condition = [0 for i in range(n)]
    NMI = [0 for i in range(n)]
    Q = [0 for i in range(n)]

    # rcon = [0 for i in range(n)]

    for i in range(n):
        # A = gaussianBF(data, c[i])
        A = MQ(data, c[i])
        # A = iMQ(data, c[i])

        # --- condition number
        # con = np.log2(condi(A))
        # rcon[i] = con
        w = np.absolute(np.linalg.eigvals(A))
        W = np.dot(A, data)

        # print(A.shape, k)
        splus, t4 = SCOREplus(W, k, c=0.1)
        # condition[i] = np.log2(max(w)/min(w))
        condition[i] = np.log2(max(w) / min(w))
        NMI4 = round(get_NMI(y, splus), 3)
        Q4 = round(modularity(splus, G), 3)
        NMI[i] = NMI4
        Q[i] = Q4

        # print(round(c[i], 4), round(con, 2), condition[i], ": ", NMI4, Q4)
        # print(round(c[i], 4), condition[i], ": ", NMI4, Q4)

    print("\nc, condition number for optimal NMI - " + fn.split('.')[0])
    q = NMI.index(max(NMI))
    print(round(c[q], 4), round(2**condition[q]), ": ", NMI[q], Q[q])

    print("\nc, condition number for optimal Modularity - " + fn.split('.')[0])
    q = Q.index(max(Q))
    print(round(c[q], 4), round(2 ** condition[q]), ": ", NMI[q], Q[q])

    # plotNMIQ(condition, NMI, Q, fn.split('.')[0])


if __name__ == "__main__":
    run_networks(fn='lesmis')

    """
    # n, p1, p2 = 800, 20, 15
    # nn, p1, p2 = 150, 10, 10
    nn, p1, p2 = 50, 5, 5

    folder = '../data/LFR/' + str(nn)
    for mu in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
    # for mu in [0.45]:
        fn = str(mu)+'_'+str(p1)+'_'+str(p2)
        filename = folder + '/LFR__' + fn
        # G = nx.read_adjlist(filename+'.adjlist', nodetype=int)
        label = pd.read_table(filename + '.labels', header=None, sep='\t')
        W = pd.read_table(filename + '.adj', header=None, sep=' ').to_numpy()
        l = label.shape[0]
        y = [label[0][i] for i in range(l)]
        # W = nx.to_pandas_adjacency(G)
        G = nx.from_numpy_matrix(W)
        data = W
        # print(len(W))
        k = max(y) - min(y) + 1

        # G = networkG(data)
        n = 50
        c = np.linspace(0, 5, n)
        N = data.shape[1]
        x = np.linspace(0, 1, N)

        condition = [0 for i in range(n)]
        NMI = [0 for i in range(n)]
        Q = [0 for i in range(n)]

        # rcon = [0 for i in range(n)]

        for i in range(n):
            A = MQ(data, c[i])
            # A = iMQ(data, c[i])

            # --- condition number
            # con = np.log2(condi(A))
            # rcon[i] = con
            w = np.absolute(np.linalg.eigvals(A))
            W = np.dot(A, data)

            # print(A.shape, k)
            splus, t4 = SCOREplus(W, k, c=0.1)
            # condition[i] = np.log2(max(w)/min(w))
            condition[i] = np.log2(max(w) / min(w))
            NMI4 = round(get_NMI(y, splus), 3)
            Q4 = round(modularity(splus, G), 3)
            NMI[i] = NMI4
            Q[i] = Q4

            # print(round(c[i], 4), round(con, 2), condition[i], ": ", NMI4, Q4)
            # print(round(c[i], 4), condition[i], ": ", NMI4, Q4)

        plotNMIQ(condition, NMI, Q, 'res/'+str(nn)+'_'+fn)
    """
