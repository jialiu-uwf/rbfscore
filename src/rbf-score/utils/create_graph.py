import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
from sklearn.cluster import KMeans
from numpy import savez_compressed
import numpy as np


def write_to(G, filename):
    communities = {frozenset(G.nodes[v]['community']) for v in G}
    k = len(communities)
    y = [i for i in range(n)]
    for m in range(k):
        for s in list(communities)[m]:
            y[s] = m
    W = nx.to_numpy_array(G)
    np.savetxt(filename+'.labels', y, fmt="%d")
    np.savetxt(filename+'.adj', W, fmt="%d")
    # nx.write_adjlist(G, filename+'.adjlist')


if __name__ == "__main__":
    n = 10000
    # for mu in [0.85]:
    for mu in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
    # for mu in [0.45]:
        tau1 = 3
        tau2 = 1.5
        ave_d = 20
        min_comm = 50
        max_d = int(0.1 * n)
        max_com = int(0.1 * n)
        G = LFR_benchmark_graph(n, tau1, tau2, mu, max_community=max_com, max_degree=max_d, min_degree=min_comm)
        folder = str(n)+'/'
        fn = 'LFR_' + '_' + str(mu) + '_' + str(ave_d) + '_' + str(min_comm)
        write_to(G, folder+fn)