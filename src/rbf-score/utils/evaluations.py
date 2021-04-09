from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import networkx as nx
from sklearn.metrics.cluster import completeness_score

def get_Completeness(y, pre):
    return completeness_score(y, pre)

def get_NMI(y, pre):
    return normalized_mutual_info_score(y, pre)

def get_ARI(y, pre):
    return adjusted_rand_score(y, pre)

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

def modularity(com, G):
    comm = [[] for i in range(len(set(com)))]
    for m in range(len(com)):
        comm[com[m]].append(m)
    edges = G.edges()
    m = len(edges)
    du = G.degree()
    ret2 = 0.0
    for c in comm:
        bian = 0
        for x in c:
            for y in c:
                if x <= y:
                    if (x, y) in edges:
                        bian = bian + 1
                else:
                    if (y, x) in edges:
                        bian = bian + 1
        duHe = 0
        for x in c:
            duHe = duHe + du[x]
        tmp = bian * 1.0 / (2 * m) - (duHe * 1.0 / (2 * m)) * (duHe * 1.0 / (2 * m))
        ret2 = ret2 + tmp
    return ret2