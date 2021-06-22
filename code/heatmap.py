import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import networkx as nx
import scipy.sparse as sp



def get_homophily(label, adj):
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = np.triu((label==label.T) & (adj==1), 1).sum(axis=0)
    d = np.triu(adj, 1).sum(axis=0)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1./ d[i])
    return np.mean(homos)


def get_O(num_class, label, adj):
    O = np.zeros((num_class, num_class))
    num_node = len(label)

    n = []
    counter = Counter(label)
    for i in range(num_class):
        n.append(counter[i])

    a = label.repeat(num_node).reshape(num_node, -1)
    for j in range(num_class):
        c = (a == j)
        for i in range(j + 1):
            b = (a == i)
            O[i,j] = np.triu((b&c.T) * adj, 1).sum()
            if i == j:
                O[j,j] = 2. / (n[j] * (n[j] - 1)) * O[j,j]
            else:
                O[i,j] = 1. / (n[i] * n[j]) * O[i,j]

    O += O.T - np.diag(O.diagonal())
    return O

path = "../data"
dataset = "sbm"
num_class = 5

with open("{}/{}.p".format(path, dataset), 'rb') as f:
    (G, feature, label) = pkl.load(f)
f.close()

with open("{}/{}_adj.p".format(path, dataset), 'rb') as f:
     estimated = pkl.load(f)
f.close()

origin = nx.adjacency_matrix(G).todense()

ho = get_homophily(label, origin)
he = get_homophily(label, estimated)

print("Homophily\tOrigin graph:{:.4f}".format(ho), "Estimated graph:{:.4f}".format(he))

oo = get_O(num_class, label, origin)
oe = get_O(num_class, label, estimated)

sns.set(font_scale=2)
sns.heatmap(oe, cmap="YlGnBu", linewidths=3, xticklabels=False, yticklabels=False,
            cbar_kws={"ticks":np.arange(0,0.5,0.1)})

plt.savefig('estimated_o.png')
