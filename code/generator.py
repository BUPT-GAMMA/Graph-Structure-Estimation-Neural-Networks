import networkx as nx
import numpy as np
import pickle as pkl



num_class = 5
num_feature = 8

size = [20 for i in range(num_class)]
prob = [[0.173, 0.078, 0.085, 0.007, 0.024],
        [0.078, 0.165, 0.101, 0.070, 0.044],
        [0.085, 0.101, 0.178, 0.159, 0.020],
        [0.007, 0.070, 0.159, 0.118, 0.090],
        [0.024, 0.044, 0.020, 0.090, 0.178]]

G = nx.stochastic_block_model(size, prob)

cov = np.triu(np.random.rand(num_feature, num_feature))
cov += cov.T - np.diag(cov.diagonal())

feature = np.zeros((20*num_class, num_feature))
label = []
for i in range(num_class):
    label.extend([i for j in range(20)])
    mean = np.random.randn(num_feature)
    feature[i*20:(i+1)*20,:] = np.random.multivariate_normal(mean, cov, 20)

label = np.array(label)

with open('../data/sbm.p', 'wb') as f:
    pkl.dump((G, feature, label), f)
f.close()

print(G.graph["partition"])