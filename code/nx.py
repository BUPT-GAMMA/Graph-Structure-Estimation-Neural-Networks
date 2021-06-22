import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt



path = "../data"
dataset = "sbm"
num_class = 5

with open("{}/{}.p".format(path, dataset), 'rb') as f:
    (_, _, label) = pkl.load(f)
f.close()

with open("{}/{}_adj.p".format(path, dataset), 'rb') as f:
    adj = pkl.load(f)
f.close()

G = nx.Graph(adj)

label_dict = dict(zip(range(len(label)), label))
nx.set_node_attributes(G, label_dict, 'label')
nx.set_node_attributes(G, label_dict, 'community')

node_list = []
for i in range(num_class):
    node_list.append(np.argwhere(label == i).flatten().tolist())

color = 0
color_map = ['red', 'blue', 'yellow', 'purple', 'black', 'green', 'pink']
for label in range(num_class):
    nx.draw(G, pos = nx.spring_layout(G, iterations = 200), nodelist = node_list[label], node_size = 30, node_color = color_map[color], width=2)
    color += 1
nx.write_gexf(G, 'sbm.gexf')
