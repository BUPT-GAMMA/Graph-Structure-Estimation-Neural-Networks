import torch
import random
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import sys
import networkx as nx
from sklearn.model_selection import ShuffleSplit
from torch_sparse import SparseTensor
from collections import Counter



def accuracy(output, label):
    """ Return accuracy of output compared to label.
    Parameters
    ----------
    output:
        output from model (torch.Tensor)
    label:
        node label (torch.Tensor)
    """
    preds = output.max(1)[1].type_as(label)
    correct = preds.eq(label).double()
    correct = correct.sum()
    return correct / len(label)


def sparse_mx_to_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    rows = torch.from_numpy(sparse_mx.row).long()
    cols = torch.from_numpy(sparse_mx.col).long()
    values = torch.from_numpy(sparse_mx.data)
    return SparseTensor(row=rows, col=cols, value=values, sparse_sizes=torch.tensor(sparse_mx.shape))


def parse_index_f(path):
    """Parse the index file.
    Parameters
    ----------
    path:
        directory of index file (str)
    """
    index = []
    for line in open(path):
        index.append(int(line.strip()))
    return index


def get_mask(idx, l):
    """Create mask.
    """
    mask = torch.zeros(l, dtype=torch.bool)
    mask[idx] = 1
    return mask


def normalize(mx):
    """Row-normalize sparse matrix.
    """
    r_sum = np.array(mx.sum(1))
    r_inv = np.power(r_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_homophily(label, adj):
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = np.triu((label==label.T) & (adj==1)).sum(axis=0)
    d = np.triu(adj).sum(axis=0)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1./ d[i])
    return np.mean(homos)


def load_data(path, dataset):
    """Load input data from directory.
    Parameters
    ----------
    path:
        directory of data (str)
    dataset:
        name of dataset (str)

    Files
    ----------
    ind.dataset.x:
        feature of trainset (sp.csr.csr_matrix)
    ind.dataset.tx:
        feature of testset (sp.csr.csr_matrix)
    ind.dataset.allx:
        feature of both labeled and unlabeled training instances (sp.csr.csr_matrix)
    ind.dataset.y:
        one-hot label of trainset (numpy.array)
    ind.dataset.ty:
        one-hot label of testset (numpy.array)
    ind.dataset.ally:
        label of instances in ind.dataset.allx (numpy.array)
    ind.dataset.graph:
        dictionary in the format {index:[index_of_neighbor_nodes]} (collections.defaultdict)
    ind.dataset.test.index:
        indices of testset for the inductive setting (list)

    All objects above must be saved using python pickle module.
    """
    print("Loading {} dataset...".format(dataset))
    if dataset in ['cora', 'citeseer', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_f("{}/ind.{}.test.index".format(path, dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[:,0] = 1
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        feature = sp.vstack((allx, tx)).tolil()
        feature = normalize(feature)
        feature[test_idx_reorder,:] = feature[test_idx_range,:]
        feature = torch.from_numpy(feature.todense()).float()

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).tolil()
        adj = sparse_mx_to_sparse_tensor(adj)
        
        label = np.vstack((ally, ty))
        label[test_idx_reorder,:] = label[test_idx_range,:]
        label = np.where(label)[1]

        num_class = len(set(label))
        num_node = len(label)
        idx_train = []
        for j in range(num_class):
            idx_train.extend([i for i,x in enumerate(label) if x==j][:20])

        label = torch.LongTensor(label)

        idx_test = test_idx_range.tolist()
        idx_val = range(len(y), len(y) + 500)

    elif dataset == 'sbm':
        with open("{}/{}.p".format(path, dataset), 'rb') as f:
            (G, feature, label) = pkl.load(f)
        f.close()

        feature = normalize(feature)
        feature = torch.from_numpy(feature).float()

        adj = nx.adjacency_matrix(G).tolil()
        adj = sparse_mx_to_sparse_tensor(adj)

        num_class = len(set(label))
        num_node = len(label)
        idx_train = []
        idx_val = []
        idx_test = []
        for j in range(num_class):
            idx_train.extend([i for i, x in enumerate(label) if x == j][:5])
            idx_val.extend([i for i, x in enumerate(label) if x == j][5:10])
            idx_test.extend([i for i, x in enumerate(label) if x == j][10:20])

        label = torch.LongTensor(label)

    else:
        G = nx.DiGraph()
        feature_dict = {}
        label_dict = {}

        with open("{}/{}_{}.txt".format(path, dataset, 'node_feature_label'), 'rb') as f:
            f.readline()
            for line in f:
                line = line.decode().rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in feature_dict and int(line[0]) not in label_dict)
                if dataset == 'actor':
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    feature_dict[int(line[0])] = feature_blank
                else:
                    feature_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                label_dict[int(line[0])] = int(line[2])

        with open("{}/{}_{}.txt".format(path, dataset, 'graph_edges'), 'rb') as f:
            f.readline()
            for line in f:
                line = line.decode().rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=feature_dict[int(line[0])], label=label_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=feature_dict[int(line[1])], label=label_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        feature = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        feature = normalize(feature)
        feature = torch.from_numpy(feature).float()

        adj = nx.adjacency_matrix(G, sorted(G.nodes())).tolil()
        adj = sparse_mx_to_sparse_tensor(adj)

        label = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

        num_class = len(set(label))
        num_node = len(label)
        idx_train = []
        for j in range(num_class):
            idx_train.extend([i for i,x in enumerate(label) if x==j][:20])
        idx_val = range(num_node - 1500, num_node - 1000)
        idx_test = range(num_node - 1000, num_node)

        label = torch.LongTensor(label)

    homophily = get_homophily(label.cpu().numpy(), adj.to_dense().cpu().numpy())

    mask_train = get_mask(idx_train, label.size(0))
    mask_val = get_mask(idx_val, label.size(0))
    mask_test = get_mask(idx_test, label.size(0))
    return DataSet(x=feature, y=label, adj=adj, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
                mask_train=mask_train, mask_val=mask_val, mask_test=mask_test, homophily=homophily)


def prob_to_adj(mx, threshold):
    mx = np.triu(mx, 1)
    mx += mx.T
    (row, col) = np.where(mx > threshold)
    adj = sp.coo_matrix((np.ones(row.shape[0]), (row,col)), shape=(mx.shape[0], mx.shape[0]), dtype=np.int64)
    adj = sparse_mx_to_sparse_tensor(adj)
    return adj


class DataSet():
    def __init__(self, x, y, adj, idx_train, idx_val, idx_test,
                mask_train, mask_val, mask_test, homophily):
        self.x = x
        self.y = y
        self.adj = adj
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.mask_train = mask_train
        self.mask_val = mask_val
        self.mask_test = mask_test
        self.num_node = x.size(0)
        self.num_feature = x.size(1)
        self.num_class = int(torch.max(y)) + 1
        self.homophily = homophily

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.adj = self.adj.to(device)
        self.mask_train = self.mask_train.to(device)
        self.mask_val = self.mask_val.to(device)
        self.mask_test = self.mask_test.to(device)
        return self
