import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
import time
from copy import deepcopy
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity as cos
import pickle as pkl

from utils import accuracy, sparse_mx_to_sparse_tensor, prob_to_adj



"""Backbone GNN Model
Parameters
----------
feature:
    feature of nodes (torch.Tensor)
adj:
    adjacency matrix (torch.Tensor)

Returns
----------
x1:
    node embedding of hidden layer (torch.Tensor)
"""
class GCN(nn.Module):
    def __init__(self, num_feature, num_class, hidden_size, dropout=0.5, activation="relu"):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_feature, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_class)

        self.dropout = dropout
        assert activation in ["relu", "leaky_relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, feature, adj):
        x1 = self.activation(self.conv1(feature, adj))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, adj)
        return x1, F.log_softmax(x2, dim=1)


"""
Parameters
----------
base_model:
    backbone GNN model in GEN
args:
    configs
device:
    "cpu" or "cuda"
"""
class GEN:
    def __init__(self, base_model, args, device):
        self.args = args
        self.device = device
        self.base_model = base_model.to(device)

        self.iter = 0
        self.num_class = 0
        self.num_node = 0

        self.best_acc_val = 0
        self.best_graph = None
        self.hidden_output = None
        self.output = None
        self.weights = None

    def fit(self, data):
        """
        Parameters
        ----------
        data
            x:
                node feature (torch.Tensor)
            adj:
                adjacency matrix (torch.Tensor)
            y:
                node label (torch.Tensor)
            mask_train：
                masked indices of trainset (torch.Tensor)
            mask_val:
                masked indices of valset (torch.Tensor)
            mask_test:
                masked indices of testset (torch.Tensor)
            idx_train:
                node indices of trainset (list)
            idx_val:
                node indices of valset (list)
            idx_test:
                node indices of testset (list)
        """
        args = self.args
        self.num_class = data.num_class
        self.num_node = data.num_node

        estimator = EstimateAdj(data)
        adj = data.adj

        # Train Model
        t_total = time.time()
        for iter in range(args.iter):
            start = time.time()
            self.train_base_model(data, adj, iter)

            estimator.reset_obs()
            estimator.update_obs(self.knn(data.x))
            estimator.update_obs(self.knn(self.hidden_output))
            estimator.update_obs(self.knn(self.output))
            
            self.iter += 1
            alpha, beta, O, Q, iterations = estimator.EM(self.output.max(1)[1].detach().cpu().numpy(), args.tolerance)
            adj = prob_to_adj(Q, args.threshold).to(self.device)

        print("***********************************************************************************************")
        print("Optimization Finished!")
        print("Total time:{:.4f}s".format(time.time() - t_total),
            "Best validation accuracy:{:.4f}".format(self.best_acc_val),
            "EM iterations:{:04d}\n".format(iterations))

        # with open('{}/{}_adj.p'.format('../data', args.dataset), 'wb') as f:
             # pkl.dump((self.best_graph.to_dense().cpu().numpy(), data.y.cpu().numpy()), f)
             # print("Save!")
        # f.close()

    def knn(self, feature):
        adj = np.zeros((self.num_node, self.num_node), dtype=np.int64)
        dist = cos(feature.detach().cpu().numpy())
        col = np.argpartition(dist, -(self.args.k + 1), axis=1)[:,-(self.args.k + 1):].flatten()
        adj[np.arange(self.num_node).repeat(self.args.k + 1), col] = 1
        return adj

    def train_base_model(self, data, adj, iter):
        best_acc_val = 0
        optimizer = optim.Adam(self.base_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        t = time.time()
        for epoch in range(self.args.epoch):
            self.base_model.train()
            optimizer.zero_grad()

            hidden_output, output = self.base_model(data.x, adj)
            loss_train = F.nll_loss(output[data.mask_train], data.y[data.mask_train])
            acc_train = accuracy(output[data.mask_train], data.y[data.mask_train])
            loss_train.backward()
            optimizer.step()

            # Evaluate valset performance (deactivate dropout)
            self.base_model.eval()
            hidden_output, output = self.base_model(data.x, adj)

            loss_val = F.nll_loss(output[data.mask_val], data.y[data.mask_val])
            acc_val = accuracy(output[data.mask_val], data.y[data.mask_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                if acc_val > self.best_acc_val:
                    self.best_acc_val = acc_val
                    self.best_graph = adj
                    self.hidden_output = hidden_output
                    self.output = output
                    self.weights = deepcopy(self.base_model.state_dict())
                    if self.args.debug:
                        print('=== Saving current graph/base_model, best_acc_val:%.4f' % self.best_acc_val.item())

            if self.args.debug:
                if epoch % 1 == 0:
                    print('Epoch {:04d}'.format(epoch + 1),
                        'loss_train:{:.4f}'.format(loss_train.item()),
                        'acc_train:{:.4f}'.format(acc_train.item()),
                        'loss_val:{:.4f}'.format(loss_val.item()),
                        'acc_val:{:.4f}'.format(acc_val.item()),
                        'time:{:.4f}s'.format(time.time() - t))

        print('Iteration {:04d}'.format(iter),
            'acc_val:{:.4f}'.format(best_acc_val.item()))


    def test(self, data):
        """Evaluate the performance on testset.
        """
        print("=== Testing ===")
        print("Picking the best model according to validation performance")
        self.base_model.load_state_dict(self.weights)

        self.base_model.eval()
        hidden_output, output = self.base_model(data.x, self.best_graph)
        loss_test = F.nll_loss(output[data.mask_test], data.y[data.mask_test])
        acc_test = accuracy(output[data.mask_test], data.y[data.mask_test])
        acc_val = accuracy(output[data.mask_val], data.y[data.mask_val])

        print("Testset results: ",
            "loss={:.4f}".format(loss_test.item()),
            "accuracy={:.4f}".format(acc_test.item()))


"""Provide adjacency matrix estimation implementation based on the Expectation-Maximization(EM) algorithm.
Parameters
----------
E:
    The actual observed number of edges between every pair of nodes (numpy.array)
"""
class EstimateAdj():
    def __init__(self, data):
        self.num_class = data.num_class
        self.num_node = data.num_node
        self.idx_train = data.idx_train
        self.label = data.y.cpu().numpy()
        self.adj = data.adj.to_dense().cpu().numpy()

        self.output = None
        self.iterations = 0

        self.homophily = data.homophily

    def reset_obs(self):
        self.N = 0
        self.E = np.zeros((self.num_node, self.num_node), dtype=np.int64)

    def update_obs(self, output):
        self.E += output
        self.N += 1

    def revise_pred(self):
        for j in range(len(self.idx_train)):
            self.output[self.idx_train[j]] = self.label[self.idx_train[j]]

    def E_step(self, Q):
        """Run the Expectation(E) step of the EM algorithm.
        Parameters
        ----------
        Q:
            The current estimation that each edge is actually present (numpy.array)
        
        Returns
        ----------
        alpha:
            The estimation of true-positive rate (float)
        beta：
            The estimation of false-positive rate (float)
        O:
            The estimation of network model parameters (numpy.array)
        """
        # Temporary variables to hold the numerators and denominators of alpha and beta
        an = Q * self.E
        an = np.triu(an, 1).sum()
        bn = (1 - Q) * self.E
        bn = np.triu(bn, 1).sum()
        ad = Q * self.N
        ad = np.triu(ad, 1).sum()
        bd = (1 - Q) * self.N
        bd = np.triu(bd, 1).sum()

        # Calculate alpha, beta
        alpha = an * 1. / (ad)
        beta = bn * 1. / (bd)

        O = np.zeros((self.num_class, self.num_class))

        n = []
        counter = Counter(self.output)
        for i in range(self.num_class):
            n.append(counter[i])

        a = self.output.repeat(self.num_node).reshape(self.num_node, -1)
        for j in range(self.num_class):
            c = (a == j)
            for i in range(j + 1):
                b = (a == i)
                O[i,j] = np.triu((b&c.T) * Q, 1).sum()
                if i == j:
                    O[j,j] = 2. / (n[j] * (n[j] - 1)) * O[j,j]
                else:
                    O[i,j] = 1. / (n[i] * n[j]) * O[i,j]
        return (alpha, beta, O)

    def M_step(self, alpha, beta, O):
        """Run the Maximization(M) step of the EM algorithm.
        """
        O += O.T - np.diag(O.diagonal())

        row = self.output.repeat(self.num_node)
        col = np.tile(self.output, self.num_node)
        tmp = O[row,col].reshape(self.num_node, -1)

        p1 = tmp * np.power(alpha, self.E) * np.power(1 - alpha, self.N - self.E)
        p2 = (1 - tmp) * np.power(beta, self.E) * np.power(1 - beta, self.N - self.E)
        Q = p1 * 1. / (p1 + p2 * 1.)
        return Q

    def EM(self, output, tolerance=.000001):
        """Run the complete EM algorithm.
        Parameters
        ----------
        tolerance:
            Determine the tolerance in the variantions of alpha, beta and O, which is acceptable to stop iterating (float)
        seed:
            seed for np.random.seed (int)

        Returns
        ----------
        iterations:
            The number of iterations to achieve the tolerance on the parameters (int)
        """
        # Record previous values to confirm convergence
        alpha_p = 0
        beta_p = 0

        self.output = output
        self.revise_pred()

        # Do an initial E-step with random alpha, beta and O
        # Beta must be smaller than alpha
        beta, alpha = np.sort(np.random.rand(2))
        O = np.triu(np.random.rand(self.num_class, self.num_class))
        
        # Calculate initial Q
        Q = self.M_step(alpha, beta, O)

        while abs(alpha_p - alpha) > tolerance or abs(beta_p - beta) > tolerance:
            alpha_p = alpha
            beta_p = beta
            alpha, beta, O = self.E_step(Q)
            Q = self.M_step(alpha, beta, O)
            self.iterations += 1

        if self.homophily > 0.5:
            Q += self.adj
        return (alpha, beta, O, Q, self.iterations)
