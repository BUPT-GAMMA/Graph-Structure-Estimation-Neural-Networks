import argparse
import torch
import numpy as np
import warnings

from utils import load_data
from models import GCN, GEN



warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=True, help='debug mode')
parser.add_argument('--base', type=str, default='gcn', choices=['gcn', 'sgc', 'gat', 'appnp', 'sage'])
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters)')
parser.add_argument('--hidden', type=int, default=16, help='hidden size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu', 'elu'])
# parser.add_argument('--num_head', type=int, default=8, help='number of heads in GAT')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'sbm'])
parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train the base model')
parser.add_argument('--iter', type=int, default=30, help='number of iterations to train the GEN')
parser.add_argument('--k', type=int, default=9, help='k of knn graph')
parser.add_argument('--threshold', type=float, default=.5, help='threshold for adjacency matrix')
parser.add_argument('--tolerance', type=float, default=.01, help='tolerance to stop EM algorithm')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

data = load_data("../data", args.dataset).to(device)

base_model_args = {"num_feature": data.num_feature, "num_class":data.num_class,
                "hidden_size": args.hidden, "dropout":args.dropout, "activation": args.activation}
if args.base == 'gcn':
    base_model = GCN(**base_model_args)
# elif args.base == 'sgc':
#     base_model = SGC(**base_model_args)
# elif args.base == 'gat':
#     base_model_args["num_head"] = args.num_head
#     base_model = GAT(**base_model_args)
# elif args.base == 'appnp':
#     base_model = PPNP(**base_model_args)
# elif args.base == 'sage':
#     base_model = SAGE(**base_model_args)

model = GEN(base_model, args, device)
model.fit(data)
model.test(data)
