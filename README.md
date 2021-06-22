# GEN

Source code for WWW2021 paper ["**Graph Structure Estimation Neural Networks**"](https://doi.org/10.1145/3442381.3449952)



## Environment Settings

* python == 3.6.9
* torch == 1.6.0



## Parameter Settings

- k: k of knn graph
- threshold: threshold for adjacency matrix
- tolerance: tolerance to stop EM algorithm
- iter: number of iterations to train the GEN
- base: backbone GNNs
- seed: random seed
- lr: learning rate
- weight_decay: weight decay (L2 loss on parameters)
- hidden: embedding dimension
- dropout: dropout rate
- activation: activation function selection
- dataset: str in ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'sbm']
- epoch: number of epochs to train the base model



## Files in the folder

~~~~
GEN/
├── code/
│   ├── train.py: training the GEN model
│   ├── models.py: implementation of GEN and backbone GNNs
│   ├── utils.py
│   ├── generator.py: generating dataset based on attribute SBM
│   ├── nx.py: saving graph structure as .gexf files for Gephi
│   └── heatmap.py: generating heatmaps of community matrices
├── data/
│   ├── ind.cora.x: cora dataset
│   ├── ind.cora.y
│   ├── ind.cora.tx
│   ├── ind.cora.ty
│   ├── ind.cora.allx
│   ├── ind.cora.ally
│   ├── ind.cora.graph
│   ├── ind.cora.test.index
│   ├── ind.citeseer.x: citeseer dataset
│   ├── ind.citeseer.y
│   ├── ind.citeseer.tx
│   ├── ind.citeseer.ty
│   ├── ind.citeseer.allx
│   ├── ind.citeseer.ally
│   ├── ind.citeseer.graph
│   ├── ind.citeseer.test.index
│   ├── ind.pubmed.x: pubmed dataset
│   ├── ind.pubmed.y
│   ├── ind.pubmed.tx
│   ├── ind.pubmed.ty
│   ├── ind.pubmed.allx
│   ├── ind.pubmed.ally
│   ├── ind.pubmed.graph
│   ├── ind.pubmed.test.index
│   ├── squirrel_node_feature_label.txt: squirrel dataset
│   ├── squirrel_graph_edges.txt
│   ├── chameleon_node_feature_label.txt: chameleon dataset
│   ├── chameleon_graph_edges.txt
│   ├── actor_node_feature_label.txt: actor dataset
│   ├── actor_graph_edges.txt
│   ├── sbm.p: synthetic dataset
│   └── sbm_adj.p: graph structure estimated by GEN
└── README.md
~~~~



## Basic Usage

~~~
python ./code/train.py 
~~~



## Hyper-parameter Tuning

There are three key hyper-parameters: *k*, *threshold* and *tolerance*.

- k: [3, 4, 5 …, 14, 15]
- threshold: [0.1, 0.2, 0.3, …, 0.8, 0.9]
- tolerance: [0.1, 0.01]

For the hyper-parameter settings of six benchmark datasets used in this paper, please refer to Section 4.4.





# Reference

```
@inproceedings{wang2021graph,
  title={Graph Structure Estimation Neural Networks},
  author={Wang, Ruijia and Mou, Shuai and Wang, Xiao and Xiao, Wanpeng and Ju, Qi and Shi, Chuan and Xie, Xing},
  booktitle={Proceedings of the Web Conference 2021},
  pages={342--353},
  year={2021}
}
```