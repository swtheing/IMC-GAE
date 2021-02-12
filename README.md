# IMC-GAE: Inductive Matrix Completion Using Graph Autoencoder for Real Recommender Systems

Paper link: 

Author's code: [https://github.com/swtheing/IMC-GAE](https://github.com/swtheing/IMC-GAE)

The implementation does not handle side-channel features and mini-epoching and thus achieves
slightly worse performance when using node features.

Credit: Wei shen ([@swtheing](https://github.com/swtheing))


## Requirements
------------

Latest tested combination: Python 3.8.1 + PyTorch 1.4.0 + DGL 0.5.2.

Install [PyTorch](https://pytorch.org/)

Install [DGL](https://github.com/dmlc/dgl)

Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm ,bidict etc.

## Data

Supported datasets: ml-100k, ml-1m, ml-10m, flixster, douban, yahoo_music

## How to run
### Train with full-graph

### Train with minibatch on a single GPU

### Train with minibatch on multi-GPU

### Train with minibatch on CPU

