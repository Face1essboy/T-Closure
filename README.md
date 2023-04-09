# pytorch implementation of T-Closure

Paper: "Understanding the Semantics of GPS-based Trajectories for Road Closure Detection"

## Contents
1. [Installation](#installation)
2. [Train_and_Test](#Train_and_Test)
3. [Datasets](#Datasets)
4. [Baselines](#Baselines)
5. [Contact](#contact)

## Installation

Clone the repository::

```
git clone https://github.com/zjs123/T-Closure.git
```

Install the anaconda and create the enviorment via:

```
conda env create -f freeze.yml
```

We use Python3.6 for implementation and the CUDA version is 10.1. 

## Train_and_Test

The user can use the following command to train our model in the sampled dataset.
```
python3 Main.py
```
Some of the important available hyper-parameters include:
```
        '-his_length', default = 10, type = int, help =' max length of traj-plan graph sequence'
	'-traffic_flow_length',  default = 50, type = int, help = ' max length of traffic flow features for each road'
	'-traj_length',  default = 50, type = int, help = ' max length of trajectory points of each road'
	'-layer', default = 1, type = int, help = 'number of layers of MVH-GNN'
	'-static_hidden', 	 default = 10,   	type=int, 	help='dimension of static features'
	'-emb_hidden', 	 default = 50,   	type=int, 	help='dimension of GRU encoded features'
	'trade-off', default = 0.1, type = float, help='trade-off between the main loss and the contrastive loss'
	'-numOfEpoch', 	default=100,	type=int	help='Train Epoches'
```

## Datasets

We provide the sampled dataset with 100 samples, and the data structure of each sample is
```
python3 Test.py
```

## Baselines

We use following public codes for baseline experiments. 

| Baselines   | Code                                                                      | Embedding size | Batch num |
|-------------|---------------------------------------------------------------------------|----------------|------------|
| TransE ([Bordes et al., 2013](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data))      | [Link](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch/openke) | 100, 200       | 100, 200       |
| TTransE ([Leblay et al., 2018](https://dl.acm.org/doi/fullHtml/10.1145/3184558.3191639))    | [Link](https://github.com/INK-USC/RE-Net/tree/master/baselines)                                  | 50, 100, 200   | 100, 200       |
| TA-TransE ([Alberto et al., 2018](https://www.aclweb.org/anthology/D18-1516.pdf))      | [Link](https://github.com/INK-USC/RE-Net/tree/master/baselines)     | 100, 200            | Default    |
| HyTE ([Dasgupta et al., 2018](http://talukdar.net/papers/emnlp2018_HyTE.pdf))        | [Link](https://github.com/malllabiisc/HyTE)                               | Default            | Default    |
| DE-DistMult ([Goel et al., 2020](https://arxiv.org/pdf/1907.03143.pdf))        | [Link](https://github.com/BorealisAI/de-simple)                               | Default            | Default    |
| TNTComplEX ([Timothee et al., 2020](https://openreview.net/pdf?id=rke2P1BFwS))        | [Link](https://github.com/facebookresearch/tkbc)                               | Default            | Default    |
| ATiSE ([Chenjin et al., 2020](https://arxiv.org/pdf/1911.07893.pdf))        | [Link](https://github.com/soledad921/ATISE)                               | Default            | Default    |

## Contact

For any questions or suggestions you can use the issues section or contact us at shengyp2011@gmail.com or zjss12358@gmail.com.
