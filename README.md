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

| Names   | Description                                                                   | Example        | Type      |
|-------------|---------------------------------------------------------------------------|----------------|-----------|
| info_id     | The ID of each sample                                                     | 00000000000    | int       |
| link_id     | The ID of target road                                                     | 00000000000000 | int       |
| status      | Label of each sample,1 is closure and 0 is not closed                     | 1    | int       |
| sub_source  | The source code of the sample                                             | 10   | int       |
| event_t     | Time of discovery of current closure event                                | 11681053604  | int       |
| exam_st      | Time for the start of manual review                                      | 11681054604  | int       |
| exam_t      | Time for the end of manual review                                         | 11681056604  | int       |
| order_info   | The lateset N related trajectories of the target road                    | [order_1, order_2, ...]  | List       |
| Traj         | The raw trajectory feature of each order in the order_info                [trajectory_sequence_1, trajectory_sequence_2, ...]  | List       |
| graph_dict   | The k-hop neighbor graph of the target road                              | [Node_set, Node_num, [edge_in, edge_out]]  | List       |
| online_nei   | The traffic_flow feature of each node in k_hop neighbor graph            | [[uv_now, uv_last_day, uv_last_week], ...]  | List       |
| static_nei   | The category feature of each node in k_hop neighbor graph                | [[0,1,1,2,1], ...]  | List       |
|order_2_traj_plan_graph_dict_list | The Traj_plan graph sequence of this sample, and in each Traj_plan graph, its structure is [Node_set, Node_num, [edge_in, edge_out], traffic_flow_feature_for_each_node, traj_feature_for_each_node, static_feature_for_each_node]        | [traj_plan_graph_for_order_1, traj_plan_graph_for_order_2, ...] |List|
## Baselines

We use following public codes for baseline experiments. 

| Baselines   | Paper                                                                      | Code |
|-------------|---------------------------------------------------------------------------|----------------|
| ASTGCN      | Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting. In AAAI. 922–929.  | implemented in pyG|
| GMAN      |  GMAN:A Graph Multi-Attention Network for Traffic Prediction. In AAAI. 1234–1241                                 | implemented in pyG|
| STGODE      |   SpatialTemporal Graph ODE Networks for Traffic Flow Forecasting. In KDD. 364–373.                            | https://github.com/square-coder/STGODE|
| DSTAGNN      |  DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting. In ICML. 11906–11917    |https://github.com/SYLan2019/DSTAGNN|
| TrajGAT |  TrajGAT: A Graph-based Long-term Dependency Modeling Approach for Trajectory Similarity Computation. In KDD. 2275–2285 | https://github.com/HuHaonan-CHN/TrajGAT|

## Contact

For any questions or suggestions you can use the issues section or contact us at zjss12358@gmail.com.
