import math
import random

from requests import head
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
import torch_scatter
from torch_sparse import SparseTensor

import numpy as np
import torch
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)

DEVICE = torch.device('cuda:0') # cuda
#DEVICE = torch.device('cpu')

class H_GAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(H_GAT, self).__init__()

        self.real_GNN = GraphConv(in_channels = in_channels, out_channels = out_channels).cuda()
        self.plan_GNN = GraphConv(in_channels = in_channels, out_channels = out_channels).cuda()
        self.other_GNN = GraphConv(in_channels = in_channels, out_channels = out_channels).cuda()

        self.gate_linear = torch.nn.Linear(2*in_channels, out_channels, bias = True)
        self.Agg_Linear = torch.nn.Linear(4*in_channels, out_channels, bias = True)

    def gated_fusion(self, A, B):
        gate = torch.sigmoid(self.gate_linear(torch.cat([A,B], -1)))
        out = (1-gate)*A + gate*B

        return out

    def forward(self, x, edge_index, edge_weight, cat_list):
        #return self.real_GNN(x, edge_index, edge_weight), torch.Tensor([0]).to(DEVICE)
        aux_loss = 0
        dst_list = edge_index[1]
        real_edge_index = []
        plan_edge_index = []
        other_edge_index = []

        real_edge_weight = []
        plan_edge_weight = []
        other_edge_weight = []
        # split the whole graph into several sub-graphs
        for index in range(len(dst_list)):
            if cat_list[dst_list[index]] == 0 :
                real_edge_index.append(edge_index[:,index].unsqueeze(-1))
                real_edge_weight.append(edge_weight[index])
            elif cat_list[dst_list[index]] == 1:
                plan_edge_index.append(edge_index[:,index].unsqueeze(-1))
                plan_edge_weight.append(edge_weight[index])
            else:
                other_edge_index.append(edge_index[:,index].unsqueeze(-1))
                other_edge_weight.append(edge_weight[index])
        
        # intra-cat feature aggregate
        if len(real_edge_weight) == 0:
            real_gnn_output = x
        else:
            #print('real')
            edge_index = torch.cat(real_edge_index, -1).to(DEVICE)
            edge_weight = torch.LongTensor(real_edge_weight).to(DEVICE)
            row, col = edge_index
            real_gnn_output = self.real_GNN(x, edge_index, edge_weight)
        
        if len(plan_edge_weight) == 0:
            plan_gnn_output = x
        else:
            #print('plan')
            edge_index = torch.cat(plan_edge_index, -1).to(DEVICE)
            edge_weight = torch.LongTensor(plan_edge_weight).to(DEVICE)
            row, col = edge_index
            adj = SparseTensor(row=row, col=col, sparse_sizes=(x.shape[0], x.shape[0]))
            plan_gnn_output = self.plan_GNN(x, edge_index, edge_weight)

        if len(other_edge_weight) == 0:
            other_gnn_output = x
        else:
            #print('other')
            edge_index = torch.cat(other_edge_index, -1).to(DEVICE)
            edge_weight = torch.LongTensor(other_edge_weight).to(DEVICE)
            other_gnn_output = self.other_GNN(x, edge_index, edge_weight)

        out1 = self.gated_fusion(real_gnn_output, plan_gnn_output) 
        out2 = self.gated_fusion(real_gnn_output, other_gnn_output)
        out3 = self.gated_fusion(plan_gnn_output, other_gnn_output)

        all_features = torch.cat([x,out1,out2,out3], -1)
        out = torch.tanh(self.Agg_Linear(all_features))

        # gated interactions
        # p&r plan_road and pass_road
        pos_traj_emb = plan_gnn_output
        neg_traj_emb = pos_traj_emb[torch.randperm(pos_traj_emb.size(0))]

        pos_exp = torch.exp(torch.sigmoid(torch.sum((real_gnn_output)*pos_traj_emb, 1)))
        neg_exp = torch.exp(torch.sigmoid(torch.sum((real_gnn_output)*neg_traj_emb, 1)))
        aux_loss += -torch.log(pos_exp/(pos_exp+neg_exp))

        
        #p&x plan_road and extend_road
        pos_traj_emb = plan_gnn_output
        neg_traj_emb = pos_traj_emb[torch.randperm(pos_traj_emb.size(0))]

        pos_exp = torch.exp(torch.sigmoid(torch.sum((other_gnn_output)*pos_traj_emb, 1)))
        neg_exp = torch.exp(torch.sigmoid(torch.sum((other_gnn_output)*neg_traj_emb, 1)))
        aux_loss += -torch.log(pos_exp/(pos_exp+neg_exp))

        #r&x pass_road and extend_road
        pos_traj_emb = real_gnn_output
        neg_traj_emb = pos_traj_emb[torch.randperm(pos_traj_emb.size(0))]

        # get graph-level aux loss
        pos_exp = torch.exp(torch.sigmoid(torch.sum((other_gnn_output)*pos_traj_emb, 1)))
        neg_exp = torch.exp(torch.sigmoid(torch.sum((other_gnn_output)*neg_traj_emb, 1)))
        aux_loss += -torch.log(pos_exp/(pos_exp+neg_exp))
        
        return  out, torch.Tensor([0]).to(DEVICE)


class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=False, **kwargs):
        super(GraphConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize        

        self.lin_l = nn.Linear(in_channels, out_channels) 
        self.lin_r = nn.Linear(in_channels, out_channels)  

        self.lin_c = nn.Linear(in_channels, in_channels) 
        self.lin_att = nn.Linear(2*in_channels, 1)

        self.edge_EMB = nn.Embedding(10, out_channels)
        torch.nn.init.uniform_(self.edge_EMB.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_c.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        prop = self.propagate(edge_index, x=(x, x), edge_weight = edge_weight, size=size)
        out = self.lin_l(x) + prop

        return prop

    def message(self, x_i, x_j, edge_weight, index, ptr): # creat message for each node for passing
        edge_weight_emb = self.edge_EMB(edge_weight) 
        
        neighbor_messge = edge_weight_emb*x_j # edge feature * node feature via ham product
        att_score = self.lin_att(torch.cat([x_i, neighbor_messge], -1))
        att_score = F.leaky_relu(att_score, -0.1)
        att_score = softmax(att_score, index, ptr) # caculate attention score
        final_messge = self.lin_r(neighbor_messge)
        return final_messge*att_score
    
    def aggregate(self, inputs, index, dim_size=None): # aggregate information for each node
        # The axis along which to index number of nodes.
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, node_dim, dim_size=dim_size)

        return out

class Seq_Encoder_GRU(torch.nn.Module): # GRU encoder for traffic flow and trajectory feature encoding
    def __init__(self, in_dim, out_dim, layer):
        super(Seq_Encoder_GRU, self).__init__()
        self.layer = layer
        self.out_dim = out_dim

        # GRU encoder
        self.LSTM = nn.GRU(in_dim, out_dim, layer, bias = False)
        self.padding_weight = torch.nn.Parameter(torch.FloatTensor(out_dim), requires_grad=True)
        torch.nn.init.uniform_(self.padding_weight, -1, 1)

        # aux task linear
        self.predictor = nn.Linear(out_dim, 1)

    def forward(self, f, padding):
        f = self.LSTM(f)[0]
        if padding == None:
            return f[-1]
        else:
            f_reshaped = f.permute(1,0,2) # n*T*f_dim
            final_index = (len(padding[0]) - torch.sum(padding, 1) - 1).long()
            
            # traj f collect
            traj_f_list = []
            for index in range(f_reshaped.size()[0]):
                if final_index[index] != -1:
                    traj_f_list.append(torch.mean(f_reshaped[index][:(final_index[index]+1).long()], 0, keepdim = False))
                else:
                    traj_f_list.append(self.padding_weight)
            f_collected = torch.stack(traj_f_list,0)

            return f_collected

class T_Closure(torch.nn.Module):
    def __init__(self, node_dim, static_dim, emb_dim, his_len, layer_num, trade):
        super(T_Closure, self).__init__()
        self.layer_num = layer_num
        self.his_len = his_len
        self.trade = trade
        # n layer GNN_LSTM for spatial and temporal modeling
        self.GNN_LSTM = [[H_GAT(in_channels = node_dim, out_channels = node_dim).to(DEVICE), nn.GRU(node_dim,node_dim,1).to(DEVICE), nn.Linear(node_dim, node_dim).to(DEVICE)] for i in range(layer_num)]

        # GRU encoder for traj feature
        self.Traj_Encoder = Seq_Encoder_GRU(3, emb_dim, 1)
        self.uv_Encoder = Seq_Encoder_GRU(3,emb_dim,1)
        
        # Linear for regress and classify
        self.linear_main_task = torch.nn.Linear(node_dim, 1, bias = False)
        self.linear_order_task = torch.nn.Linear(node_dim, 1, bias = False)
        
        # Init Embeddings

        self.lane_EMB = nn.Embedding(10, static_dim)
        torch.nn.init.uniform_(self.lane_EMB.weight.data)

        self.direction_EMB = nn.Embedding(10, static_dim)
        torch.nn.init.uniform_(self.direction_EMB.weight.data)

        self.fc_EMB = nn.Embedding(10, static_dim)
        torch.nn.init.uniform_(self.fc_EMB.weight.data)

        self.speed_class_EMB = nn.Embedding(10, static_dim)
        torch.nn.init.uniform_(self.speed_class_EMB.weight.data)

        self.park_EMB = nn.Embedding(10, static_dim)
        torch.nn.init.uniform_(self.park_EMB.weight.data)

        # aux task loss
        self.aux_linear = torch.nn.Linear(node_dim, node_dim, bias = False)
        self.criterion = torch.nn.BCELoss()

        #uv traj emb
        self.uv_embs = []
        self.traj_embs = []
        

    def graph_pooling(self, batch_index, h): # select the central node from each graph for sequence modeling
        batch_len = list(batch_index.bincount())
        output = torch.stack([i[0] for i in h.split(batch_len, 0)]) # batch_size*F
        return output, 0
    
    def GNN_LSTM_block(self, x, edge_index, edge_weight, batch_list, order_list, cat_list, graph_aux_sample_batch):
        graph_aux_loss = 0
        GNN_outputs = [] # T*N*32
        order_list = order_list.permute(1,0)
        for step in range(len(x)):
            node_f = x[step]
            edge = edge_index[step]
            weight = edge_weight[step]
            cat = cat_list[step]
            for i in range(self.layer_num):
                # spatial modeling via MVH-GNN
                node_f, aux_loss = self.GNN_LSTM[i][0](node_f, edge, weight, cat.to(DEVICE))
                graph_aux_loss += torch.mean(aux_loss)
            graph_emb, aux_loss = self.graph_pooling(batch_list[step], node_f)
            GNN_outputs.append((graph_emb).unsqueeze(0))

        spatial_embeddings = torch.cat(GNN_outputs, 0)
        h, _ = self.GNN_LSTM[0][1](spatial_embeddings)# T*batch_size*32
        
        
        # get random seq
        # used for sequence level aux loss which generates embedding via random perburtations
        GNN_outputs = [] # T*N*32
        order_list = order_list.permute(1,0)
        seq_list = np.array([i for i in range(len(x))])
        Type = random.random()
        if Type <= 0.5:
            noise_index = random.sample([i for i in range(len(x))], int(0.2*len(seq_list)))
        else:
            noise_index = random.sample([i for i in range(len(x))], int(0.4*len(seq_list)))
        seq_list[noise_index] = 0
        for step in seq_list:
            node_f = x[step]
            edge = edge_index[step]
            weight = edge_weight[step]
            cat = cat_list[step]
            for i in range(self.layer_num):
                node_f, aux_loss = self.GNN_LSTM[i][0](node_f, edge, weight, cat.to(DEVICE))
            graph_emb, aux_loss = self.graph_pooling(batch_list[step], node_f)
            GNN_outputs.append((graph_emb).unsqueeze(0))

        spatial_embeddings = torch.cat(GNN_outputs, 0)
        rand_h, _ = self.GNN_LSTM[0][1](spatial_embeddings)
        
        return h, spatial_embeddings, graph_aux_loss, rand_h
    
    def get_aux_task(self, uv_emb,traj_emb): # get contrastive learning loss for each aux task
        
        pos_traj_emb = traj_emb
        neg_traj_emb = pos_traj_emb[torch.randperm(pos_traj_emb.size(0))]

        # caculate the loss
        pos_exp = torch.exp(torch.sigmoid(torch.sum(uv_emb*pos_traj_emb, 1)))
        neg_exp = torch.exp(torch.sigmoid(torch.sum(uv_emb*neg_traj_emb, 1)))
        aux_loss = -torch.log(pos_exp/(pos_exp+neg_exp))

        return torch.mean(aux_loss)

    def forward(self, batch, Type):
        """
        features = Node features for T time steps T*N(batched by snapshot)*(1+uv_f_dim)
        edge_indices = Graph edge indices T*2*E(batched by snapshot)
        edge_weight = Graph edge weight T*E(batched by snapshot)
        edge_weights = Batch split for T time steps T*N(batched by snapshot)
        targets = label for each node in T time steps T*N(batched by snapshot)*1
        traj_f = raw traj features in T time steps T*N(batched by snapshot)*6*point_sample
        traj_padding = padding index for raw traj features, 1:pad, 0:unpad T*N(batched by snapshot)*point_sample
        seq_padding = final index of each order squence (batch_size)
        """
        
        seq_padding = batch[1].to(DEVICE).long()
        order_type_list = batch[5].to(DEVICE)
        time_steps = len(batch[0].edge_indices)
        graph_aux_sample_batch = batch[3]

        
        edge_index_list = []
        edge_weight_list = []
        target_list = []
        batch_list =  []
        f_list = [] # T*N*F
        cat_list = []

        aux_loss = 0

        # perpare the input features 
        for step in range(time_steps):
            edge_index = torch.Tensor(batch[0].edge_indices[step]).long().to(DEVICE)
            edge_weight = torch.Tensor(batch[0].edge_weights[step]).long().to(DEVICE)
            features = torch.Tensor(batch[0].features[step]).to(DEVICE)
            batch_index = torch.Tensor(batch[0].batches[step]).long().to(DEVICE)
            target = torch.Tensor(batch[0].targets[step]).to(DEVICE)

            # encode the raw traj feature
            traj_raw_feature = torch.Tensor(batch[0].traj_f[step]).to(DEVICE) # N(batched by snapshot)*6*point_sample
            traj_raw_feature = traj_raw_feature.permute(2,0,1) # point_sample*N(batched by snapshot)*6
            
            traj_raw_feature_num = traj_raw_feature[:,:,2:-1] # point_sample*N(batched by snapshot)*5
            traj_raw_feature = traj_raw_feature_num #torch.cat([traj_raw_feature_num, status_emb], -1)
            traj_padding = torch.Tensor(batch[0].traj_padding[step]).float().to(DEVICE) # N(batched by snapshot)*point_sample

            traj_feature = self.Traj_Encoder(traj_raw_feature, traj_padding) # N(batched by snapshot)*(traj_emb)

            # encode the raw uv feature
            uv_feature = features[:,1:31] # N*uv_dim
            uv_feature = uv_feature.reshape(-1, 3, 10) # N*3*point_sample
            uv_feature = uv_feature.permute(2,0,1) # point_sample*N*3
            #uv_feature = uv_feature[-4:]
            uv_feature = self.uv_Encoder(uv_feature, None) # N(batched by snapshot)*(uv_emb)
            
            if step == time_steps-1:
                batch_len = list(batch_list[-1].bincount())
                central_uv = torch.stack([i[0] for i in uv_feature.split(batch_len, 0)])
                self.uv_embs.append(central_uv)
                central_traj = torch.stack([i[0] for i in traj_feature.split(batch_len, 0)])
                self.traj_embs.append(central_traj)
            

            # look up the category embedding
            static_feature = features[:,31:] # N*7
            cat_list.append(features[:,0].long())

            lane_emb = self.lane_EMB(static_feature[:,0].long()) # N*node_dim
            direction_emb = self.direction_EMB(static_feature[:,1].long())
            fc_emb = self.fc_EMB(static_feature[:,4].long())
            speed_class_emb = self.speed_class_EMB(static_feature[:,5].long())
            park_emb = self.park_EMB(static_feature[:,6].long())

            # concate the features
            h = torch.cat([lane_emb, direction_emb, fc_emb, speed_class_emb, park_emb, traj_feature, uv_feature], -1) # N*(node_dim*4+uv_dim)

            # caculate the encoder aux loss
            aux_loss += self.trade*self.get_aux_task(uv_feature, traj_feature)

            edge_index_list.append(edge_index)
            edge_weight_list.append(edge_weight)
            target_list.append(target)
            batch_list.append(batch_index)
            f_list.append(h)

        # encode the data via n layer GNN and n layer LSTM
        LSTM_outputs, GNN_outputs, graph_aux_loss, rand_LSTM_outputs = self.GNN_LSTM_block(f_list, edge_index_list, edge_weight_list, batch_list, order_type_list, cat_list, graph_aux_sample_batch) # T*batch_size*dim
        aux_loss += graph_aux_loss

        # collect the output features
        GNN_outputs_reshaped = GNN_outputs.permute(1,0,2)
        LSTM_outputs_reshaped = LSTM_outputs.permute(1,0,2) # batch_size*T*dim
        rand_LSTM_outputs_reshaped = rand_LSTM_outputs.permute(1,0,2)
        predict_emb = []
        rand_predict_emb = []

        seq_aux_target = []
        seq_aux_sample = []

        for batch in range(GNN_outputs_reshaped.size()[0]):
            if Type == 'test':
                predict_emb.append(LSTM_outputs_reshaped[batch][seq_padding[batch]-1].unsqueeze(0))
            else:
                predict_emb.append(LSTM_outputs_reshaped[batch][seq_padding[batch]-1].unsqueeze(0))
            rand_predict_emb.append(rand_LSTM_outputs_reshaped[batch][seq_padding[batch]-1].unsqueeze(0))

            seq_aux_target.append(LSTM_outputs_reshaped[batch][seq_padding[batch]-1].unsqueeze(0))
            if seq_padding[batch]-2 <0:
                seq_aux_sample.append(LSTM_outputs_reshaped[batch][seq_padding[batch]-1].unsqueeze(0))
            else:
                seq_aux_sample.append(LSTM_outputs_reshaped[batch][seq_padding[batch]-2].unsqueeze(0))
            

        predict_emb = torch.cat(predict_emb, 0)
        rand_predict_emb = torch.cat(rand_predict_emb, 0)
        aux_loss += self.trade*self.get_aux_task(predict_emb, rand_predict_emb)
        
        predict_emb = F.dropout(predict_emb, p = 0.3, training=self.training) # batch_size*F

        # main task loss
        batch_len = list(batch_list[-1].bincount())
        target_main = torch.stack([i[0] for i in target_list[-1].split(batch_len, 0)])

        y_main = torch.sigmoid(self.linear_main_task(predict_emb)) # batch_size*1

        # seq task loss
        seq_aux_target_emb = torch.cat(seq_aux_target, 0)
        seq_aux_sample_emb = torch.cat(seq_aux_sample, 0)
        aux_loss += self.trade*self.get_aux_task(seq_aux_target_emb, seq_aux_sample_emb)

        return y_main, target_main, aux_loss
