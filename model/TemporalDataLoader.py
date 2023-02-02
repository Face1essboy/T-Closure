from typing import Union, List, Optional
import numpy as np
from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, HeteroData, Dataset, Batch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal as SGTS
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal as DGTS
from torch_geometric_temporal.signal import DynamicGraphStaticSignal as DGSS
from torch_geometric_temporal.signal import StaticGraphTemporalSignalBatch as SGTSBatch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch as DGTSBatch
from torch_geometric_temporal.signal import DynamicGraphStaticSignalBatch as DGSSBatch


def collate_temporal_signal(
        signal_list: List[Union[SGTS, DGTS, DGSS]],
        signal_batch_class: Union[SGTSBatch, DGTSBatch, DGSSBatch],
        follow_batch: Optional[Union[List[str]]] = None,
        exclude_keys: Optional[Union[List[str]]] = None,
):
    # determine output type
    in_type = type(signal_list[0][0])
    signal_batch_type = None
    if in_type == SGTS:
        signal_batch_type = SGTSBatch
    elif in_type == DGTS:
        signal_batch_type = DGTSBatch
    elif in_type == DGSS:
        signal_batch_type = DGSSBatch

    # create list(samples) of list(temporal sequence) of torch_geometric.Data/HeteroData
    
    graph_list = [sample[0] for sample in signal_list]
    seq_padding_list = [sample[1] for sample in signal_list]
    seq_span_list = [sample[4] for sample in signal_list]
    order_type_list = [sample[5] for sample in signal_list]
    
    traj_aux_sample_list = []
    for i in range(len(signal_list[0][2])):
        tmp_list = []
        for batch_index in range(len(signal_list)):
            tmp_list.append(signal_list[batch_index][2][i])
        traj_aux_sample_list.append(tmp_list)
    
    graph_aux_sample_list = []
    for i in range(len(signal_list[0][3])):
        tmp_list = []
        for batch_index in range(len(signal_list)):
            tmp_list.append(signal_list[batch_index][3][i])
        graph_aux_sample_list.append(tmp_list)


    all_graphs = [[g for g in iter(signal)] for signal in graph_list]

    # create diagonalized torch_geometric.Batch objects by timestep
    batches_by_timestep = [Batch.from_data_list([sample[t] for sample in all_graphs], follow_batch, exclude_keys)
                    for t in range(len(all_graphs[0]))]

    # generate graph batch
    graph_batch = signal_batch_type(
        edge_indices = [batch["edge_index"].numpy() for batch in batches_by_timestep],
        edge_weights = [batch["edge_attr"].numpy() for batch in batches_by_timestep],
        features = [batch["x"].numpy() for batch in batches_by_timestep],
        targets = [batch["y"].numpy() for batch in batches_by_timestep],
        batches = [batch["batch"].numpy() for batch in batches_by_timestep],
        traj_f = [batch["traj_f"].numpy() for batch in batches_by_timestep], 
        traj_padding = [batch["traj_padding"].numpy() for batch in batches_by_timestep]
    )

    traj_sample = []
    for step in range(len(graph_batch.batches)):
        batch_index = torch.Tensor(graph_batch.batches[step]).long()
        batch_len = list(batch_index.bincount())
        collected_traj_sample_step = []
        for batch_index in range(len(traj_aux_sample_list[step])):
            for sample in traj_aux_sample_list[step][batch_index]:
                batched_target_index = sum(batch_len[:batch_index]) + sample[0]
                batched_pos_index = sum(batch_len[:batch_index]) + sample[1]
                batched_neg_index = sum(batch_len[:batch_index]) + sample[2]
                collected_traj_sample_step.append([batched_target_index, batched_pos_index, batched_neg_index])
        traj_sample.append(collected_traj_sample_step)

    
    graph_sample = []
    for step in range(len(graph_batch.batches)):
        batch_index = torch.Tensor(graph_batch.batches[step]).long()
        batch_len = list(batch_index.bincount())
        collected_graph_sample_step = []
        for batch_index in range(len(graph_aux_sample_list[step])):
            real_index = graph_aux_sample_list[step][batch_index][0]
            plan_index = graph_aux_sample_list[step][batch_index][1]
            if plan_index == [0]:
                new_plan_index = []
            else:
                new_plan_index = [int(sum(batch_len[:batch_index]) + i) for i in plan_index]
            new_real_index = [int(sum(batch_len[:batch_index]) + i) for i in real_index]
            collected_graph_sample_step.append([new_real_index, new_plan_index])
        graph_sample.append(collected_graph_sample_step)
    

    return [graph_batch, torch.LongTensor(seq_padding_list), traj_sample, graph_sample, torch.FloatTensor(seq_span_list), torch.FloatTensor(order_type_list)]

class PYGTCollater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        elem = batch[0][0]
        if isinstance(elem, SGTS) or isinstance(elem, DGTS) or isinstance(elem, DGSS):
            return collate_temporal_signal(batch, self.follow_batch, self.exclude_keys)
        elif isinstance(elem, Data) or isinstance(elem, HeteroData):
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)

class DataLoader(torch.utils.data.DataLoader):

    def __init__(
        self,
        dataset: Union[Dataset, List[Data], List[HeteroData], List[SGTS], List[DGTS], List[DGSS]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: List[str] = [],
        exclude_keys: List[str] = [],
        **kwargs,
    ):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=PYGTCollater(follow_batch, exclude_keys), **kwargs)