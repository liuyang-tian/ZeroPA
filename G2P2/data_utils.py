import math
import numpy as np

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import k_hop_subgraph, to_undirected, to_networkx
import copy
from tqdm import tqdm
from torch_sparse import SparseTensor
import networkx as nx


def load_dataset(root, name):
    data = torch.load(f"{root}/{name.lower()}.pt", map_location='cpu')
    data.num_nodes = data.x.shape[0]
    if name=='ogbn-arxiv':
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.x.shape[0])
    return data


class MergedGraphDataset:
    def __init__(self, data_list, neigh_num):
        self.neigh_num = neigh_num
        self.neighs_list = []
        self.data_list = data_list
        self.merged_data = self._merge_graphs()
        self._parse_source_data()

    def _merge_graphs(self):
        x_list = []
        edge_index_list = []
        y_list = []
        raw_texts = []

        node_offset = 0
        for data in self.data_list:
            num_nodes = data.x.size(0)

            x_list.append(data.x)

            edge_index = data.edge_index + node_offset
            edge_index_list.append(edge_index)

            if hasattr(data, 'y') and data.y is not None:
                y_list.append(data.y)

            if hasattr(data, 'raw_texts'):
                raw_texts.extend(data.raw_texts)

            node_offset += num_nodes

        x = torch.cat(x_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        y = torch.cat(y_list, dim=0) if y_list else None

        merged_data = Data(x=x, edge_index=edge_index, y=y)
        merged_data.raw_texts = raw_texts
        return merged_data

    def _parse_source_data(self):
        for idx in tqdm(range(self.merged_data.x.shape[0])):
            neig_idx, _, _, _ = k_hop_subgraph([idx], 1, self.merged_data.edge_index, relabel_nodes=False)
            neig_idx = neig_idx[1:]
            self.neighs_list.append(neig_idx)
    
    def __len__(self):
        return self.merged_data.x.shape[0]
    
    def __getitem__(self, idx):
        neighs = self.neighs_list[idx]
        if len(neighs) > self.neigh_num:
            t_n = np.random.choice(neighs, self.neigh_num, replace=False)
        else:
            if len(neighs) > 0:
                t_n = np.random.choice(neighs, self.neigh_num, replace=True)
            else:
                t_n = None
        
        if t_n is None:
            t_n = [idx, idx, idx]
        else:
            t_n = t_n.tolist()
        
        sample = {
            'node_idx': idx,
            's_n_text': self.merged_data.raw_texts[idx],
            't_n_text': [self.merged_data.raw_texts[i] for i in t_n]
        }

        return sample
    

class SourceGraphDataset(Dataset):
    def __init__(self, data, node_idx, hop_num=3, transform=None):
        self.data = data
        self.hop_num = hop_num
        self.transform = transform
        self.path_length = 0

        graphs = self._parse_source_data()
        self.graphs = [graphs[i] for i in node_idx]

    def _parse_source_data(self):
        collected_graph_data = []
        for idx in tqdm(range(self.data.num_nodes)):
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph([idx], self.hop_num, self.data.edge_index, relabel_nodes=False)

            node_idx = torch.unique(sub_edge_index)
            node_idx_map = {j: i for i, j in enumerate(node_idx.tolist())}

            sources_idx = list(map(node_idx_map.get, sub_edge_index[0].tolist()))
            target_idx = list(map(node_idx_map.get, sub_edge_index[1].tolist()))

            edge_index = torch.tensor([sources_idx, target_idx], dtype=torch.long)

            if len(node_idx)==0:
                graph = Data(
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    x=self.data.x[idx].unsqueeze(0),
                    y=self.data.y[idx],
                    root_n_index=0,
                )
                spd = torch.tensor([0])

            else:
                graph = Data(
                    edge_index=edge_index,
                    x=self.data.x[node_idx],
                    y=self.data.y[idx],
                    root_n_index=node_idx_map[idx],
                )

                G = to_networkx(graph, to_undirected=True)
                lengths = dict(nx.single_source_shortest_path_length(G, graph.root_n_index))
                spd = torch.full((graph.num_nodes,), -1, dtype=torch.long)
                for node_id, dist in lengths.items():
                    spd[node_id] = dist
            
            graph.spd = spd
            if graph.spd.max().item() > self.path_length:
                self.path_length = graph.spd.max().item()

            if self.transform:
                graph = self.transform(graph)

            collected_graph_data.append(graph)
        return collected_graph_data

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        # data_new = Data(x=data.x, edge_index=data.edge_index, y=data.y, root_n_index=data.root_n_index)

        return data


class SourceGraphDataset_Wiki(Dataset):
    def __init__(self, data, node_idx, length, hop_num=3, transform=None):
        self.data = data
        self.hop_num = hop_num
        self.transform = transform

        self.path_length = 0
        graphs = self._parse_source_data(length)
        self.graphs = [graphs[i] for i in node_idx]


    def _parse_source_data(self, length):
        collected_graph_data = []
        node_ids = torch.arange(self.data.num_nodes)
        node_list, edge_list = self.pyg_random_walk(node_ids, self.data, length, restart_prob=0.5)

        for idx in tqdm(range(self.data.num_nodes)):
            node_idx = node_list[idx]
            node_idx_map = {j: i for i, j in enumerate(node_idx.tolist())}

            sources_idx = list(map(node_idx_map.get, edge_list[idx][0].tolist()))
            target_idx = list(map(node_idx_map.get, edge_list[idx][1].tolist()))

            edge_index = torch.tensor([sources_idx, target_idx], dtype=torch.long)

            if len(node_idx)==0:
                graph = Data(
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    x=self.data.x[idx].unsqueeze(0),
                    y=self.data.y[idx],
                    root_n_index=0,
                )
                spd = torch.tensor([0])

            else:
                graph = Data(
                    edge_index=edge_index,
                    x=self.data.x[node_idx],
                    y=self.data.y[idx],
                    root_n_index=node_idx_map[idx],
                )

                G = to_networkx(graph, to_undirected=True)
                lengths = dict(nx.single_source_shortest_path_length(G, graph.root_n_index))
                spd = torch.full((graph.num_nodes,), -1, dtype=torch.long)
                for node_id, dist in lengths.items():
                    spd[node_id] = dist
            
                graph.spd = spd
                if graph.spd.max().item() > self.path_length:
                    self.path_length = graph.spd.max().item()

            if self.transform:
                graph = self.transform(graph)

            collected_graph_data.append(graph)
        return collected_graph_data
    

    def pyg_random_walk(self, seeds, graph, length, restart_prob=0.8):
        edge_index = graph.edge_index
        node_num = graph.y.shape[0]
        start_nodes = seeds
        graph_num = start_nodes.shape[0]

        value = torch.arange(edge_index.size(1))

        if type(edge_index) == SparseTensor:
            adj_t = edge_index
        else:
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                        value=value,
                                        sparse_sizes=(node_num, node_num)).t()
            
        current_nodes = start_nodes.clone()

        history = start_nodes.clone().unsqueeze(0)
        signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
        for i in range(length):
            seed = torch.rand([graph_num])
            nei = adj_t.sample(1, current_nodes).squeeze()
            sign = seed < restart_prob
            nei[sign] = start_nodes[sign]
            history = torch.cat((history, nei.unsqueeze(0)), dim=0)
            signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
            current_nodes = nei
        history = history.T
        signs = signs.T

        node_list = []
        edge_list = []
        for i in range(graph_num):
            path = history[i]
            sign = signs[i]
            node_idx = path.unique()
            node_list.append(node_idx)

            sources = path[:-1].numpy().tolist()
            targets = path[1:].numpy().tolist()
            sub_edges = torch.IntTensor([sources, targets]).long()
            sub_edges = sub_edges.T[~sign[1:]].T
            # undirectional
            if sub_edges.shape[1] != 0:
                sub_edges = to_undirected(sub_edges)
            edge_list.append(sub_edges)
        return node_list, edge_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        return data


# class GenerateSubGraph(Dataset):
#     def __init__(self, data, num_hops=1, transform=None):
#         self.num_nodes = data.num_nodes
#         self.num_hops = num_hops
#         self.transform = transform

#         self.neighs_list = []
#         # self.map_list = []
#         self.raw_texts = data.raw_texts
#         self.graphs = self._parse_source_data(data)

#     def _parse_source_data(self, data):
#         collected_graph_data = []
#         for idx in tqdm(range(self.num_nodes)):
#             neig_idx, _, _, _ = k_hop_subgraph([idx], 1, data.edge_index, relabel_nodes=False)
#             neig_idx = neig_idx[1:]
#             self.neighs_list.append(neig_idx)

#             subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph([idx], self.num_hops, data.edge_index, relabel_nodes=False)
#             node_idx = torch.unique(sub_edge_index)
#             node_idx_map = {j: i for i, j in enumerate(node_idx.tolist())}
#             # self.map_list.append(node_idx_map)

#             sources_idx = list(map(node_idx_map.get, sub_edge_index[0].tolist()))
#             target_idx = list(map(node_idx_map.get, sub_edge_index[1].tolist()))

#             edge_index = torch.tensor([sources_idx, target_idx], dtype=torch.long)

#             if len(node_idx)==0:
#                 graph = Data(
#                     edge_index=torch.empty((2, 0), dtype=torch.long),
#                     x=data.x[idx].unsqueeze(0),
#                     y=data.y[idx],
#                     root_n_index=0,
#                 )
#                 # continue
#             else:
#                 graph = Data(
#                     edge_index=edge_index,
#                     x=data.x[node_idx],
#                     y=data.y[idx],
#                     root_n_index=node_idx_map[idx],
#                 )

#             if self.transform:
#                 graph = self.transform(graph)

#             collected_graph_data.append(graph)
#         return collected_graph_data


# class SourceGraphDataset(Dataset):
#     def __init__(self, data, neigh_num=3):
#         self.neigh_num = neigh_num
#         self.raw_texts = data.raw_texts
#         self.neighs_list = []
        
#         self.parse_source_data(data)
    
#     def parse_source_data(self, data):
#         for idx in tqdm(range(data.num_nodes)):
#             neig_idx, _, _, _ = k_hop_subgraph([idx], 1, data.edge_index, relabel_nodes=False)
#             neig_idx = neig_idx[1:]
#             self.neighs_list.append(neig_idx)

#     def __len__(self):
#         return len(self.neighs_list)

#     def __getitem__(self, idx):
#         neighs = self.neighs_list[idx]
#         if len(neighs) > self.neigh_num:
#             t_n = np.random.choice(neighs, self.neigh_num, replace=False)
#         else:
#             if len(neighs) > 0:
#                 t_n = np.random.choice(neighs, self.neigh_num, replace=True)
#             else:
#                 t_n = None
        
#         if t_n is None:
#             t_n = [idx, idx, idx]
#         else:
#             t_n = t_n.tolist()

#         return self.raw_texts[idx], [self.raw_texts[i] for i in t_n]


# class SourceGraphDataset(Dataset):
#     def __init__(self, data, num_hops=1, neigh_num=3, transform=None):
#         self.data = data
#         self.num_hops = num_hops
#         self.neigh_num = neigh_num
#         self.transform = transform

#         self.neighs_list = []
#         self.graphs = self._parse_source_data()


#     def _parse_source_data(self):
#         collected_graph_data = []
#         for idx in tqdm(range(self.data.num_nodes)):
#             neig_idx, _, _, _ = k_hop_subgraph([idx], 1, self.data.edge_index, relabel_nodes=False)
#             subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph([idx], self.num_hops, self.data.edge_index, relabel_nodes=False)

#             node_idx = torch.unique(sub_edge_index)
#             node_idx_map = {j: i for i, j in enumerate(node_idx.tolist())}
#             # self.map_list.append(node_idx_map)

#             neig_idx = neig_idx[1:]
#             neig_idx = list(map(node_idx_map.get, neig_idx.tolist()))
#             self.neighs_list.append(neig_idx)

#             sources_idx = list(map(node_idx_map.get, sub_edge_index[0].tolist()))
#             target_idx = list(map(node_idx_map.get, sub_edge_index[1].tolist()))

#             edge_index = torch.tensor([sources_idx, target_idx], dtype=torch.long)

#             if len(node_idx)==0:
#                 graph = Data(
#                     edge_index=torch.empty((2, 0), dtype=torch.long),
#                     x=self.data.x[idx].unsqueeze(0),
#                     y=self.data.y[idx],
#                     root_n_index=0,
#                     raw_texts=[self.data.raw_texts[idx]]
#                 )
#             else:
#                 graph = Data(
#                     edge_index=edge_index,
#                     x=self.data.x[node_idx],
#                     y=self.data.y[idx],
#                     root_n_index=node_idx_map[idx],
#                     raw_texts=[self.data.raw_texts[i] for i in node_idx.tolist()]
#                 )

#             if self.transform:
#                 graph = self.transform(graph)

#             collected_graph_data.append(graph)
#         return collected_graph_data

#     def __len__(self):
#         return len(self.graphs)

#     def __getitem__(self, idx):
#         data = self.graphs[idx]
#         data_new = Data(x=data.x, edge_index=data.edge_index, y=data.y, root_n_index=data.root_n_index)

#         neighs = self.neighs_list[idx]
#         if len(neighs) > self.neigh_num:
#             t_n = np.random.choice(neighs, self.neigh_num, replace=False)
#         else:
#             if len(neighs) > 0:
#                 t_n = np.random.choice(neighs, self.neigh_num, replace=True)
#             else:
#                 t_n = None
        
#         if t_n is None:
#             t_n = [data.root_n_index, data.root_n_index, data.root_n_index]
#         else:
#             t_n = t_n.tolist()

#         data_new.s_n_text = data.raw_texts[data.root_n_index]
#         data_new.t_n_text = [data.raw_texts[i] for i in t_n]

#         return data_new
