# -*- coding: utf-8 -*-
# @Filename: gnn_block
# @Date: 2022-07-12 09:56
# @Author: Leo Xu
# @Email: leoxc1571@163.com


import torch
import torch.nn as nn
import dgl
import dgl.function as fn

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

# def read_rows(data, index):
#     if data is None:
#         return None
#     elif isinstance(data, dict):
#         new_data = {}
#         for key, value in data.items():
#             new_data[key] = read_rows(value, index)
#         return new_data

class RowReader(object):
    """Memory Efficient RowReader
    """

    def __init__(self, nfeat, index):
        self.nfeat = nfeat
        self.loaded_nfeat = {}
        self.index = index

    def __getitem__(self, key):
        if key not in self.loaded_nfeat:
            # self.loaded_nfeat[key] = self.read_rows(self.nfeat[key], self.index)
            self.loaded_nfeat[key] = self.nfeat[key][self.index, :]
        return self.loaded_nfeat[key]

    def read_rows(self, data, index):
        new_data = {}
        for key, value in data:
            new_data[key] = self.read_rows(value, index)
        return new_data

class RowReader2(object):
    """Memory Efficient RowReader
    """

    def __init__(self, nfeat, index):
        self.nfeat = nfeat
        self.loaded_nfeat = {}
        self.index = index

    def __getitem__(self, key):
        if key not in self.loaded_nfeat:
            # self.loaded_nfeat[key] = self.read_rows(self.nfeat[key], self.index)
            self.loaded_nfeat[key] = self.nfeat[key][self.index, :]
        return self.loaded_nfeat[key]

    def read_rows(self, data, index):
        new_data = {}
        for key, value in data:
            new_data[key] = self.read_rows(value, index)
        return new_data


class GraphNorm(nn.Module):
    """Implementation of graph normalization. Each node features is divied by sqrt(num_nodes) per graphs.

    Args:
        graph: the graph object from (:code:`Graph`)
        feature: A tensor with shape (num_nodes, feature_size).

    Return:
        A tensor with shape (num_nodes, hidden_size)

    References:

    [1] BENCHMARKING GRAPH NEURAL NETWORKS. https://arxiv.org/abs/2003.00982

    """

    def __init__(self):
        super(GraphNorm, self).__init__()
        self.graph_pool = SumPooling()

    def forward(self, graph, feature):
        """graph norm"""
        nodes = torch.ones(graph.num_nodes(), 1)
        norm = self.graph_pool(graph, nodes)
        norm = torch.sqrt(norm)
        norm = torch.gather(norm, graph.graph_node_id) #TODO: torch.gather and paddle.gather
        return feature / norm

def send(graph,
         message_func,
         edge_feat=None,
         node_feat=None, ):
    src_feat_temp = {}
    dst_feat_temp = {}

    src_feat_temp.update(node_feat)
    dst_feat_temp.update(node_feat)

    edge_feat_temp = {}

    edge_feat_temp.update(edge_feat)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    src_feat = RowReader(src_feat_temp, src)
    dst_feat = RowReader(dst_feat_temp, dst)
    msg = message_func(src_feat, dst_feat, edge_feat_temp)

    return msg




def recv(graph, reduce_func, msg, recv_mode="dst"):

    # dst, src, eid = self.adj_dst_index.triples()
    # src = graph.edges(form='all')[0]
    dst = graph.edges(form='all')[1]
    eid = graph.edges(form='all')[2]

    msg = RowReader2(msg, eid)

    uniq_ind, segment_ids = torch.unique(dst, return_inverse=True)

    # bucketed_msg = Message(msg, segment_ids)
    # output = reduce_func(bucketed_msg)
    # output = math.segment_sum(msg, segment_ids)
    output = torch.zeros(size=(uniq_ind.shape[0], msg.nfeat['h'].shape[1])).to(msg.nfeat['h'].device)
    for idx in segment_ids:
        output[idx] += msg['h'][idx]

    # output_dim = output.shape[-1]
    # init_output = torch.zeros(size=(dst.shape[0], output_dim), dtype=output.dtype)
    # final_output = torch.scatter(init_output, uniq_ind, output)

    return output


# class MeanPool(nn.Layer):
#     """
#     TODO: temporary class due to pgl mean pooling
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.graph_pool = pgl.nn.GraphPool(pool_type="sum")
#
#     def forward(self, graph, node_feat):
#         """
#         mean pooling
#         """
#         sum_pooled = self.graph_pool(graph, node_feat)
#         ones_sum_pooled = self.graph_pool(
#             graph,
#             paddle.ones_like(node_feat, dtype="float32"))
#         pooled = sum_pooled / ones_sum_pooled
#         return pooled


class GIN(nn.Module):
    """
    Implementation of Graph Isomorphism Network (GIN) layer with edge features
    """

    def __init__(self, hidden_size):
        super(GIN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, graph, node_feat, edge_feat):
        """
        Args:
            node_feat(tensor): node features with shape (num_nodes, feature_size).
            edge_feat(tensor): edges features with shape (num_edges, feature_size).
        """

        def _send_func(src_feat, dst_feat, edge_feat):
            x = src_feat['h'] + edge_feat['h']
            return {'h': x}

        def _recv_func(msg):
            return msg.reduce_sum(msg['h'])

        # msg = graph.send(
        #     message_func=_send_func,
        #     node_feat={'h': node_feat},
        #     edge_feat={'h': edge_feat})
        # node_feat = graph.recv(reduce_func=_recv_func, msg=msg)
        # node_feat = self.mlp(node_feat)

        msg = send(
            graph,
            message_func=_send_func,
            node_feat={'h': node_feat},
            edge_feat={'h': edge_feat})
        node_feat = recv(graph, reduce_func=_recv_func, msg=msg)
        node_feat = self.mlp(node_feat)

        # msg = graph.send_and_recv(graph.edges(), fn.src_mul_edge(), fn.reducer.sum())
        # node_feat = self.mlp(msg)

        return node_feat


