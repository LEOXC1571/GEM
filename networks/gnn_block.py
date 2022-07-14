# -*- coding: utf-8 -*-
# @Filename: gnn_block
# @Date: 2022-07-12 09:56
# @Author: Leo Xu
# @Email: leoxc1571@163.com


import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling

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
        nodes = torch.ones(graph.num_nodes, 1)
        norm = self.graph_pool(graph, nodes)
        norm = torch.sqrt(norm)
        norm = torch.gather(norm, graph.graph_node_id) #TODO: torch.gather and paddle.gather
        return feature / norm


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

        msg = graph.send(
            message_func=_send_func,
            node_feat={'h': node_feat},
            edge_feat={'h': edge_feat})
        node_feat = graph.recv(reduce_func=_recv_func, msg=msg)
        node_feat = self.mlp(node_feat)
        return node_feat
