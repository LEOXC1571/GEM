# -*- coding: utf-8 -*-
# @Filename: MoleculeDataset
# @Date: 2022-07-18 10:06
# @Author: Leo Xu
# @Email: leoxc1571@163.com

import os
from glob import glob
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_path, smile2graph=None):
        super(MoleculeDataset, self).__init__()
        self.data_path = data_path
        self.smile2graph = smile2graph
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.folder = os.path.join(root, dataset)

        # if os.path.isdir(self.folder):

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def read_data2list(self):
        print('Loading SMILES strings ......')
        files = sorted(glob('%s/*' % self.data_path))
        smiles_list = []
        for file in files:
            with open(file, 'r') as f:
                tmp_data_list = [line.strip() for line in f.readlines()]
            smiles_list.extend(tmp_data_list)
        return smiles_list

    def smile2graph(self, smiles_list):
        length = len(smiles_list)
        data_list = []
        for i in tqdm(range(len(length))):
            data = Data()
            smiles = smiles_list[i]
            graph = smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)

            data_list.append(data)

        split_dict = self.get_idx_split()
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert (all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        assert (all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(os.path.join(self.root, 'split_dict.pt')))
        return split_dict



