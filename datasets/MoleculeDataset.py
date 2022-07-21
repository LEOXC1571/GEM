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

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

class  MoleculeDataset(InMemoryDataset):
    def __init__(self, data_path, smile2graph=None):
        super(MoleculeDataset, self).__init__()
        self.data_path = data_path
        self.smile2graph = smile2graph
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.folder = os.path.join(root, dataset)

        # if os.path.isdir(self.folder):

    def processed_file_names(self):
        return 'geometric_dta_processed.pt'

    def process(self):
        files = sorted(glob('%s/*' % self.data_path))
        data_list = []
        for file in files:
            with open(file, 'r') as f:
                tmp_data_list = [line.strip() for line in f.readlines()]
            data_list.extend(tmp_data_list)
