#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:54:10 2022

@author: brochetc

DataSet class from Importance_Sampled images

"""

import os

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4}


################
class ISDataset(Dataset):

    def __init__(self, config,path, add_coords=False):
        self.data_dir = path
        self.labels = pd.read_csv(os.path.join(path, 'IS_method_labels.csv'))

        self.CI = config.crop
        self.VI = [var_dict[var] for var in config.var_indexes]

        ## adding 'positional encoding'
        self.add_coords = add_coords
        try:
            Means = np.load(os.path.join(self.data_dir, config.mean_file))[self.VI]
            Maxs = np.load(os.path.join(self.data_dir, config.max_file))[self.VI]
        except (FileNotFoundError, KeyError):
            try:
                Means = np.load(config.mean_file)[self.VI]
                Maxs = np.load(config.max_file)[self.VI]
            except (FileNotFoundError, KeyError):
                raise ValueError(
                    'The mean_file and max_file must be specified in the parser using --mean_file and --max_file options')

        self.means = list(tuple(Means))
        self.stds = list(tuple((1.0 / 0.95) * (Maxs)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # idx=idx+19
        sample_path = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
        sample = np.float32(np.load(sample_path + '.npy')) \
            [self.VI, self.CI[0]:self.CI[1], self.CI[2]:self.CI[3]]

        ## transpose to get off with transform.Normalize builtin transposition
        sample = sample.transpose((1, 2, 0))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),
            ]
        )
        sample = self.transform(sample)
        return sample

#
# class ISData_Loader():
#
#     def __init__(self, path, batch_size, var_indexes, crop_indexes, \
#                  shuf=False, add_coords=False, device='cuda'):
#         self.path = path
#         self.batch = batch_size
#
#         self.shuf = shuf  # shuffle performed once per epoch
#
#         self.VI = var_indexes
#         self.CI = crop_indexes
#         # self.img_size=img_size
#
#         Means = np.load(path + 'mean_with_orog.npy')[self.VI]
#         Maxs = np.load(path + 'max_with_orog.npy')[self.VI]
#
#         self.means = list(tuple(Means))
#         self.stds = list(tuple((1.0 / 0.95) * (Maxs)))
#         self.add_coords = add_coords
#
#     def loader(self):
#         from multiprocessing import cpu_count
#         self.device = 'cuda'
#         dataset = ISDataset(self.path, 'IS_method_labels.csv', self.VI, self.CI, self.device)
#
#         loader = DataLoader(dataset=dataset,
#                             batch_size=self.batch,
#                             num_workers=cpu_count(),
#                             pin_memory=True,
#                             shuffle=True,
#                             drop_last=True,
#                             )
#         return loader, dataset
