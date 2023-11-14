#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:54:10 2022

@author: brochetc

DataSet class from Importance_Sampled images

"""

import os
import re

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset

################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4}


################
class ISDataset(Dataset):

    def __init__(self, config, path, csv_file, add_coords=False):
        self.data_dir = path
        self.labels = pd.read_csv(os.path.join(path, csv_file))

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
        file_name = self.labels.iloc[idx, 0]
        sample_path = os.path.join(self.data_dir, file_name)
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
        sample_id = re.search(r'\d+', file_name).group()
        return sample, sample_id
