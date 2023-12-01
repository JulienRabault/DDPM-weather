#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:44:08 2022

@authors: gandonb, rabaultj, brochetc


DataSet/DataLoader classes from Importance_Sampled images
DataSet:DataLoader classes for test samples

"""

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import Dataset

################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4, 'z500': 5, 't850': 6, 'tpw850': 7}

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data

class MultiOptionNormalize(object):
    def __init__(self, value_sup, value_inf, data_config, config):
        self.value_sup = value_sup
        self.value_inf = value_inf
        self.data_config = data_config
        self.config = config
        if 'rr' in self.config.var_indexes:
            self.gaussian_std = self.data_config.rr_transform['gaussian_std']
            if self.gaussian_std:
                for _ in range(self.data_config.rr_transform['log_transform_iteration']):
                    self.gaussian_std = np.log(1 + self.gaussian_std)
                self.gaussian_std_map = np.random.choice([-1, 1], size=(self.config.image_size, self.config.image_size)) * self.gaussian_std
                self.gaussian_noise = np.mod(np.random.normal(0, self.gaussian_std, size=(self.config.image_size, self.config.image_size)), self.gaussian_std_map)
        if np.ndim(self.value_sup) > 1:
            if 'rr' in self.config.var_indexes:
                if self.data_config.normalization['for_rr']['blur_iteration'] > 0:
                    gaussian_filter = np.float32([[1, 4,  6,  4,  1],
                                                [4, 16, 24, 16, 4],
                                                [6, 24, 36, 24, 6],
                                                [4, 16, 24, 16, 4],
                                                [1, 4,  6,  4,  1]]) / 256.0
                    for _ in range(self.data_config.normalization['for_rr']['blur_iteration']):
                        self.value_sup[0] = scipy.ndimage.convolve(self.value_sup[0], gaussian_filter, mode='mirror')
            self.value_inf = torch.umpy(self.value_inf)
            self.value_sup = torch.from_numpy(self.value_sup)
        else:
            self.value_inf = torch.from_numpy(self.value_inf).view(-1, 1, 1)
            self.value_sup = torch.from_numpy(self.value_sup).view(-1, 1, 1)

    def __call__(self, sample):
        if not isinstance(sample, torch.Tensor):
            raise TypeError(f'Input sample should be a torch tensor. Got {type(sample)}.')
        if sample.ndim < 3:
            raise ValueError(f'Expected sample to be a tensor image of size (..., C, H, W). Got tensor.size() = {sample.size()}.')
        
        if 'rr' in self.config.var_indexes:
            for _ in range(self.data_config.rr_transform['log_transform_iteration']):
                sample[0] = torch.log(1 + sample[0])
            if self.data_config.rr_transform['symetrization'] and np.random.random() <= 0.5:
                sample[0] = -sample[0]
            if self.gaussian_std != 0:
                mask_no_rr = (sample[0].numpy() <= self.gaussian_std)
                sample[0] = sample[0].add_(from_numpy(self.gaussian_noise * mask_no_rr))
        if self.data_config.normalization['type'] == 'mean':
            sample = (sample - self.value_inf) / self.value_sup
        elif self.data_config.normalization['type'] == 'minmax':
            sample = -1 + 2 * ((sample - self.value_inf) / (self.value_sup - self.value_inf))
        return sample

class Dataset_config:
    def __init__(self, dataset_config_file):
        # Load YAML configuration file and initialize logger
        yaml_config = load_yaml(dataset_config_file)
        for prop, value in yaml_config.items():
            setattr(self, prop, value)

################
class ISDataset(Dataset):
    def __init__(self, config, add_coords=False):
        """
        Initialize the ISDataset.
        Args:
            config: Configuration settings.
            path (str): Directory path containing data.
            csv_file (str): CSV file containing labels and information.
            add_coords (bool): Whether to add positional encoding.
        """
        self.data_dir = Path(config.data_dir)
        self.labels = pd.read_csv(self.data_dir / config.csv_file)
        self.config = config
        self.CI = config.crop
        self.VI = [var_dict[var] for var in config.var_indexes]
        self.ensembles = None
        
        # Group labels by guiding column if specified
        if self.config.guiding_col is not None:
            if 'Unnamed: 0' in self.labels:
                self.labels = self.labels.drop('Unnamed: 0', axis=1)
            self.ensembles = self.labels.groupby([self.config.guiding_col]).agg(lambda x: x)
            self.ensembles = self.ensembles['Name']

        # Add positional encoding
        self.add_coords = add_coords

        self.data_config = Dataset_config(config.dataset_handler_config)

        # Depending on the normalization, value_sup is max or std or Q90... value_min is min or mean or Q10...
        self.value_sup, self.value_inf = self.init_normalization()

        transformations = self.prepare_tranformations()
        self.transform = transforms.Compose(transformations)
    
    def prepare_tranformations(self):
        transformations = []
        normalization = self.data_config.normalization['type']
        if normalization != 'None':
            if 'rr' in self.config.var_indexes and self.data_config.rr_transform['symetrization']: #applying transformations on rr only if selected
                if normalization == 'means':
                    # mean of rr is 0
                    self.value_inf[var_dict['rr']] = np.zeros_like(self.value_inf[var_dict['rr']])
                elif normalization == 'minmax':
                    # min of 'negative rain' is -max
                    self.value_inf[var_dict['rr']] = -self.value_sup[var_dict['rr']]
        transformations.append(transforms.ToTensor())
        transformations.append(MultiOptionNormalize(self.value_sup, self.value_inf, self.data_config, self.config))
        return transformations


    def init_normalization(self):
        normalization_type = self.data_config.normalization['type']
        if normalization_type == 'mean':
            stds, means = self.load_stat_files(normalization_type, 'std', 'mean')
            return stds[self.VI] * 1.0 / 0.95, means[self.VI]

        if normalization_type == 'minmax':
            maxs, mins = self.load_stat_files(normalization_type, 'max', 'min')
            return maxs[self.VI], mins[self.VI]

        print('No normalization set')
        return None, None

    def load_stat_files(self, normalization_type, str_sup, str_inf):
        print(f'Normalization set to {normalization_type}')
        norm_vars = []
        for name in (str_sup, str_inf):
            filename = f'{name}_{self.data_config.stat_version}'
            filename += '_log' * self.data_config.rr_transform['log_transform_iteration']
            
            if self.data_config.normalization['per_pixel']:
                filename += '_ppx'

            filename += '.npy'

            try:
                path = self.data_dir / self.data_config.stat_folder / filename
                norm_var = np.load(path).astype('float32')
            except FileNotFoundError as err:
                raise FileNotFoundError(f'{name} file was not found at this location: {path}')
            norm_vars.append(norm_var)
        return norm_vars

    def __len__(self):
        """
        Get the length of the dataset.
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Args:
            idx (int): Index of the sample.
        Returns:
            dict: Dictionary containing 'img' (sample), 'img_id' (sample ID), and 'condition' (conditional sample).
        """
        file_name = self.labels.iloc[idx, 0]
        sample = self.file_to_torch(file_name)

        # Get conditional sample if ensembles are specified
        if self.ensembles is not None:
            ensemble_id = self.labels.loc[idx, self.config.guiding_col]
            try:
                ensemble = self.ensembles[ensemble_id].tolist()
                ensemble.remove(self.labels.iloc[idx, 0])
                ens = random.sample(ensemble, 1)
            except:
                ens = self.ensembles[ensemble_id]
            condition = self.file_to_torch(ens)
        else:
            condition = torch.empty(0)

        sample_id = re.search(r'\d+', file_name).group()
        return {'img': sample, 'img_id': sample_id, 'condition': condition}

    def file_to_torch(self, file_name):
        """
        Convert a file to a torch tensor.
        Args:
            file_name (str or list): Name of the file or list of file names.
        Returns:
            torch.Tensor: Torch tensor representing the sample.
        """
        if type(file_name) == list:
            file_name = file_name[0]
        sample_path = self.data_dir / f'{file_name}.npy'
        sample = np.float32(np.load(sample_path)) \
            [self.VI, self.CI[0]:self.CI[1], self.CI[2]:self.CI[3]]
        sample = sample.transpose((1, 2, 0))
        sample = self.transform(sample)
        return sample

