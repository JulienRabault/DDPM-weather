#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:44:08 2022

@authors: gandonb, rabaultj, brochetc


DataSet/DataLoader classes from Importance_Sampled images
DataSet:DataLoader classes for test samples

"""

import os
import pandas as pd

import numpy as np
import pandas as pd
import scipy.ndimage
import yaml
from filelock import FileLock
from torch import Tensor, from_numpy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Normalize, Compose
from data.statsMasked import normalizeUnderMask
from data.normalize_funcs import MultiOptionNormalize
from multiprocessing import Manager

################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4, 'z500': 5, 't850': 6, 'tpw850': 7}


class DatasetCache(object):
    def __init__(self, use_cache=False):
        self.use_cache = use_cache
        self.manager = Manager()
        self._dict = self.manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get function is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, sample, importance, pos):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (sample, importance, pos)


################
class ISDataset(Dataset):

    def __init__(self, config, dataset_handler_yaml, sample_method, variable_indices,
     transform, use_cache=False):
        self.config = config
        self.dataset_handler_yaml = dataset_handler_yaml
        self.sample_method = sample_method
        self.VI = variable_indices
        self.transform = transform
        self.labels = pd.read_csv(f"{self.config.csv_file}")
        self.cache = DatasetCache(use_cache=use_cache)
        if use_cache:
            import resource
            resource.setrlimit(
                resource.RLIMIT_CORE,
                (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            
        ## choosing the sampling method (either random crop or fixed coordinates)

        assert sample_method in ['random', 'coords']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.config.data_dir, self.labels.iloc[idx]["Name"])
        if self.sample_method=='coords':
            sample = np.float32(np.load(f"{sample_path}.npy"))[self.VI, self.config.crop[0]:self.config.crop[1], self.config.crop[2]:self.config.crop[3]]
            position = self.config.crop
           
        # importance = self.labels.iloc[idx]["Importance"]
        #### IMPORTANCE_ERROR
        importance = 0

        if 'rr' in self.config.var_indexes: #applying transformations on rr only if selected
            for _ in range(self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]):
                sample[0] = np.log(1 + sample[0])
            if self.dataset_handler_yaml["rr_transform"]["symetrization"] and np.random.random() <= 0.5:
                sample[0] = -sample[0]
            
        ## transpose to get off with transform.Normalize builtin transposition
        sample = sample.transpose((1,2,0))
        sample = self.transform(sample)

        self.cache.cache(idx, sample, importance, position)

        return sample, importance, position


class ISData_Loader():

    def __init__(self, dataset_type, config, shuf=False):
        print(f"{dataset_type} files data loader...")
        self.config = config
        self.batch_size = self.config.batch_size
        self.VI = [var_dict[var] for var in self.config.var_indexes]
        self.sampled_indices = {var: i for i, var in enumerate(self.config.var_indexes)} # corresponding indices in the prepared data (after sampling)

        self.shuf = shuf #shuffle performed once per epoch

        self.dataset_handler_yaml = self.read_dataset_handler_config_file()
        self.maxs, self.mins, self.means, self.stds = self.init_normalization()

        if self.stds is not None:
            self.stds *= 1.0 / 0.95

    def read_dataset_handler_config_file(self):
        print(f"{self.config.dataset_handler_config}")
        with open(f"{self.config.dataset_handler_config}", "r") as dataset_handler_config_file:
            print(f"{self.config.dataset_handler_config} opened...")
            return yaml.safe_load(dataset_handler_config_file)

    def init_normalization(self):
        normalization_type = self.dataset_handler_yaml["normalization"]["type"]
        if normalization_type == "mean":
            means, stds = self.load_stat_files(normalization_type, "mean", "std")
            return None, None, means[self.VI], stds[self.VI]
        if normalization_type == "minmax":
            maxs, mins = self.load_stat_files(normalization_type, "max", "min")
            return maxs[self.VI], mins[self.VI], None, None
        print("No normalization set")
        return None, None, None, None

    def load_stat_files(self, normalization_type, str1, str2):
        mean_or_max_filename = f"{str1}_{self.dataset_handler_yaml['stat_version']}"
        mean_or_max_filename += "_log" * self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]
        std_or_min_filename = f"{str2}_{self.dataset_handler_yaml['stat_version']}"
        std_or_min_filename += "_log" * self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]
        if self.dataset_handler_yaml["normalization"]["per_pixel"]:
            mean_or_max_filename += "_ppx"
            std_or_min_filename += "_ppx"
        mean_or_max_filename += ".npy"
        std_or_min_filename += ".npy"
        print(f"Normalization set to {normalization_type}")
        means_or_maxs = np.load(f"{self.config.data_dir}{self.dataset_handler_yaml['stat_folder']}{mean_or_max_filename}").astype('float32')
        print(f"{str1} file found")
        stds_or_mins = np.load(f"{self.config.data_dir}{self.dataset_handler_yaml['stat_folder']}{std_or_min_filename}").astype('float32')
        print(f"{str2} file found")
        return means_or_maxs, stds_or_mins

    def transform(self):
        options = [ToTensor()]
        normalization = self.dataset_handler_yaml["normalization"]["type"]
        if normalization != "None":
            if 'rr' in self.config.var_indexes and self.dataset_handler_yaml["rr_transform"]["symetrization"]: #applying transformations on rr only if selected
                if normalization == "means":
                    self.means[0] = np.zeros_like(self.means[0])
                elif normalization == "minmax":
                    self.mins[0] = -self.maxs[0]
        options.append(MultiOptionNormalize(self.means, self.stds, self.maxs, self.mins, self.config, self.dataset_handler_yaml))
        transform = Compose(options)
        return transform

    def loader(self, world_size=None, local_rank=None, kwargs=None):

        if kwargs is not None:
            with FileLock(os.path.expanduser("~/.horovod_lock")):  # if absent, causes SIGSEGV error
                sample_method = 'coords'
                dataset = ISDataset(self.config, self.dataset_handler_yaml, sample_method, self.VI, self.transform()) # coordinates system

        self.sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
        if kwargs is not None:
            loader = DataLoader(dataset = dataset,
                            batch_size = self.batch_size,
                            shuffle = self.shuf,
                            sampler = self.sampler,
                            drop_last = True,
                            num_workers=1,
                            **kwargs)
        else:
            loader = DataLoader(dataset = dataset,
                            batch_size = self.batch_size,
                            shuffle = self.shuf,
                            sampler = self.sampler,
                            drop_last = True,
                            num_workers=1)
        return loader
