import warnings

import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from distributed import get_rank, is_main_gpu

import torch.distributed as dist
import torch.nn as nn


class Ddpm_base:
    def __init__(
            self,
            model: torch.nn.Module,
            config,
            dataloader=None,) -> None:
        """
        Initialize the Trainer.
        Args:
            model (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): The optimizer for model parameters.
            config: Configuration settings.
        """
        self.optimizer = None
        self.scheduler = None
        self.config = config
        self.gpu_id = get_rank()
        self.timesteps = model.num_timesteps
        print(self.gpu_id)
        self.model = model.to(self.gpu_id)
        self.dataloader = dataloader
        self.snapshot_path = self.config.model_path
        if self.snapshot_path is not None:
            if is_main_gpu():
                print(f"#INFO : Loading snapshot")
            self._load_snapshot(self.snapshot_path)
        if self.dataloader is not None:
            self.train_dataset = self.dataloader.dataset
            self.stds = self.train_dataset.stds
            self.means = self.train_dataset.means
        if config.invert_norm:
            self.transforms_func = transforms.Compose([
                transforms.Normalize(mean=[0.] * len(self.config.var_indexes),
                                     std=[1 / el for el in self.stds]),
                transforms.Normalize(mean=[-el for el in self.means],
                                     std=[1.] * len(self.config.var_indexes)),
            ])
        else:
            def transforms_func(x): return x
            self.transforms_func = transforms_func
        if dist.is_initialized():
            dist.barrier()
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_id],
                                                             output_device=self.gpu_id)

    def _load_snapshot(self, snapshot_path):
        """
        Load the snapshot of the training progress.
        Args:
            snapshot_path: Path to the snapshot file.
        """
        loc = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        snapshot = torch.load(snapshot_path, map_location=loc)
        if "SCHEDULER_STATE" in snapshot and self.scheduler is not None:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        if "WANDB_ID" in snapshot:
            self.wandb_id = snapshot["WANDB_ID"]
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        if self.optimizer is not None:
            self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.best_loss = snapshot["BEST_LOSS"]
        try:
            data_config = snapshot["DATA"]
            if data_config['V_IDX'] != self.config.var_indexes or data_config['CROP'] != self.config.crop:
                raise ValueError("The variable indexes or crop of the snapshot do not match the current config")
        except KeyError:
            warnings.warn("The snapshot does not contain data config, assuming it is the same as the current config")
        self.stds = snapshot["STDS"]
        self.means = snapshot["MEANS"]
        if self.config.debug_log:
            print(
                f"\n#LOG : [GPU{self.gpu_id}] Resuming training from {snapshot_path} at Epoch {self.epochs_run}")
        elif is_main_gpu():
            print(
                f"#INFO : Resuming training from {snapshot_path} at Epoch {self.epochs_run}")
        self.epochs_run += 1

    def _sample_batch(self, nb_img=4):
        if nb_img <= 0:
            return []
        sampled_images = self.model.module.sample(
            batch_size=nb_img) if dist.is_initialized() else self.model.sample(batch_size=nb_img)
        sampled_images = self.transforms_func(sampled_images)
        return sampled_images.cpu().numpy()
