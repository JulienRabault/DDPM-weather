import logging
import os
import warnings

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib
from torchvision.transforms import transforms

from utils.distributed import get_rank, is_main_gpu

matplotlib.use("Agg")
logger = logging.getLogger('logddp')


class Ddpm_base:
    def __init__(
            self,
            model: torch.nn.Module,
            config,
            dataloader=None) -> None:
        """
        Initialize the Trainer.
        Args:
            model (torch.nn.Module): The neural network model.
            config: Configuration settings.
        """
        self.optimizer = None
        self.scheduler = None
        self.config = config
        self.gpu_id = get_rank()
        self.timesteps = model.num_timesteps
        self.dataloader = dataloader
        self.snapshot_path = self.config.model_path
        self.model = model
        if self.snapshot_path is not None:
            if is_main_gpu():
                logger.info(f"Loading snapshot")
            self._load_snapshot(self.snapshot_path)
        model.to(torch.device(self.gpu_id))
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
            def transforms_func(x):
                return x

            self.transforms_func = transforms_func

        if torch.cuda.device_count() >= 2:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_id],
                                                             output_device=self.gpu_id)
            self.model = self.model.module

    def _load_snapshot(self, snapshot_path):
        """
        Load the snapshot of the training progress.
        Args:
            snapshot_path: Path to the snapshot file.
        """
        snapshot = torch.load(snapshot_path, map_location=get_rank())
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
        self.stds = data_config["STDS"]
        self.means = data_config["MEANS"]
        if is_main_gpu():
            logger.info(
                f" Resuming model from {snapshot_path} at Epoch {self.epochs_run}")
        self.epochs_run += 1

    def _sample_batch(self, nb_img=4, condition=None):
        """
        Sample a batch of images.
        Args:
            nb_img (int): Number of images to sample.
        Returns:
            numpy.ndarray: Array of sampled images.
        """
        if nb_img <= 0:
            return []
        if condition is None:
            sampled_images = self.model.sample(batch_size=nb_img)
        else:
            sampled_images = self.model.sample(batch_size=nb_img, condition=condition)
        sampled_images = self.transforms_func(sampled_images)
        return sampled_images.cpu().numpy()

    def plot_grid(self, file_name, np_img):
        """
        Plot a grid of images.
        Args:
            file_name (str): Name of the file to save the plot.
            np_img (numpy.ndarray): Array of images to plot.
        """
        nb_image = len(np_img)
        fig, axes = plt.subplots(nrows=min(6, nb_image), ncols=len(
            self.config.var_indexes), figsize=(10, 10))
        for i in range(min(6, nb_image)):
            for j in range(len(self.config.var_indexes)):
                cmap = 'viridis' if self.config.var_indexes[j] != 't2m' else 'bwr'
                image = np_img[i, j]
                if len(self.config.var_indexes) > 1 and min(6, nb_image) > 1:
                    im = axes[i, j].imshow(
                        image, cmap=cmap, origin='lower')
                    axes[i, j].axis('off')
                    fig.colorbar(im, ax=axes[i, j])
                else:
                    im = axes[i].imshow(image, cmap=cmap, origin='lower')
                    axes[i].axis('off')
                    fig.colorbar(im, ax=axes[i])
        plt.savefig(os.path.join(f"{self.config.run_name}", "samples", file_name),
                    bbox_inches='tight')
        plt.close()
