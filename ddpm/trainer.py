import csv
import logging
import os.path
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import distributed as dist
from tqdm import tqdm

from ddpm.ddpm_base import Ddpm_base
from utils.distributed import is_main_gpu

logger = logging.getLogger('logddp')


class Trainer(Ddpm_base):

    def __init__(self, model, config, dataloader=None,
                 optimizer=None):
        """
        Initialize the Trainer class.
        Args:
            model: The neural network model for training.
            config: Configuration settings for training.
            dataloader: The data loader for training data.
            optimizer: The optimizer for model parameter updates.
        """
        super().__init__(model, config, dataloader)
        self.optimizer = optimizer
        self.epochs_run = 0
        self.best_loss = float('inf')
        self.guided_diffusion = self.config.guiding_col is not None
        if self.config.scheduler:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.config.lr,
                                                                 epochs=self.config.scheduler_epoch,
                                                                 steps_per_epoch=len(
                                                                     self.dataloader),
                                                                 anneal_strategy="cos",
                                                                 pct_start=0.1,
                                                                 div_factor=15.0,
                                                                 final_div_factor=1500.0)
        else:
            self.scheduler = None
    
    def _prepare_batch(self, batch, key_get, convert_keys={}):
        """
        Prepare the batch for training.
        Args:
            batch: Input batch for training.
            key_get: Keys to extract from the batch.
            convert_keys: Keys to convert to tensors.
            convert_tensor: Whether to convert the batch to tensors.
        Returns:
            dict: The prepared batch.
        """
        batch = {key: batch[key] for key in key_get}
        for key in batch.keys():
            if key in convert_keys:
                batch[convert_keys[key]] = batch[key].to(self.gpu_id)
                del batch[key]
            else:
                batch[key] = batch[key].to(self.gpu_id)
        # check if batch is empty and raise
        # if len(batch) != len(key_get):
        #     raise ValueError("Batch is empty")
        # assert len(batch) == len(key_get)
        return batch

    def _run_batch(self, batch):
        """
        Run a single training batch.
        Args:
            batch: Input batch for training.
        Returns:
            float: Loss value for the batch.
        """
        self.optimizer.zero_grad()
        loss = self.model(**batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        """
        Run a training epoch.
        Args:
            epoch (int): Current epoch number.
        Returns:
            float: Average loss for the epoch.
        """
        iters = len(self.dataloader)
        if dist.is_initialized():
            self.dataloader.sampler.set_epoch(epoch)
        total_loss = 0
        loop = tqdm(enumerate(self.dataloader), total=iters,
                             desc=f"Epoch {epoch}/{self.config.epochs + self.epochs_run}", unit="batch",
                             leave=False, postfix="", disable=not is_main_gpu())
        for i, batch in loop:
            needs_keys = ['img'] + (['condition'] if self.guided_diffusion else [])
            batch = self._prepare_batch(batch,needs_keys)
            loss = self._run_batch(batch)
            total_loss += loss
            if self.config.scheduler:
                self.scheduler.step()
            if is_main_gpu():
                loop.set_postfix_str(f"Loss : {total_loss / (i + 1):.6f}")
        logger.debug(
            f"Epoch {epoch} | Batchsize: {self.config.batch_size} | Steps: {len(self.dataloader)} | "
            f"Last loss: {total_loss / len(self.dataloader)} | "
            f"Lr : {self.scheduler.get_last_lr()[0] if self.config.scheduler else self.config.lr}")

        if epoch % self.config.any_time == 0.0 and is_main_gpu():
            condition = self._prepare_batch(batch,['condition']if self.guided_diffusion else [])
            self.sample_train(str(epoch), self.config.n_sample,condition)

        return total_loss / len(self.dataloader)

    def _save_snapshot(self, epoch, path, loss):
        """
        Save a snapshot of the training progress.
        Args:
            epoch (int): Current epoch number.
            path: Path to save the snapshot.
            loss: Loss value at the epoch.

        Returns:
            None
        """
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            'OPTIMIZER_STATE': self.optimizer.state_dict(),
            'BEST_LOSS': loss,
            'TIMESTAMP': self.timesteps,
            'GUIDED_DIFFUSION': self.guided_diffusion,
            'DATA': {
                'STDS': self.stds,
                'MEANS': self.means,
                'V_IDX': self.config.var_indexes,
                'CROP': self.config.crop,
            }
        }
        if self.config.scheduler:
            snapshot["SCHEDULER_STATE"] = self.scheduler.state_dict()
        if self.config.use_wandb:
            snapshot["WANDB_ID"] = wandb.run.id
        torch.save(snapshot, path)
        logger.info(
            f"Epoch {epoch} | Training snapshot saved at {path} | Loss: {loss}")

    def _init_wandb(self):
        """
        Initialize WandB for logger training progress.
        Returns:
            None
        """
        if self.gpu_id != 0:
            return

        t = time.strftime("%d-%m-%y_%H-%M", time.localtime(time.time()))
        wandb.init(project=self.config.wp, resume="auto" if self.config.resume else None, mode=os.environ['WANDB_MODE'],
                   entity=self.config.entityWDB,
                   name=f"{self.config.run_name}_{t}/",
                   config={**vars(self.config), **{"optimizer": self.optimizer.__class__,
                                                   "scheduler": self.scheduler.__class__,
                                                   "lr_base": self.optimizer.param_groups[0]["lr"],
                                                   "weight_decay": self.optimizer.param_groups[0][
                                                       "weight_decay"], }})

    def train(self):
        """
        Start the training process.
        Returns:
            None
        """
        filename_format = "sample_epoch{epoch}_{i}.npy"
        if is_main_gpu():
            self._init_wandb()
            loop = tqdm(range(self.epochs_run, self.config.epochs + self.epochs_run),
                        desc=f"Training...", unit="epoch", postfix="")
        else:
            loop = range(self.epochs_run, self.config.epochs + self.epochs_run)

        for epoch in loop:
            avg_loss = self._run_epoch(epoch)
            if is_main_gpu():
                loop.set_postfix_str(
                    f"Epoch loss : {avg_loss:.5f} | Lr : {(self.scheduler.get_last_lr()[0] if self.config.scheduler else self.config.lr):.6f}")

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self._save_snapshot(epoch, os.path.join(
                        f"{self.config.run_name}", "best.pt"), avg_loss)

                if epoch % self.config.any_time == 0.0:
                    self._save_snapshot(epoch, os.path.join(
                        f"{self.config.run_name}", f"save_{epoch}.pt"), avg_loss)

                log = {"avg_loss": avg_loss,
                       "lr": self.scheduler.get_last_lr()[0] if self.config.scheduler else self.config.lr}
                self._log(epoch, log)
                self._save_snapshot(epoch, os.path.join(
                    f"{self.config.run_name}", "last.pt"), avg_loss)

        if is_main_gpu():
            wandb.finish()
            logger.info(
                f"Training finished, best loss : {self.best_loss:.6f}, lr : f{self.scheduler.get_last_lr()[0]}, "
                f"saved at {os.path.join(f'{self.config.run_name}', 'best.pt')}")

    def sample_train(self, ep=None, nb_img=4, condition=None):
        """
        Generate and save sample images during training.
        Args:
            ep (str): Epoch identifier for filename.
            nb_img (int): Number of images to generate.
            condition (torch.Tensor): (optional) Condition to use for sampling.
        Returns:
            None
        """
        if self.gpu_id != 0:
            return
        if nb_img > 6:
            Warning(
                "Sampling more than 6 images may long to compute because sampling use only main GPU.")

        logger.info(f"Sampling {nb_img} images...")
        samples = super()._sample_batch(nb_img=nb_img, condition=condition)
        for i, img in enumerate(samples):
            filename = f"_sample_{ep}_{i}.npy" if ep is not None else f"_sample_{i}.npy"
            save_path = os.path.join(self.config.run_name, "samples", filename)
            np.save(save_path, img)
        if self.config.plot:
            self.plot_grid(f"samples_grid_{ep}.jpg", samples)
        logger.info(
            f"Sampling done. Images saved in {self.config.run_name}/samples/")

    def _log(self, epoch, log_dict):
        """
        Log training metrics.
        Args:
            epoch (int): Current epoch number.
            log_dict (dict): Dictionary containing log data.
        Returns:
            None
        """
        if self.config.use_wandb:
            wandb.log(log_dict, step=epoch)
        csv_filename = os.path.join(
            f"{self.config.run_name}", "logs_train.csv")
        file_exists = Path(csv_filename).is_file()
        with open(csv_filename, 'a' if file_exists else 'w', newline='') as csvfile:
            fieldnames = ['epoch'] + list(log_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({**{'epoch': epoch}, **log_dict})
