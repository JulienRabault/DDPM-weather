import csv
import os.path
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch import distributed as dist, nn as nn
from torchvision import transforms
from tqdm import tqdm

from Ddpm_base import Ddpm_base
from distributed import is_main_gpu


def image_basic_loss(images, target_image):
    """
    Given a target image, return a loss for how far away on average
    the images' pixels are from that image.
    """
    error = torch.abs(images - target_image).mean()
    return error


def save_gray_image(grid, outfile, colormap):
    plt.imshow(grid, cmap=colormap)
    plt.colorbar()
    plt.savefig(outfile)
    plt.close()


class Trainer(Ddpm_base):

    def __init__(self, model, config, dataloader=None,
                 optimizer=None):
        super().__init__(model, config, dataloader)
        self.optimizer = optimizer
        self.epochs_run = 0
        self.best_loss = float('inf')

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

    def _run_batch(self, batch):
        """
        Run a single training batch.
        Args:
            batch: Input batch for training.
        Returns:
            float: Loss value for the batch.
        """
        self.optimizer.zero_grad()
        print("batch", batch.shape)

        loss = self.model(batch)
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
        b_sz = len(next(iter(self.dataloader)))
        iters = len(self.dataloader)
        if dist.is_initialized():
            self.dataloader.sampler.set_epoch(epoch)
        total_loss = 0
        if is_main_gpu():
            loop = tqdm(enumerate(self.dataloader), total=iters,
                        desc=f"Epoch {epoch}/{self.config.epochs + self.epochs_run}", unit="batch",
                        leave=False, postfix="")
        else:
            loop = enumerate(self.dataloader)
        for i, batch in loop:
            batch = batch.to(self.gpu_id)
            loss = self._run_batch(batch)
            total_loss += loss
            if self.config.scheduler:
                self.scheduler.step()
            if is_main_gpu():
                loop.set_postfix_str(f"Loss : {total_loss / (i + 1):.6f}")
        if self.config.debug_log:
            print(
                f"\n#LOG : [GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.dataloader)} | Last loss: {total_loss / len(self.dataloader)} | Lr : {self.scheduler.get_last_lr()[0] if self.config.scheduler else self.config.lr}")

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
            "MODEL_STATE": self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            'OPTIMIZER_STATE': self.optimizer.state_dict(),
            'BEST_LOSS': loss,
            'TIMESTAMP': self.timesteps,
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
        print(
            f"#INFO : Epoch {epoch} | Training snapshot saved at {path} | Loss: {loss}")

    def _init_wandb(self):
        """
        Initialize WandB for logging training progress.
        Returns:
            None
        """
        if self.gpu_id != 0:
            return

        t = time.strftime("%d-%m-%y_%H-%M", time.localtime(time.time()))
        if self.config.resume:
            wandb.init(project=self.config.wp, resume="auto", mode=os.environ['WANDB_MODE'],
                       entity=self.config.entityWDB,
                       name=f"{self.config.train_name}_{t}/",
                       config={**vars(self.config), **{"optimizer": self.optimizer.__class__,
                                                       "scheduler": self.scheduler.__class__,
                                                       "lr_base": self.optimizer.param_groups[0]["lr"],
                                                       "weight_decay": self.optimizer.param_groups[0][
                                                           "weight_decay"], }})
        else:
            wandb.init(project=self.config.wp, entity=self.config.entityWDB, mode=os.environ['WANDB_MODE'],
                       name=f"{self.config.train_name}_{t}/",
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
                        f"{self.config.train_name}", "best.pt"), avg_loss)

                if epoch % self.config.any_time == 0.0:
                    samples = self._sample_batch(self.config.n_sample)
                    for i, s in enumerate(samples):
                        filename = filename_format.format(epoch = epoch, i = i)
                        save_path = os.path.join(self.config.train_name, "samples", filename)
                        np.save(save_path, s)
                    self._save_snapshot(epoch, os.path.join(
                        f"{self.config.train_name}", f"save_{epoch}.pt"), avg_loss)

                log = {"avg_loss": avg_loss,
                       "lr": self.scheduler.get_last_lr()[0] if self.config.scheduler else self.config.lr}
                self._log(epoch, log)
                self._save_snapshot(epoch, os.path.join(
                    f"{self.config.train_name}", "last.pt"), avg_loss)

        if is_main_gpu():
            wandb.finish()
            print(
                f"#INFO : Training finished, best loss : {self.best_loss:.6f}, lr : f{self.scheduler.get_last_lr()[0]}, saved at {os.path.join(f'{self.config.train_name}', 'best.pt')}")

    def reconstruct_images(self, nb_batch=1, t_noise=999):
        """
        Generate and save sample images of the dataset.
        Args:
            nb_batch (int): Number of batch to reconstruct.
            t_noise (int): Noise t.
        Returns:
            None
        """
        # torch.manual_seed(0)
        if self.config.invert_norm:
            transforms_func = transforms.Compose([
                transforms.Normalize(mean=[0.] * len(self.config.var_indexes),
                                     std=[1 / el for el in self.stds]),
                transforms.Normalize(mean=[-el for el in self.means],
                                     std=[1.] * len(self.config.var_indexes)),
            ])
        else:
            def transforms_func(x):
                return x  # identity

        batchs = []

        for i in tqdm(range(nb_batch)):

            batch = next(iter(self.dataloader))
            print("batch", batch.shape)
            batchs.append(batch)
            image_batch = transforms_func(batch)
            for img1 in image_batch:
                self.plot_img(img1, i, "batch")

        # ==============================================
        # plot recons images from previous dataset batch
        # ==============================================

        for i, batch in tqdm(enumerate(batchs)):

            # get a t such as  0 < t   and t < num_timesteps=1000
            t = torch.ones((batch.shape[0],),
                           device=self.gpu_id).long() * t_noise

            guidance_loss_scale = 20

            # normalize batch
            x_start = self.model.normalize(batch).to(self.gpu_id)

            # get a random noise
            noise = torch.randn_like(x_start).to(self.gpu_id)

            # Forward Diffusion : add noise to x_start
            # img = self.model.q_sample(x_start=x_start, t=t, noise=noise)
            img = noise

            for t in tqdm(reversed(range(0, t.item())), desc='sampling loop time step', total=t.item()):
                img, _ = self.model.p_sample(img, t, None)
                img = img.detach().requires_grad_()

                loss = image_basic_loss(img, batch) * guidance_loss_scale
                if t % 10 == 0:
                    print(i, "loss:", loss.item())
                cond_grad = -torch.autograd.grad(loss, img)[0]

                img = img.detach() + cond_grad

            # unnormalize img
            sampled_images = self.model.unnormalize(img)

            # back to original values
            sampled_images_unnorm = transforms_func(sampled_images)
            np_img = sampled_images_unnorm.cpu()

            # plot reconsruction images
            for img1 in np_img:
                self.plot_img(img1, i, "recon_20_")
                # self.plot_img(img1, i, "reconwithoutguid")

    def sample_train(self, ep=None, nb_img=4):
        """
        Generate and save sample images during training.
        Args:
            ep (str): Epoch identifier for filename.
            nb_image (int): Number of images to generate.
        Returns:
            None
        """
        if nb_img > 6:
            Warning(
                "Sampling more than 6 images may long to compute because sampling use only main GPU.")

        print(f"Sampling {nb_img} images...")
        samples = super()._sample_batch(nb_img=nb_img)
        for i, img in enumerate(samples):
            filename = f"_sample_{ep}_{i}.npy" if ep is not None else f"_sample_{i}.npy"
            save_path = os.path.join(self.config.train_name, "samples", filename)
            np.save(save_path, img)
        self.plot_grid(ep, samples)
        print(
            f"\nSampling done. Images saved in {self.config.train_name}/samples/")

    def plot_grid(self, ep, np_img):
        nb_image = len(np_img)
        fig, axes = plt.subplots(nrows=min(6, nb_image), ncols=len(
            self.config.var_indexes), figsize=(10, 10))
        for i in range(min(6, nb_image)):
            for j in range(len(self.config.var_indexes)):
                cmap = 'viridis' if self.config.var_indexes[j] != 't2m' else 'bwr'
                image = np_img[i, j]
                if len(self.config.var_indexes) > 1:
                    im = axes[i, j].imshow(
                        image, cmap=cmap, origin='lower')
                    axes[i, j].axis('off')
                    fig.colorbar(im, ax=axes[i, j])
                else:
                    im = axes[i].imshow(image, cmap=cmap, origin='lower')
                    axes[i].axis('off')
                    fig.colorbar(im, ax=axes[i])
        plt.savefig(os.path.join(f"{self.config.train_name}", "samples", f"all_images_grid_{ep}.jpg"),
                    bbox_inches='tight')
        plt.close()

    def _log(self, epoch, log_dict):
        """
        Log training metrics.
        Args:
            epoch (int): Current epoch number.
            log_dict (dict): Dictionary containing log data.
        Returns:
            None
        """
        wandb.log(log_dict, step=epoch)
        csv_filename = os.path.join(
            f"{self.config.train_name}", "logs_train.csv")
        file_exists = Path(csv_filename).is_file()
        with open(csv_filename, 'a' if file_exists else 'w', newline='') as csvfile:
            fieldnames = ['epoch'] + list(log_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({**{'epoch': epoch}, **log_dict})
