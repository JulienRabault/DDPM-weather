import csv
import os.path
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torchvision import transforms
from tqdm import tqdm

from distributed import get_rank, is_main_gpu


def save_gray_image(grid, outfile, colormap):
    plt.imshow(grid, cmap=colormap)
    plt.colorbar()
    plt.savefig(outfile)
    plt.close()


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            config,
            dataloader=None,
            optimizer=None,
    ) -> None:
        """
        Initialize the Trainer.
        Args:
            model (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): The optimizer for model parameters.
            config: Configuration settings.
        """
        self.config = config
        self.gpu_id = get_rank()
        self.model = model.to(self.gpu_id)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.epochs_run = 0

        self.snapshot_path = self.config.model_path
        if self.dataloader is not None:
            self.train_dataset = self.dataloader.dataset
            self.stds = self.train_dataset.stds
            self.means = self.train_dataset.means
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

        if self.snapshot_path is not None:
            if is_main_gpu():
                print(f"#INFO : Loading snapshot")
            self._load_snapshot(self.snapshot_path)

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
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        if "SCHEDULER_STATE" in snapshot and self.scheduler is not None:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        if "WANDB_ID" in snapshot:
            self.wandb_id = snapshot["WANDB_ID"]
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.best_loss = snapshot["BEST_LOSS"]
        self.stds = snapshot["STDS"]
        self.means = snapshot["MEANS"]
        if self.config.debug_log:
            print(
                f"\n#LOG : [GPU{self.gpu_id}] Resuming training from {snapshot_path} at Epoch {self.epochs_run}")
        elif is_main_gpu():
            print(
                f"#INFO : Resuming training from {snapshot_path} at Epoch {self.epochs_run}")
        self.epochs_run += 1

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
            'STDS': self.stds,
            'MEANS': self.means,
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
            wandb.init(project=self.config.wp, resume="auto", mode=os.environ['WANDB_MODE'], entity=self.config.entityWDB,
                       name=f"{self.config.train_name}_{t}/",
                       config={**vars(self.config), **{"optimizer": self.optimizer.__class__,
                                                       "scheduler": self.scheduler.__class__,
                                                       "lr_base": self.optimizer.param_groups[0]["lr"],
                                                       "weight_decay": self.optimizer.param_groups[0]["weight_decay"], }})
        else:
            wandb.init(project=self.config.wp, entity=self.config.entityWDB, mode=os.environ['WANDB_MODE'],
                       name=f"{self.config.train_name}_{t}/",
                       config={**vars(self.config), **{"optimizer": self.optimizer.__class__,
                                                       "scheduler": self.scheduler.__class__,
                                                       "lr_base": self.optimizer.param_groups[0]["lr"],
                                                       "weight_decay": self.optimizer.param_groups[0]["weight_decay"], }})

    def train(self):
        """
        Start the training process.
        Returns:
            None
        """
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
                    self.sample_images(
                        ep=str(epoch), nb_image=self.config.n_sample)
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
            config: Configuration settings.
            ep (str): Epoch identifier for filename.
            nb_image (int): Number of images to generate.
        Returns:
            None
        """
        # transformation to get original data values
        invTrans = transforms.Compose([
            transforms.Normalize(mean=[0.] * len(self.config.var_indexes),
                                 std=[1 / el for el in self.stds]),
            transforms.Normalize(mean=[-el for el in self.means],
                                 std=[1.] * len(self.config.var_indexes)),
        ])

        # =========================
        # plot batch dataset images
        # =========================

        batchs = []

        for i in tqdm(range(nb_batch)):

            batch = next(iter(self.dataloader))
            print("batch", batch.shape)
            batchs.append(batch)
            image_batch = invTrans(batch)
            for img1 in image_batch:
                self.plot_img(img1, i, "batch")

        # ==============================================
        # plot recons images from previous dataset batch
        # ==============================================

        for i, batch in tqdm(enumerate(batchs)):

            # get a t such as  0 < t   and t < num_timesteps=1000
            t = torch.ones((batch.shape[0],),
                           device=self.gpu_id).long() * t_noise

            # normalize batch
            x_start = self.model.normalize(batch).to(self.gpu_id)

            # get a random noise
            noise = torch.randn_like(x_start).to(self.gpu_id)

            # Forward Diffusion : add noise to x_start
            img = self.model.q_sample(x_start=x_start, t=t, noise=noise)

            # Reverse Diffusion : add noise to x_start
            for t in tqdm(reversed(range(0, t.item())), desc='sampling loop time step', total=t.item()):
                img, _ = self.model.p_sample(img, t, None)

            # unnormalize img
            sampled_images = self.model.unnormalize(img)

            # back to original values
            sampled_images_unnorm = invTrans(sampled_images)
            np_img = sampled_images_unnorm.cpu()

            # plot reconsruction images
            for img1 in np_img:
                self.plot_img(img1, i, "recon")

    def sample_images(self, ep=None, nb_image=4):
        """
        Generate and save sample images during training.
        Args:
            ep (str): Epoch identifier for filename.
            nb_image (int): Number of images to generate.
        Returns:
            None
        """
        if is_main_gpu():
            print(f"Sampling {nb_image * torch.cuda.device_count()} images...")

        if self.config.invert_norm:
            transforms_func = transforms.Compose([
                transforms.Normalize(mean=[0.] * len(self.config.var_indexes),
                                     std=[1 / el for el in self.stds]),
                transforms.Normalize(mean=[-el for el in self.means],
                                     std=[1.] * len(self.config.var_indexes)),
            ])
        else:
            # identity
            def transforms_func(x): return x
        b = 0
        i = self.gpu_id

        with tqdm(total=nb_image // self.config.batch_size, desc="Sampling ", unit="batch") as pbar:
            while b < nb_image:
                batch_size = min(nb_image - b, self.config.batch_size)

                sampled_images = self.model.module.sample(
                    batch_size=batch_size) if dist.is_initialized() else self.model.sample(batch_size=batch_size)
                print("sampled_images", sampled_images.shape)

                b += batch_size
                sampled_images = transforms_func(sampled_images)
                np_img = sampled_images.cpu().numpy()
                for img1 in np_img:
                    if ep is not None:
                        np.save(os.path.join(
                            f"{self.config.train_name}", "samples", f"_sample_{ep}_{i}.npy"), img1.numpy())
                    else:

                        np.save(os.path.join(
                            f"{self.config.train_name}", "samples", f"_sample_{i}.npy"), img1.numpy())
                        if self.config.plot_image:

                            self.plot_img(img1, i, "sample")
                            # img1 = torch.transpose(img1, 0, 2)
                            # img1 = torch.fliplr(img1)
                            # img1 = torch.transpose(img1, 0, 2)

                            # save_gray_image(img1[0, :, :].numpy(), os.path.join(
                            #     f"{self.config.train_name}", "samples", f"_sample_{i}_u.png"), 'viridis')
                            # save_gray_image(img1[1, :, :].numpy(), os.path.join(
                            #     f"{self.config.train_name}", "samples", f"_sample_{i}_v.png"), 'viridis')
                            # save_gray_image(img1[2, :, :].numpy(), os.path.join(
                            #     f"{self.config.train_name}", "samples", f"_sample_{i}_t.png"), 'RdBu_r')

                    if torch.cuda.device_count() > 1 and self.config.mode == "Test":
                        i += torch.cuda.device_count()
                    else:
                        i += 1
                pbar.update(1)

        # Plotting images for evolution
        if is_main_gpu() and self.config.mode == 'Train':
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

        print(
            f"\nSampling done. Images saved in {self.config.train_name}/samples/")

    def plot_img(self, img1, i, name):
        img1 = torch.transpose(img1, 0, 2)
        img1 = torch.fliplr(img1)
        img1 = torch.transpose(img1, 0, 2)
        save_gray_image(img1[0, :, :].numpy(), os.path.join(
            f"{self.config.train_name}", "samples", f"_{name}_{i}_u.png"), 'viridis')
        save_gray_image(img1[1, :, :].numpy(), os.path.join(
            f"{self.config.train_name}", "samples", f"_{name}_{i}_v.png"), 'viridis')
        save_gray_image(img1[2, :, :].numpy(), os.path.join(
            f"{self.config.train_name}", "samples", f"_{name}_{i}_t.png"), 'RdBu_r')

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
