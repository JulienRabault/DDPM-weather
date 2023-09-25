import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from Ddpm_base import Ddpm_base
from distributed import get_rank, is_main_gpu


def image_basic_loss(images, target_image):
    """
    Given a target image, return a loss for how far away on average
    the images' pixels are from that image.
    """
    error = torch.abs(images - target_image).mean()
    return error


class Sampler(Ddpm_base):

    def __init__(self, model: torch.nn.Module, config, dataloader=None) -> None:
        super().__init__(model, config, dataloader)
        self.loss_func = image_basic_loss

    def _guided_sample_batch(self, truth_sample_batch,  timesteps=1000, guidance_loss_scale = 20):

        # get a random noise
        sample = torch.randn_like(truth_sample_batch).to(self.gpu_id)

        for i, t in enumerate(reversed(range(0, timesteps))):
            sample, _ = self.model.p_sample(sample, t, None)

            sample = sample.detach().requires_grad_()

            loss = image_basic_loss(sample, truth_sample_batch) * guidance_loss_scale
            # if i % 10 == 0:
            #     print(i, "loss:", loss.item())
            cond_grad = -torch.autograd.grad(loss, sample)[0]

            sample = sample.detach() + cond_grad

        # unnormalize img
        sampled_images = self.model.unnormalize(sample)

        # back to original values
        sampled_images_unnorm = self.transforms_func(sampled_images).cpu().numpy()
        return sampled_images_unnorm

    def sample(self, filename_format="_sample_{i}.npy", nb_img=4):
        """
        Generate and save sample images during training.
        Args:
            filename_format (str): Format of the filename to save the images.
            nb_img (int): Number of images to generate.
        Returns:
            None
        """
        if is_main_gpu():
            print(f"Sampling {nb_img * torch.cuda.device_count()} images...")

        i = self.gpu_id if type(self.gpu_id) is int else 0
        b = 0
        with tqdm(total=nb_img // self.config.batch_size, desc="Sampling ", unit="batch",
                  disable=is_main_gpu()) as pbar:
            while b < nb_img:
                batch_size = min(nb_img - b, self.config.batch_size)
                print("batch_size :",batch_size)
                samples = super()._sample_batch(nb_img=batch_size)
                for s in samples:
                    filename = filename_format.format(i=str(i))
                    save_path = os.path.join(self.config.train_name, "samples", filename)
                    np.save(save_path, s)
                    i += max(torch.cuda.device_count(), 1)
                b += batch_size
                pbar.update(1)
        self.plot_grid(0, samples)
        print(
            f"\nSampling done. Images saved in {self.config.train_name}/samples/")

    def guided_sample(self,dataloader , filename_format="F_samble_Diff_{i}.npy"):
        iters = len(dataloader)
        if is_main_gpu():
            print(f"Sampling {iters * torch.cuda.device_count()} images...")
            loop = tqdm(enumerate(dataloader), total=iters,
                        desc=f"Sampling", unit="batch",
                        leave=False, postfix="")
        else:
            loop = enumerate(self.dataloader)
        i = self.gpu_id if type(self.gpu_id) == int else 0

        for _, batch in loop:
            print("ok")
            self.plot_grid(0, batch.numpy())
            batch = batch.to(self.gpu_id)
            samples = self._guided_sample_batch(batch)
            for s in samples:
                filename = filename_format.format(i=str(i))
                save_path = os.path.join(self.config.train_name, "samples", filename)
                np.save(save_path, s)
                i += max(torch.cuda.device_count(), 1)
        self.plot_grid(0, samples)
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
                if len(self.config.var_indexes) > 1 and min(6, nb_image) > 1:
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



