import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from ddpm.ddpm_base import Ddpm_base
from utils.distributed import is_main_gpu
from utils.guided_loss import loss_dict

logger = logging.getLogger()


class Sampler(Ddpm_base):
    def __init__(self, model: torch.nn.Module, config, dataloader=None) -> None:
        """
        Initialize the Sampler class.
        Args:
            model: The neural network model for sampling.
            config: Configuration settings for sampling.
            dataloader: The data loader for input data (optional).
        """
        super().__init__(model, config, dataloader)
        self.loss_func = loss_dict["L1Loss"]


    def _simple_guided_sample_batch(self, truth_sample_batch, guidance_loss_scale=100, random_noise=False):
        """
        Perform guided sampling of a batch of images.
        Args:
            truth_sample_batch: Ground truth image batch for guidance.
            guidance_loss_scale: Scaling factor for the guidance loss between [0 - 100].
            random_noise: Whether to use random noise as the initial sample.
        Returns:
            numpy.ndarray: Array of sampled images.
        """
        assert 0 <= guidance_loss_scale <= 100, "Guidance loss scale must be between 0 and 100."
        noise = torch.randn_like(truth_sample_batch).to(self.gpu_id)
        t_l = torch.ones((truth_sample_batch.shape[0])).to(self.gpu_id).long() * (self.timesteps - 1)

        if not random_noise:
            sample = self.model.q_sample(x_start=truth_sample_batch, t=t_l, noise=noise)
        else:
            sample = noise

        for t in reversed(range(0, self.timesteps)):
            sample, _ = self.model.p_sample(sample, t, None)
            sample = sample.detach().requires_grad_()
            loss = self.loss_func(sample, truth_sample_batch) * guidance_loss_scale
            # Compute the gradient of the loss and update the sample
            cond_grad = -torch.autograd.grad(loss, sample)[0]
            sample = sample.detach() + cond_grad
        sampled_images_unnorm = self.transforms_func(sample).cpu().numpy()
        return sampled_images_unnorm

    def sample(self, filename_format="_sample_{i}.npy", nb_img=4, plot=False, sampling_mode="simple",
               random_noise=False):
        """
        Generate and save sample images during training.
        Args:
            filename_format (str): Format of the filename to save the images.
            nb_img (int): Number of images to generate.
            plot (bool): Whether to plot and save sampled images during the process.
            sampling_mode (str): Sampling mode, choose between "simple", "guided", or "simple_guided".
            random_noise (bool): Whether to use random noise as the initial sample.
        Returns:
            None
        """
        if is_main_gpu():
            logger.info(
                f"Sampling {nb_img * (torch.cuda.device_count() if torch.cuda.is_available() else 1)} images...")

        i = self.gpu_id if type(self.gpu_id) is int else 0
        if sampling_mode != "simple":
            dataloader_iter = iter(self.dataloader)
        b = 0
        with tqdm(total=nb_img // self.config.batch_size, desc="Sampling ", unit="batch",
                  disable=is_main_gpu()) as pbar:
            while b < nb_img:
                batch_size = min(nb_img - b, self.config.batch_size)
                if sampling_mode == "simple":
                    samples = super()._sample_batch(nb_img=batch_size)
                elif sampling_mode == "guided":
                    samples = self._sample_batch(nb_img=batch_size, condition=next(dataloader_iter)['img'])
                elif sampling_mode == "simple_guided":
                    samples = self._simple_guided_sample_batch(next(dataloader_iter)['img'], random_noise=random_noise)
                else:
                    raise ValueError(f"Sampling mode {sampling_mode} not supported.")
                for s in samples:
                    filename = filename_format.format(i=str(i))
                    save_path = os.path.join(self.config.run_name, "samples", filename)
                    np.save(save_path, s)
                    i += max(torch.cuda.device_count(), 1)
                b += batch_size
                pbar.update(1)

        if plot:
            self.plot_grid("last_samples.jpg", samples)

        logger.info(
            f"Sampling done. Images saved in {self.config.run_name}/samples/")
