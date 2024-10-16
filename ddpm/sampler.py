import os

import numpy as np
import torch
from tqdm import tqdm
import logging
from ddpm.ddpm_base import Ddpm_base
from utils.distributed import is_main_gpu
from utils.guided_loss import loss_dict


class Sampler(Ddpm_base):
    def __init__(
        self,
        model: torch.nn.Module,
        config,
        dataloader=None,
        inversion_transforms=None,
    ) -> None:
        """
        Initialize the Sampler class.
        Args:
            model (torch.nn.Module): The neural network model for sampling.
            config: Configuration settings for sampling.
            dataloader: The data loader for input data (optional).
        """
        super().__init__(model, config, dataloader, inversion_transforms)
        self.loss_func = loss_dict["L1Loss"]

    @torch.no_grad()
    def _guided_sample_batch(
        self, truth_sample_batch, guidance_loss_scale=100, random_noise=False
    ):
        """
        Perform guided sampling of a batch of images.
        Args:
            truth_sample_batch (torch.Tensor): Ground truth image batch for guidance.
            guidance_loss_scale (float): Scaling factor for the guidance loss between [0 - 100].
            random_noise (bool): Whether to use random noise as the initial sample.
        Returns:
            numpy.ndarray: Array of sampled images.
        """
        assert (
            0 <= guidance_loss_scale <= 100
        ), "Guidance loss scale must be between 0 and 100."
        noise = torch.randn_like(truth_sample_batch).to(self.gpu_id)
        t_l = torch.ones((truth_sample_batch.shape[0])).to(
            self.gpu_id
        ).long() * (self.timesteps - 1)

        if not random_noise:
            sample = self.model.q_sample(
                x_start=truth_sample_batch, t=t_l, noise=noise
            )
        else:
            sample = noise

        for t in reversed(range(0, self.timesteps)):
            sample, _ = self.model.p_sample(sample, t, None)
            sample = sample.detach().requires_grad_()
            loss = (
                self.loss_func(sample, truth_sample_batch)
                * guidance_loss_scale
            )
            # Compute the gradient of the loss and update the sample
            cond_grad = -torch.autograd.grad(loss, sample)[0]
            sample = sample.detach() + cond_grad
        sampled_images_unnorm = self.transforms_func(sample).cpu().numpy()
        return sampled_images_unnorm

    @torch.no_grad()
    def sample(self, filename_format="_sample_{i}.npy"):
        """
        Generate and save sample images during training.
        Args:
            filename_format (str): Format of the filename to save the images.
        Returns:
            None
        """

        i = self.gpu_id if type(self.gpu_id) is int else 0

        if self.config.sampling_mode == "simple":

            if is_main_gpu():
                self.logger.info(
                    f"Sampling {self.config.n_sample * (torch.cuda.device_count() if torch.cuda.is_available() else 1)} images...")
            with tqdm(total=self.config.n_sample // self.config.batch_size, desc="Sampling ", unit="batch",
                      disable= not is_main_gpu()) as pbar:
                b = 0
                while b < self.config.n_sample:
                    batch_size = min(self.config.n_sample - b, self.config.batch_size)
                    samples = super()._sample_batch(nb_img=batch_size)
                    for s in samples:
                        filename = filename_format.format(i=str(i))
                        save_path = os.path.join(self.config.output_dir ,self.config.run_name, "samples", filename)
                        np.save(save_path, s)
                        i += max(torch.cuda.device_count(), 1)
                    b += batch_size
                    pbar.update(1)
        elif "conditioned" in self.config.sampling_mode:
            if is_main_gpu():
                self.logger.info(
                    f"Sampling {len(self.dataloader) * self.config.batch_size * (torch.cuda.device_count() if torch.cuda.is_available() else 1)} images...")
            for batch_idx, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Sampling ", unit="batch"):
                cond = batch['img'].to(self.gpu_id)
                ids = batch['img_id']
                samples = self._sample_batch(nb_img=len(cond), condition=cond)
                # if self.config.sampling_mode == "conditioned":
                #     samples = self._sample_batch(nb_img=len(cond), condition=cond)
                # elif self.config.sampling_mode == "guided":
                #     samples = self._guided_sample_batch(cond, random_noise=self.config.random_noise)
                for s, img_id in zip(samples, ids):
                    filename = filename_format.format(i=img_id)
                    save_path = os.path.join(self.config.output_dir, self.config.run_name, "samples", filename)
                    np.save(save_path, s)

        else:
            raise ValueError(f"Sampling mode {self.config.sampling_mode} not supported.")

        if self.config.plot and is_main_gpu():
            self.plot_grid("last_samples.jpg", samples)

        self.logger.info(
            f"Sampling done. Images saved in {self.config.output_dir}/{self.config.run_name}/samples/")
