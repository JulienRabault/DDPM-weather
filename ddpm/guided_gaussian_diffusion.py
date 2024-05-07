import torch
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    default,
    extract,
)
from einops import reduce, rearrange
from torch.nn.functional import mse_loss
from tqdm import tqdm


class GuidedGaussianDiffusion(GaussianDiffusion):
    def __init__(self, *args, with_grad=False, **kwargs):
        """
        Initialize the GuidedGaussianDiffusion.
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.with_grad = with_grad

    @torch.no_grad()
    def sample(self, batch_size, return_all_timesteps=False, condition=None):
        """
        Generate samples using guided diffusion.
        Args:
            batch_size (int): Number of samples to generate.
            return_all_timesteps (bool): Whether to return samples at all timesteps.
            condition: Additional conditioning information.
        Returns:
            torch.Tensor: Generated samples.
        """
        image_size, channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop
            if not self.is_ddim_sampling
            else self.ddim_sample
        )
        return sample_fn(
            (batch_size, channels, image_size, image_size),
            return_all_timesteps=return_all_timesteps,
            condition=condition,
        )

    # @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False, condition=None):
        """
        Sample from guided diffusion using a loop over timesteps.
        Args:
            shape: Shape of the samples to generate.
            return_all_timesteps (bool): Whether to return samples at all timesteps.
            condition: Additional conditioning information.
        Returns:
            torch.Tensor: Generated samples.
        """
        batch, device = shape[0], self.device
        img = torch.randn(shape, device=device)
        imgs = [img]
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
            leave=False,
        ):
            if self.with_grad:
                img, x_start = self.p_sample(img, t, condition)
            else:
                with torch.inference_mode():
                    img, x_start = self.p_sample(img, t, condition)

            imgs.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        return ret

    def p_sample(self, x, t: int, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        if self.with_grad:
            preds = self.model_predictions(x, t, x_self_cond)
        else:
            with torch.inference_mode():
                preds = self.model_predictions(x, t, x_self_cond)

        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = (
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps=False, condition=None):
        """
        Sample from guided diffusion using ddim sampling.
        Args:
            shape: Shape of the samples to generate.
            return_all_timesteps (bool): Whether to return samples at all timesteps.
            condition: Additional conditioning information.
        Returns:
            torch.Tensor: Generated samples.
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )
        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)
        imgs = [img]
        for time, time_next in tqdm(
            time_pairs,
            desc="sampling loop time step",
            leave=False,
        ):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long
            )

            if self.with_grad:
                pred_noise, x_start, *_ = self.model_predictions(
                    img,
                    time_cond,
                    condition,
                    clip_x_start=True,
                    rederive_pred_noise=True,
                )
            else:
                with torch.inference_mode():
                    pred_noise, x_start, *_ = self.model_predictions(
                        img,
                        time_cond,
                        condition,
                        clip_x_start=True,
                        rederive_pred_noise=True,
                    )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta
                * (
                    (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
                ).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        return ret

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        offset_noise_strength=None,
        condition=None,
    ):
        """
        Calculate pixel-wise loss for guided diffusion.
        Args:
            x_start: Starting image tensor.
            t (int): Timestep.
            noise: Noise tensor.
            offset_noise_strength: Strength of offset noise.
            condition: Additional conditioning information.
        Returns:
            torch.Tensor: Pixel-wise loss.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        offset_noise_strength = default(
            offset_noise_strength, self.offset_noise_strength
        )

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(
                offset_noise, "b c -> b c 1 1"
            )

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_self_cond = condition

        model_out = self.model(x, t, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """
        Forward pass for guided diffusion.
        Args:
            img: Input image tensor.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            torch.Tensor: Forward pass result.
        """
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *img.shape,
            img.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
