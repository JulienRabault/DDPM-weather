from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce

# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# tensor helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# main class


class ElucidatedDiffusion(nn.Module):
    def __init__(
        self,
        net,
        *,
        image_size,
        channels=3,
        num_sample_steps=100,  # number of sampling steps
        sigma_min=0.01,  # min noise level
        sigma_max=0.141,  # max noise level
        sigma_data=0.5,  # standard deviation of data distribution
        rho=7,  # controls the sampling schedule
        P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin=0.05,
        S_tmax=50,
        S_noise=0.0,
    ):
        super().__init__()

        # assert net.random_or_learned_sinusoidal_cond
        self.self_condition = net.self_condition
        print(f"self condition {self.self_condition}")
        self.net = net

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = (
            num_sample_steps  # otherwise known as N in the paper
        )

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma, option="edm"):
        if option=="edm":
            return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)
        elif option=="ddim":
            return 1.0

    def c_out(self, sigma, option="edm"):

        if option=="edm":
            return (
                sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5
            )
        elif option=="ddim":
            return -1 * sigma

    def c_in(self, sigma, option="edm"):
        if option=="edm":
            return 1 * (sigma**2 + self.sigma_data**2) ** -0.5
        elif option=="ddim":
            return 1 * (sigma**2 + 1.0) ** -0.5

    def c_noise(self, sigma, option="edm",max_steps=None):
        if option=="edm":
            return log(sigma) * 0.25
        elif option=="ddim":
            # here the aim is to retrieve the "t" index corresponding to sigma in the training
            # if we assume linear schedule for beta_t and sigma_t**2 = beta_t
            return float(max_steps - 1) * (sigma ** 2 - self.sigma_min ** 2) / (self.sigma_max ** 2 - self.sigma_min ** 2)



    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(
        self, noised_images, sigma, self_cond=None, clamp=False
    ):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full(
                (batch,), sigma, device=device, dtype=torch.float
            )

        padded_sigma = rearrange(sigma, "b -> b 1 1 1")
        c_in = self.c_in(padded_sigma,option="ddim").squeeze()[0]
        c_noise = self.c_noise(sigma,option="ddim",max_steps=1000).squeeze()[0]
        c_skip = self.c_skip(padded_sigma,option="ddim")
        c_out = self.c_out(padded_sigma,option="ddim").squeeze()[0]
        print("c_noise", c_noise)
        v = self.net.model(
            self.c_in(padded_sigma,option="ddim") * noised_images,
            self.c_noise(sigma,option="ddim",max_steps=1000).int(),
            self_cond,
        )
        print("compute x_start")
        print(self.net.sqrt_alphas_cumprod.shape)

        x_start = self.net.predict_start_from_v(noised_images,self.c_noise(sigma,option="ddim",max_steps=1000).long(),v)
        
        #print(x_start[0,0,0,0])
        #print("compute noise")
        #noise = self.net.predict_noise_from_start(noised_images,self.c_noise(sigma,option="ddim",max_steps=1000).long(),x_start)
        #print(noise[0,0,0,0])
        #print("compute out")
        #print(sigma)
        
        return x_start

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sample_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (N - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(
            sigmas, (0, 1), value=0.0
        )  # last step is sigma value of 0.

        return sigmas

    @torch.no_grad()
    def sample(self, batch_size=16, num_sample_steps=None, cond=None, clamp=False):

        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]
        # images = init_sigma * torch.randn(shape, device=self.device)
        images = init_sigma * torch.randn(shape, device=self.device)

        print(
            f"init mean : {torch.mean(images).item()}, max : {torch.max(images).item()}"
        )

        # for self conditioning

        x_start = None

        # gradually denoise

        for sigma, sigma_next, gamma in tqdm(
            sigmas_and_gammas, desc="sampling time step"
        ):
            sigma, sigma_next, gamma = map(
                lambda t: t.item(), (sigma, sigma_next, gamma)
            )
            eps = self.S_noise * torch.randn(
                shape, device=self.device
            )  # stochastic sampling

            sigma_hat = sigma #+ gamma * sigma
            images_hat = images # + sqrt(sigma_hat**2 - sigma**2) * eps
            self_cond = cond if self.self_condition else None

            model_output = self.preconditioned_network_forward(
                images_hat, sigma_hat, self_cond, clamp=clamp
            )

            print(
                f"mean : {torch.mean(model_output).item()}, max : {torch.max(model_output).item()}"
            )

            denoised_over_sigma = (images_hat - model_output) / sigma_hat
            print(
                f" denoising magnitude mean : {torch.mean(denoised_over_sigma).item()}, max : {torch.max(denoised_over_sigma).item()}, std : {torch.std(denoised_over_sigma).item()}"
            )
            images_next = (
                images_hat + (sigma_next - sigma_hat) * denoised_over_sigma
            )
            print("img diff std",(images_next - images_hat).std())

            # second order correction, if not the last timestep

            if sigma_next != 0:
                self_cond = cond if self.self_condition else None

                model_output_next = self.preconditioned_network_forward(
                    images_next, sigma_next, self_cond, clamp=clamp
                )
                denoised_prime_over_sigma = (
                    images_next - model_output_next
                ) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma
                )

            images = images_next
            x_start = model_output_next if sigma_next != 0 else model_output

        # images = images.clamp(-1.0, 1.0)
        # images = unnormalize_to_zero_to_one(images)

        #images = self.net.unnormalize(images)
        return images

    @torch.no_grad()
    def sample_using_dpmpp(self, batch_size=16, num_sample_steps=None):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """

        device, num_sample_steps = self.device, default(
            num_sample_steps, self.num_sample_steps
        )

        sigmas = self.sample_schedule(num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        images = sigmas[0] * torch.randn(shape, device=device)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(
                images, sigmas[i].item()
            )
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = -1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (
                -h
            ).expm1() * denoised_d
            old_denoised = denoised

        images = images.clamp(-1.0, 1.0)
        return unnormalize_to_zero_to_one(images)

    # training

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) * (
            sigma * self.sigma_data
        ) ** -2

    def noise_distribution(self, batch_size):
        return (
            self.P_mean
            + self.P_std * torch.randn((batch_size,), device=self.device)
        ).exp()

    def forward(self, images):
        batch_size, c, h, w, device, image_size, channels = (
            *images.shape,
            images.device,
            self.image_size,
            self.channels,
        )

        assert (
            h == image_size and w == image_size
        ), f"height and width of image must be {image_size}"
        assert c == channels, "mismatch of image channels"

        images = normalize_to_neg_one_to_one(images)

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1 1")

        noise = torch.randn_like(images)

        noised_images = (
            images + padded_sigmas * noise
        )  # alphas are 1. in the paper

        self_cond = None

        if self.self_condition and random() < 0.5:
            # from hinton's group's bit diffusion paper
            with torch.no_grad():
                self_cond = self.preconditioned_network_forward(
                    noised_images, sigmas
                )
                self_cond.detach_()

        denoised = self.preconditioned_network_forward(
            noised_images, sigmas, self_cond
        )

        losses = F.mse_loss(denoised, images, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        losses = losses * self.loss_weight(sigmas)

        print("forward")

        return losses.mean()
