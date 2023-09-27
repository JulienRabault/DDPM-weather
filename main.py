import argparse
import gc
import os
import time
import warnings
from multiprocessing import cpu_count

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch import distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import dataSet_Handler
from distributed import get_rank, is_main_gpu, get_rank_num
from sampler import Sampler
from trainer import Trainer

warnings.filterwarnings(
    "ignore", message="This DataLoader will create .* worker processes in total.*")
gc.collect()
torch.cuda.empty_cache()


def ddp_setup(config):
    """
    Configuration for Distributed Data Parallel (DDP).
    Args:
        config (Namespace): Configuration parameters.
    """
    if torch.cuda.device_count() < 2:
        return
    init_process_group(
        'nccl' if dist.is_nccl_available() else 'gloo',
        world_size=torch.cuda.device_count())
    if config.debug_log:
        print(
            f"\n#LOG : [GPU{get_rank_num()}] init_process_group(backend={'nccl' if dist.is_nccl_available() else 'gloo'})")
    torch.cuda.set_device(get_rank())


def load_train_objs(config):
    """
    Load training objects.
    Args:
        config (Namespace): Configuration parameters.
    Returns:
        tuple: model, optimizer.
    """

    umodel = Unet(
        dim=int(config.image_size / 2),
        dim_mults=(1, 2, 4, 8),
        channels=len(config.var_indexes)
    )
    model = GaussianDiffusion(
        umodel,
        image_size=config.image_size,
        timesteps=1000,
        beta_schedule=config.beta_schedule,
        auto_normalize=config.auto_normalize,
        sampling_timesteps=config.ddim_timesteps,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, betas=config.adam_betas)
    return model, optimizer


def prepare_dataloader(config, path):
    """
    Prepare the data loader.
    Args:
        config (Namespace): Configuration parameters.

    Returns:
        DataLoader: Data loader.
    """
    train_set = dataSet_Handler.ISDataset(config, path)

    return DataLoader(
        train_set,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=not torch.cuda.device_count() >= 2,
        num_workers=cpu_count(),
        sampler=DistributedSampler(train_set, rank=get_rank_num(), shuffle=True,
                                   drop_last=False) if torch.cuda.device_count() >= 2 else None,
        drop_last=False
    )


def print_config(config):
    """
    Print the configuration.
    Args:
        config (Namespace): Configuration parameters.
    """
    print("Configuration:")
    for arg in vars(config):
        print(f"\t{arg}: {getattr(config, arg)}")
    print(f"\tWANDB_MODE: {os.environ['WANDB_MODE']}")
    print()


def save_config(config):
    """
    Save the configuration to a text file.
    Args:
        config (Namespace): Configuration parameters.
    """
    with open(f"{config.run_name}/config.txt", 'w') as f:
        for arg in vars(config):
            f.write(f"\t{arg}: {getattr(config, arg)}\n")


def check_config(config):
    """
    Check and configure the provided settings.
    Args:
        config (Namespace): Configuration parameters.
    Returns:
        Namespace: Updated configuration.
    """

    if config.invert_norm and config.mode != 'Train' and config.model_path is None:
        raise ValueError(
            "If --invert_norm is specified in Sample mode, --model_path must be defined.")

    if config.scheduler:
        warnings.warn(
            f"scheduler_epoch is set to {config.scheduler_epoch} (default: 150). The scheduler is a OneCycleLR scheduler in PyTorch, and it is saved in the .pt file. You must provide the total number of training epochs when using a scheduler.")

    if not torch.cuda.is_available():
        warnings.warn(
            f"Sampling on CPU may be slow. It is recommended to use one or more GPUs for faster sampling.")

    # Check if resuming training and model path exists
    check_path(config)

    # Adjust sample count if using multiple GPUs
    if torch.cuda.device_count() > 1 and config.mode != 'Train':
        world_size = torch.cuda.device_count()
        n_sample = config.n_sample

        if n_sample % world_size != 0:
            raise ValueError(
                f"n_sample={n_sample} is not divisible by world_size gpus={torch.cuda.device_count()}")

        config.n_sample = n_sample // world_size

    config.var_indexes = ['t2m'] if config.v_i == 1 else [
        'u', 'v'] if config.v_i == 2 else ['u', 'v', 't2m']

    # Save configuration and print if running on local rank 0
    if get_rank() == 0:
        save_config(config)

    return config


def check_path(config):
    if config.resume:
        if (config.model_path is None or not os.path.isfile(config.model_path)):
            raise FileNotFoundError(
                f"config.resume={config.resume} but snapshot_path={config.model_path} is None or doesn't exist")
        if config.mode != 'Train':
            raise ValueError("--r flag can only be used in Train mode.")
    # Print selected mode if running on local rank 0
    if get_rank() == 0:
        if config.mode == 'Train':
            print('#INFO: Mode Train selected')
        elif config.mode != 'Train':
            print('#INFO: Mode Sample selected')
    paths = [
        f"{config.run_name}/",
        f"{config.run_name}/samples/",
    ]
    if config.mode == 'Train':
        paths.append(f"{config.run_name}/WANDB/")
        paths.append(f"{config.run_name}/WANDB/cache")
    # Check paths if resuming, else create them
    if is_main_gpu():
        if config.resume:
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"The following directories do not exist: {path}")
        else:
            train_num = 1
            train_name = config.run_name
            while os.path.exists(train_name):
                if f"_{train_num}" in train_name:
                    train_name = "_".join(train_name.split(
                        '_')[:-1]) + f"_{train_num + 1}"
                    train_num += 1
                else:
                    train_name = f"{train_name}_{train_num}"
            config.run_name = train_name
            paths = [
                f"{config.run_name}/",
                f"{config.run_name}/samples/",
            ]
            if config.mode == 'Train':
                paths.append(f"{config.run_name}/WANDB/")
                paths.append(f"{config.run_name}/WANDB/cache")
            for path in paths:
                print(f"Creating directory {path}")
                os.makedirs(path, exist_ok=True)


def main_train(config):
    """
    Main function for training.
    Args:
        config (Namespace): Configuration parameters.
    """
    model, optimizer = load_train_objs(config)
    train_data = prepare_dataloader(config, path=config.data_dir)
    start = time.time()
    trainer = Trainer(model, config, dataloader=train_data,
                      optimizer=optimizer)
    trainer.train()
    end = time.time()
    total_time = end - start
    if config.debug_log:
        print(f"\n#LOG: Training execution time: {total_time} seconds")
    if is_main_gpu():
        # Sample best model
        config.model_path = f"{config.run_name}/best.pt"
        model, _ = load_train_objs(config)
        sampler = Sampler(model, config)
        sampler.sample(filename_format="sample_best_{i}.npy", nb_img=config.n_sample, plot=config.plot)


def main_sample(config):
    """
    Main function for testing.
    Args:
        config (Namespace): Configuration parameters.
    """
    model, _ = load_train_objs(config)
    sampler = Sampler(model, config)
    if config.guided is None:
        sampler.sample(nb_img=config.n_sample, plot=config.plot)
    else:
        train_data = prepare_dataloader(config, path=config.guided)
        sampler.guided_sample(train_data, plot=config.plot, random_noise=config.random_noise)


def get_parser():
    parser = argparse.ArgumentParser(description='Deep Learning Training and Testing Script')

    # General parameters
    general_args = parser.add_argument_group('General Parameters')
    general_args.add_argument(dest='mode', choices=['Train', 'Sample'],
                              help='Execution mode: Choose between Train or Sample')
    general_args.add_argument('--run_name', type=str, default='run', help='Name for the training run')
    general_args.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    general_args.add_argument('-a', '--any_time', type=int, default=400,
                              help='Every how many epochs to save and sample')
    general_args.add_argument('-mp', '--model_path', type=str, default=None,
                              help='Path to the model for loading and resuming training if necessary (no path will start training from scratch)')
    general_args.add_argument("--debug", dest="debug_log", default=False, action="store_true",
                              help="Enable debug logs")

    # Sampling parameters
    sample_args = parser.add_argument_group('Sampling Parameters')
    sample_args.add_argument('-ddim', '--ddim_timesteps', type=int, default=None,
                             help='If not None, will sample from the ddim methode with the specified number of timesteps')
    sample_args.add_argument("-p", "--plot", dest="plot", default=False, action="store_true")
    sample_args.add_argument("-g", "--guide_data", dest="guided", type=str, default=None, help="Path to guided data")
    sample_args.add_argument('-s', '--n_sample', type=int, default=10, help='Number of samples to generate')
    sample_args.add_argument('-rn', '--random_noise', type=bool, default=False,
                             help='Use random noise for x_start in guided sampling')

    # Data parameters
    data_args = parser.add_argument_group('Data Parameters')
    data_args.add_argument('-dd', '--data_dir', type=str, default=None, help='Directory containing the data')
    data_args.add_argument('-nvi', '--v_i', type=int, default=3, help='Number of variable indices')
    data_args.add_argument('-vi', '--var_indexes', type=list, default=['u', 'v', 't2m'],
                           help='List of variable indices')
    data_args.add_argument('-c', '--crop', type=list, default=[78, 206, 55, 183], help='Crop parameters for images')
    data_args.add_argument("--auto_normalize", dest="auto_normalize", default=False, action="store_true",
                           help="Automatically normalize")
    data_args.add_argument("--invert_norm", dest="invert_norm", default=False, action="store_true",
                           help="Invert normalization of image samples")
    data_args.add_argument('-is', '--image_size', type=int, default=128, help='Size of the image')
    data_args.add_argument('-mf', '--mean_file', type=str, default='mean_with_orog.npy', help='Mean file path')
    data_args.add_argument('-xf', '--max_file', type=str, default='max_with_orog.npy', help='Max file path')

    # Model parameters
    training_args = parser.add_argument_group('Train Parameters')
    training_args.add_argument("--scheduler", dest="scheduler", default=False, action="store_true",
                               help="Use scheduler for learning rate")
    training_args.add_argument('-se', '--scheduler_epoch', type=int, default=150,
                               help='Number of epochs for scheduler to downscale (save for resume)')
    training_args.add_argument("-r", "--resume", dest="resume", default=False, action="store_true",
                               help="Resume from checkpoint")
    training_args.add_argument('-lr', '--lr', type=float, default=5e-4, help='Learning rate')
    training_args.add_argument('-ab', '--adam_betas', type=tuple, default=(0.9, 0.99),
                               help='Betas for the Adam optimizer')
    training_args.add_argument('-e', '--epochs', type=int, default=150, help='Number of epochs to train for')
    training_args.add_argument("--beta_schedule", type=str, default="cosine",
                               help="Beta schedule type (cosine or linear)")

    # Tracking parameters
    tracking_args = parser.add_argument_group('Tracking Parameters')
    tracking_args.add_argument("--wandbproject", dest="wp", type=str, default="meteoDDPM", help="Wandb project name")
    tracking_args.add_argument("-w", "--wandb", dest="use_wandb", default=False, action="store_true",
                               help="Use wandb for logging")
    tracking_args.add_argument("--entityWDB", type=str, default="jrabault", help="Wandb entity name")
    return parser


if __name__ == "__main__":

    config = get_parser().parse_args()
    # assert config.n_sample <= config.batch_size, 'can only work with n_sample <=  batch_size'
    ddp_setup(config)
    local_rank = get_rank()
    print("local_rank : ", local_rank)

    config = check_config(config)

    if not config.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    else:
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_CACHE_DIR'] = f"{config.run_name}/WANDB/cache"
        os.environ['WANDB_DIR'] = f"{config.run_name}/WANDB/"

    if is_main_gpu():
        print_config(config)

    if config.mode == 'Train':
        main_train(config)
    elif config.mode != 'Train':
        main_sample(config)

    if dist.is_initialized():
        destroy_process_group()
