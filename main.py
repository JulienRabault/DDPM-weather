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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import DataSet_Handler
from DataSet_Handler import ISData_Loader
from distributed import get_rank, is_main_gpu
from trainer import Trainer

warnings.filterwarnings("ignore", message="This DataLoader will create .* worker processes in total.*")
gc.collect()
torch.cuda.empty_cache()


def ddp_setup(config):
    """
    Configuration for Distributed Data Parallel (DDP).
    Args:
        config (Namespace): Configuration parameters.
    """
    if torch.cuda.device_count() < 2 or config.mode == 'Test':
        return
    init_process_group(
        'nccl' if dist.is_nccl_available() else 'gloo',
        rank=local_rank,
        world_size=torch.cuda.device_count())
    if config.debug_log:
        print(
            f"\n#LOG : [GPU{get_rank()}] init_process_group(backend={'nccl' if dist.is_nccl_available() else 'gloo'})")
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
        auto_normalize=config.auto_normalize
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.adam_betas)
    return model, optimizer


def prepare_dataloader(config):
    """
    Prepare the data loader.
    Args:
        dataset (Dataset): Dataset.
        config (Namespace): Configuration parameters.

    Returns:
        DataLoader: Data loader.
    """
    train_set = ISData_Loader(config.data_dir, config.batch_size,
                              [DataSet_Handler.var_dict[var] for var in config.var_indexes], config.crop).loader()[1]
    return DataLoader(
        train_set,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=not dist.is_initialized(),
        num_workers=cpu_count(),
        sampler=DistributedSampler(train_set, rank=get_rank(), shuffle=True,
                                   drop_last=False) if dist.is_initialized() else None,
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
    with open(f"{config.train_name}/config.txt", 'w') as f:
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
    if config.invert_norm and config.mode == 'Test' and config.model_path is None:
        raise ValueError("If --invert_norm is specified in Test mode, --model_path must be defined.")

    if config.scheduler:
        warnings.warn(f"scheduler_epoch is set to {config.scheduler_epoch} (default: 150). The scheduler is a OneCycleLR scheduler in PyTorch, and it is saved in the .pt file. You must provide the total number of training epochs when using a scheduler.")

    if not torch.cuda.is_available():
        warnings.warn(f"Sampling on CPU may be slow. It is recommended to use one or more GPUs for faster sampling.")

    # Check if resuming training and model path exists
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
        elif config.mode == 'Test':
            print('#INFO: Mode Test selected')

    # Define paths
    paths = [
        f"{config.train_name}/",
        f"{config.train_name}/samples/",
        f"{config.train_name}/WANDB/",
        f"{config.train_name}/WANDB/cache",
    ]

    # Check paths if resuming, else create them
    if get_rank() == 0:
        if config.resume:
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"The following directories do not exist: {path}")
        else:
            train_num = 1
            train_name = config.train_name
            while os.path.exists(train_name):
                if f"_{train_num}" in train_name:
                    train_name = "_".join(train_name.split('_')[:-1]) + f"_{train_num + 1}"
                    train_num += 1
                else:
                    train_name = f"{train_name}_{train_num}"
            config.train_name = train_name
            paths = [
                f"{config.train_name}/",
                f"{config.train_name}/samples/",
                f"{config.train_name}/WANDB/",
                f"{config.train_name}/WANDB/cache",
            ]
            for path in paths:
                os.makedirs(path, exist_ok=True)

    # Adjust sample count if using multiple GPUs
    if torch.cuda.device_count() > 1 and config.mode == 'Test':
        world_size = torch.cuda.device_count()
        n_sample = config.n_sample

        if n_sample % world_size != 0:
            raise ValueError(f"n_sample={n_sample} is not divisible by world_size gpus={torch.cuda.device_count()}")

        config.n_sample = n_sample // world_size

    # Determine variable indexes
    config.var_indexes = ['t2m'] if config.v_i == 1 else ['u', 'v'] if config.v_i == 2 else ['u', 'v', 't2m']

    # Save configuration and print if running on local rank 0
    if get_rank() == 0:
        save_config(config)

    return config


def main_train(config):
    """
    Main function for training.
    Args:
        config (Namespace): Configuration parameters.
    """
    model, optimizer = load_train_objs(config)
    train_data = prepare_dataloader(config)
    start = time.time()
    trainer = Trainer(model, config, dataloader=train_data, optimizer=optimizer)
    trainer.train(config)
    end = time.time()
    total_time = end - start
    if config.debug_log:
        print(f"\n#LOG: Training execution time: {total_time} seconds")
    if is_main_gpu():
        trainer.sample_images("last", nb_image=config.n_sample)


def main_test(config):
    """
    Main function for testing.
    Args:
        config (Namespace): Configuration parameters.
    """
    model, _ = load_train_objs(config)
    trainer = Trainer(model, config)
    trainer.sample_images(nb_image=config.n_sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['Train', 'Test'], help='Execution mode: Choose between Train or Test')
    parser.add_argument('--train_name', type=str, default='train', help='Name for the training run')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_sample', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--any_time', type=int, default=400, help='Every how many epochs to save and sample')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model for loading and resuming training if necessary (no path will start training from scratch)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.99), help='Betas for the Adam optimizer')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train for')
    parser.add_argument('--image_size', type=int, default=128, help='Size of the image')
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory containing the data')
    parser.add_argument('--v_i', type=int, default=3, help='Number of variable indices')
    parser.add_argument('--var_indexes', type=list, default=['u', 'v', 't2m'], help='List of variable indices')
    parser.add_argument('--crop', type=list, default=[78, 206, 55, 183], help='Crop parameters for images')
    parser.add_argument("--wandbproject", dest="wp", type=str, default="meteoDDPM", help="Wandb project name")
    parser.add_argument("-w", "--wandb", dest="use_wandb", default=False, action="store_true",
                        help="Use wandb for logging")
    parser.add_argument("--entityWDB", type=str, default="jrabault", help="Wandb entity name")
    parser.add_argument("--invert_norm", dest="invert_norm", default=False, action="store_true",
                        help="Invert normalization of images samples")
    parser.add_argument("--beta_schedule", type=str, default="cosine", help="Beta schedule type (cosine or linear)")
    parser.add_argument("--auto_normalize", dest="auto_normalize", default=False, action="store_true",
                        help="Automatically normalize")
    parser.add_argument("--scheduler", dest="scheduler", default=False, action="store_true",
                        help="Use scheduler for learning rate")
    parser.add_argument('--scheduler_epoch', type=int, default=150, help='Number of epochs for scheduler to down scale (save for resume')
    parser.add_argument("-r", "--resume", dest="resume", default=False, action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--debug", dest="debug_log", default=False, action="store_true", help="Enable debug logs")
    config = parser.parse_args()

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        local_rank = 0

    ddp_setup(config)

    config = check_config(config)

    if not config.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    else:
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_CACHE_DIR'] = f"{config.train_name}/WANDB/cache"
        os.environ['WANDB_DIR'] = f"{config.train_name}/WANDB/"

    if is_main_gpu():
        print_config(config)

    if config.mode == 'Train':
        main_train(config)
    elif config.mode == 'Test':
        main_test(config)

    if dist.is_initialized():
        destroy_process_group()
