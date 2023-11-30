import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from multiprocessing import cpu_count

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch import distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ddpm import dataSet_Handler
from ddpm.guided_gaussian_diffusion import GuidedGaussianDiffusion
from ddpm.sampler import Sampler
from ddpm.trainer import Trainer
from utils.config import Config
from utils.distributed import get_rank_num, get_rank, is_main_gpu, synchronize
from ddpm import dataset_handler_ddp

warnings.filterwarnings(
    "ignore", message="This DataLoader will create .* worker processes in total.*")
gc.collect()
torch.cuda.empty_cache()


def setup_logger(config, log_file="ddpm.log"):
    """
    Configure a logger with specified console and file handlers.
    Args:
        config: The configuration object.
        log_file (str): The name of the log file.
    Returns:
        logging.Logger: The configured logger.
    """
    # Use a logger specific to the GPU rank
    console_format = f'[GPU {get_rank_num()}] %(asctime)s - %(levelname)s - %(message)s' if torch.cuda.device_count() > 1 \
        else '%(asctime)s - %(levelname)s - %(message)s'

    logger = logging.getLogger(f'logddp_{get_rank_num()}')
    logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
    logger.propagate = False  # Prevent double printing

    # Console handler for printing log messages to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if config.debug else logging.INFO)
    console_formatter = logging.Formatter(console_format)
    console_handler.setFormatter(console_formatter)

    # File handler for saving log messages to a file
    file_handler = logging.FileHandler(os.path.join(config.run_name, log_file), mode='w+')
    file_handler.setLevel(logging.DEBUG if config.debug else logging.INFO)
    file_formatter = logging.Formatter(console_format)
    file_handler.setFormatter(file_formatter)

    # Add both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.getLogger('wandb').setLevel(logging.WARNING)

    return logger


def ddp_setup():
    """
    Configuration for Distributed Data Parallel (DDP).
    """
    if torch.cuda.device_count() < 2:
        return
    # Initialize the process group for DDP
    init_process_group(
        'nccl' if dist.is_nccl_available() else 'gloo',
        world_size=torch.cuda.device_count())
    torch.cuda.set_device(get_rank())


def load_train_objs(config):
    """
    Load training objects.
    Args:
        config (Namespace): Configuration parameters.
    Returns:
        tuple: model, optimizer.
    """
    # Create a U-Net model and a diffusion model based on configuration
    umodel = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=len(config.var_indexes),
        self_condition=config.guiding_col is not None,
    )
    if config.guiding_col is not None:
        cls = GuidedGaussianDiffusion
    else:
        cls = GaussianDiffusion
    model = cls(
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


def prepare_dataloader(config, path, csv_file):
    """
    Prepare the data loader.
    Args:
        config (Namespace): Configuration parameters.
    Returns:
        DataLoader: Data loader.
    """

    if config.dataloader_rr:
        dataloader = dataset_handler_ddp.ISData_Loader("Train", config)
        kwargs = {'pin_memory': True}
        return dataloader.loader(get_world_size(), get_rank_num(), kwargs)
    else:
        # Load the dataset and create a DataLoader with distributed sampling if using multiple GPUs
        train_set = dataSet_Handler.ISDataset(config, path, csv_file)
        return DataLoader(
            train_set,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=not torch.cuda.device_count() >= 2,
            num_workers=cpu_count(),
            sampler=DistributedSampler(train_set, rank=get_rank_num(), shuffle=False,
                                    drop_last=True) if torch.cuda.device_count() >= 2 else None,
            drop_last=True
        )

def main_train(config):
    """
    Main function for training.
    Args:
        config (Namespace): Configuration parameters.
    """
    # Load training objects and start the training process
    model, optimizer = load_train_objs(config)
    train_data = prepare_dataloader(config, path=config.data_dir, csv_file=config.csv_file)
    start = time.time()
    trainer = Trainer(model, config, dataloader=train_data, optimizer=optimizer)
    trainer.train()
    end = time.time()
    total_time = end - start
    logging.debug(f"Training execution time: {total_time} seconds")
    synchronize()
    # Sample the best model
    sample_data = None if config.guiding_col is None else train_data
    config.model_path = os.path.join(config.run_name, "best.pt")
    model, _ = load_train_objs(config)
    sampler = Sampler(model, config, dataloader=sample_data)
    sampler.sample(filename_format="sample_best_{i}.npy")
    logging.info(f"Training completed and best model sampled. You can check log and results in {config.run_name}")


def main_sample(config):
    """
    Main function for testing.
    Args:
        config (Namespace): Configuration parameters.
    """
    # Load the model and start the sampling process
    model, _ = load_train_objs(config)
    if config.sampling_mode != "simple":
        sample_data = prepare_dataloader(config, path=config.data_dir, csv_file=config.csv_file)
    else:
        sample_data = None
    sampler = Sampler(model, config, dataloader=sample_data)
    sampler.sample()


if __name__ == "__main__":
    # Parse command line arguments and load configuration
    parser = argparse.ArgumentParser(description='Deep Learning Training and Testing Script')
    parser.add_argument('--yaml_path', type=str, default='config_train.yml', help='Path to YAML configuration file')
    parser.add_argument('--debug', action='store_true', help='Debug logging')

    with open('utils/config_schema.json', 'r') as schema_file:
        schema = json.load(schema_file)
    ddp_setup()

    Config.create_arguments(parser, schema)
    args = parser.parse_args()
    config = Config.from_args_and_yaml(args)

    local_rank = get_rank()

    # Configure logging and synchronize processes
    if not config.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    else:
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_CACHE_DIR'] = f"{config.run_name}/WANDB/cache"
        os.environ['WANDB_DIR'] = f"{config.run_name}/WANDB/"
    synchronize()
    setup_logger(config)
    logger = logging.getLogger(f'logddp_{get_rank_num()}')
    if is_main_gpu():
        config.save(f"{config.run_name}/config_train.yml")
        logger.info(config)
        logger.info(f'Mode {config.mode} selected')

    synchronize()
    logger.debug(f"Local_rank: {local_rank}")

    # Execute the main training or sampling function based on the mode
    if config.mode == 'Train':
        main_train(config)
    elif config.mode != 'Train':
        main_sample(config)

    # Clean up distributed processes if initialized
    if dist.is_initialized():
        destroy_process_group()
