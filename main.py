import argparse
import gc
import json
import os
import time
import warnings
from multiprocessing import cpu_count

import torch
import yaml
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch import distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import logging

from ddpm import dataSet_Handler
from ddpm.guided_gaussian_diffusion import GuidedGaussianDiffusion
from ddpm.sampler import Sampler
from ddpm.trainer import Trainer
from utils.config import Config
from utils.distributed import get_rank_num, get_rank, is_main_gpu

warnings.filterwarnings(
    "ignore", message="This DataLoader will create .* worker processes in total.*")
gc.collect()
torch.cuda.empty_cache()


def setup_logger(config, log_file="ddpm.log"):
    """
    Set up a logger with specified console and file handlers.
    Args:
        config: The configuration object.
        log_file (str): The name of the log file.
    Returns:
        logging.Logger: The configured logger.
    """

    del logging.getLogger().handlers[:]
    # logger = logging.getLogger('logddp')

    # If the logger is not already configured, configure it
    console_format = f'[GPU {get_rank_num()}] %(asctime)s - %(levelname)s - %(message)s' if torch.cuda.device_count() > 1 \
        else '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=console_format)
    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(console_format)
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)



def ddp_setup():
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
    logging.debug(
        f"init_process_group(backend={'nccl' if dist.is_nccl_available() else 'gloo'})")
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
    train_set = dataSet_Handler.ISDataset(config, path, csv_file)

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


def main_train(config):
    """
    Main function for training.
    Args:
        config (Namespace): Configuration parameters.
    """
    model, optimizer = load_train_objs(config)
    train_data = prepare_dataloader(config, path=config.data_dir, csv_file=config.csv_file, )
    start = time.time()
    trainer = Trainer(model, config, dataloader=train_data,
                      optimizer=optimizer)
    trainer.train()
    end = time.time()
    total_time = end - start
    logging.debug(f"Training execution time: {total_time} seconds")
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
    if config.data_dir is not None:
        sample_data = prepare_dataloader(config, path=config.data_dir, csv_file=config.csv_file)
    else:
        sample_data = None
    sampler = Sampler(model, config, dataloader=sample_data)
    sampler.sample(plot=config.plot, random_noise=config.random_noise)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script d\'entraînement et de test de Deep Learning')
    parser.add_argument('--yaml_path', type=str, default='config.yml', help='Path to YAML configuration file')
    parser.add_argument('--debug', action='store_true', help='Debug logging')

    # Load the schema from a file
    with open('utils/config_schema.json', 'r') as schema_file:
        schema = json.load(schema_file)
 
    ddp_setup()

    Config.create_arguments(parser, schema)
    args = parser.parse_args()
    config = Config.from_args_and_yaml(args)

    setup_logger(config)
    # assert config.n_sample <= config.batch_size, 'can only work with n_sample <= batch_size'
    local_rank = get_rank()

    if not config.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    else:
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_CACHE_DIR'] = f"{config.run_name}/WANDB/cache"
        os.environ['WANDB_DIR'] = f"{config.run_name}/WANDB/"
    # setup_logger(config)
    # logger = logging.getLogger('logddp')


    logger = logging.getLogger()

    if is_main_gpu():
        config.save(f"{config.run_name}/config.yml")
        logger.info(config)
        logger.info(f'Mode {config.mode} selected')

    logger.debug(f"local_rank: {local_rank}")

    if config.mode == 'Train':
        main_train(config)
    elif config.mode != 'Train':
        main_sample(config)

    if dist.is_initialized():
        destroy_process_group()
