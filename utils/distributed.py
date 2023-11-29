import torch
from torch import distributed as dist

def get_rank():
    """
    Get the rank of the current process.
    Returns:
        str: 'cpu' if CUDA is not available, 'cuda' if there is a single GPU, or 'cuda:{rank}' for distributed GPUs.
    """
    if not torch.cuda.is_available():
        return 'cpu'
    if torch.cuda.device_count() < 2:
        return 'cuda'
    return "cuda:" + str(dist.get_rank())


def get_rank_num():
    """
    Get the rank number of the current process.
    Returns:
        int: 0 if CUDA is not available or there is a single GPU, or the rank number for distributed GPUs.
    """
    if not torch.cuda.is_available():
        return 0
    if torch.cuda.device_count() < 2:
        return 0
    return dist.get_rank()


def is_main_gpu():
    """
    Check if the current process is running on the main GPU.
    Returns:
        bool: True if CUDA is not available or there is a single GPU, or if the current process is the main process in distributed training.
    """
    if not torch.cuda.is_available():
        return True
    if torch.cuda.device_count() < 2:
        return True
    return dist.get_rank() == 0


def synchronize():
    """
    Synchronize all processes in distributed training.
    This function is a barrier that ensures all processes reach this point before proceeding.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size():
    """
    Get the total number of processes in distributed training.
    Returns:
        int: 1 if CUDA is not available or not in distributed mode, or the total number of processes in distributed training.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_sum(tensor):
    """
    Perform distributed sum reduction on the input tensor.
    Args:
        tensor (torch.Tensor): Input tensor to be summed across all processes.

    Returns:
        torch.Tensor: Resulting tensor after the sum reduction.
    """
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
