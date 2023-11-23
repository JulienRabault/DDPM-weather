import torch
from torch import distributed as dist


def get_rank():
    if not torch.cuda.is_available():
        return 'cpu'
    if torch.cuda.device_count() < 2:
        return 'cuda'
    return "cuda:" + str(dist.get_rank())


def get_rank_num():
    if not torch.cuda.is_available():
        return 0
    if torch.cuda.device_count() < 2:
        return 0
    return dist.get_rank()


def is_main_gpu():
    if not torch.cuda.is_available():
        return True
    if torch.cuda.device_count() < 2:
        return True
    return dist.get_rank() == 0

def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


