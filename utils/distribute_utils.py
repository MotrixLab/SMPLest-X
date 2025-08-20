import os
import os.path as osp
import pickle
import shutil
import tempfile
import time
import torch
import torch.distributed as dist
import random
import numpy as np
from .device_utils import get_device, synchronize


def get_dist_info():
    """
    Get the rank and world size in the current distributed training setup.

    Returns:
        tuple: (rank, world_size)
            rank: int, the rank of the current process.
            world_size: int, the total number of processes in the group.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_synchronized():
    synchronize()  # Use centralized synchronization
    return time.time()


def setup_for_distributed(is_master):
    """This function disables printing when not in master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(port = None, master_port=29500):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    # import pdb; pdb.set_trace()
    if torch.cuda.is_available():
        dist_backend = 'nccl'
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
    else:
        dist_backend = 'gloo'  # Use gloo backend for non-CUDA devices
        rank = int(os.environ['RANK'])
        num_gpus = 1  # For MPS or CPU

    dist.init_process_group(backend=dist_backend)
    distributed = True
    gpu_idx = rank % num_gpus

    return distributed, gpu_idx


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_process_groups():
    world_size = int(os.environ['WORLD_SIZE'])
    ranks = list(range(world_size))
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
    num_nodes = world_size // num_gpus
    if world_size % num_gpus != 0:
        raise NotImplementedError('Not implemented for node not fully used.')

    groups = []
    for node_idx in range(num_nodes):
        groups.append(ranks[node_idx*num_gpus : (node_idx+1)*num_gpus])
    process_groups = [torch.distributed.new_group(group) for group in groups]

    return process_groups

def get_group_idx():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
    proc_id = get_rank()
    group_idx = proc_id // num_gpus

    return group_idx


def is_main_process():
    return get_rank() == 0

def cleanup():
    dist.destroy_process_group()

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data:
            Any picklable object
    Returns:
        data_list(list):
            List of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    device = get_device()  # Use centralized device selection
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.empty((max_size, ), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size, ), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
