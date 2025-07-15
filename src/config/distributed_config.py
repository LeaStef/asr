import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl') -> None:
    """Initialize distributed training"""
    if world_size > 1:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process"""
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reduce tensor across all processes"""
    if world_size == 1:
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_distributed_config() -> tuple:
    """Get distributed configuration from environment"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    return rank, world_size, local_rank