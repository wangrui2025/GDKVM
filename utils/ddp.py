import os
import logging
import torch
import torch.distributed as dist

log = logging.getLogger(__name__)

def distributed_setup(backend: str = "nccl"):
    """
    Initialize torch.distributed from torchrun-provided env.
    Returns:
        local_rank (int), world_size (int)
    """
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return local_rank, world_size

    # torchrun 
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)

    log.info(f"[dist setup] rank={rank}, local_rank={local_rank}, world_size={world_size}, backend={backend}")
    return local_rank, world_size

def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def info_if_rank_zero(msg: str):
    if is_main_process():
        log.info(msg)

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
