import os
import logging
import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def distributed_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    log.info(f'[dist setup] local_rank={local_rank}, world_size={world_size}')
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def info_if_rank_zero(msg):
    if dist.get_rank() == 0:
        log.info(msg)
