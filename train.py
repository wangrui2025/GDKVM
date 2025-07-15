import os
import math
import time
import datetime
import logging
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from utils.ddp import distributed_setup, info_if_rank_zero

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf import DictConfig
import wandb
from tqdm import tqdm 
from dataset.vos_dataset_0709_2010 import TenCamusDataset
from model.trainer_super import Trainer
from utils.logger import TensorboardLogger

log = logging.getLogger(__name__)


@hydra.main(version_base='1.3.2', config_path='config', config_name='gdkvm_0709_2010.yaml')
def train(cfg: DictConfig):
    os.environ["WANDB_MODE"] = "offline"

    # local_rank, world_size = distributed_setup()
    # local_rank 和 world_size 将从 torchrun/DDP 环境中自动获取
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)


    if local_rank == 0:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        cfg_dict_wandb = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project="GDKVM_20250702",
            entity="team-gdkvm", 
            name=f"run_{timestamp}", 
            config=cfg_dict_wandb,  # 记录所有配置参数
            reinit=True  # 如果需要多次调用 wandb.init()
        )
        wandb.config.update(cfg_dict_wandb, allow_val_change=True)  # 确保配置被正确记录
    
    try:
        run_dir = HydraConfig.get().run.dir
        if local_rank == 0:
            info_if_rank_zero(f'All configuration: {cfg}')
            info_if_rank_zero(f'Number of detected GPUs: {world_size}')
            info_if_rank_zero(f'Run dir: {run_dir}')

        if cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        if local_rank == 0:
            info_if_rank_zero(f'batch_size={cfg.main_training.batch_size}')
        cfg.main_training.batch_size = max(cfg.main_training.batch_size // world_size, 1)
        if local_rank == 0:
            info_if_rank_zero(f'batch_size(per-GPU)={cfg.main_training.batch_size}')

        if local_rank == 0:
            info_if_rank_zero(f'num_workers={cfg.main_training.num_workers}')
        cfg.main_training.num_workers = max(cfg.main_training.num_workers // world_size, 1)
        if local_rank == 0:
            info_if_rank_zero(f'num_workers(per-GPU)={cfg.main_training.num_workers}')

        # logger (可选)
        log_writer = TensorboardLogger(run_dir, logging.getLogger(), enabled_tb=(local_rank == 0))

        # ========== 准备数据集与 DataLoader ==========
        # 创建一个辅助函数来安全地创建 DataLoader
        def create_safe_dataloader(dataset, sampler, batch_size, num_workers, pin_memory=False, drop_last=False):
            """创建一个安全的 DataLoader，在多进程出现问题时自动降级"""
            try:
                return data.DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=False,
                    drop_last=drop_last
                )
            except Exception as e:
                if local_rank == 0:
                    log.warning(f"Failed to create DataLoader with num_workers={num_workers}, trying with num_workers=0: {e}")
                return data.DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=0,  # 降级到单进程
                    pin_memory=False,
                    shuffle=False,
                    drop_last=drop_last
                )
        
        # 训练集
        train_dataset = TenCamusDataset(
            filepath=cfg.data_path, 
            mode='train',
            seq_length=cfg.main_training.seq_length,
            max_num_obj=cfg.main_training.num_objects,
            size=cfg.main_training.crop_size[0]
        )
        train_sampler = data.distributed.DistributedSampler(train_dataset)
        train_loader = create_safe_dataloader(
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=cfg.main_training.batch_size,
            num_workers=cfg.main_training.num_workers,
            pin_memory=False,
            drop_last=True
        )

        # 验证集
        val_dataset = TenCamusDataset(
            filepath=cfg.data_path, 
            mode='val',
            seq_length=cfg.main_training.seq_length,
            max_num_obj=cfg.main_training.num_objects,
            size=cfg.main_training.crop_size[0]
        )
        val_sampler = data.distributed.DistributedSampler(val_dataset)
        val_loader = create_safe_dataloader(
            dataset=val_dataset,
            sampler=val_sampler,
            batch_size=cfg.main_training.batch_size,
            num_workers=cfg.main_training.num_workers,
            pin_memory=False,
            drop_last=False
        )

        test_dataset = TenCamusDataset(
            filepath=cfg.data_path, 
            mode='test',
            seq_length=cfg.main_training.seq_length,
            max_num_obj=cfg.main_training.num_objects,
            size=cfg.main_training.crop_size[0]
        )
        test_sampler = data.distributed.DistributedSampler(test_dataset)
        test_loader = create_safe_dataloader(
            dataset=test_dataset,
            sampler=test_sampler,
            batch_size=cfg.main_training.batch_size,
            num_workers=cfg.main_training.num_workers,
            pin_memory=False,
            drop_last=False
        )

        # ========== 准备 Trainer ==========
        stage_cfg = cfg.main_training
        trainer = Trainer(
            cfg=cfg,
            stage_cfg=stage_cfg,
            log=log_writer,
            run_path=run_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )

        total_iterations = stage_cfg.num_iterations
        it = 0
        current_epoch = 0
        max_epoch = math.ceil(total_iterations / len(train_loader))
        if local_rank == 0:
            log.info(f"Total iterations: {total_iterations}")
            log.info(f"train_loader length (batches per epoch): {len(train_loader)}")
            info_if_rank_zero(f"Total iteration={total_iterations}, est. epochs={max_epoch} ...")

        # ========== 创建一个 tqdm 进度条（仅在 local_rank=0）==========
        if local_rank == 0:
            pbar = tqdm(total=total_iterations, desc="Training", ncols=120)
        else:
            pbar = None  # 非主进程不需要进度条


        # iteration-based 训练
        steps_per_epoch = len(train_loader)
        data_iter = iter(train_loader)

        for it in range(total_iterations):
            # 若迭代器已取完一个 epoch，就重置（相当于自然epoch结束）
            current_epoch = it // steps_per_epoch + 1

            try:
                batch_data = next(data_iter)
            except StopIteration:
                train_sampler.set_epoch(current_epoch)
                data_iter = iter(train_loader)
                batch_data = next(data_iter)


            loss_val = trainer.do_pass(batch_data, it)
                
            # ========== 保存 checkpoint ==========
            if cfg.save == 1:
                if local_rank == 0 :
                    if it % cfg.save_weights_interval == 0 and it > 0:
                        trainer.save_weights(it)
                        log.info(f"Weights saved at iteration {it} and epoch {current_epoch}")
                    if it % cfg.save_checkpoint_interval == 0 and it > 0:
                        trainer.save_checkpoint(it)
                        log.info(f"Checkpoint saved at iteration {it} and epoch {current_epoch}")

            # 用 tqdm 更新进度
            if local_rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    'iter': it + 1,
                    'loss': f"{loss_val:.4f}"
                })

            # ========== 每隔固定 iteration 做验证 ==========
            if (it + 1) % cfg.eval_stage.eval_interval == 0:
                trainer.evaluate(
                    val_loader=val_loader,
                    epoch=current_epoch,    # 这里的 epoch 参数只是给 eval_vos 里打印或给 sampler 用的，你也可改成别的
                    local_rank=local_rank,
                    world_size=world_size,
                    run_path=run_dir,
                    it=it + 1
                )

                trainer.test(
                    test_loader=test_loader,
                    epoch=current_epoch,    # 这里的 epoch 参数只是给 eval_vos 里打印或给 sampler 用的，你也可改成别的
                    local_rank=local_rank,
                    world_size=world_size,
                    run_path=run_dir,
                    it=it + 1
                )

        info_if_rank_zero("Training completed. ")

        

    finally:
        if local_rank == 0:
            wandb.finish()
        dist.destroy_process_group()


if __name__ == '__main__':
    train()
