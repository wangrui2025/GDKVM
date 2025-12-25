import os
import math
import datetime
import logging
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

from utils.ddp import distributed_setup, info_if_rank_zero, is_main_process, barrier

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
import wandb
from tqdm import tqdm

from dataset.vos_dataset import TenCamusDataset
from model.trainer import Trainer
from utils.logger import TensorboardLogger

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3.2", config_path="config", config_name="config_gdkvm_01.yaml")
def train(cfg: DictConfig):
    os.environ.setdefault("WANDB_MODE", cfg.get("wandb_mode", "offline"))

    # -------- DDP Initialization --------
    local_rank, world_size = distributed_setup(backend="nccl")
    main_process = is_main_process()

    # Initialize wandb only on the main process
    if main_process:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        cfg_dict_wandb = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project="GDKVM_2025",
            entity="team-wangrui",
            name=f"run_{timestamp}",
            config=cfg_dict_wandb,
            reinit="finish_previous",
        )
        wandb.config.update(cfg_dict_wandb, allow_val_change=True)

    try:
        run_dir = HydraConfig.get().run.dir

        # Ensure configuration is printed only once by the main process
            
        info_if_rank_zero(f"All configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
        info_if_rank_zero(f"Number of detected GPUs: {world_size}")
        info_if_rank_zero(f"Run dir: {run_dir}")

        if cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # -------- Random Seed (Offset by rank to avoid identical augmentations) --------
        base_seed = int(cfg.seed)
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.manual_seed(base_seed + rank)
        np.random.seed(base_seed + rank)
        random.seed(base_seed + rank)

        # -------- Adjust per-GPU batch size and workers based on world_size --------
        stage_cfg = cfg.main_training
        info_if_rank_zero(f"batch_size={stage_cfg.batch_size}")
        stage_cfg.batch_size = max(stage_cfg.batch_size // world_size, 1)
        info_if_rank_zero(f"batch_size(per-GPU)={stage_cfg.batch_size}")

        info_if_rank_zero(f"num_workers={stage_cfg.num_workers}")
        stage_cfg.num_workers = max(stage_cfg.num_workers // world_size, 1)
        info_if_rank_zero(f"num_workers(per-GPU)={stage_cfg.num_workers}")

        # -------- Logging: Only main process writes to TensorBoard --------
        log_writer = TensorboardLogger(run_dir, logging.getLogger(), enabled_tb=main_process)

        # -------- DataLoader Factory (Robust with fallback) --------
        def create_safe_dataloader(
            dataset,
            sampler,
            batch_size,
            num_workers,
            *,
            pin_memory=True,
            drop_last=False,
        ):
            try:
                return data.DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                    shuffle=False,
                    drop_last=drop_last,
                    worker_init_fn=lambda wid: np.random.seed(torch.initial_seed() % (2**32)),
                )
            except Exception as e:
                if main_process:
                    log.warning(
                        f"DataLoader failed with num_workers={num_workers}, fallback to 0: {e}"
                    )
                return data.DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=0,
                    pin_memory=False,
                    shuffle=False,
                    drop_last=drop_last,
                )

        def build_loader(mode, *, shuffle, drop_last):
            dataset = TenCamusDataset(
                filepath=cfg.data_path,
                mode=mode,
                seq_length=stage_cfg.seq_length,
                max_num_obj=stage_cfg.num_objects,
                size=stage_cfg.crop_size[0],
            )
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            loader = create_safe_dataloader(
                dataset=dataset,
                sampler=sampler,
                batch_size=stage_cfg.batch_size,
                num_workers=stage_cfg.num_workers,
                pin_memory=True,
                drop_last=drop_last,
            )
            return loader, sampler

        # -------- Dataset / Sampler / DataLoader --------
        train_loader, train_sampler = build_loader("train", shuffle=True, drop_last=True)
        val_loader, val_sampler = build_loader("val", shuffle=False, drop_last=False)
        test_loader, test_sampler = build_loader("test", shuffle=False, drop_last=False)

        # -------- Trainer --------
        trainer = Trainer(
            cfg=cfg,
            stage_cfg=stage_cfg,
            log=log_writer,
            run_path=run_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        total_iterations = int(stage_cfg.num_iterations)
        steps_per_epoch = len(train_loader)
        max_epoch = math.ceil(total_iterations / max(steps_per_epoch, 1))
        info_if_rank_zero(f"Total iterations: {total_iterations}")
        info_if_rank_zero(f"train_loader length (batches per epoch): {steps_per_epoch}")
        info_if_rank_zero(f"Total iteration={total_iterations}, est. epochs={max_epoch} ...")

        # Display progress bar only on main process
        pbar = tqdm(total=total_iterations, desc="Training", ncols=120) if main_process else None

        save_enabled = getattr(cfg, "save", 0) == 1
        weights_interval = getattr(cfg, "save_weights_interval", 0)
        checkpoint_interval = getattr(cfg, "save_checkpoint_interval", 0)
        eval_interval = getattr(cfg.eval_stage, "eval_interval", 0)

        # -------- Iteration-based Training Loop --------
        data_iter = None
        for it in range(total_iterations):
            epoch = it // steps_per_epoch
            # Set seed and rebuild iterator at the start of each epoch
            if it % steps_per_epoch == 0:
                train_sampler.set_epoch(epoch)
                data_iter = iter(train_loader)

            try:
                batch_data = next(data_iter)
            except StopIteration:
                # Fallback safety mechanism (rarely triggered)
                train_sampler.set_epoch(epoch)
                data_iter = iter(train_loader)
                batch_data = next(data_iter)

            loss_val = trainer.do_pass(batch_data, it)

            # ----- Save (Main process only), barrier used to prevent concurrency issues -----
            if save_enabled and it > 0:
                if weights_interval and it % weights_interval == 0:
                    if main_process:
                        trainer.save_weights(it)
                        log.info(f"Weights saved at iteration {it} (epoch {epoch+1})")
                    barrier()

                if checkpoint_interval and it % checkpoint_interval == 0:
                    if main_process:
                        trainer.save_checkpoint(it)
                        log.info(f"Checkpoint saved at iteration {it} (epoch {epoch+1})")
                    barrier()

            # Update progress bar only on main process
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"iter": it + 1, "loss": f"{loss_val:.4f}"})

            # ----- Periodic Evaluation/Testing: Fix eval shard before evaluation -----
            if eval_interval and (it + 1) % eval_interval == 0:
                if pbar is not None:
                    print()
                eval_seed = epoch  # Or fixed to 0
                if isinstance(val_sampler, DistributedSampler):
                    val_sampler.set_epoch(eval_seed)
                if isinstance(test_sampler, DistributedSampler):
                    test_sampler.set_epoch(eval_seed)

                trainer.evaluate(
                    val_loader=val_loader,
                    epoch=epoch + 1,
                    local_rank=local_rank,
                    world_size=world_size,
                    run_path=run_dir,
                    it=it + 1,
                )

                trainer.test(
                    test_loader=test_loader,
                    epoch=epoch + 1,
                    local_rank=local_rank,
                    world_size=world_size,
                    run_path=run_dir,
                    it=it + 1,
                )

        info_if_rank_zero("Training completed.")

    finally:
        # Synchronize all processes before closing resources
        barrier()
        if main_process:
            try:
                wandb.finish()
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    train()