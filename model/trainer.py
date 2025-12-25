import os
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import wandb
from omegaconf import DictConfig

from utils.logger import TensorboardLogger
from utils.log_integrator import Integrator
from utils.time_estimator import TimeEstimator
from model.gdkvm01 import GDKVM
from model.utils.parameter_groups import get_parameter_groups
from model.losses import LossComputer
from vis.vis_0730 import visualize_sequence

from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
)

log = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        stage_cfg: DictConfig,
        log: TensorboardLogger,
        run_path: str,
        train_loader,
        val_loader,
        test_loader,
    ):
        self.cfg = cfg
        self.stage_cfg = stage_cfg
        self.log = log
        self.run_path = Path(run_path)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.exp_id = cfg["exp_id"]
        self.stage = stage_cfg["name"]
        self.use_amp = stage_cfg.amp
        self.crop_size = stage_cfg["crop_size"]

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")

        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.main_process = self.rank == 0

        model = GDKVM().to(self.device)
        model = model.to(memory_format=torch.channels_last)

        self.model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        if self.main_process:
            try:
                wandb.watch(self.model, log="all", log_freq=100)
            except Exception:
                pass

            try:
                param_count = sum(p.nelement() for p in self.model.parameters()) / 1e6
                self.log.info(f"Model Parameters: {param_count:.2f}M")
            except Exception:
                self.log.info("Model Parameters: Count failed")

        self.train_integrator = Integrator(self.log, distributed=True)
        self._is_train = True

        parameter_groups = get_parameter_groups(
            self.model, stage_cfg, print_log=self.main_process
        )
        self.optimizer = optim.AdamW(
            parameter_groups,
            lr=stage_cfg["learning_rate"],
            weight_decay=stage_cfg["weight_decay"],
            eps=1e-6 if self.use_amp else 1e-8,
            foreach=True,
        )
        self.loss_computer = LossComputer(cfg, stage_cfg)
        self.scaler = torch.amp.GradScaler(
            "cuda", init_scale=8192, enabled=self.use_amp
        )
        self.clip_grad_norm = stage_cfg["clip_grad_norm"]

        self._init_scheduler(stage_cfg)

        self.log_text_interval = cfg.get("log_text_interval", 100)
        self.log_image_interval = cfg.get("log_image_interval", 500)
        if cfg.get("debug", False):
            self.log_text_interval = self.log_image_interval = 1

        self.log.time_estimator = TimeEstimator(
            stage_cfg.get("num_iterations", 3000), self.log_text_interval
        )

        self._init_metrics()

    def _init_scheduler(self, stage_cfg):
        if stage_cfg["lr_schedule"] == "constant":
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda _: 1.0
            )
        elif stage_cfg["lr_schedule"] == "poly":
            total_num_iter = stage_cfg["num_iterations"]
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda x: (1 - (x / total_num_iter)) ** 0.9
            )
        elif stage_cfg["lr_schedule"] == "step":
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                stage_cfg["lr_schedule_steps"],
                stage_cfg["lr_schedule_gamma"],
            )
        else:
            raise NotImplementedError(
                f"Scheduler {stage_cfg['lr_schedule']} not implemented"
            )

    def _init_metrics(self):
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.iou_metric  = MeanIoU(include_background=False, reduction="mean")
        self.hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        self.assd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")

        self.conf_metric_names = [
            "precision",
            "recall",
            "accuracy",
            "specificity",
            "f1 score",
        ]
        self.conf_metric = ConfusionMatrixMetric(
            include_background=False, 
            metric_name=self.conf_metric_names,
            reduction="mean",
        )

    def _move_to_device(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 4:
                    batch[key] = value.to(
                        self.device,
                        memory_format=torch.channels_last,
                        non_blocking=True,
                    )
                elif value.ndim == 5:
                    batch[key] = value.to(
                        self.device,
                        memory_format=torch.channels_last_3d,
                        non_blocking=True,
                    )
                else:
                    batch[key] = value.to(self.device, non_blocking=True)
        return batch

    def _ensure_finite_outputs(self, outputs):
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    outputs[key] = torch.nan_to_num(
                        value, nan=0.0, posinf=1e4, neginf=-1e4
                    )
        return outputs

    def train(self):
        self._is_train = True
        self.model.train()
        return self

    def val(self):
        self._is_train = False
        self.model.eval()
        return self

    def do_pass(self, data, it=0):
        torch.set_grad_enabled(self._is_train)
        self._move_to_device(data)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.model(data)
            out = self._ensure_finite_outputs(out)

            num_objects = out.get("num_objects", [1] * data["rgb"].shape[0])
            data.update(out)

            T = data["rgb"].shape[1]
            supervised_indices = torch.tensor([0, T - 1], device=self.device)

            all_logits_keys = [f"logits_{ti}" for ti in supervised_indices.tolist()]
            if not all(k in data for k in all_logits_keys):
                raise KeyError(
                    f"Missing logits keys. Expected {all_logits_keys}, found {list(data.keys())}"
                )

            new_pred = torch.stack([data[k] for k in all_logits_keys], dim=1)
            new_gt = torch.index_select(data["cls_gt"], 1, supervised_indices)
            data.update({"logits": new_pred, "masks": new_gt})

            losses = self.loss_computer.compute(data, num_objects)
            loss = losses["total_loss"]

        if not torch.isfinite(loss):
            if self.main_process:
                self.log.warning(
                    f"[Trainer] Loss is NaN/Inf at iter {it}, skipping batch."
                )
            return torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)

        self.scaler.scale(loss).backward()

        if self.clip_grad_norm > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        if it % self.log_text_interval == 0:
            loss_val = loss.item()
            self.log.log_scalar("loss", loss_val, it)

            if self.main_process:
                self._wandb_log(losses, loss_val, it)
                if it != 0:
                    self.log.log_scalar("lr", self.scheduler.get_last_lr()[0], it)

        return loss.detach()

    def _wandb_log(self, losses, total_loss, it):
        try:
            log_dict = {
                "loss": total_loss,
                "lr": self.scheduler.get_last_lr()[0],
            }
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    log_dict[k] = v.item()
            wandb.log(log_dict, step=it)
        except Exception:
            pass

    def evaluate(
        self, val_loader, epoch, run_path, it, local_rank=None, world_size=None
    ):
        return self._run_evaluation(val_loader, "val", epoch, run_path, it)

    def test(self, test_loader, epoch, run_path, it, local_rank=None, world_size=None):
        return self._run_evaluation(test_loader, "test", epoch, run_path, it)

    def _reset_metrics(self):
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.hd_metric.reset()
        self.assd_metric.reset()
        self.conf_metric.reset()

    def _run_evaluation(self, data_loader, mode, epoch, run_path, it):
        if self.is_distributed:
            dist.barrier()

        if self.main_process:
            self.log.info(
                f"[{mode.capitalize()}] Iter {it} Epoch {epoch}: Start Evaluation..."
            )

        prev_mode = self.model.training
        self.model.eval()
        self._reset_metrics()

        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                self._move_to_device(batch_data)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = self.model(batch_data)
                    out = self._ensure_finite_outputs(out)

                T = batch_data["rgb"].shape[1]
                mask_keys = [f"masks_{0}", f"masks_{T - 1}"]

                if not all(k in out for k in mask_keys):
                    continue

                masks_first = out[mask_keys[0]]
                masks_last = out[mask_keys[1]]

                if masks_first.shape[1] > 1:
                    masks_first = masks_first[:, 1:2, ...]
                    masks_last = masks_last[:, 1:2, ...]

                gt = batch_data["cls_gt"]
                if gt.dim() == 5:
                    gt = gt.squeeze(2)

                gt_first = gt[:, 0, ...].unsqueeze(1)
                gt_last = gt[:, -1, ...].unsqueeze(1)

                preds_concat = torch.cat([masks_first, masks_last], dim=0)
                gts_concat = torch.cat([gt_first, gt_last], dim=0)

                preds_bin = (preds_concat > 0.5).float()
                gts_bin = (gts_concat > 0.5).float()

                self.dice_metric(y_pred=preds_bin, y=gts_bin)
                self.iou_metric(y_pred=preds_bin, y=gts_bin)
                self.hd_metric(y_pred=preds_bin, y=gts_bin)
                self.assd_metric(y_pred=preds_bin, y=gts_bin)
                self.conf_metric(y_pred=preds_bin, y=gts_bin)

                vis_limit = self.cfg.get("eval_stage", {}).get("num_vis", 0)
                if vis_limit == 0:
                    vis_limit = self.cfg.get("num_vis", 0)

                if self.main_process and batch_idx < vis_limit:
                    self._visualize_batch(batch_data, out, batch_idx, it, epoch, mode)

        local_counts = len(self.dice_metric.get_buffer())

        metrics_map = {
            "dice": self.dice_metric,
            "iou": self.iou_metric,
            "hd95": self.hd_metric,
            "assd": self.assd_metric,
        }

        local_results = {}
        for name, metric in metrics_map.items():
            try:
                res = metric.aggregate()
                if isinstance(res, torch.Tensor):
                    res = res.item()
                local_results[name] = res if np.isfinite(res) else 0.0
            except Exception:
                local_results[name] = 0.0

        try:
            conf_res = self.conf_metric.aggregate()
            conf_names = ["precision", "recall", "acc", "sp", "F1"]
            for i, name in enumerate(conf_names):
                val = conf_res[i].item()
                local_results[name] = val if np.isfinite(val) else 0.0
        except Exception:
            pass

        global_metrics = self._reduce_metrics_weighted(local_results, local_counts)

        if self.main_process:
            self._log_final_metrics(global_metrics, mode, it, epoch)

        if self.is_distributed:
            dist.barrier()

        self.model.train(prev_mode)
        return global_metrics

    def _visualize_batch(self, batch_data, out, batch_idx, it, epoch, mode):
        try:
            rgb_seq = batch_data["rgb"][0].cpu().numpy()
            cls_gt_seq = batch_data["cls_gt"][0].cpu().numpy()

            patient_name = f"b{batch_idx}"
            if "info" in batch_data and "name" in batch_data["info"]:
                patient_name = str(batch_data["info"]["name"][0])

            visualize_sequence(
                rgb_seq,
                cls_gt_seq,
                out,
                str(self.run_path),
                f"vis_idx_{batch_idx}",
                iteration=it,
                epoch=epoch,
                patient_id=patient_name,
                mode=mode,
            )
        except Exception as e:
            self.log.warning(f"Vis failed: {e}")

    def _reduce_metrics_weighted(self, local_metrics: dict, local_count: int):
        if not self.is_distributed:
            return local_metrics

        keys = list(local_metrics.keys())
        local_vec = [float(local_count)]
        for k in keys:
            local_vec.append(local_metrics[k] * local_count)

        tensor_vec = torch.tensor(local_vec, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor_vec, op=dist.ReduceOp.SUM)

        total_count = tensor_vec[0].item()
        global_metrics = {}

        if total_count > 0:
            for i, k in enumerate(keys):
                global_metrics[k] = tensor_vec[i + 1].item() / total_count
        else:
            for k in keys:
                global_metrics[k] = 0.0

        return global_metrics

    def _log_final_metrics(self, metrics, mode, it, epoch):
        log_items = []
        for k, v in metrics.items():
            log_items.append(f"{k.upper()}={v:.4f}")
        
        log_str = f"[{mode.capitalize()}] Iter={it} | " + " | ".join(log_items)
        self.log.info(log_str)

        try:
            w_metrics = {f"{mode}/{k}": v for k, v in metrics.items()}
            w_metrics["epoch"] = epoch
            wandb.log(w_metrics, step=it)
        except Exception:
            pass

    def save_checkpoint(self, it: int):
        if not self.main_process:
            return
        self.run_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.run_path / f"{self.exp_id}_{self.stage}_ckpt_{it}.pth"

        torch.save(
            {
                "it": it,
                "model": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            ckpt_path,
        )
        self.log.info(f"Saved checkpoint: {ckpt_path}")

    def load_checkpoint(self, path: str):
        self.log.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        return ckpt["it"]