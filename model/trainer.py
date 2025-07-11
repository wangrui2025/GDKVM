import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np

import wandb
from omegaconf import DictConfig
from utils.logger import TensorboardLogger
from utils.log_integrator import Integrator
from utils.time_estimator import TimeEstimator
import PIL

from model.gdkvm01 import GDKVM
from model.utils.parameter_groups import get_parameter_groups
from model.losses import LossComputer
from metr.dice import dice_coefficient, sespiou_coefficient
from vis.vis0709_1020 import visualize_sequence

class Trainer:
    def __init__(self,
                 cfg: DictConfig,
                 stage_cfg: DictConfig,
                 log: TensorboardLogger,
                 run_path: str,
                 train_loader,
                 val_loader,
                 test_loader):
        self.cfg = cfg
        self.stage_cfg = stage_cfg
        self.log = log
        self.run_path = run_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.exp_id = cfg['exp_id']
        self.stage = stage_cfg['name']
        self.use_amp = stage_cfg.amp

        local_rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        self.model = nn.parallel.DistributedDataParallel(
            GDKVM().cuda(),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False
        )
        self.size = stage_cfg['crop_size']

        if local_rank == 0:
            wandb.watch(self.model, log="all", log_freq=100)

        self.log = log
        self.run_path = run_path
        if local_rank == 0:
            self.log.info(f'model_size: {sum(param.nelement() for param in self.model.parameters())}')
            self.log.info(f'number_of_parameters_that_require_gradient: {sum(param.nelement() for param in filter(lambda p: p.requires_grad, self.model.parameters()))}')
            self.log.info(f'torch version: {torch.__version__}')
            self.log.info(f'PIL version: {PIL.__version__}')

        self.train_integrator = Integrator(self.log, distributed=True)

        self.train()
        parameter_groups = get_parameter_groups(self.model, stage_cfg, print_log=(local_rank == 0))

        self.optimizer = optim.AdamW(
            parameter_groups,
            lr=stage_cfg['learning_rate'],
            weight_decay=stage_cfg['weight_decay'],
            eps=1e-6 if self.use_amp else 1e-8,
            foreach=True)

        self.loss_computer = LossComputer(cfg, stage_cfg)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(init_scale=8192)
        self.clip_grad_norm = stage_cfg['clip_grad_norm']

        if stage_cfg['lr_schedule'] == 'constant':
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
        elif stage_cfg['lr_schedule'] == 'poly':
            total_num_iter = stage_cfg['num_iterations']
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: (1 - (x / total_num_iter))**0.9)
        elif stage_cfg['lr_schedule'] == 'step':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, stage_cfg['lr_schedule_steps'], stage_cfg['lr_schedule_gamma'])
        else:
            raise NotImplementedError

        self.log_text_interval = cfg['log_text_interval']
        self.log_image_interval = cfg['log_image_interval']
        if cfg['debug']:
            self.log_text_interval = self.log_image_interval = 1

        self.log.time_estimator = TimeEstimator(3000, self.log_text_interval)
        self.train_integrator = Integrator(self.log, distributed=True)
        self._is_train = True
        self.clip_grad_norm = stage_cfg['clip_grad_norm']

    def do_pass(self, data, it=0):
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda(non_blocking=True)

        out = self.model(data)

        num_objects = out.get('num_objects', [1] * data['rgb'].shape[0])
        data.update(out)

        T = data['rgb'].shape[1]
        supervised_indices = torch.tensor([0, T - 1], device=data['rgb'].device)
        all_logits_keys = [f'logits_{ti}' for ti in supervised_indices.tolist()]
        
        if not all(k in data for k in all_logits_keys):
            raise KeyError(f"Missing keys in data: {all_logits_keys}")

        new_pred = torch.stack([data[key] for key in all_logits_keys], dim=1)
        new_gt = torch.index_select(data['cls_gt'], 1, supervised_indices)
        data.update({'logits': new_pred, 'masks': new_gt})

        if self.use_amp:
            with torch.amp.autocast(device_type='cuda'):
                losses = self.loss_computer.compute(data, num_objects)
        else:
            losses = self.loss_computer.compute(data, num_objects)
        loss = losses['total_loss']

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"WARNING: NaN or inf detected in loss: {loss}. Skipping batch.")
            return 0.0

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

        self.scheduler.step()

        loss_val = loss.item()
        self.log.log_scalar('loss', loss_val, it)
        if self.local_rank == 0:
            wandb.log({
                "loss": loss_val,
                "lr": self.scheduler.get_last_lr()[0],
                **{k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)}
            }, step=it)

        if self.local_rank == 0 and it % self.log_text_interval == 0 and it != 0:
            lr = self.scheduler.get_last_lr()[0]
            self.log.log_scalar('lr', lr, it)

        return loss_val

    def _extract_first_last_masks(self, out, batch_idx):
        masks_first = None
        masks_last = None

        if isinstance(out, dict):
            masks_keys = sorted([key for key in out.keys() if key.startswith('masks_')], key=lambda x: int(x.split('_')[-1]))
            if masks_keys:
                first_key, last_key = masks_keys[0], masks_keys[-1]
                masks_first, masks_last = out.get(first_key), out.get(last_key)
            else:
                if self.local_rank == 0: self.log.warning(f"[Val] Batch {batch_idx}: No 'masks_X' keys found in model output.")
        elif isinstance(out, torch.Tensor):
            masks_first = masks_last = out
        else:
            if self.local_rank == 0: self.log.warning(f"[Val] Batch {batch_idx}: Unexpected model output type: {type(out)}")

        if masks_first is not None and masks_first.dim() == 4 and masks_first.shape[1] > 1:
            masks_first = masks_first[:, 1, :, :].unsqueeze(1)
            masks_last = masks_last[:, 1, :, :].unsqueeze(1)
        
        if masks_first is None or masks_last is None:
            self.log.warning(f"[Val] Batch {batch_idx}: 'masks_first' or 'masks_last' not found or shape unexpected.")
            return None, None

        return masks_first, masks_last

    def evaluate(self, val_loader, epoch, local_rank, world_size, run_path, it):
        return self._run_evaluation(data_loader=val_loader, mode='val', epoch=epoch, local_rank=local_rank, world_size=world_size, run_path=run_path, it=it)

    def test(self, test_loader, epoch, local_rank, world_size, run_path, it):
        return self._run_evaluation(data_loader=test_loader, mode='test', epoch=epoch, local_rank=local_rank, world_size=world_size, run_path=run_path, it=it)

    def _run_evaluation(self, data_loader, mode, epoch, local_rank, world_size, run_path, it):
        if self.local_rank == 0:
            self.log.info(f"[{mode.capitalize()}] Iter {it} Epoch {epoch}: Evaluating...")

        self.model.eval()

        global_metrics = {"dice": 0.0, "iou": 0.0}
        total_count = 0

        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                metrics = {"dice": 0.0, "iou": 0.0}
                count = 0

                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = value.cuda(non_blocking=True)

                out = self.model(batch_data)
                masks_first, masks_last = self._extract_first_last_masks(out, it)
                if masks_first is None or masks_last is None:
                    self.log.warning(f"[{mode.capitalize()}] Iter {it}: Skipping batch due to missing masks.")
                    continue

                gt_BTCHW = batch_data.get('cls_gt')
                if gt_BTCHW is None or gt_BTCHW.dim() != 5 or gt_BTCHW.shape[2] != 1:
                    self.log.warning(f"[{mode.capitalize()}] Iter {it}: 'cls_gt' shape unexpected or missing.")
                    continue
                
                gt = gt_BTCHW.squeeze(2)
                gt_first = gt[:, 0, :, :]
                gt_last = gt[:, -1, :, :]
                pred_first = (masks_first > 0.5).long()
                pred_last = (masks_last > 0.5).long()

                pred_first_np = pred_first.squeeze(1).cpu().numpy()
                pred_last_np = pred_last.squeeze(1).cpu().numpy()
                gt_first_np = gt_first.cpu().numpy()
                gt_last_np = gt_last.cpu().numpy()

                for i in range(pred_first_np.shape[0]):
                    try:
                        dice_b = dice_coefficient(pred_first_np[i], gt_first_np[i])
                        dice_b_last = dice_coefficient(pred_last_np[i], gt_last_np[i])
                        
                        iou_b = sespiou_coefficient(pred_first_np[i], gt_first_np[i])
                        iou_b_last = sespiou_coefficient(pred_last_np[i], gt_last_np[i])

                        metrics["dice"] += (dice_b + dice_b_last) / 2
                        metrics["iou"] += (iou_b + iou_b_last) / 2
                        count += 1
                    except Exception as e:
                        self.log.error(f"[{mode.capitalize()}] Iter {it}, Sample {i}: Error computing metrics: {e}")
                
                if count > 0:
                    for key in metrics: metrics[key] /= count
                    for k in global_metrics.keys(): global_metrics[k] += metrics[k]
                    total_count += 1

                self.log.debug(f"[Val] Iter {it}: Metrics this batch: {metrics}, Count: {count}")

                if self.local_rank == 0 and batch_idx < self.cfg.eval_stage.num_vis:
                    rgb_seq = batch_data['rgb'][0].cpu().numpy()
                    cls_gt_seq = batch_data['cls_gt'][0].cpu().numpy()
                    visualize_sequence(rgb_seq, cls_gt_seq, out, self.run_path, f"It_{it}_E_{epoch}_{mode}_idx_{batch_idx}")

            if world_size > 1:
                for k in global_metrics.keys():
                    tensor_val = torch.tensor(global_metrics[k], dtype=torch.float32, device='cuda')
                    dist.all_reduce(tensor_val, op=dist.ReduceOp.SUM)
                    global_metrics[k] = tensor_val.item()

                total_count_t = torch.tensor(total_count, dtype=torch.float32, device='cuda')
                dist.all_reduce(total_count_t, op=dist.ReduceOp.SUM)
                total_count_all = int(total_count_t.item())
            else:
                total_count_all = total_count

            if self.local_rank == 0:
                if total_count_all > 0:
                    for k in global_metrics.keys():
                        global_metrics[k] /= total_count_all
                    
                    log_str = f"[{mode.capitalize()}] Iter={it}, "
                    for k, v in global_metrics.items():
                        log_str += f"{k.upper()}={v:.4f}, "
                    self.log.info(log_str[:-2])
                    
                    wandb_metrics = {f"{mode}_{k}": v for k, v in global_metrics.items()}
                    wandb_metrics["epoch"] = epoch
                    wandb.log(wandb_metrics, step=it)

                    for k, v in global_metrics.items():
                        wandb.summary[f"{mode}_{k}"] = v
                else:
                    self.log.warning(f"[{mode.capitalize()}] No valid samples were evaluated.")

        self.model.train()
        return global_metrics

    def save_weights(self, it):
        if self.local_rank != 0: return
        os.makedirs(self.run_path, exist_ok=True)
        save_path = os.path.join(self.run_path, f'{self.exp_id}_{self.stage}_{it}.pth')
        torch.save(self.model.module.state_dict(), save_path)
        self.log.info(f'[Trainer] Weights saved: {save_path}')

    def save_checkpoint(self, it):
        if self.local_rank != 0: return
        os.makedirs(self.run_path, exist_ok=True)
        ckpt = {
            'it': it,
            'weights': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        save_path = os.path.join(self.run_path, f'{self.exp_id}_{self.stage}_ckpt_{it}.pth')
        torch.save(ckpt, save_path)
        self.log.info(f'[Trainer] Checkpoint saved: {save_path}')

    def load_checkpoint(self, path):
        map_loc = f'cuda:{self.local_rank}'
        checkpoint = torch.load(path, map_location=map_loc)
        it = checkpoint['it']
        self.model.module.load_state_dict(checkpoint['weights'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.log.info(f'[Trainer] Loaded checkpoint from {path}, it={it}')
        return it

    def load_weights(self, path):
        map_loc = f'cuda:{self.local_rank}'
        weights = torch.load(path, map_location=map_loc)
        self.model.module.load_state_dict(weights)
        self.log.info(f'[Trainer] Loaded weights from {path}')

    def train(self):
        self._is_train = True
        self.model.train()
        return self

    def val(self):
        self._is_train = False
        self.model.eval()
        return self
