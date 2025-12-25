import os
import math
import random
import logging
import logging.handlers
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF

import SimpleITK as sitk
from thop import profile

# External libraries for metrics only
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.transforms import AsDiscrete

# -----------------------------------------------------------------------------
# 1. Reproducibility & Logging
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Enforces deterministic behavior for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic settings
    cudnn.benchmark = False  # Set to False for strict reproducibility
    cudnn.deterministic = True

def get_logger(name: str, log_dir: str) -> logging.Logger:
    """
    Configures a logger with both file and console handlers.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File Handler
        info_path = os.path.join(log_dir, f'{name}.info.log')
        file_handler = logging.handlers.TimedRotatingFileHandler(
            info_path, when='D', encoding='utf-8'
        )
        formatter = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def log_config_info(config, logger: logging.Logger) -> None:
    """Logs the configuration dictionary."""
    logger.info('#---------- Configuration Summary ----------#')
    for k, v in config.__dict__.items():
        if not k.startswith('_'):
            logger.info(f'{k}: {v}')

def cal_params_flops(model: nn.Module, size: int, logger: logging.Logger) -> None:
    """Calculates and logs FLOPs and Parameters."""
    device = next(model.parameters()).device
    input_tensor = torch.randn(1, 3, size, size).to(device)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    logger.info(f'Input Resolution: {size}x{size}')
    logger.info(f'FLOPs: {flops/1e9:.4f}G')
    logger.info(f'Params: {params/1e6:.4f}M')

# -----------------------------------------------------------------------------
# 2. Optimizer & Scheduler
# -----------------------------------------------------------------------------

def get_optimizer(config, model: nn.Module) -> torch.optim.Optimizer:
    """Factory function for optimizers."""
    if not hasattr(torch.optim, config.opt):
        raise ValueError(f'Unsupported optimizer: {config.opt}')
    
    optim_args = {'lr': config.lr, 'weight_decay': config.weight_decay}
    
    if config.opt in ['Adam', 'AdamW']:
        optim_args.update({'betas': config.betas, 'eps': config.eps, 'amsgrad': config.amsgrad})
    elif config.opt == 'SGD':
        optim_args.update({'momentum': config.momentum, 'nesterov': config.nesterov})
    elif config.opt == 'RMSprop':
        optim_args.update({'momentum': config.momentum, 'alpha': config.alpha, 'eps': config.eps})
    
    optimizer_cls = getattr(torch.optim, config.opt)
    return optimizer_cls(model.parameters(), **optim_args)

def get_scheduler(config, optimizer: torch.optim.Optimizer):
    """Factory function for learning rate schedulers."""
    
    if config.sch == 'WP_MultiStepLR':
        def warm_up_multi_step(epoch):
            if epoch <= config.warm_up_epochs:
                return epoch / config.warm_up_epochs
            milestones_passed = len([m for m in config.milestones if m <= epoch])
            return config.gamma ** milestones_passed
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_multi_step)
    
    elif config.sch == 'WP_CosineLR':
        def warm_up_cosine(epoch):
            if epoch <= config.warm_up_epochs:
                return epoch / config.warm_up_epochs
            progress = (epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs)
            return 0.5 * (math.cos(progress * math.pi) + 1)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_cosine)

    elif hasattr(torch.optim.lr_scheduler, config.sch):
        scheduler_cls = getattr(torch.optim.lr_scheduler, config.sch)
        if config.sch == 'StepLR':
            return scheduler_cls(optimizer, step_size=config.step_size, gamma=config.gamma)
        elif config.sch == 'MultiStepLR':
            return scheduler_cls(optimizer, milestones=config.milestones, gamma=config.gamma)
        elif config.sch == 'CosineAnnealingLR':
            return scheduler_cls(optimizer, T_max=config.T_max, eta_min=config.eta_min)
        elif config.sch == 'ReduceLROnPlateau':
            return scheduler_cls(optimizer, mode=config.mode, factor=config.factor, 
                                 patience=config.patience, min_lr=config.min_lr)
    
    raise ValueError(f'Unsupported scheduler: {config.sch}')

# -----------------------------------------------------------------------------
# 3. Loss Functions
# -----------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Standard Dice Coefficient Loss.
    Expects logits as input (applies sigmoid internally).
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.contiguous().view(pred.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class BceDiceLoss(nn.Module):
    """
    Weighted combination of BCE and Dice Loss.
    Uses BCEWithLogitsLoss for numerical stability.
    """
    def __init__(self, wb: float = 1.0, wd: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss() 
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.wb * self.bce(pred, target) + self.wd * self.dice(pred, target)

class GT_BceDiceLoss(nn.Module):
    """
    Deep Supervision Loss combining BceDiceLoss at multiple scales.
    """
    def __init__(self, wb: float = 1.0, wd: float = 1.0):
        super().__init__()
        self.criterion = BceDiceLoss(wb, wd)
        self.weights = [0.5, 0.4, 0.3, 0.2, 0.1] 

    def forward(self, preds: List[torch.Tensor], out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        main_loss = self.criterion(out, target)
        deep_loss = 0.0
        
        for i, pred in enumerate(preds):
            if i < len(self.weights):
                if pred.shape[2:] != target.shape[2:]:
                    pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
                deep_loss += self.criterion(pred, target) * self.weights[-(i+1)]
                
        return main_loss + deep_loss

# -----------------------------------------------------------------------------
# 4. Data Transforms
# -----------------------------------------------------------------------------

class MyToTensor:
    """Converts numpy arrays to FloatTensors and permutes axes."""
    def __call__(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1).float(), torch.tensor(mask).permute(2, 0, 1).float()

class MyResize:
    def __init__(self, size: Tuple[int, int] = (256, 256)):
        self.size = size
    
    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = data
        return TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR), \
               TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

class MyRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        return image, mask

class MyRandomRotation:
    def __init__(self, p: float = 0.5, degrees: Tuple[int, int] = (0, 360)):
        self.p = p
        self.degrees = degrees
    
    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(*self.degrees)
            return TF.rotate(image, angle), TF.rotate(mask, angle)
        return image, mask

class MyNormalize:
    """Standard Z-Score Normalization."""
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        img, msk = data
        img_norm = (img - self.mean) / (self.std + 1e-8)
        return img_norm, msk

# -----------------------------------------------------------------------------
# 5. Inference & Metrics
# -----------------------------------------------------------------------------

def test_single_volume(image: torch.Tensor, label: torch.Tensor, model: nn.Module, 
                       classes: int, patch_size: List[int] = [256, 256], 
                       save_path: Optional[str] = None, case: Optional[str] = None, 
                       z_spacing: int = 1, batch_size: int = 32) -> List[Tuple[float, float]]:
    """
    Performs inference on a 3D volume using 2D batch processing and calculates MONAI metrics.
    """
    image_np = image.squeeze(0).cpu().detach().numpy() # (S, H, W)
    label_np = label.squeeze(0).cpu().detach().numpy() # (S, H, W)
    
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Batch Inference
    prediction_logits = []
    
    if len(image_np.shape) == 3:
        input_tensor = torch.from_numpy(image_np).unsqueeze(1).float().to(device) # (S, 1, H, W)
        
        target_h, target_w = patch_size[0], patch_size[1]
        orig_h, orig_w = input_tensor.shape[2], input_tensor.shape[3]

        with torch.no_grad():
            for i in range(0, input_tensor.shape[0], batch_size):
                batch_img = input_tensor[i : i + batch_size]
                
                # Pre-processing resize
                if (orig_h, orig_w) != (target_h, target_w):
                    batch_img = F.interpolate(batch_img, size=(target_h, target_w), 
                                              mode='bilinear', align_corners=False)
                
                logits = model(batch_img)
                
                # Post-processing resize
                if (orig_h, orig_w) != (target_h, target_w):
                    logits = F.interpolate(logits, size=(orig_h, orig_w), 
                                           mode='bilinear', align_corners=False)
                
                prediction_logits.append(logits)
        
        full_logits = torch.cat(prediction_logits, dim=0) # (S, C, H, W)
        
    else:
        # Fallback for 2D inputs
        input_t = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            full_logits = model(input_t)

    # 2. Metric Calculation (MONAI)
    full_pred_mask = torch.argmax(full_logits, dim=1).cpu().numpy()
    
    # Prepare data for MONAI: (Batch, Channel, Spatial...)
    y_pred_vol = full_logits.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, S, H, W)
    y_gt_vol = torch.from_numpy(label_np).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, S, H, W)

    post_pred = AsDiscrete(argmax=True, to_onehot=classes)
    post_label = AsDiscrete(to_onehot=classes)
    
    y_pred_onehot = post_pred(y_pred_vol)
    y_gt_onehot = post_label(y_gt_vol)
    
    metric_list = []
    
    for c in range(1, classes):
        sub_pred = y_pred_onehot[:, c:c+1, ...]
        sub_gt = y_gt_onehot[:, c:c+1, ...]
        
        # Dice
        dice_score = compute_dice(y_pred=sub_pred, y=sub_gt, include_background=True).item()
        
        # Hausdorff Distance (HD95)
        if sub_pred.sum() > 0 and sub_gt.sum() > 0:
            hd95_score = compute_hausdorff_distance(
                y_pred=sub_pred, y=sub_gt, include_background=True, percentile=95
            ).item()
        elif sub_pred.sum() == 0 and sub_gt.sum() == 0:
            hd95_score = 0.0
        else:
            hd95_score = 100.0

        metric_list.append((dice_score, hd95_score))

    # 3. Save Results (NIfTI)
    if save_path is not None:
        img_itk = sitk.GetImageFromArray(image_np.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(full_pred_mask.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label_np.astype(np.float32))
        
        spacing = (1.0, 1.0, float(z_spacing))
        img_itk.SetSpacing(spacing)
        prd_itk.SetSpacing(spacing)
        lab_itk.SetSpacing(spacing)
        
        sitk.WriteImage(prd_itk, os.path.join(save_path, f"{case}_pred.nii.gz"))
        sitk.WriteImage(img_itk, os.path.join(save_path, f"{case}_img.nii.gz"))
        sitk.WriteImage(lab_itk, os.path.join(save_path, f"{case}_gt.nii.gz"))
        
    return metric_list

def save_imgs(img: torch.Tensor, msk: torch.Tensor, msk_pred: torch.Tensor, 
              i: int, save_path: str, datasets: str, threshold: float = 0.5, 
              test_data_name: Optional[str] = None) -> None:
    """Visualizes and saves prediction results."""
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
    # Min-Max Normalize for visualization
    if img.min() < 0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
    msk = np.squeeze(msk, axis=0)
    msk_pred = np.squeeze(msk_pred, axis=0)

    if datasets != 'retinal':
        msk = (msk > 0.5).astype(int)
        msk_pred = (msk_pred > threshold).astype(int)

    plt.figure(figsize=(7, 15))
    
    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Image')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')
    plt.title('Ground Truth')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')
    plt.title('Prediction')

    filename = f"{test_data_name}_{i}.png" if test_data_name else f"{i}.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.close()