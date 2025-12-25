import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optimization 1: Utilize JIT compilation to accelerate recurrent update steps 
@torch.jit.script
def recurrent_update(
    h_BNCHW: torch.Tensor, 
    values_BNCHW: torch.Tensor
) -> torch.Tensor:
    """
    """
    # Optimization 2: Employ `torch.chunk` instead of index slicing for improved 
    # code clarity and to facilitate backend optimization.
    chunks = torch.chunk(values_BNCHW, 3, dim=2)
    forget_gate_BNCHW = torch.sigmoid(chunks[0])
    update_gate_BNCHW = torch.sigmoid(chunks[1])
    new_values_BNCHW = torch.tanh(chunks[2])
    
    new_h_BNCHW = forget_gate_BNCHW * h_BNCHW * (1. - update_gate_BNCHW) + \
        update_gate_BNCHW * new_values_BNCHW

    return new_h_BNCHW


class KPFF(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, linear_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.layer2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)
        self.gate_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, key_DCHW: torch.Tensor, pixfeat_DCHW: torch.Tensor) -> torch.Tensor:
        """
        Args:
            key_DCHW: [B, C, H, W] Feature map 1
            pixfeat_DCHW: [B, C, H, W] Feature map 2
        Returns:
            fused_DCHW: [B, output_dim, H, W] Fused feature map
        """
        pixfeat_processed = self.layer2(F.relu(self.layer1(pixfeat_DCHW)))

        # Optimization 3: Eliminate unnecessary .clone() operations and manual expansion.
        # Leverage PyTorch's broadcasting mechanism for direct addition of (B, C, H, W) and (B, C, 1, 1).
        global_key_features = self.global_pool(key_DCHW)
        
        # If in-place addition is unsafe, use: key = key_DCHW + global_key_features
        # Direct addition is used here to avoid the memory overhead of an additional clone.
        key = key_DCHW + global_key_features

        gate_weight = torch.sigmoid(self.gate_layer(key))

        fused_DCHW = key * gate_weight + pixfeat_processed * (1 - gate_weight)

        return fused_DCHW

class KeyProj(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.layer2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(self.layer1(x))
        return x


class PixProj(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()        
        self.layer = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x

class CAResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

        t = int((abs(math.log2(out_dim)) + 1) // 2)
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        if self.residual:
            if in_dim == out_dim:
                self.downsample = nn.Identity()
            else:
                self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))

        b, c = x.shape[:2]
        
        # Optimization 4: Simplify dimension transformation operations by using `view`
        # instead of `transpose` to avoid non-contiguous memory issues.
        w = self.pool(x).view(b, 1, c)
        w = self.conv(w)
        w = w.view(b, c, 1, 1).sigmoid() # Changed from transpose+unsqueeze to view

        if self.residual:
            x = x * w + self.downsample(r)
        else:
            x = x * w

        return x


class FeatureFuser(nn.Module):
    def __init__(self, input_x_dim: int, input_y_dim: int, out_dim: int):
        super().__init__()
        self.x_transform = nn.Conv2d(input_x_dim, out_dim, kernel_size=1)
        self.y_transform = nn.Conv2d(input_y_dim, out_dim, kernel_size=1)

        self.layer1 = CAResBlock(out_dim, out_dim)
        self.layer2 = CAResBlock(out_dim, out_dim)

    def forward(self, x_BCHW: torch.Tensor, y_BNCHW: torch.Tensor) -> torch.Tensor:
        """ Fuse pixel and value features 
        Args:
            x_BCHW: pixel feature 
            y_BNCHW: value feature
        """
        B, N = y_BNCHW.shape[:2]

        x_BCHW = self.x_transform(x_BCHW)
        x_BNCHW = x_BCHW.unsqueeze(1).expand(-1, N, -1, -1, -1)
        x_DCHW = x_BNCHW.flatten(start_dim=0, end_dim=1)

        y_DCHW = y_BNCHW.flatten(start_dim=0, end_dim=1)
        y_DCHW = self.y_transform(y_DCHW)

        # Add x and y
        fused_DCHW = x_DCHW + y_DCHW
        fused_DCHW = self.layer2(self.layer1(fused_DCHW))
        fused_BNCHW = fused_DCHW.view(B, N, *fused_DCHW.shape[1:])

        return fused_BNCHW


class PixelFuser(nn.Module):
    """ Fuse pixel feature with value readout, as well as sensory.
    """
    def __init__(self, pix_dim, embed_dim, value_dim, sensory_dim):
        super().__init__()
        self.fuser = FeatureFuser(pix_dim, embed_dim, value_dim)
        self.conv = nn.Conv2d(sensory_dim + 2, value_dim, kernel_size=1)

    def forward(
        self, 
        pixfeat_BCHW: torch.Tensor, 
        readout_BNCHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        masks_BNHW: torch.Tensor,
        masks_others_BNHW: torch.Tensor,
        *,
        chunk_size: int=-1,
    ) -> torch.Tensor:      
        N = masks_BNHW.shape[1]        
        chunk_size = N if chunk_size < 0 else chunk_size
        masks_BN2HW = torch.stack([masks_BNHW, masks_others_BNHW], dim=2)

        # chunk-by-chunk inference
        X_all = []
        for i in range(0, N, chunk_size):
            sensory_chunk_BNCHW = sensory_BNCHW[:, i:i+chunk_size]
            masks_chunk_BN2HW = masks_BN2HW[:, i:i+chunk_size]
            
            sensory_chunk_BNCHW = torch.cat(
                [sensory_chunk_BNCHW, masks_chunk_BN2HW], dim=2)

            B, M = sensory_chunk_BNCHW.shape[:2]
            sensory_chunk_DCHW = sensory_chunk_BNCHW.flatten(start_dim=0, end_dim=1)
            sensory_chunk_DCHW = self.conv(sensory_chunk_DCHW)
            sensory_chunk_BNCHW = \
                sensory_chunk_DCHW.view(B, M, *sensory_chunk_DCHW.shape[1:])

            X_BNCHW = readout_BNCHW[:, i:i + chunk_size] + sensory_chunk_BNCHW
            X_BNCHW = self.fuser(pixfeat_BCHW, X_BNCHW)
            X_all.append(X_BNCHW)
        fused = torch.cat(X_all, dim=1)

        return fused


class MultiscaleSensoryUpdater(nn.Module):
    """ Used in the mask decoder with multi-scale feature and GRU.
    """
    def __init__(self, g_dims: List[int], mid_dim: int, sensory_dim: int):
        super().__init__()
        self.g16_conv = nn.Conv2d(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = nn.Conv2d(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = nn.Conv2d(g_dims[2] + 1, mid_dim, kernel_size=1)
        self.transform = nn.Conv2d(
            mid_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def downsample(
            self,
            input_BCHW: torch.Tensor,
            ratio: float,
            mode: str = 'area',
            align_corners: bool = None
    ) -> torch.Tensor:
        """
        """
        output_BCHW = F.interpolate(
            input_BCHW, scale_factor=ratio, mode=mode, align_corners=align_corners)
        return output_BCHW

    def forward(self, g: List[torch.Tensor], h_BNCHW: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: multiscale features list
            h_BNCHW: sensory

        Returns:
            None
        """
        B, N = g[0].shape[:2]
        
        # Optimization 5: List comprehension.
        g16_DCHW, g8_DCHW, g4_DCHW = [x.flatten(start_dim=0, end_dim=1) for x in g]

        # Muti-scale fusion
        g_DCHW = self.g16_conv(g16_DCHW) + \
                 self.g8_conv(self.downsample(g8_DCHW, ratio=1 / 2)) + \
                 self.g4_conv(self.downsample(g4_DCHW, ratio=1 / 4))

        h_DCHW = h_BNCHW.flatten(start_dim=0, end_dim=1)
        values_DCHW = self.transform(torch.cat([g_DCHW, h_DCHW], dim=1))
        values_BNCHW = values_DCHW.view(B, N, *values_DCHW.shape[1:])
        
        # Optimization 6: Remove .clone() and pass the tensor directly.
        new_h_BNCHW = recurrent_update(h_BNCHW, values_BNCHW)

        return new_h_BNCHW


class SensoryUpdater(nn.Module):
    """ Used in the mask encoder for deep update.
    """
    def __init__(self, value_dim: int, sensory_dim: int):
        super().__init__()
        self.transform = nn.Conv2d(
            value_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)
        nn.init.xavier_normal_(self.transform.weight)

    # Note: This redundant method definition is retained as per requirements.
    def recurrent_update(
        self, h_BNCHW: torch.Tensor, values_BNCHW: torch.Tensor
    ) -> torch.Tensor:
        dim = values_BNCHW.shape[2] // 3
        forget_gate_BNCHW = torch.sigmoid(values_BNCHW[:, :, :dim])
        update_gate_BNCHW = torch.sigmoid(values_BNCHW[:, :, dim:dim*2])
        new_values_BNCHW = torch.tanh(values_BNCHW[:, :, dim*2:])
        new_h_BNCHW = forget_gate_BNCHW * h_BNCHW * (1. - update_gate_BNCHW) + \
            update_gate_BNCHW * new_values_BNCHW

        return new_h_BNCHW

    def forward(self, g_BNCHW: torch.Tensor, h_BNCHW: torch.Tensor) -> torch.Tensor:
        B, N = g_BNCHW.shape[:2]

        B, N_g = g_BNCHW.shape[:2]
        B, N_h = h_BNCHW.shape[:2]
        if N_h == 1 and N_h < N_g:
            # Expand h if it has single object dimension but g has multiple.
            h_BNCHW = h_BNCHW.expand(B, N_g, -1, -1, -1)
        
        gh_BNCHW = torch.cat([g_BNCHW, h_BNCHW], dim=2)
        gh_DCHW = gh_BNCHW.flatten(start_dim=0, end_dim=1)
        values_DCHW = self.transform(gh_DCHW)
        values_BNCHW = values_DCHW.view(B, N, *values_DCHW.shape[1:])
        
        # Optimization 7: Remove .clone() and pass h_BNCHW directly.
        # Note: The global `recurrent_update` is called here, not the internal class method.
        new_h_BNCHW = recurrent_update(h_BNCHW, values_BNCHW)

        return new_h_BNCHW