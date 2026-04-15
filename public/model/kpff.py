import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KPFF"]


class KPFF(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, linear_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.layer2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)
        self.gate_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, key_DCHW: torch.Tensor, pixfeat_DCHW: torch.Tensor) -> torch.Tensor:
        pixfeat_processed = self.layer2(F.relu(self.layer1(pixfeat_DCHW)))
        key = key_DCHW.clone()
        global_key_features = self.global_pool(key_DCHW)
        global_key_features = global_key_features.view(global_key_features.size(0), -1)
        global_key_features = global_key_features.unsqueeze(-1).unsqueeze(-1)
        global_key_features = global_key_features.expand(-1, -1, key_DCHW.size(2), key_DCHW.size(3))
        key = key + global_key_features
        gate_weight = torch.sigmoid(self.gate_layer(key))
        fused_DCHW = key * gate_weight + pixfeat_processed * (1 - gate_weight)
        return fused_DCHW
