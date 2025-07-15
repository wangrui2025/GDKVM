import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from typing import Optional

def fft_conv2d(
    x: torch.Tensor,    # [B, C, H, W]
    k: torch.Tensor,    # [C, H, W] 或 [C, Hk, Wk]，视情况而定
    dropout_mask: Optional[torch.Tensor] = None,
    gelu: bool = True,
    residual: bool = True,
):
    """
    使用 2D FFT 对张量 x 做卷积 (深度/通道逐点或与 k 相同通道数的情况)。
    可以根据需求修改剩余连接 (residual) 以及激活函数 (gelu) 等逻辑。

    Args:
        x: [B, C, H, W]
        k: [C, Hk, Wk]，如果需要“全尺寸”卷积，一般 Hk = H，Wk = W；或者你也可以在外部手动 pad。
        dropout_mask: shape = [B, C], 用于通道级的 dropout 掩码（可选）。
        gelu: 是否使用 gelu 激活（可选）。
        residual: 是否加入 x 的残差（可选）。

    Returns:
        out: [B, C, H, W]，和 x 同尺寸
    """

    B, C, H, W = x.shape
    # 假设 k.shape = [C, Hk, Wk]
    Hk, Wk = k.shape[-2], k.shape[-1]

    # 选择一个至少能覆盖 H+Hk-1, W+Wk-1 的 FFT 尺寸
    # 为简单，这里直接取 2 倍大小，也可根据需求/效率微调
    fft_height = 2 * max(H, Hk)
    fft_width = 2 * max(W, Wk)

    # 先对核 k 做 zero-pad
    # 如果只需要逐通道深度卷积 (depthwise)，通常 k 的 shape 是 [C, something, something]
    # 这里为了简单，假设“逐通道”/深度可分离的场景
    k_f = torch.fft.rfft2(k.to(x.dtype), s=(fft_height, fft_width))  # [C, fft_height, fft_width//2+1]
    # 注意：除以 fft_height * fft_width 这一步放到后面与 x_f 相乘之后也可以做 (等价)，
    # 也可以提前除，细节因实现风格而异
    
    # 对输入 x 做 FFT
    x_f = torch.fft.rfft2(x, s=(fft_height, fft_width))  # [B, C, fft_height, fft_width//2+1]

    # 逐通道点乘
    # x_f: [B, C, Hf, Wf], k_f: [C, Hf, Wf] -> broadcasting
    y_f = x_f * k_f.unsqueeze(0)
    # 逆变换
    # irfft2 输出大小固定为 s=(fft_height, fft_width)，然后再裁剪回原大小
    y = torch.fft.irfft2(
        y_f,
        s=(fft_height, fft_width),
        norm="forward"  # 或者根据需要 "backward"/"forward"
    )
    # 截取到原图大小
    # 如果要做 full-convolution 或 same-convolution，可以根据需求裁剪
    y = y[..., :H, :W]  # [B, C, H, W]

    # 可选：加残差 + 激活
    if residual:
        out = y + x
    else:
        out = y
    if gelu:
        out = F.gelu(out)

    # 如果需要通道级别的 dropout_mask
    # dropout_mask: [B, C], broadcasting 到 [B, C, 1, 1]
    if dropout_mask is not None:
        out = out * dropout_mask.unsqueeze(-1).unsqueeze(-1)

    return out


class ShortConvolution2D(nn.Conv2d):
    """
    用于图像的短卷积 (深度可分离)，类似原先的 nn.Conv1d 版本。
    默认 groups = hidden_size，实现逐通道卷积。
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        bias: bool = False,
        activation: Optional[str] = 'silu',
    ):
        # 这里的 padding 策略可根据自己需求调整。
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,     # 保持与 hidden_size 相同，实现深度可分离
            bias=bias,
            padding=kernel_size // 2  # 做 same 卷积
        )

        self.hidden_size = hidden_size
        self.activation = None
        if activation is not None:
            if activation not in ['silu', 'swish', 'relu', 'gelu']:
                raise ValueError(f"Activation `{activation}` not supported yet.")
            self.activation = activation

    def forward(
        self,
        x: torch.Tensor,               # [B, C, H, W]
        mask: Optional[torch.Tensor] = None,  # 可选的 mask
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            mask: [B, H, W] 或者 [B, 1, H, W]，表示哪些位置是有效的。
                  也可以按需广播到 [B, C, H, W]。

        Returns:
            out: [B, C, H, W]
        """
        if mask is not None:
            # 假设 mask 形状兼容 [B, 1, H, W] 或 [B, C, H, W]
            x = x * mask

        # 普通 2D 卷积
        out = self._conv_forward(x, self.weight, self.bias)

        # activation
        if self.activation == 'silu' or self.activation == 'swish':
            out = F.silu(out)
        elif self.activation == 'relu':
            out = F.relu(out)
        elif self.activation == 'gelu':
            out = F.gelu(out)

        return out

class LongConvolution2D(nn.Module):
    """
    在图像上做 “长卷积”：
    - 直接学一个 [C, H, W] 或 [C, H_max, W_max] 的卷积核（取决于最大分辨率）
    - 前向时使用 2D FFT 做卷积
    """

    def __init__(
        self,
        hidden_size: int,
        max_h: int,
        max_w: int,
    ):
        """
        Args:
            hidden_size: 通道数 C
            max_h, max_w: 最大高宽 (假设训练时图像不会超过这个大小)
        """
        super().__init__()
        # 这里的 filter 尺寸是 [C, max_h, max_w]
        # 如果你想做更灵活的情况，比如每个通道学不同大小，也可以再细化
        self.hidden_size = hidden_size
        self.filter = nn.Parameter(
            torch.randn(self.hidden_size, max_h, max_w),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            y: [B, C, H, W]
        """
        # 直接用 2D FFT 卷积
        y = fft_conv2d(x, self.filter, dropout_mask=None, gelu=False, residual=True)
        return y

import math

class PositionalEmbedding2D(nn.Module):
    """
    2D 的位置编码举例。最简单可以把 (x/H, y/W) 做若干三角函数展开。
    下面的写法只是一个示例，可根据需要自行设计。
    """

    def __init__(self, emb_dim: int, max_h: int, max_w: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_h = max_h
        self.max_w = max_w

        # 预先存一个 2D 网格: [H, W, 2] -> (row, col)
        # 然后再做一些频率展开
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, max_h),
            torch.linspace(0, 1, max_w),
            indexing='ij'
        )
        # grid: [max_h, max_w, 2]
        grid = torch.stack([grid_y, grid_x], dim=-1)  # [H, W, 2]
        # 这里只做基础的 (x, y)，若要更多频率可自行扩展
        # 例如加 sin(2pi k x)、cos(...) 等
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, h: int, w: int):
        """
        只返回前 h,w 的位置编码。形状: [h, w, something]
        """
        # 这里直接返回 self.grid[:h, :w] 作为最简单的 “位置编码”
        # 如果要做额外的线性/非线性映射，也可在此添加
        return self.grid[:h, :w]


class ImplicitLongConvolution2D(nn.Module):
    """
    使用 MLP 隐式生成 2D 卷积核，然后用 FFT 做卷积
    """

    def __init__(
        self,
        hidden_size: int,   # 通道数 C
        max_h: int,
        max_w: int,
        d_emb: int = 4,     # 位置编码维度 (示例)
        d_hidden: int = 16, # MLP 隐藏层大小
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_emb = d_emb

        # 2D 位置编码
        self.pos_emb = PositionalEmbedding2D(d_emb, max_h, max_w)

        # 一个简单的 MLP： (d_emb) -> (d_hidden) -> (C)
        # 注意，这里希望输出是 [C]，但实际上要生成 2D 卷积核 [C, h, w]，
        # 所以我们 MLP 需要对网格上的每个像素独立映射，然后再拼起来
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_emb, d_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(d_hidden, hidden_size),
        )

    def generate_filter(self, h: int, w: int):
        """
        根据输入的 h, w 动态生成 [C, h, w] 的卷积核
        """
        # 位置编码: [h, w, d_emb]
        pe = self.pos_emb(h, w)  # [h, w, 2] (或更多)
        # reshape: [h*w, d_emb]
        pe_flat = pe.view(-1, self.d_emb)

        # 过 MLP: [h*w, hidden_size]
        out_flat = self.mlp(pe_flat)  # [h*w, C]

        # reshape 回 [C, h, w]
        k = out_flat.transpose(0, 1).reshape(self.hidden_size, h, w)
        return k

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # 生成对应大小的卷积核 k: [C, H, W]
        k = self.generate_filter(H, W)

        # 用 fft_conv2d 卷积
        y = fft_conv2d(
            x, k,
            dropout_mask=None,
            gelu=False,
            residual=True
        )
        return y
