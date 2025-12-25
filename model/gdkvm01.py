import logging
import math
from typing import Tuple, Iterable, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import resnet
from model.aux_modules import AuxComputer
from model.transformer.object_transformer import QueryTransformer
from model.transformer.object_summarizer import ObjectSummarizer
from utils.tensor_utils import aggregate

log = logging.getLogger()

from model.kpff import KPFF, KeyProj, PixProj, PixelFuser, MultiscaleSensoryUpdater, SensoryUpdater, FeatureFuser

class ImageEncoder(nn.Module):
    def __init__(self, encoder_type: str = 'resnet50'):
        super().__init__()
        if encoder_type == 'resnet18':
            network = resnet.resnet18(pretrained=True)
        elif encoder_type == 'resnet50':
            network = resnet.resnet50(pretrained=True)
        else:
            raise NotImplementedError
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (tesnor): shape of B, C, H, W
        Returns:
            f16 (tensor): shape of B, C, H/16, w/16. C={1024, 256} for resnet{50, 18}.
            f8  (tesnor): shape of B, C, H/8, w/8. C={512, 128} for resnet{50, 18}.
            f4  (tensor): shape of B, C, H/4, w/4. C={256, 64} for resnet{50, 18}.
        """
        f2 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f4 = self.layer1(f2)
        f8 = self.layer2(f4)
        f16 = self.layer3(f8)

        return f16, f8, f4


class MaskEncoder(nn.Module):
    def __init__(
        self,
        pix_dim: int,
        value_dim: int,
        sensory_dim: int,
        encoder_type: str = 'resnet18'
    ):
        super().__init__()
        if encoder_type == 'resnet18':
            network = resnet.resnet18(pretrained=True, extra_dim=2)
        elif encoder_type == 'resnet50':
            network = resnet.resnet50(pretrained=True, extra_dim=2)
        else:
            raise NotImplementedError

        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

        embed_dim = {'resnet18': 256, 'resnet50': 1024}[encoder_type]
        self.fuser = FeatureFuser(pix_dim, embed_dim, value_dim)
        self.sensory_updater = SensoryUpdater(value_dim, sensory_dim)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(
        self,
        image_BCHW: torch.Tensor,
        pixfeat_BCHW: torch.Tensor,
        masks_BO2HW: torch.Tensor,
        sensory_BOCHW: torch.Tensor,
        *,
        deep_update: bool = False,
        chunk_size=-1
    ) -> torch.Tensor:
 
        B, N, K, H, W = masks_BO2HW.shape
        assert K == 2, f"Expected K=2 for masks, but got K={K}"

        image_BO1HW = image_BCHW.unsqueeze(1).expand(-1, N, -1, -1, -1)
        input_BO3HW = torch.cat([image_BO1HW, masks_BO2HW], dim=2)

        if chunk_size < 1 or chunk_size > N:
            chunk_size = N

        # Create new sensory container if deep update is required and chunking is used.
        if deep_update and chunk_size != N:
            new_sensory_BOCHW = torch.empty_like(sensory_BOCHW)
        else:
            new_sensory_BOCHW = sensory_BOCHW.clone()

        X_chunks = []
        for i in range(0, N, chunk_size):
            X_BM3HW = input_BO3HW[:, i:i+chunk_size] # M = chunk size
            X_D5HW = X_BM3HW.flatten(start_dim=0, end_dim=1) # D=B*N
            X_DCHW = self.maxpool(self.relu(self.bn1(self.conv1(X_D5HW))))
            X_DCHW = self.layer1(X_DCHW) # 1/4
            X_DCHW = self.layer2(X_DCHW) # 1/8
            X_DCHW = self.layer3(X_DCHW) # 1/16
            X_BMCHW = X_DCHW.view(B, X_BM3HW.shape[1], *X_DCHW.shape[1:])
            X_BMCHW = self.fuser(pixfeat_BCHW, X_BMCHW)
            X_chunks.append(X_BMCHW)

            if deep_update:
                sensory_chunk_BMCHW = sensory_BOCHW[:, i:i+chunk_size]
                new_sensory_BOCHW[:, i:i+chunk_size] = \
                    self.sensory_updater(X_BMCHW, sensory_chunk_BMCHW)

        X_BNCHW = torch.cat(X_chunks, dim=1)

        return X_BNCHW, new_sensory_BOCHW


class MaskUpsampleBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale_factor: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        if in_dim == out_dim:
            self.linear_proj = nn.Identity()
        else:
            self.linear_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        self.scale_factor = scale_factor

    def upsample(
            self, feat: torch.Tensor, ratio: float,
            mode: str = 'bilinear', align_corners: bool = False
    ) -> torch.Tensor:
        B, N = feat.shape[:2]
        feat = F.interpolate(
            feat.flatten(start_dim=0, end_dim=1),
            scale_factor=ratio, mode=mode, align_corners=align_corners)
        feat = feat.view(B, N, *feat.shape[1:])
        return feat

    def forward(self, feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat (tensor): B,N,C,H,W
            skip_feat (tensor): B,C,2H,2W
        """
        B, N = feat.shape[:2]
        feat_BNCHW, skip_BCHW = feat, skip_feat
        feat_BNCHW = self.upsample(feat_BNCHW, ratio=self.scale_factor)
        skip_BNCHW = skip_BCHW.unsqueeze(1).expand(-1, N, -1, -1, -1)
        feat_BNCHW = feat_BNCHW + skip_BNCHW
        feat_DCHW = feat_BNCHW.flatten(start_dim=0, end_dim=1)
        feat_DCHW = self.linear_proj(feat_DCHW) + \
                    self.act2(self.conv2(self.act1(self.conv1(feat_DCHW))))
        feat_BNCHW = feat_DCHW.view(B, N, *feat_DCHW.shape[1:])

        return feat_BNCHW


class MaskDecoder(nn.Module):
    def __init__(self, ms_dims, up_dims, sensory_dim):
        super().__init__()
        self.linear_proj_f8 = nn.Conv2d(ms_dims[1], up_dims[0], kernel_size=1)
        self.linear_proj_f4 = nn.Conv2d(ms_dims[2], up_dims[1], kernel_size=1)
        self.upsample_16_to_8 = MaskUpsampleBlock(up_dims[0], up_dims[1])
        self.upsample_8_to_4 = MaskUpsampleBlock(up_dims[1], up_dims[2])
        self.pred = nn.Conv2d(up_dims[-1], 1, kernel_size=1)

        self.sensory_updater = MultiscaleSensoryUpdater(
            up_dims, sensory_dim, sensory_dim)


    def forward(
        self, 
        ms_feats: Iterable[torch.Tensor],
        readout_BNCHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        *,
        update_sensory: bool=False,
        chunk_size: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = readout_BNCHW.shape[:2]
        v16_BNCHW = readout_BNCHW
        f8_BCHW, f4_BCHW = ms_feats[1:] 
        f8_BCHW = self.linear_proj_f8(f8_BCHW)
        f4_BCHW = self.linear_proj_f4(f4_BCHW)

        chunk_size = N if chunk_size < 1 or chunk_size > N else chunk_size
        
        if update_sensory:
            if chunk_size != N:
                # Create a new tensor for chunked updates
                new_sensory_BNCHW = torch.empty_like(sensory_BNCHW)
            else:
                # Clone to avoid in-place modification errors
                new_sensory_BNCHW = sensory_BNCHW.clone()
        else:
            new_sensory_BNCHW = sensory_BNCHW

        logits_chunks = []
        for i in range(0, N, chunk_size):
            v16_chunk_BMCHW = v16_BNCHW[:, i:i+chunk_size]
            M = v16_chunk_BMCHW.shape[1] 
            v8_chunk_BMCHW = self.upsample_16_to_8(v16_chunk_BMCHW, f8_BCHW)
            v4_chunk_BMCHW = self.upsample_8_to_4(v8_chunk_BMCHW, f4_BCHW)
            v4_chunk_DCHW = v4_chunk_BMCHW.flatten(start_dim=0, end_dim=1)
            logits_chunk_D1HW = self.pred(v4_chunk_DCHW) 
            logits_chunk_BMHW = \
                logits_chunk_D1HW.view(B, M, *logits_chunk_D1HW.shape[-2:])

            if update_sensory:
                logits_chunk_BM1HW = logits_chunk_BMHW.unsqueeze(2)
                v4_chunk_BMCHW = torch.cat([v4_chunk_BMCHW, logits_chunk_BM1HW], dim=2)
                new_sensory_BNCHW[:, i:i+chunk_size] = self.sensory_updater(
                    [v16_chunk_BMCHW, v8_chunk_BMCHW, v4_chunk_BMCHW], 
                    sensory_BNCHW[:, i:i+chunk_size])

            logits_chunks.append(logits_chunk_BMHW)
        logits_BNCHW = torch.cat(logits_chunks, dim=1)

        return logits_BNCHW, new_sensory_BNCHW


class GDKVM(nn.Module):
    def __init__(
            self,
            model_type='base',
            image_encoder_type='resnet50',
            mask_encoder_type='resnet18',
    ) -> None:
        super().__init__()
        self.ms_dims = {
            'resnet50': [1024, 512, 256], 
            'resnet18': [256, 128, 64],
        }[image_encoder_type]

        self.up_dims = {
            'base': [256, 128, 128],
            'small': [256, 128, 64]
        }[model_type]

        self.key_dim = 64
        self.value_dim = 256
        self.pixel_dim = 256
        self.sensory_dim = 256
        self.embed_dim = 256

        # Parameters for the query transformer
        self.num_blocks = 3
        self.num_heads = 8
        self.num_queries = 16
        self.ff_dim = 2048

        self.image_encoder = ImageEncoder(encoder_type=image_encoder_type)
        self.mask_encoder = MaskEncoder(
            self.pixel_dim, self.value_dim, self.sensory_dim, 
            encoder_type=mask_encoder_type)
        
        self.key_projector = KeyProj(self.ms_dims[0], self.pixel_dim, self.key_dim)
        self.pix_projector = PixProj(self.ms_dims[0], self.pixel_dim)
        self.KPFF          = KPFF   (self.ms_dims[0], self.pixel_dim, self.key_dim)
        self.mask_decoder = MaskDecoder(self.ms_dims, self.up_dims, self.sensory_dim)
        self.pixel_fuser = PixelFuser(
            self.pixel_dim, self.embed_dim, self.value_dim, self.sensory_dim)

        self.object_transformer = QueryTransformer(
            self.value_dim, self.embed_dim, self.ff_dim, self.num_blocks,
            self.num_heads, self.num_queries)

        self.object_summarizer = ObjectSummarizer(
            self.value_dim, self.embed_dim, self.num_queries, add_pe=True)

        self.aux_computer = AuxComputer(self.sensory_dim, self.embed_dim)

        self.register_buffer(
            "pixel_mean", torch.Tensor([0.5]).view(-1, 1, 1), False)
        self.register_buffer(
            "pixel_std", torch.Tensor([0.5]).view(-1, 1, 1), False)
        
        self.b_proj = nn.Linear(self.value_dim, self.num_heads, bias=False)
        self.a_proj = nn.Linear(self.value_dim, self.num_heads, bias=False)
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True
        

    def encode_image(
        self, image: torch.Tensor
    ) -> Tuple[Iterable[torch.Tensor], torch.Tensor]:
        
        image = (image - self.pixel_mean) / self.pixel_std
        multiscale_feats = self.image_encoder(image)
        return multiscale_feats

    def encode_mask(
        self, 
        image_BCHW: torch.Tensor, 
        pixfeat_BCHW: torch.Tensor, 
        mask_BNHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        *,
        deep_update: bool=False,
        chunk_size: int=-1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image (tensor): B, C, H, W.
            pixfeat (tensor): B, C, H, W.
            mask (tensor): B, N, H, W. No background in the mask.
        Returns:
            value (tensor): shape of BNCHW.
        """
        image_BCHW = (image_BCHW - self.pixel_mean) / self.pixel_std
        mask_other_BNHW = (mask_BNHW.float().sum(dim=1, keepdim=True) - mask_BNHW.float()).clamp(0, 1)
        masks_BN2HW = torch.stack([mask_BNHW, mask_other_BNHW], dim=2)
        mask_value_BNCHW, sensory_BNCHW = self.mask_encoder(
            image_BCHW, pixfeat_BCHW, masks_BN2HW, sensory_BNCHW, 
            deep_update=deep_update, chunk_size=chunk_size)

        object_memory_BNQC = self.object_summarizer(mask_BNHW, mask_value_BNCHW)

        return mask_value_BNCHW, sensory_BNCHW, object_memory_BNQC

    def pixel_fusion(
        self, 
        pixfeat_BCHW: torch.Tensor, 
        readout_BNCHW: torch.Tensor, 
        sensory_BNCHW: torch.Tensor,
        last_masks_BNHW: torch.Tensor,
    ) -> torch.Tensor:
        masks_BNHW = F.interpolate(
            last_masks_BNHW.float(), size=readout_BNCHW.shape[-2:], mode='area')
        masks_sum_B1HW = masks_BNHW.sum(dim=1, keepdim=True)
        masks_others_BNHW = (masks_sum_B1HW - masks_BNHW).clamp(0, 1)

        readout_fused_BNCHW = self.pixel_fuser(
            pixfeat_BCHW, readout_BNCHW, sensory_BNCHW, masks_BNHW, masks_others_BNHW)
        return readout_fused_BNCHW

    def segment(
        self, 
        ms_feats: Tuple[torch.Tensor], 
        readout_BNCHW: torch.Tensor, 
        pixfeat_BCHW: torch.Tensor,
        last_masks_BNHW: torch.Tensor,
        sensory_BNCHW: torch.Tensor,
        object_memory_BNQC: torch.Tensor,
        *,
        selector: Optional[torch.Tensor] = None,
        update_sensory: bool=False,
        chunk_size: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            ms_feats: multi-scale features extracted from image
            value (tensor): mask value
        """
        readout_BNCHW = self.pixel_fusion(
            pixfeat_BCHW, readout_BNCHW, sensory_BNCHW, last_masks_BNHW)

        # Enhance the readout with object memory
        readout_BNCHW, aux_features = self.object_transformer(
            readout_BNCHW, object_memory_BNQC, selector=selector)

        logits, sensory_BNCHW = self.mask_decoder(
            ms_feats, readout_BNCHW, sensory_BNCHW,
            chunk_size=chunk_size, update_sensory=update_sensory)

        masks = torch.sigmoid(logits)
        if selector is not None:
            masks = masks * selector

        # Softmax over all objects
        logits = aggregate(masks, dim=1)
        logits = F.interpolate(
            logits, scale_factor=4, mode='bilinear', align_corners=False)
        masks = F.softmax(logits, dim=1)

        return logits, masks, sensory_BNCHW, aux_features

    def forward(self, data: Dict):
        out                = {}
        num_objects        = [num.item() for num in data['info']['num_objects']]
        out['num_objects'] = num_objects
        
        max_num_objects = max(num_objects)

        images_BTCHW = data['rgb']
        B, T         = images_BTCHW.shape[:2]
        
        # Optimization: Use channels_last memory format for acceleration.
        images_DCHW  = images_BTCHW.reshape(B*T, *images_BTCHW.shape[2:]) 
        if images_DCHW.device.type == 'cuda':
            images_DCHW = images_DCHW.to(memory_format=torch.channels_last)
        
        ms_feats      = self.encode_image(images_DCHW)

        # Optimization: Reshape features outside the loop.
        ms_feats_reshaped = [f.view(B, T, *f.shape[1:]) for f in ms_feats]

        feat_DCHW     = ms_feats[0]
        pixfeat_DCHW  = self.pix_projector(feat_DCHW)
        pixfeat_BTCHW = pixfeat_DCHW.view(B, T, *pixfeat_DCHW.shape[1:])

        # 其他投影等
        key_DCHW  = self.key_projector(feat_DCHW)
        key_DCHW = self.KPFF(key_DCHW, pixfeat_DCHW)
        key_BTCHW = key_DCHW.view(B, T, *key_DCHW.shape[1:])

        # Extract 0-th frame image and mask.
        first_frame_image_BCHW    = images_BTCHW[:, 0]
        first_frame_pixfeat_BCHW  = pixfeat_BTCHW[:, 0]

        first_frame_mask_B1MHW = data['ff_gt']
        first_frame_mask_B1MHW = torch.zeros_like(first_frame_mask_B1MHW)
        first_frame_mask_BNHW  = first_frame_mask_B1MHW[:, 0, :max_num_objects]

        # Initialize sensory.
        N, Cs = max_num_objects, self.sensory_dim
        Hk, Wk = key_DCHW.shape[2:]
        sensory_BNCHW = torch.zeros(B, N, Cs, Hk, Wk, device=key_DCHW.device)
        
        # Encode mask for the 0-th frame.
        value_BNCHW, sensory_BNCHW, object_memory_BNQC = self.encode_mask(
            first_frame_image_BCHW, first_frame_pixfeat_BCHW,
            first_frame_mask_BNHW, sensory_BNCHW, deep_update=True
        )

        # Obtain initial object memory.
        object_memory_sum_BNQC = object_memory_BNQC.clone()

        # Compute initial state.
        key_BCHW = key_BTCHW[:, 0]
        key_max_B1HW = torch.max(key_BCHW, dim=1, keepdim=True).values
        key_BCHW = (key_BCHW - key_max_B1HW).softmax(dim=1)
        state_BNCC = torch.einsum('bkhw,bnvhw->bnkv', key_BCHW, value_BNCHW)

        last_masks_BNHW = first_frame_mask_BNHW.clone()

        # t0
        this_key_BCHW = key_BTCHW[:, 0]
        this_key_max_B1HW = torch.max(this_key_BCHW, dim=1, keepdim=True).values
        this_key_BCHW = (this_key_BCHW - this_key_max_B1HW).softmax(dim=1)
        this_readout_BNCHW = torch.einsum(
            'bkhw,bnkv->bnvhw', this_key_BCHW, state_BNCC).contiguous()

        this_pixfeat_BCHW = pixfeat_BTCHW[:, 0]
        # Optimization: Slice pre-processed features.
        this_ms_feats = [f[:, 0] for f in ms_feats_reshaped]
        this_logits_BNHW, this_masks_BNHW, sensory_BNCHW, aux = self.segment(
                this_ms_feats, 
                this_readout_BNCHW, 
                this_pixfeat_BCHW, 
                last_masks_BNHW, 
                sensory_BNCHW, 
                object_memory_sum_BNQC, 
                update_sensory=True)
        this_aux_output = self.aux_computer(this_pixfeat_BCHW, sensory_BNCHW, aux)
        last_masks_BNHW = this_masks_BNHW[:, 1:]
        out[f'logits_{0}'] = this_logits_BNHW
        out[f'masks_{0}'] = this_masks_BNHW[:, 1:]  # remove the background
        out[f'aux_{0}'] = this_aux_output

        for i in range(1, T):
            # t-1
            state_t_1_BNCC = state_BNCC.clone()

            # Eraser term
            this_key_BCHW = key_BTCHW[:, i]
            this_key_max_B1HW = torch.max(this_key_BCHW, dim=1, keepdim=True).values
            this_key_BCHW = (this_key_BCHW - this_key_max_B1HW).softmax(dim=1)
            
            v_old = torch.einsum('bkhw,bnkv->bnvhw', this_key_BCHW, state_t_1_BNCC).contiguous()
            v_k = torch.einsum('bkhw,bnvhw->bnkv', this_key_BCHW, v_old)
            
            beta_t = self.b_proj(state_t_1_BNCC).sigmoid()
            beta_t_expan_b = beta_t.repeat_interleave(256 // self.num_heads, dim=3)
            eraser = torch.einsum('bnkv,bnkv->bnkv', beta_t_expan_b, v_k).contiguous()

            # New information
            this_image_BCHW = images_BTCHW[:, i]
            this_value_BNCHW, sensory_BNCHW, object_memory_BNQC = self.encode_mask(
                this_image_BCHW, this_pixfeat_BCHW,
                last_masks_BNHW,
                sensory_BNCHW, deep_update=True)
            
            vk_t = torch.einsum('bkhw,bnvhw->bnkv', this_key_BCHW, this_value_BNCHW).contiguous()
            new = torch.einsum('bnkv,bnkv->bnkv', beta_t_expan_b, vk_t).contiguous()

            # Alpha
            old = state_t_1_BNCC - eraser
            alpha = -self.A_log.float().exp() * F.softplus(self.a_proj(state_t_1_BNCC).float() + self.dt_bias)
            
            # Use repeat_interleave for physical copy, consistent with beta_t logic.
            alpha_expanded = alpha.repeat_interleave(256 // 8, dim=3)
            old = torch.einsum('bnkv,bnkv->bnkv', alpha_expanded, old).contiguous()

            # End
            state_BNCC = old + new

            # Readout
            this_readout_BNCHW = torch.einsum(
                'bkhw,bnkv->bnvhw', this_key_BCHW, state_BNCC).contiguous()

            # Segment
            this_pixfeat_BCHW = pixfeat_BTCHW[:, i]
            # Optimization: Slice directly.
            this_ms_feats = [f[:, i] for f in ms_feats_reshaped]
            
            this_logits_BNHW, this_masks_BNHW, sensory_BNCHW, aux = self.segment(
                this_ms_feats, 
                this_readout_BNCHW, 
                this_pixfeat_BCHW, 
                last_masks_BNHW, 
                sensory_BNCHW, 
                object_memory_sum_BNQC, 
                update_sensory=True)
            
            # Auxilary task
            this_aux_output = self.aux_computer(this_pixfeat_BCHW, sensory_BNCHW, aux)

            last_masks_BNHW = this_masks_BNHW[:, 1:]  # remove the background

            # Accumulate object memory.
            object_memory_sum_BNQC = object_memory_sum_BNQC + object_memory_BNQC

            out[f'logits_{i}'] = this_logits_BNHW
            out[f'masks_{i}'] = this_masks_BNHW[:, 1:]  # remove the background
            out[f'aux_{i}'] = this_aux_output
        
        return out
    def load_weights(self, src_dict, init_as_zero_if_needed=False) -> None:
        # Map single-object weight to multi-object weight (4->5 out channels in conv1)
        for k in list(src_dict.keys()):
            if k == 'mask_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    log.info(f'Converting {k} from single object to multiple objects.')
                    pads = torch.zeros((64, 1, 7, 7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        nn.init.orthogonal_(pads)
                        log.info(f'Randomly initialized padding for {k}.')
                    else:
                        log.info(f'Zero-initialized padding for {k}.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)
            elif k == 'pixel_fuser.sensory_compress.weight':
                if src_dict[k].shape[1] == self.sensory_dim + 1:
                    log.info(f'Converting {k} from single object to multiple objects.')
                    pads = torch.zeros((self.value_dim, 1, 1, 1), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        nn.init.orthogonal_(pads)
                        log.info(f'Randomly initialized padding for {k}.')
                    else:
                        log.info(f'Zero-initialized padding for {k}.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        for k in src_dict:
            if k not in self.state_dict():
                log.info(f'Key {k} found in src_dict but not in self.state_dict()!!!')
        for k in self.state_dict():
            if k not in src_dict:
                log.info(f'Key {k} found in self.state_dict() but not in src_dict!!!')

        self.load_state_dict(src_dict, strict=False)
