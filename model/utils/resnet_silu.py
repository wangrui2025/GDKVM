"""
resnet.py - A modified ResNet structure
We append extra channels to the first conv by some network surgery
"""

from collections import OrderedDict
import math
import logging

import torch
import torch.nn as nn
from torch.utils import model_zoo

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_weights_add_extra_dim(target, source_state, extra_dim=0, image_channels=1):
    """
    Load pretrained weights into the target model, adjusting the first convolution layer's weights based on extra_dim and image_channels.

    Parameters:
    - target: The target model instance.
    - source_state: The source model's state_dict (pretrained weights).
    - extra_dim: Number of extra channels. If 0 and image_channels=1, adjust to 1 channel.
    - image_channels: Number of input channels for the target model (excluding extra_dim).
    """
    
    # log.info(f"Executing load_weights_add_extra_dim function, image_channels={image_channels}, extra_dim={extra_dim}, total input channels={image_channels + extra_dim}")
    
    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if 'num_batches_tracked' in k1:
            continue  # Skip BatchNorm tracking parameters

        if k1 not in source_state:
            log.warning(f"Warning: Key {k1} missing in source state_dict. Using target model's default weights.")
            new_dict[k1] = v1
            continue

        tar_v = source_state[k1]

        if k1 == 'conv1.weight':
            # Handle first convolution layer weights
            src_c, src_in_channels, w, h = tar_v.shape
            tgt_c, tgt_in_channels, tw, th = v1.shape

            expected_in_channels = image_channels + extra_dim

            # log.info(f"Processing {k1}:")
            # log.info(f"    Source weights shape: {tar_v.shape}")
            # log.info(f"    Target weights shape: {v1.shape}")
            # log.info(f"    Expected input channels: {expected_in_channels}")

            if src_in_channels != expected_in_channels:
                # log.info(f"Adjusting input channels of {k1} from {src_in_channels} to {expected_in_channels}.")
                # Initialize extra channels by averaging the pretrained weights across the existing channels
                tar_v_extra = tar_v.mean(dim=1, keepdim=True)  # Generate a grayscale channel
                if expected_in_channels > 1:
                    # Initialize additional channel weights
                    extra_weights = torch.zeros((src_c, expected_in_channels - 1, w, h), device=tar_v.device)
                    nn.init.orthogonal_(extra_weights)
                    tar_v = torch.cat([tar_v_extra, extra_weights], dim=1)
                else:
                    tar_v = tar_v_extra
                # log.info(f"    Adjusted {k1} weights shape: {tar_v.shape}")
            else:
                log.info(f"No adjustment needed for {k1} input channels.")

            new_dict[k1] = tar_v
        else:
            if v1.shape == tar_v.shape:
                # Directly load weights for other layers if shapes match
                new_dict[k1] = tar_v
            else:
                log.warning(f"Warning: Shape mismatch for key {k1}. Target shape {v1.shape}, source shape {tar_v.shape}. Using target model's default weights.")
                new_dict[k1] = v1

    # Confirm adjusted weights
    # if 'conv1.weight' in new_dict:
        # log.info(f"Final adjusted conv1.weight shape: {new_dict['conv1.weight'].shape}")

    # Load the new state_dict into the target model, allowing missing or extra keys
    target.load_state_dict(new_dict, strict=False)
    log.info("Pretrained weights loaded successfully.")

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes,
                    out_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    dilation=dilation,
                    bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.SiLU = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.SiLU(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.SiLU(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               dilation=dilation,
                               padding=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.SiLU = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.SiLU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.SiLU(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.SiLU(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3), extra_dim=0, image_channels=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # Adjust the number of input channels for the first convolution layer
        self.conv1 = nn.Conv2d(image_channels + extra_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.SiLU = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)


def resnet18(pretrained=True, extra_dim=0, image_channels=1):
    model = ResNet(BasicBlock, [2, 2, 2, 2], extra_dim, image_channels)
    if pretrained:
        # log.info("Loading pretrained weights...")
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        load_weights_add_extra_dim(model, pretrained_state_dict, extra_dim, image_channels)
    return model


def resnet50(pretrained=True, extra_dim=0, image_channels=1):
    model = ResNet(Bottleneck, [3, 4, 6, 3], extra_dim, image_channels)
    if pretrained:
        # log.info("Loading pretrained weights...")
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        load_weights_add_extra_dim(model, pretrained_state_dict, extra_dim, image_channels)
    return model
