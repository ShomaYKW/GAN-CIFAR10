"""
models/blocks.py
SN-GAN ResNet blocks (Miyato et al. 2018), pre-activation style.
Generator blocks use BatchNorm; discriminator blocks use no normalization
(WGAN-GP penalizes per-sample gradients, which BatchNorm would couple across the batch).
"""
import torch.nn as nn
import torch.nn.functional as F


def _init(m, gain=1.0):
    nn.init.xavier_uniform_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class ResBlockUp(nn.Module):
    """G block: BN-ReLU-Up-Conv3x3-BN-ReLU-Conv3x3  +  Up-Conv1x1 shortcut."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = _init(nn.Conv2d(in_ch, out_ch, 3, 1, 1), 2 ** 0.5)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = _init(nn.Conv2d(out_ch, out_ch, 3, 1, 1), 2 ** 0.5)
        self.sc = _init(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

    def forward(self, x):
        h = F.interpolate(F.relu(self.bn1(x)), scale_factor=2, mode="nearest")
        h = self.conv2(F.relu(self.bn2(self.conv1(h))))
        return h + self.sc(F.interpolate(x, scale_factor=2, mode="nearest"))


class ResBlockDownFirst(nn.Module):
    """First D block (input = image, so no pre-ReLU):
    Conv3x3-ReLU-Conv3x3-AvgPool  +  AvgPool-Conv1x1 shortcut."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = _init(nn.Conv2d(in_ch, out_ch, 3, 1, 1), 2 ** 0.5)
        self.conv2 = _init(nn.Conv2d(out_ch, out_ch, 3, 1, 1), 2 ** 0.5)
        self.sc = _init(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

    def forward(self, x):
        h = F.avg_pool2d(self.conv2(F.relu(self.conv1(x))), 2)
        return h + self.sc(F.avg_pool2d(x, 2))


class ResBlockDown(nn.Module):
    """D block: ReLU-Conv3x3-ReLU-Conv3x3-AvgPool  +  Conv1x1-AvgPool shortcut."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = _init(nn.Conv2d(in_ch, out_ch, 3, 1, 1), 2 ** 0.5)
        self.conv2 = _init(nn.Conv2d(out_ch, out_ch, 3, 1, 1), 2 ** 0.5)
        self.sc = _init(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

    def forward(self, x):
        h = F.avg_pool2d(self.conv2(F.relu(self.conv1(F.relu(x)))), 2)
        return h + F.avg_pool2d(self.sc(x), 2)


class ResBlock(nn.Module):
    """D block, same resolution: ReLU-Conv3x3-ReLU-Conv3x3  +  identity."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = _init(nn.Conv2d(ch, ch, 3, 1, 1), 2 ** 0.5)
        self.conv2 = _init(nn.Conv2d(ch, ch, 3, 1, 1), 2 ** 0.5)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))
