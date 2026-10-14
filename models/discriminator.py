"""
models/discriminator.py
SN-GAN CIFAR ResNet critic (Figure 2), without spectral norm (WGAN-GP supplies the Lipschitz constraint).
x[3,32,32] -> Down128 -> Down128 -> Res128 -> Res128 -> ReLU -> global SUM pool -> dense 1
"""
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResBlock, ResBlockDown, ResBlockDownFirst, _init


class Discriminator(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.blocks = nn.Sequential(
            ResBlockDownFirst(3, ch),  # 32 -> 16
            ResBlockDown(ch, ch),      # 16 -> 8
            ResBlock(ch),
            ResBlock(ch),
        )
        self.fc = _init(nn.Linear(ch, 1))

    def forward(self, x):
        h = F.relu(self.blocks(x))
        h = h.sum(dim=(2, 3))  # global sum pooling
        return self.fc(h).squeeze(1)  # [B] critic score (unbounded, no sigmoid)
