"""
models/generator.py
SN-GAN CIFAR ResNet generator (Figure 1). ch=256 (variants 1, 3) or 128 (variant 2).
z(128) -> dense 4x4xch -> 3x ResBlockUp -> BN, ReLU, Conv3x3 -> 3, Tanh  => [B,3,32,32]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResBlockUp, _init


class Generator(nn.Module):
    def __init__(self, z_dim=128, ch=256):
        super().__init__()
        self.z_dim, self.ch = z_dim, ch
        self.fc = _init(nn.Linear(z_dim, 4 * 4 * ch))
        self.blocks = nn.Sequential(
            ResBlockUp(ch, ch),  # 4 -> 8
            ResBlockUp(ch, ch),  # 8 -> 16
            ResBlockUp(ch, ch),  # 16 -> 32
        )
        self.bn = nn.BatchNorm2d(ch)
        self.out = _init(nn.Conv2d(ch, 3, 3, 1, 1))

    def forward(self, z):
        h = self.fc(z).view(-1, self.ch, 4, 4)
        h = self.blocks(h)
        return torch.tanh(self.out(F.relu(self.bn(h))))

    @torch.no_grad()
    def sample(self, n, device):
        return self(torch.randn(n, self.z_dim, device=device))
