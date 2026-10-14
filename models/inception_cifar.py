"""
models/inception_cifar.py
Small Inception for CIFAR-10 (Zhang et al. 2017, "Understanding deep learning requires
rethinking generalization"), trained from scratch. Used for IS (softmax) and FID (336-d features).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionModule(nn.Module):
    """1x1 branch || 3x3 branch, concatenated on channels."""

    def __init__(self, in_ch, c1, c3):
        super().__init__()
        self.b1 = ConvBN(in_ch, c1, 1)
        self.b3 = ConvBN(in_ch, c3, 3)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x)], 1)


class DownsampleModule(nn.Module):
    """3x3 stride-2 conv || 3x3 stride-2 max-pool, concatenated."""

    def __init__(self, in_ch, c3):
        super().__init__()
        self.conv = ConvBN(in_ch, c3, 3, 2)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        return torch.cat([self.conv(x), self.pool(x)], 1)


class InceptionCIFAR(nn.Module):
    FEAT_DIM = 336

    def __init__(self, num_classes=10):
        super().__init__()
        self.features_net = nn.Sequential(
            ConvBN(3, 96, 3),                                   # 32x32, 96
            InceptionModule(96, 32, 32),                        # 64
            InceptionModule(64, 32, 48),                        # 80
            DownsampleModule(80, 80),                           # 16x16, 160
            InceptionModule(160, 112, 48),                      # 160
            InceptionModule(160, 96, 64),                       # 160
            InceptionModule(160, 80, 80),                       # 160
            InceptionModule(160, 48, 96),                       # 144
            DownsampleModule(144, 96),                          # 8x8, 240
            InceptionModule(240, 176, 160),                     # 336
            InceptionModule(336, 176, 160),                     # 336
        )
        self.fc = nn.Linear(self.FEAT_DIM, num_classes)

    def features(self, x):
        """Penultimate 336-d embedding (FID)."""
        return F.adaptive_avg_pool2d(self.features_net(x), 1).flatten(1)

    def forward(self, x):
        """Logits (train with CE; softmax for IS)."""
        return self.fc(self.features(x))
