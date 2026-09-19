"""Small, dependency-free PyTorch segmentation models for FireWatch."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int, maximum: int = 8) -> int:
    """Return a GroupNorm group count that divides *channels*."""
    for value in range(min(maximum, channels), 0, -1):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return F.silu(x + residual)


class SmallResUNet(nn.Module):
    """A compact U-Net robust to batches of one through GroupNorm.

    This intentionally has no pretrained encoder: competition chips have a
    sensor-specific channel layout and external image pretraining is unsafe.
    """
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 24) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        b = base_channels
        self.stem = ResidualBlock(in_channels, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(b * 2, b * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(b * 4, b * 8))
        self.mid = ResidualBlock(b * 8, b * 8)
        self.up2 = ResidualBlock(b * 8 + b * 4, b * 4)
        self.up1 = ResidualBlock(b * 4 + b * 2, b * 2)
        self.up0 = ResidualBlock(b * 2 + b, b)
        self.head = nn.Conv2d(b, out_channels, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.stem(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        x = self.mid(e3)
        x = self.up2(self._up(x, e2))
        x = self.up1(self._up(x, e1))
        x = self.up0(self._up(x, e0))
        return self.head(x)


def build_model(in_channels: int, task: str, base_channels: int = 24) -> SmallResUNet:
    """Build an AF binary or BS 4-class model."""
    task = task.lower()
    if task not in {"af", "bs"}:
        raise ValueError("task must be 'af' or 'bs'")
    return SmallResUNet(in_channels, 1 if task == "af" else 4, base_channels)
