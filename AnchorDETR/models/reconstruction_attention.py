# ------------------------------------------------------------------------
# Reconstruction Attention module for enhancing high-resolution features
# using low-resolution semantic cues.
# ------------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PMA(nn.Module):
    """Pooling-based channel attention (similar to SE block)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fc(self.pool(x))
        return x * weight


class ReconstructionAttention(nn.Module):
    """
    同时为 F1、F2 提供互补信息的重构注意力:
        mask_f1 = Sigmoid(F1 - Conv(Up(F2)))
        mask_f2 = Sigmoid(F2 - Conv(Down(F1)))
    """

    def __init__(
        self,
        c_f1: int,
        c_f2: int,
        upsample_mode: str = "bilinear",
        downsample_mode: str = "bilinear",
        pma_reduction: int = 16,
    ):
        super().__init__()
        self.align_low_to_high = nn.Sequential(
            nn.Conv2d(c_f2, c_f1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_f1),
        )
        self.align_high_to_low = nn.Sequential(
            nn.Conv2d(c_f1, c_f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_f2),
        )
        self.upsample_mode = upsample_mode
        self.downsample_mode = downsample_mode
        self.pma_f1 = PMA(c_f1, reduction=pma_reduction)
        self.pma_f2 = PMA(c_f2, reduction=pma_reduction)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f2_up = F.interpolate(
            f2,
            size=f1.shape[-2:],
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in {"bilinear", "bicubic"} else None,
        )
        f2_recon = self.align_low_to_high(f2_up)
        mask_f1 = torch.sigmoid(f1 - f2_recon)
        enhanced_f1 = self.pma_f1(f1 * mask_f1)

        f1_down = F.interpolate(
            f1,
            size=f2.shape[-2:],
            mode=self.downsample_mode,
            align_corners=False if self.downsample_mode in {"bilinear", "bicubic"} else None,
        )
        f1_recon = self.align_high_to_low(f1_down)
        mask_f2 = torch.sigmoid(f2 - f1_recon)
        enhanced_f2 = self.pma_f2(f2 * mask_f2)

        return enhanced_f1, enhanced_f2

if __name__ == "__main__":
    f1 = torch.randn(1, 256, 64, 64)
    f2 = torch.randn(1, 128, 128, 128)
    ra = ReconstructionAttention(256, 128)
    enhanced1,enhanced2 = ra(f1, f2)
    print(enhanced1.shape,enhanced2.shape)

