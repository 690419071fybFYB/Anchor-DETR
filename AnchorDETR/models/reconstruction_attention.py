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
    实现文中描述的 F2 -> F1 重构注意力:
        mask = Sigmoid(F1 - Conv1x1(Up(F2)))
        RA(F1, F2) = PMA(F1 * mask)
    """

    def __init__(
        self,
        c_f1: int,
        c_f2: int,
        upsample_mode: str = "bilinear",
        pma_reduction: int = 16,
    ):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(c_f2, c_f1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_f1),
        )
        self.upsample_mode = upsample_mode
        self.pma = PMA(c_f1, reduction=pma_reduction)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        f2_up = F.interpolate(
            f2,
            size=f1.shape[-2:],
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in {"bilinear", "bicubic"} else None,
        )
        f2_recon = self.align(f2_up)
        mask = torch.sigmoid(f1 - f2_recon)
        enhanced = f1 * mask
        return self.pma(enhanced)

if __name__ == "__main__":
    f1 = torch.randn(1, 256, 64, 64)
    f2 = torch.randn(1, 128, 128, 128)
    ra = ReconstructionAttention(256, 128)
    enhanced = ra(f1, f2)
    print(enhanced.shape)