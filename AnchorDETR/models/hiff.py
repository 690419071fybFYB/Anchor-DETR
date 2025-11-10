# ------------------------------------------------------------------------
# Copyright (c) 2024.
# ------------------------------------------------------------------------
"""
Hierarchical Image Feature Fusion (HIFF).

This module implements the fusion strategy described in the prompt: a top-down
hierarchical fusion where high-level features are reshaped to the spatial
resolution of the lower stage, summed, globally pooled, and reweighted through
modality-specific MLPs with Softmax normalization on the channel dimension.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["HIFF", "HIFFBlock"]


class MLP(nn.Module):
    """Generic multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers: List[nn.Module] = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class HIFFBlock(nn.Module):
    """
    HIFF fusion block operating on a pair of feature maps (low-level & high-level).
    """

    def __init__(
        self,
        channels: int,
        mlp_hidden_dim: int = 512,
        temperature: float = 1.0,
        resize_mode: str = "bilinear",
        class_frequency_dim: int | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.channels = channels
        self.temperature = temperature
        self.resize_mode = resize_mode
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.context_fc = nn.Linear(channels, channels)
        self.high_mlp = MLP(channels, mlp_hidden_dim, channels)
        self.low_mlp = MLP(channels, mlp_hidden_dim, channels)
        self.class_freq_proj = (
            nn.Linear(class_frequency_dim, channels) if class_frequency_dim is not None else None
        )

    def forward(
        self, low_feat: Tensor, high_feat: Tensor, class_frequency: Tensor | None = None
    ) -> Tensor:
        if low_feat.shape[1] != self.channels or high_feat.shape[1] != self.channels:
            raise ValueError("Feature channels must match the configured HIFF block channels.")

        high_upsampled = F.interpolate(
            high_feat,
            size=low_feat.shape[-2:],
            mode=self.resize_mode,
            align_corners=False if self.resize_mode in {"bilinear", "bicubic"} else None,
        )

        combined = low_feat + high_upsampled
        context = self.pool(combined).flatten(1)
        context = self.context_fc(context)

        if class_frequency is not None and self.class_freq_proj is not None:
            # Accept either [B, K] or [K] statistics and broadcast to the batch.
            if class_frequency.dim() == 1:
                class_frequency = class_frequency.unsqueeze(0).expand(context.size(0), -1)
            freq_bias = self.class_freq_proj(class_frequency)
            context = context + freq_bias

        weight_low = self._compute_weights(self.low_mlp(context))
        weight_high = self._compute_weights(self.high_mlp(context))

        fused = weight_low * low_feat + weight_high * high_upsampled
        return fused

    def _compute_weights(self, logits: Tensor) -> Tensor:
        logits = logits / self.temperature
        weights = torch.softmax(logits, dim=-1)
        return weights.unsqueeze(-1).unsqueeze(-1)


class HIFF(nn.Module):
    """
    Hierarchical Image Feature Fusion (HIFF) model.

    Args:
        in_channels: Iterable containing the number of channels for each input scale ordered
            from high-resolution (lowest stride) to low-resolution (highest stride).
        embed_dim: Target channel dimension for every fusion stage.
        mlp_hidden_dim: Hidden width for the modality-specific MLPs (default: 512).
        temperature: Softmax temperature used when generating channel weights.
        resize_mode: Interpolation mode used when aligning spatial resolutions.
        class_frequency_dim: Optional size of the class-frequency vector used for reweighting.
    """

    def __init__(
        self,
        in_channels: Sequence[int] | int,
        embed_dim: int,
        mlp_hidden_dim: int = 512,
        temperature: float = 1.0,
        resize_mode: str = "bilinear",
        class_frequency_dim: int | None = None,
    ) -> None:
        super().__init__()

        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if len(in_channels) < 2:
            raise ValueError("HIFF requires at least two feature levels.")

        self.embed_dim = embed_dim

        self.projections = nn.ModuleList(
            [
                nn.Identity() if c == embed_dim else nn.Conv2d(c, embed_dim, kernel_size=1)
                for c in in_channels
            ]
        )

        self.blocks = nn.ModuleList(
            [
                HIFFBlock(
                    embed_dim,
                    mlp_hidden_dim=mlp_hidden_dim,
                    temperature=temperature,
                    resize_mode=resize_mode,
                    class_frequency_dim=class_frequency_dim,
                )
                for _ in range(len(in_channels) - 1)
            ]
        )

    def forward(
        self, features: Iterable[Tensor], class_frequency: Tensor | None = None
    ) -> List[Tensor]:
        """
        Perform hierarchical fusion.

        Args:
            features: Iterable of tensors ordered from high-resolution to low-resolution.
            class_frequency: Optional tensor containing dataset statistics that will be injected
                into every HIFF block to boost rare classes.

        Returns:
            List of fused feature maps preserving the input order.
        """
        features = list(features)
        if len(features) != len(self.projections):
            raise ValueError("Number of input features does not match HIFF configuration.")

        #通道数投影至一致
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        fused_features = list(projected)
        running = fused_features[-1]
        for idx in reversed(range(len(fused_features) - 1)):
            running = self.blocks[idx](fused_features[idx], running, class_frequency)
            fused_features[idx] = running

        return fused_features
if __name__ == "__main__":
    hiff = HIFF(in_channels=[256, 512, 1024], embed_dim=256)
    feature1=torch.randn(1, 256, 160, 160)
    feature2=torch.randn(1, 512, 80, 80)
    feature3=torch.randn(1, 1024, 40, 40)
    features = [feature1,feature2,feature3]
    fused_features = hiff(features)
    print([f.shape for f in fused_features])
