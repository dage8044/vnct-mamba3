"""Multi-range local-perception path for stage feature maps."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvFFN(nn.Module):
    """Pointwise feed-forward network that preserves the spatial grid."""

    def __init__(self, channels: int, ratio: float = 3.0) -> None:
        super().__init__()
        hidden_channels = int(channels * ratio)
        self.input = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.output = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.output(F.gelu(self.input(feature)))


class MultiRangeLocalMixer(nn.Module):
    """Gated local mixer with parallel standard and dilated depthwise paths.

    Input and output both have shape ``[B, C, H, W]``.  The input projection is
    initialized as an identity.  The identity profile uses a zero residual
    scale; the training profile uses a small non-zero scale for immediate
    branch gradients.
    """

    def __init__(
        self,
        channels: int,
        *,
        dilation: int = 2,
        ffn_ratio: float = 3.0,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")

        self.input_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.feature_near = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.feature_wide = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.feature_fusion = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.gate_near = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.gate_wide = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.gate_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.ffn = ConvFFN(channels, ratio=ffn_ratio)
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

        nn.init.dirac_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError("local mixer input must have shape [B, C, H, W]")
        projected = self.input_projection(feature)
        value = self.feature_fusion(
            torch.cat(
                (self.feature_near(projected), self.feature_wide(projected)), dim=1
            )
        )
        gate = torch.sigmoid(
            self.gate_projection(self.gate_wide(self.gate_near(projected)))
        )
        update = self.output_projection(self.ffn(F.gelu(value) * gate))
        return projected + self.residual_scale.to(dtype=update.dtype) * update


__all__ = ["MultiRangeLocalMixer"]
