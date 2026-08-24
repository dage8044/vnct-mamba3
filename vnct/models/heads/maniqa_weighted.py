"""MANIQA-style patch-weighted regression for multi-stage BIQA features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vnct.models.heads.common import QualityHeadOutput


class MANIQAPatchWeightedHead(nn.Module):
    """Regress patch quality and positive patch weights, then normalize.

    This keeps MANIQA's two-branch score/weight readout while adapting its
    single-resolution token input to four VSSD stages.  Stage-specific 1x1
    projections align the channels, adaptive pooling supplies an equal token
    grid per stage, and one shared pair of MLPs processes every token.  There
    is no joint Transformer or separately learned stage weighting function.
    """

    def __init__(
        self,
        channels: tuple[int, int, int, int],
        *,
        embed_dim: int = 192,
        grid_size: int = 7,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain four positive dimensions")
        if embed_dim <= 0 or grid_size <= 0:
            raise ValueError("embed_dim and grid_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.channels = tuple(int(channel) for channel in channels)
        self.embed_dim = int(embed_dim)
        self.grid_size = int(grid_size)
        self.eps = float(eps)
        self.stage_projections = nn.ModuleList(
            nn.Conv2d(channel, embed_dim, kernel_size=1) for channel in channels
        )
        self.token_norm = nn.LayerNorm(embed_dim)
        self.quality_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.ReLU(),
        )
        self.weight_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        stage_features: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> QualityHeadOutput:
        if len(stage_features) != 4:
            raise ValueError("exactly four stage features are required")

        stage_tokens: list[torch.Tensor] = []
        for stage, (feature, projection, expected_channels) in enumerate(
            zip(stage_features, self.stage_projections, self.channels, strict=True)
        ):
            if feature.ndim != 4 or feature.shape[1] != expected_channels:
                raise ValueError(
                    f"stage {stage + 1} must have shape [B, {expected_channels}, H, W]"
                )
            projected = projection(feature)
            pooled = F.adaptive_avg_pool2d(
                projected, (self.grid_size, self.grid_size)
            )
            stage_tokens.append(pooled.flatten(2).transpose(1, 2))

        tokens = self.token_norm(torch.stack(stage_tokens, dim=1))
        token_quality = self.quality_predictor(tokens).squeeze(-1)
        positive_weights = self.weight_predictor(tokens).squeeze(-1)

        # Accumulate in float32 so the epsilon remains meaningful under AMP.
        quality_float = token_quality.float()
        weights_float = positive_weights.float()
        stage_weight_sums = weights_float.sum(dim=-1)
        stage_denominator = stage_weight_sums.clamp_min(self.eps)
        stage_scores = (quality_float * weights_float).sum(dim=-1) / stage_denominator
        spatial_weights = weights_float / stage_denominator.unsqueeze(-1)

        total_denominator = stage_weight_sums.sum(dim=-1).clamp_min(self.eps)
        stage_weights = stage_weight_sums / total_denominator.unsqueeze(-1)
        score = (stage_weights * stage_scores).sum(dim=-1)
        return QualityHeadOutput(
            score=score,
            stage_scores=stage_scores,
            stage_weights=stage_weights,
            spatial_weights=spatial_weights,
        )


__all__ = ["MANIQAPatchWeightedHead"]
