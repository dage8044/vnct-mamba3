"""Parallel joint-stage regression head for no-reference IQA."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vnct.models.heads.common import QualityHeadOutput
from vnct.models.layers.position import (
    continuous_2d_sincos,
    normalized_grid_coordinates,
)


# Backward-compatible public name used by existing checkpoints and callers.
JointStageHeadOutput = QualityHeadOutput


class JointStageQualityHead(nn.Module):
    """Fuse all four stage token sets jointly, then regress one quality score."""

    def __init__(
        self,
        channels: tuple[int, int, int, int],
        *,
        embed_dim: int = 192,
        grid_size: int = 7,
        num_heads: int = 4,
        depth: int = 1,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain four positive dimensions")
        if embed_dim <= 0 or embed_dim % num_heads != 0 or embed_dim % 4 != 0:
            raise ValueError(
                "embed_dim must be positive and divisible by num_heads and four"
            )
        if grid_size <= 0 or depth <= 0:
            raise ValueError("grid_size and depth must be positive")

        self.channels = tuple(int(channel) for channel in channels)
        self.embed_dim = int(embed_dim)
        self.grid_size = int(grid_size)
        self.stage_projections = nn.ModuleList(
            nn.Conv2d(channel, embed_dim, kernel_size=1) for channel in channels
        )
        self.stage_embedding = nn.Parameter(torch.zeros(4, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.joint_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )
        hidden_dim = max(embed_dim // 2, 1)
        self.quality_predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.importance_predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.stage_predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        nn.init.zeros_(self.importance_predictor[-1].weight)
        nn.init.zeros_(self.importance_predictor[-1].bias)
        nn.init.zeros_(self.stage_predictor[-1].weight)
        nn.init.zeros_(self.stage_predictor[-1].bias)

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
        token_count = self.grid_size**2
        coordinates = normalized_grid_coordinates(
            self.grid_size,
            self.grid_size,
            device=stage_features[0].device,
            dtype=stage_features[0].dtype,
        )
        position = continuous_2d_sincos(coordinates, self.embed_dim)
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
            tokens = pooled.flatten(2).transpose(1, 2)
            tokens = tokens + position + self.stage_embedding[stage]
            stage_tokens.append(tokens)

        batch = stage_tokens[0].shape[0]
        joint_tokens = self.joint_encoder(torch.cat(stage_tokens, dim=1))
        joint_tokens = joint_tokens.reshape(batch, 4, token_count, self.embed_dim)
        token_quality = self.quality_predictor(joint_tokens).squeeze(-1)
        importance_logits = self.importance_predictor(joint_tokens).squeeze(-1)
        spatial_weights = importance_logits.softmax(dim=-1)
        stage_scores = (spatial_weights * token_quality).sum(dim=-1)
        stage_descriptors = joint_tokens.mean(dim=2)
        stage_logits = self.stage_predictor(stage_descriptors).squeeze(-1)
        stage_weights = stage_logits.softmax(dim=-1)
        score = (stage_weights * stage_scores).sum(dim=-1)
        return QualityHeadOutput(
            score=score,
            stage_scores=stage_scores,
            stage_weights=stage_weights,
            spatial_weights=spatial_weights,
        )


__all__ = ["JointStageHeadOutput", "JointStageQualityHead"]
