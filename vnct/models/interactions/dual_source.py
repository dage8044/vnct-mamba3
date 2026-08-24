"""Unified local and regional evidence interaction for BIQA."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vnct.models.layers.position import (
    continuous_2d_sincos,
    normalized_grid_coordinates,
)


class SpatialReducedCrossAttention(nn.Module):
    """Cross-attend native-resolution queries to one masked evidence bank."""

    def __init__(
        self,
        query_channels: int,
        source_channels: int,
        *,
        inner_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if inner_dim <= 0 or inner_dim % num_heads != 0:
            raise ValueError("inner_dim must be positive and divisible by num_heads")
        if inner_dim % 4 != 0:
            raise ValueError("inner_dim must be divisible by four for 2-D positions")
        self.query_channels = int(query_channels)
        self.source_channels = int(source_channels)
        self.inner_dim = int(inner_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.query_norm = nn.LayerNorm(query_channels)
        self.source_norm = nn.LayerNorm(source_channels)
        self.query_projection = nn.Linear(query_channels, inner_dim)
        self.key_projection = nn.Linear(source_channels, inner_dim)
        self.value_projection = nn.Linear(source_channels, inner_dim)
        self.output_projection = nn.Linear(inner_dim, query_channels)

    def forward(
        self,
        query_feature: torch.Tensor,
        source_tokens: torch.Tensor,
        source_position: torch.Tensor,
        *,
        source_mask: torch.Tensor | None = None,
        source_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_feature.ndim != 4:
            raise ValueError("query_feature must have shape [B, C, H, W]")
        if source_tokens.ndim != 3:
            raise ValueError("source_tokens must have shape [B, M, C]")
        batch, channels, height, width = query_feature.shape
        if channels != self.query_channels:
            raise ValueError(f"expected {self.query_channels} query channels")
        if source_tokens.shape[0] != batch or source_tokens.shape[2] != self.source_channels:
            raise ValueError("source token batch or channel dimension is invalid")
        if source_position.shape[-2:] != (source_tokens.shape[1], self.inner_dim):
            raise ValueError("source_position must have shape [M,D] or [B,M,D]")
        if source_mask is not None and source_mask.shape != source_tokens.shape[:2]:
            raise ValueError("source_mask must have shape [B, M]")
        if source_type is not None and source_type.shape[-2:] != (
            source_tokens.shape[1], self.inner_dim
        ):
            raise ValueError("source_type must have shape [M,D] or [B,M,D]")

        query_tokens = query_feature.flatten(2).transpose(1, 2)
        query_coordinates = normalized_grid_coordinates(
            height,
            width,
            device=query_feature.device,
            dtype=query_feature.dtype,
        )
        query_position = continuous_2d_sincos(query_coordinates, self.inner_dim)
        normalized_source = self.source_norm(source_tokens)
        query = self.query_projection(self.query_norm(query_tokens)) + query_position
        key = self.key_projection(normalized_source) + source_position
        value = self.value_projection(normalized_source)
        if source_type is not None:
            key = key + source_type
            value = value + source_type

        head_dim = self.inner_dim // self.num_heads
        query = query.view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        attention_mask = None
        if source_mask is not None:
            attention_mask = torch.zeros(
                (batch, 1, 1, source_tokens.shape[1]),
                device=query.device,
                dtype=query.dtype,
            ).masked_fill(~source_mask[:, None, None, :], -torch.inf)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, height * width, -1)
        output = self.output_projection(attended)
        return output.transpose(1, 2).reshape(batch, channels, height, width)


class UnifiedEvidenceInteraction(nn.Module):
    """Generate a stage feature from main queries and one evidence bank."""

    def __init__(
        self,
        channels: int,
        *,
        inner_dim: int,
        num_heads: int = 4,
        local_grid_size: int = 7,
        has_enhancement: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if local_grid_size <= 0:
            raise ValueError("local_grid_size must be positive")
        self.channels = int(channels)
        self.inner_dim = int(inner_dim)
        self.local_grid_size = int(local_grid_size)
        self.has_enhancement = bool(has_enhancement)
        self.attention = SpatialReducedCrossAttention(
            channels,
            channels,
            inner_dim=inner_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.source_type_embedding = nn.Parameter(torch.zeros(3, inner_dim))
        nn.init.normal_(self.source_type_embedding, std=0.02)
        self.summary_position = nn.Parameter(torch.zeros(inner_dim))
        nn.init.normal_(self.summary_position, std=0.02)
        self.fusion = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.last_stats: dict[str, torch.Tensor] = {}

    def _build_evidence(
        self,
        local_feature: torch.Tensor,
        *,
        importance_map: torch.Tensor | None,
        enhancement_tokens: torch.Tensor | None,
        enhancement_coordinates: torch.Tensor | None,
        enhancement_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = local_feature.shape[0]
        pooled_local = F.adaptive_avg_pool2d(
            local_feature, (self.local_grid_size, self.local_grid_size)
        )
        local_tokens = pooled_local.flatten(2).transpose(1, 2)
        local_coordinates = normalized_grid_coordinates(
            self.local_grid_size,
            self.local_grid_size,
            device=local_feature.device,
            dtype=local_feature.dtype,
        )
        local_position = continuous_2d_sincos(local_coordinates, self.inner_dim)
        local_position = local_position.unsqueeze(0).expand(batch, -1, -1)
        local_type = self.source_type_embedding[0].view(1, 1, -1).expand(
            batch, local_tokens.shape[1], -1
        )
        parts = [local_tokens]
        positions = [local_position]
        types = [local_type]
        masks = [
            torch.ones(
                (batch, local_tokens.shape[1]),
                device=local_feature.device,
                dtype=torch.bool,
            )
        ]

        if self.has_enhancement:
            if (
                enhancement_tokens is None
                or enhancement_coordinates is None
                or enhancement_mask is None
            ):
                raise ValueError("regional tokens, coordinates, and mask are required")
            if importance_map is not None:
                if importance_map.shape != (
                    batch,
                    1,
                    local_feature.shape[2],
                    local_feature.shape[3],
                ):
                    raise ValueError("importance_map must match the local feature grid")
                summary = (
                    local_feature.float() * importance_map.float()
                ).sum(dim=(-2, -1)).to(dtype=local_feature.dtype).unsqueeze(1)
                summary_position = self.summary_position.view(1, 1, -1).expand(
                    batch, 1, -1
                )
                summary_type = self.source_type_embedding[1].view(1, 1, -1).expand(
                    batch, 1, -1
                )
                parts.append(summary)
                positions.append(summary_position)
                types.append(summary_type)
                masks.append(
                    torch.ones(
                        (batch, 1), device=local_feature.device, dtype=torch.bool
                    )
                )
            regional_position = continuous_2d_sincos(
                enhancement_coordinates, self.inner_dim
            )
            regional_type = self.source_type_embedding[2].view(1, 1, -1).expand(
                batch, enhancement_tokens.shape[1], -1
            )
            parts.append(enhancement_tokens)
            positions.append(regional_position)
            types.append(regional_type)
            masks.append(enhancement_mask)
        elif any(
            item is not None
            for item in (
                importance_map,
                enhancement_tokens,
                enhancement_coordinates,
                enhancement_mask,
            )
        ):
            raise ValueError("stage without enhancement accepts local evidence only")

        return (
            torch.cat(parts, dim=1),
            torch.cat(positions, dim=1),
            torch.cat(types, dim=1),
            torch.cat(masks, dim=1),
        )

    def forward(
        self,
        main_feature: torch.Tensor,
        local_feature: torch.Tensor,
        *,
        importance_map: torch.Tensor | None = None,
        enhancement_tokens: torch.Tensor | None = None,
        enhancement_coordinates: torch.Tensor | None = None,
        enhancement_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if main_feature.shape != local_feature.shape:
            raise ValueError("main_feature and local_feature must have the same shape")
        tokens, position, source_type, source_mask = self._build_evidence(
            local_feature,
            importance_map=importance_map,
            enhancement_tokens=enhancement_tokens,
            enhancement_coordinates=enhancement_coordinates,
            enhancement_mask=enhancement_mask,
        )
        attended = self.attention(
            main_feature,
            tokens,
            position,
            source_mask=source_mask,
            source_type=source_type,
        )
        main_tokens = main_feature.flatten(2).transpose(1, 2)
        attended_tokens = attended.flatten(2).transpose(1, 2)
        output_tokens = self.fusion(torch.cat((main_tokens, attended_tokens), dim=-1))
        output = output_tokens.transpose(1, 2).reshape_as(main_feature)
        with torch.no_grad():
            reference = main_feature.detach().float().abs().mean().clamp_min(1e-6)
            self.last_stats = {
                "evidence_tokens_mean": source_mask.sum(dim=1).float().mean(),
                "fusion_change_ratio": (
                    output.detach().float() - main_feature.detach().float()
                ).abs().mean()
                / reference,
            }
        return output


# Compatibility alias for checkpoints/configs created before the interaction
# was renamed. It is the same implementation, not the former gated module.
DualSourceInteraction = UnifiedEvidenceInteraction


__all__ = [
    "DualSourceInteraction",
    "SpatialReducedCrossAttention",
    "UnifiedEvidenceInteraction",
]
