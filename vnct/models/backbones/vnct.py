"""Hierarchical VNCT backbone for downstream BIQA feature extraction."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from vnct.models.backbones.vssd import AttentionMixer, DropPath, MLP
from vnct.models.layers.nc_mamba3 import NCMamba3


class OverlapPatchEmbed(nn.Module):
    """Paper-specified 7x7, stride-4 overlapping image embedding."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 7, stride=4, padding=3)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.proj(image)
        return self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class StageDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.proj(feature)
        return self.norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LocalPerceptionUnit(nn.Module):
    """Depthwise local residual from VNCT equation (22)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        feature = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
        local = self.conv(feature).flatten(2).transpose(1, 2)
        return tokens + self.norm(local)


class SinusoidalPosition2D(nn.Module):
    """Parameter-free absolute 2D sinusoidal position signal."""

    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"2D sinusoidal encoding requires dim divisible by 4, got {dim}")
        self.dim = dim
        quarter = dim // 4
        inv_freq = base ** (-torch.arange(quarter, dtype=torch.float32) / quarter)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows = torch.arange(height, device=device, dtype=self.inv_freq.dtype)
        cols = torch.arange(width, device=device, dtype=self.inv_freq.dtype)
        row_phase = torch.outer(rows, self.inv_freq)
        col_phase = torch.outer(cols, self.inv_freq)
        row = torch.cat((row_phase.sin(), row_phase.cos()), dim=-1)
        col = torch.cat((col_phase.sin(), col_phase.cos()), dim=-1)
        position = torch.cat(
            (
                row[:, None, :].expand(-1, width, -1),
                col[None, :, :].expand(height, -1, -1),
            ),
            dim=-1,
        )
        return position.reshape(1, height * width, self.dim).to(dtype=dtype)


class VNCTBlock(nn.Module):
    """LPU, Pos2D, global mixer and FFN sequence from equations (22)--(25)."""

    def __init__(self, dim: int, mixer: nn.Module, mlp_ratio: float, drop_path: float) -> None:
        super().__init__()
        self.lpu = LocalPerceptionUnit(dim)
        self.position = SinusoidalPosition2D(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = mixer
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)
        self.layer_scale1 = nn.Parameter(torch.full((dim,), 1e-5))
        self.layer_scale2 = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        tokens = self.lpu(tokens, height, width)
        tokens = tokens + self.position(
            height, width, device=tokens.device, dtype=tokens.dtype
        )
        tokens = tokens + self.drop_path(
            self.layer_scale1 * self.mixer(self.norm1(tokens), height, width)
        )
        return tokens + self.drop_path(self.layer_scale2 * self.mlp(self.norm2(tokens)))


class VNCTBackbone(nn.Module):
    """Four-stage, classification-head-free VNCT feature extractor."""

    def __init__(
        self,
        channels: Sequence[int] = (96, 192, 384, 768),
        depths: Sequence[int] = (2, 2, 9, 2),
        d_state: int = 64,
        ssm_head_dim: int = 64,
        attention_heads: int = 24,
        mimo_rank: int = 4,
        mlp_ratio: float = 3.0,
        expand: float = 2.0,
        rope_fraction: float = 0.5,
        chunk_size: int = 256,
        drop_path_rate: float = 0.2,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(depths) != 4:
            raise ValueError("VNCTBackbone expects four channel and depth entries")
        if channels[3] % attention_heads != 0:
            raise ValueError(
                f"stage-4 channels={channels[3]} must be divisible by "
                f"attention_heads={attention_heads}"
            )
        for dim in channels[:3]:
            if int(dim * expand) % ssm_head_dim != 0:
                raise ValueError(
                    f"expanded stage dimension {int(dim * expand)} must be divisible "
                    f"by ssm_head_dim={ssm_head_dim}"
                )

        self.channels = tuple(channels)
        self.patch_embed = OverlapPatchEmbed(in_channels, channels[0])
        total_blocks = sum(depths)
        drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cursor = 0
        stages = []
        downsamples = []
        for stage_index, (dim, depth) in enumerate(zip(channels, depths, strict=True)):
            blocks = []
            for _ in range(depth):
                if stage_index < 3:
                    mixer = NCMamba3(
                        dim,
                        d_state=d_state,
                        expand=expand,
                        headdim=ssm_head_dim,
                        mimo_rank=mimo_rank,
                        rope_fraction=rope_fraction,
                        chunk_size=chunk_size,
                    )
                else:
                    mixer = AttentionMixer(dim, attention_heads)
                blocks.append(VNCTBlock(dim, mixer, mlp_ratio, drop_rates[cursor]))
                cursor += 1
            stages.append(nn.ModuleList(blocks))
            if stage_index < 3:
                downsamples.append(StageDownsample(dim, channels[stage_index + 1]))

        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)
        self.output_norms = nn.ModuleList(nn.LayerNorm(dim) for dim in channels)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feature = self.patch_embed(image)
        outputs = []
        for stage_index, blocks in enumerate(self.stages):
            batch, channels, height, width = feature.shape
            tokens = feature.flatten(2).transpose(1, 2)
            for block in blocks:
                tokens = block(tokens, height, width)
            tokens = self.output_norms[stage_index](tokens)
            feature = tokens.transpose(1, 2).reshape(batch, channels, height, width)
            outputs.append(feature)
            if stage_index < len(self.downsamples):
                feature = self.downsamples[stage_index](feature)
        return tuple(outputs)


def vnct_tiny(**kwargs: object) -> VNCTBackbone:
    """VNCT-Tiny structural preset from supplementary Table 9."""
    return VNCTBackbone(**kwargs)


def vnct_debug(**kwargs: object) -> VNCTBackbone:
    """Reduced preset for unit tests and BIQA pipeline integration."""
    defaults = dict(
        channels=(16, 32, 64, 128),
        depths=(1, 1, 1, 1),
        d_state=16,
        ssm_head_dim=16,
        attention_heads=8,
        mimo_rank=2,
        mlp_ratio=2.0,
        chunk_size=32,
        drop_path_rate=0.0,
    )
    defaults.update(kwargs)
    return VNCTBackbone(**defaults)
