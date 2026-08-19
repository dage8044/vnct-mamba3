"""BIQA-ready multi-scale VSSD backbone without a classification head."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from vnct.models.layers.nc_ssd import NCSSD


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep) / keep


class ConvPatchEmbed(nn.Module):
    """Overlapping convolutional stem with output stride four."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = out_channels // 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConvDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialResidual(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        feature = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
        return tokens + self.conv(feature).flatten(2).transpose(1, 2)


class AttentionMixer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        del height, width
        return self.attention(tokens, tokens, tokens, need_weights=False)[0]


class VSSDBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        d_state: int,
        mlp_ratio: float,
        mixer_type: str,
        drop_path: float,
        expand: float,
    ) -> None:
        super().__init__()
        self.local1 = SpatialResidual(dim)
        self.norm1 = nn.LayerNorm(dim)
        if mixer_type == "ncssd":
            self.mixer = NCSSD(
                dim,
                d_state=d_state,
                expand=expand,
                headdim=int(dim * expand) // num_heads,
            )
        elif mixer_type == "attention":
            self.mixer = AttentionMixer(dim, num_heads)
        else:
            raise ValueError(f"unsupported mixer type: {mixer_type}")
        self.local2 = SpatialResidual(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)
        self.layer_scale1 = nn.Parameter(torch.full((dim,), 1e-5))
        self.layer_scale2 = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        tokens = self.local1(tokens, height, width)
        tokens = tokens + self.drop_path(
            self.layer_scale1 * self.mixer(self.norm1(tokens), height, width)
        )
        tokens = self.local2(tokens, height, width)
        return tokens + self.drop_path(self.layer_scale2 * self.mlp(self.norm2(tokens)))


class VSSDBackbone(nn.Module):
    """Four-stage VSSD feature extractor for downstream BIQA models.

    ``forward`` returns four NCHW feature maps at strides 4, 8, 16 and 32.
    No classification or quality-regression head is included.
    """

    def __init__(
        self,
        channels: Sequence[int] = (96, 192, 384, 768),
        depths: Sequence[int] = (2, 2, 8, 2),
        num_heads: Sequence[int] = (4, 4, 8, 16),
        state_dims: Sequence[int] = (12, 24, 48, 64),
        mlp_ratio: float = 3.0,
        expand: float = 1.0,
        drop_path_rate: float = 0.2,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(depths) == len(num_heads) == len(state_dims) == 4):
            raise ValueError("VSSDBackbone expects four entries for every stage setting")
        self.channels = tuple(channels)
        self.patch_embed = ConvPatchEmbed(in_channels, channels[0])
        total_blocks = sum(depths)
        drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cursor = 0
        stages = []
        downsamples = []
        for index, (dim, depth, heads, d_state) in enumerate(
            zip(channels, depths, num_heads, state_dims, strict=True)
        ):
            mixer_type = "attention" if index == 3 else "ncssd"
            blocks = []
            for _ in range(depth):
                blocks.append(
                    VSSDBlock(
                        dim,
                        heads,
                        d_state,
                        mlp_ratio,
                        mixer_type,
                        drop_rates[cursor],
                        expand,
                    )
                )
                cursor += 1
            stages.append(nn.ModuleList(blocks))
            if index < 3:
                downsamples.append(ConvDownsample(dim, channels[index + 1]))
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)
        self.output_norms = nn.ModuleList(nn.LayerNorm(dim) for dim in channels)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feature = self.patch_embed(image)
        outputs = []
        for index, blocks in enumerate(self.stages):
            batch, channels, height, width = feature.shape
            tokens = feature.flatten(2).transpose(1, 2)
            for block in blocks:
                tokens = block(tokens, height, width)
            tokens = self.output_norms[index](tokens)
            feature = tokens.transpose(1, 2).reshape(batch, channels, height, width)
            outputs.append(feature)
            if index < len(self.downsamples):
                feature = self.downsamples[index](feature)
        return tuple(outputs)


def vssd_tiny(**kwargs: object) -> VSSDBackbone:
    """ICCV-style Tiny structural preset, without a classifier."""
    return VSSDBackbone(**kwargs)


def vssd_debug(**kwargs: object) -> VSSDBackbone:
    """Small preset for unit tests and NC-M3 integration work."""
    defaults = dict(
        channels=(16, 32, 64, 128),
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 4, 8),
        state_dims=(8, 8, 16, 16),
        mlp_ratio=2.0,
        drop_path_rate=0.0,
    )
    defaults.update(kwargs)
    return VSSDBackbone(**defaults)
