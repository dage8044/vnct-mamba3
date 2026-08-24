"""Position helpers shared by sparse and dense spatial token modules."""

from __future__ import annotations

import math

import torch


def normalized_grid_coordinates(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return flattened cell-center coordinates in normalized ``(x, y)`` form."""
    if height <= 0 or width <= 0:
        raise ValueError("grid height and width must be positive")
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)


def continuous_2d_sincos(
    coordinates: torch.Tensor,
    embedding_dim: int,
    *,
    temperature: float = 10_000.0,
) -> torch.Tensor:
    """Encode normalized ``(..., 2)`` coordinates without learned grid limits."""
    if coordinates.shape[-1] != 2:
        raise ValueError("coordinates must end with normalized (x, y)")
    if embedding_dim <= 0 or embedding_dim % 4 != 0:
        raise ValueError("embedding_dim must be positive and divisible by four")

    quarter = embedding_dim // 4
    frequency = torch.arange(
        quarter,
        device=coordinates.device,
        dtype=torch.float32,
    )
    denominator = max(quarter - 1, 1)
    frequency = temperature ** (-frequency / denominator)
    angles = coordinates.float().unsqueeze(-1) * (2.0 * math.pi * frequency)
    x_angle, y_angle = angles.unbind(dim=-2)
    embedding = torch.cat(
        (x_angle.sin(), x_angle.cos(), y_angle.sin(), y_angle.cos()), dim=-1
    )
    return embedding.to(dtype=coordinates.dtype)


__all__ = ["continuous_2d_sincos", "normalized_grid_coordinates"]
