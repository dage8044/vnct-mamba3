"""Two-dimensional rotary positional embeddings."""

from __future__ import annotations

import torch
from torch import nn


class RoPE2D(nn.Module):
    """Apply independent rotary embeddings along image rows and columns.

    The input must have shape ``(batch, heads, height, width, dim)``. Half
    of the channels encode row offsets and half encode column offsets.
    """

    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f"RoPE2D requires dim divisible by 4, got {dim}")
        self.dim = dim
        self.base = base
        pairs_per_axis = dim // 4
        inv_freq = base ** (-torch.arange(pairs_per_axis, dtype=torch.float32) / pairs_per_axis)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        x_pair = x.reshape(*x.shape[:-1], -1, 2)
        real, imag = x_pair.unbind(dim=-1)
        cos = angles.cos().to(dtype=x.dtype)
        sin = angles.sin().to(dtype=x.dtype)
        return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[-1] != self.dim:
            raise ValueError(
                f"expected (B, heads, H, W, {self.dim}), got {tuple(x.shape)}"
            )

        _, _, height, width, _ = x.shape
        row_x, col_x = x.chunk(2, dim=-1)
        row_pos = torch.arange(height, device=x.device, dtype=self.inv_freq.dtype)
        col_pos = torch.arange(width, device=x.device, dtype=self.inv_freq.dtype)
        row_angles = torch.outer(row_pos, self.inv_freq)[None, None, :, None, :]
        col_angles = torch.outer(col_pos, self.inv_freq)[None, None, None, :, :]
        return torch.cat((self._rotate(row_x, row_angles), self._rotate(col_x, col_angles)), dim=-1)
