"""A clean, scan-free NC-SSD reference layer for vision experiments.

This module is an independent implementation of the global-state operator
described in the VSSD paper. The upstream VSSD repository is retained as a
read-only submodule under ``third_party/VSSD`` for provenance and comparison.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from vnct.models.layers.rope2d import RoPE2D


class NCSSD(nn.Module):
    """First-order non-causal SSD token mixer.

    Args:
        d_model: Input and output channel dimension.
        d_state: Global state feature dimension. Must be divisible by four
            when 2D RoPE is enabled.
        expand: Expansion ratio of the value stream.
        headdim: Per-head value dimension.
        d_conv: Kernel size of the depthwise local projection.
        use_rope: Apply 2D RoPE to source and query features.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: float = 1.0,
        headdim: int = 24,
        d_conv: int = 3,
        use_rope: bool = True,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)
        self.headdim = headdim
        self.eps = eps
        if self.d_inner % headdim != 0:
            raise ValueError(f"d_inner={self.d_inner} must be divisible by headdim={headdim}")
        self.nheads = self.d_inner // headdim

        # Streams follow the VSSD/Mamba convention: gate, value, B/K, C/Q, dt.
        projection_dim = 2 * self.d_inner + 2 * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, projection_dim, bias=False)
        conv_dim = self.d_inner + 2 * d_state
        self.local_proj = nn.Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=conv_dim,
        )

        dt = torch.exp(torch.empty(self.nheads).uniform_(math.log(dt_min), math.log(dt_max)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.nheads).uniform_(0.0, math.log(16.0)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.rope = RoPE2D(d_state) if use_rope else None
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Mix flattened image tokens without a directional scan."""
        batch, length, _ = tokens.shape
        if length != height * width:
            raise ValueError(f"token length {length} does not match H*W={height * width}")

        projected = self.in_proj(tokens)
        z, xbc, dt = torch.split(
            projected,
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads],
            dim=-1,
        )
        xbc = rearrange(xbc, "b (h w) c -> b c h w", h=height, w=width)
        xbc = F.silu(self.local_proj(xbc))
        xbc = rearrange(xbc, "b c h w -> b (h w) c")
        value, source, query = torch.split(
            xbc, [self.d_inner, self.d_state, self.d_state], dim=-1
        )

        value = rearrange(value, "b l (h p) -> b h l p", h=self.nheads)
        source = (F.elu(source) + 1.0)[:, None, :, :]
        query = (F.elu(query) + 1.0)[:, None, :, :]

        # VSSD repurposes the SSD decay as a source-token contribution weight.
        dt = F.softplus(dt + self.dt_bias)
        decay = -torch.exp(self.A_log)
        source_logits = -(dt * decay).transpose(1, 2)
        source_weight = source_logits.softmax(dim=-1).unsqueeze(-1) * length
        weighted_source = source * source_weight

        if self.rope is not None:
            query_rope = self.rope(
                query.reshape(batch, 1, height, width, self.d_state)
            ).reshape(batch, 1, length, self.d_state)
            weighted_source = self.rope(
                weighted_source.reshape(batch, self.nheads, height, width, self.d_state)
            ).reshape(batch, self.nheads, length, self.d_state)
        else:
            query_rope = query

        # G = sum_j m_j B_j outer-product x_j; every target reads the same G.
        length_scale = length**-0.5
        global_state = torch.matmul(
            weighted_source.transpose(-2, -1) * length_scale,
            value * length_scale,
        )
        query_scale = float(self.nheads * self.headdim)
        output = torch.matmul(query_rope.expand(-1, self.nheads, -1, -1) / query_scale, global_state)

        # Positive-feature normalization used by the ICCV VSSD implementation.
        denominator = torch.matmul(
            query / query_scale,
            source.mean(dim=-2, keepdim=True).transpose(-2, -1),
        ).clamp_min(self.eps)
        output = output / denominator
        output = output + value * self.D[None, :, None, None]
        output = rearrange(output, "b h l p -> b l (h p)")
        output = self.norm(output) * F.silu(z)
        return self.out_proj(output)
