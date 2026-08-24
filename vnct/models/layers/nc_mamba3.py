"""Second-order, scan-free Non-Causal Mamba-3 token mixing.

This is an independent PyTorch implementation of equations (11)--(20) in
the VNCT paper. It uses ordinary PyTorch operations so the math can be
inspected and tested before introducing a fused CUDA kernel.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from vnct.models.layers.rope2d import RoPE2D


class NCMamba3(nn.Module):
    """Non-causal lift of Mamba-3's trapezoidal state dynamics."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: float = 2.0,
        headdim: int = 64,
        mimo_rank: int = 4,
        rope_fraction: float = 0.5,
        chunk_size: int = 256,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        a_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)
        self.headdim = headdim
        self.mimo_rank = mimo_rank
        self.chunk_size = chunk_size
        self.a_floor = a_floor

        if self.d_inner % headdim != 0:
            raise ValueError(f"d_inner={self.d_inner} must be divisible by headdim={headdim}")
        if mimo_rank < 1:
            raise ValueError(f"mimo_rank must be positive, got {mimo_rank}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not 0.0 <= rope_fraction <= 1.0:
            raise ValueError(f"rope_fraction must be in [0, 1], got {rope_fraction}")

        self.nheads = self.d_inner // headdim
        rotary_dim = int(d_state * rope_fraction)
        rotary_dim -= rotary_dim % 4
        self.rotary_dim = rotary_dim

        # z, x, B, C, delta, data-dependent A, trapezoidal interpolation.
        projection_dim = 2 * self.d_inner + 2 * mimo_rank * d_state + 3 * self.nheads
        self.in_proj = nn.Linear(d_model, projection_dim, bias=False)

        dt = torch.exp(torch.empty(self.nheads).uniform_(math.log(dt_min), math.log(dt_max)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.B_norm = nn.RMSNorm(d_state)
        self.C_norm = nn.RMSNorm(d_state)
        self.B_bias = nn.Parameter(torch.ones(self.nheads, mimo_rank, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, mimo_rank, d_state))

        if mimo_rank > 1:
            self.mimo_in = nn.Parameter(
                torch.full((self.nheads, mimo_rank, headdim), 1.0 / mimo_rank)
            )
            self.mimo_out = nn.Parameter(
                torch.full((self.nheads, mimo_rank, headdim), 1.0 / mimo_rank)
            )
        else:
            self.register_parameter("mimo_in", None)
            self.register_parameter("mimo_out", None)

        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True
        self.rope = RoPE2D(rotary_dim) if rotary_dim else None
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _coefficients(
        self,
        delta_logit: torch.Tensor,
        decay_logit: torch.Tensor,
        trap_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return float32 ``alpha``, ``beta`` and ``gamma`` from Mamba-3."""
        dt = F.softplus(delta_logit.float() + self.dt_bias.float())
        decay = -F.softplus(decay_logit.float()).clamp_min(self.a_floor)
        alpha = torch.exp(decay * dt)
        trap = torch.sigmoid(trap_logit.float())
        gamma = trap * dt
        beta = (1.0 - trap) * dt * alpha
        return alpha, beta, gamma

    def _apply_rope(
        self,
        state: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Apply grid RoPE to ``(B, L, H, R, N)`` state features."""
        if self.rope is None:
            return state
        _, length, heads, ranks, _ = state.shape
        if length != height * width:
            raise ValueError(f"token length {length} does not match H*W={height * width}")
        rotary, remainder = state.split(
            [self.rotary_dim, self.d_state - self.rotary_dim], dim=-1
        )
        rotary = rearrange(
            rotary,
            "b (y x) h r n -> b (h r) y x n",
            y=height,
            x=width,
        )
        rotary = self.rope(rotary)
        rotary = rearrange(
            rotary,
            "b (h r) y x n -> b (y x) h r n",
            h=heads,
            r=ranks,
        )
        return rotary if remainder.shape[-1] == 0 else torch.cat((rotary, remainder), dim=-1)

    @staticmethod
    def _shift_beta(beta: torch.Tensor) -> torch.Tensor:
        """Map ``beta[j + 1]`` to source token ``j`` without wrap-around."""
        beta_plus = torch.zeros_like(beta)
        beta_plus[:, :-1] = beta[:, 1:]
        return beta_plus

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, length, _ = tokens.shape
        if length != height * width:
            raise ValueError(f"token length {length} does not match H*W={height * width}")

        projected = self.in_proj(tokens)
        z, x, source, query, delta, decay, trap = torch.split(
            projected,
            [
                self.d_inner,
                self.d_inner,
                self.mimo_rank * self.d_state,
                self.mimo_rank * self.d_state,
                self.nheads,
                self.nheads,
                self.nheads,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", h=self.nheads)
        x = rearrange(x, "b l (h p) -> b l h p", h=self.nheads)
        source = rearrange(source, "b l (r n) -> b l r n", r=self.mimo_rank)
        query = rearrange(query, "b l (r n) -> b l r n", r=self.mimo_rank)
        source = self.B_norm(source)[:, :, None] + self.B_bias[None, None]
        query = self.C_norm(query)[:, :, None] + self.C_bias[None, None]
        source = self._apply_rope(source, height, width)
        query = self._apply_rope(query, height, width)

        _, beta, gamma = self._coefficients(delta, decay, trap)
        beta_plus = self._shift_beta(beta)
        state_scale = math.sqrt(self.d_state)
        source_weight = (
            torch.softmax(gamma / state_scale, dim=1)
            + torch.softmax(beta_plus / state_scale, dim=1)
        )

        if self.mimo_in is None:
            value_rank = x.unsqueeze(3)
        else:
            value_rank = x.unsqueeze(3) * self.mimo_in[None, None]

        # Eq. (17): accumulate the shared global sufficient statistic.
        global_state = torch.zeros(
            batch,
            self.nheads,
            self.headdim,
            self.mimo_rank,
            self.d_state,
            device=tokens.device,
            dtype=torch.float32,
        )
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            weighted_source = (
                source[:, start:stop].float()
                * source_weight[:, start:stop, :, None, None]
            )
            global_state = global_state + torch.einsum(
                "bkhrp,bkhrn->bhprn",
                value_rank[:, start:stop].float(),
                weighted_source,
            )

        # Eq. (18): every target token reads the same global state.
        chunks = []
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            rank_output = torch.einsum(
                "bhprn,bkhrn->bkhpr",
                global_state,
                query[:, start:stop].float(),
            )
            if self.mimo_out is None:
                chunk = rank_output.squeeze(-1)
            else:
                chunk = torch.einsum(
                    "bkhpr,hrp->bkhp", rank_output, self.mimo_out.float()
                )
            chunks.append(chunk)
        output = torch.cat(chunks, dim=1).to(dtype=x.dtype)

        output = output + x * self.D.to(dtype=x.dtype)[None, None, :, None]
        output = output * F.silu(z)
        output = rearrange(output, "b l h p -> b l (h p)")
        return self.out_proj(output)
