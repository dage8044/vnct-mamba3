"""Checkpoint-compatible VSSD-Small backbone with a rank-M NC-M3 bridge.

The module subclasses the pinned camera-ready VSSD implementation instead of
copying its hierarchy.  Consequently, all pretrained VSSD parameters retain
their original state-dict paths and tensor shapes.  The only additional mixer
parameters are rank adapters, data-dependent decay/trapezoidal projections,
and a residual transition scale.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def _load_upstream_vssd() -> ModuleType:
    """Load the pinned VSSD model package without claiming ``models`` globally."""
    package_name = "_vnct_upstream_vssd_models"
    module_name = f"{package_name}.mamba2"
    if module_name in sys.modules:
        return sys.modules[module_name]

    repository = Path(__file__).resolve().parents[3]
    package_dir = repository / "third_party" / "VSSD" / "classification" / "models"
    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        raise ImportError(
            "The pinned VSSD source is missing. Run "
            "`git submodule update --init --recursive`."
        )

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load upstream VSSD package from {init_file}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return sys.modules[module_name]


_VSSD = _load_upstream_vssd()


class VSSDCheckpointNCM3(_VSSD.Mamba2):
    """NC-M3 transition that preserves every original VSSD mixer parameter.

    ``ncm3_scale=0`` is exactly the original NC-SSD operator.  The trainable
    scalar can then move the loaded model toward the second-order trapezoidal
    operator without discarding the ImageNet initialization.  Rank zero uses
    the inherited VSSD B/C streams.  Additional ranks are small learned
    perturbations, so the inherited projection shapes and checkpoint keys do
    not change.
    """

    def __init__(
        self,
        *args: Any,
        ncm3_scale_init: float = 0.0,
        mimo_rank: int = 4,
        adapter_init_std: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.ngroups != 1:
            raise ValueError("The VSSD checkpoint bridge currently requires ssd_ngroups=1")
        if not self.linear_attn_duality:
            raise ValueError("The VSSD checkpoint bridge requires LINEAR_ATTN_DUALITY=True")
        if mimo_rank < 1:
            raise ValueError("mimo_rank must be positive")
        if adapter_init_std <= 0.0:
            raise ValueError("adapter_init_std must be positive")

        self.mimo_rank = int(mimo_rank)
        self.adapter_init_std = float(adapter_init_std)

        # Raw parameters avoid being reinitialized by VMAMBA2._init_weights.
        # Small non-zero projections make decay and theta data-dependent from
        # the first optimization step.  The outer transition scale controls
        # feature drift from the inherited VSSD function.
        projection_std = self.adapter_init_std / math.sqrt(self.d_inner)
        self.trap_weight = nn.Parameter(
            torch.empty(self.nheads, self.headdim).normal_(std=projection_std)
        )
        self.trap_bias = nn.Parameter(torch.zeros(self.nheads))
        self.decay_weight = nn.Parameter(
            torch.empty(self.nheads, self.d_inner).normal_(std=projection_std)
        )
        self.decay_bias = nn.Parameter(torch.zeros(self.nheads))
        if self.mimo_rank > 1:
            extra_dimension = (self.mimo_rank - 1) * self.d_state
            self.mimo_source_weight = nn.Parameter(
                torch.empty(extra_dimension, self.d_inner).normal_(std=projection_std)
            )
            self.mimo_query_weight = nn.Parameter(
                torch.empty(extra_dimension, self.d_inner).normal_(std=projection_std)
            )
        else:
            self.register_parameter("mimo_source_weight", None)
            self.register_parameter("mimo_query_weight", None)
        mixing = self.mimo_rank**-0.5
        self.mimo_in = nn.Parameter(
            torch.full((self.nheads, self.mimo_rank, self.headdim), mixing)
        )
        self.mimo_out = nn.Parameter(
            torch.full((self.nheads, self.mimo_rank, self.headdim), mixing)
        )
        self.ncm3_scale = nn.Parameter(torch.tensor(float(ncm3_scale_init)))

    @staticmethod
    def _shift_beta(beta: torch.Tensor) -> torch.Tensor:
        beta_plus = torch.zeros_like(beta)
        beta_plus[..., :-1] = beta[..., 1:]
        return beta_plus

    def _apply_upstream_rope(
        self,
        tensor: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Apply the inherited 2-D RoPE to ``[B, L, H, R, N]``."""
        if self.ropes is None:
            return tensor
        batch, length, heads, ranks, d_state = tensor.shape
        feature = tensor.permute(0, 2, 3, 1, 4).reshape(
            batch * heads * ranks, height, width, d_state
        )
        if height != self.ropes.rotations.shape[0] or width != self.ropes.rotations.shape[1]:
            feature = _VSSD.rope(feature, (height, width, d_state))
        else:
            feature = self.ropes(feature)
        return (
            feature.reshape(batch, heads, ranks, length, d_state)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )

    def _rank_states(
        self,
        base: torch.Tensor,
        value: torch.Tensor,
        weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Lift one inherited B/C stream into checkpoint-compatible MIMO ranks."""
        if self.mimo_rank == 1:
            return base.unsqueeze(2)
        if weight is None:
            raise RuntimeError("rank adapters are missing")
        batch, length, _, _ = value.shape
        perturbation = F.linear(value.flatten(2), weight).view(
            batch, length, self.mimo_rank - 1, self.d_state
        )
        return torch.cat((base.unsqueeze(2), base.unsqueeze(2) + perturbation), dim=2)

    def _data_dependent_decay(
        self,
        value: torch.Tensor,
        static_decay: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate inherited negative A while preserving its sign and scale."""
        magnitude = (-static_decay.float()).clamp_min(1e-4)
        base_logit = magnitude + torch.log(-torch.expm1(-magnitude))
        modulation = F.linear(
            value.flatten(2).float(),
            self.decay_weight.float(),
            self.decay_bias.float(),
        )
        return -F.softplus(base_logit.view(1, 1, -1) + modulation)

    def _trapezoidal_linear_attn(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank-M scan-free form of the VNCT trapezoidal state update."""
        batch, length, heads, head_dim = x.shape
        d_state = B.shape[-1]
        value = x
        source = self._rank_states(B, value, self.mimo_source_weight)
        query = self._rank_states(C, value, self.mimo_query_weight)
        source = (self.elu(source) + 1.0).unsqueeze(2).expand(
            -1, -1, heads, -1, -1
        )
        query = (self.elu(query) + 1.0).unsqueeze(2).expand(
            -1, -1, heads, -1, -1
        )

        # Mamba-3 trapezoidal coefficients with token-dependent negative A.
        decay = self._data_dependent_decay(value, A)
        alpha = torch.exp(dt.float() * decay)
        theta = torch.sigmoid(
            torch.einsum("blhp,hp->blh", x.float(), self.trap_weight.float())
            + self.trap_bias.float().view(1, 1, heads)
        )
        gamma = theta * dt.float()
        beta = (1.0 - theta) * dt.float() * alpha
        beta_plus = self._shift_beta(beta.transpose(1, 2)).transpose(1, 2)

        # Both branches are spatial distributions. Averaging and multiplying
        # by L keeps the source magnitude aligned with the pretrained path.
        state_scale = math.sqrt(d_state)
        source_weight = 0.5 * (
            torch.softmax(gamma / state_scale, dim=1)
            + torch.softmax(beta_plus / state_scale, dim=1)
        ) * length
        weighted_source = source * source_weight[..., None, None].to(source.dtype)
        query_rope = self._apply_upstream_rope(query, height, width)
        source_rope = self._apply_upstream_rope(
            weighted_source, height, width
        )

        query_scale = float(heads * head_dim)
        query_rope = query_rope / query_scale
        query_plain = query / query_scale
        value_rank = value.unsqueeze(3) * self.mimo_in.to(value.dtype)[None, None]
        length_scale = length**-0.5
        global_state = torch.einsum(
            "blhrn,blhrp->bhrnp",
            source_rope.float() * length_scale,
            value_rank.float() * length_scale,
        )
        rank_output = torch.einsum(
            "blhrn,bhrnp->blhrp", query_rope.float(), global_state
        )
        mean_source = weighted_source.float().mean(dim=1)
        denominator = torch.einsum(
            "blhrn,bhrn->blhr", query_plain.float(), mean_source
        )
        rank_output = rank_output / denominator.clamp_min(1e-6).unsqueeze(-1)
        mixing = self.mimo_out.float()[None, None]
        output = (rank_output * mixing).sum(dim=3).to(value.dtype)
        output = output + value * D.view(1, 1, -1, 1)
        mixed_state = (
            global_state * self.mimo_out.float()[None, :, :, None, :]
        ).sum(dim=2)
        return output.contiguous(), mixed_state

    def non_casual_linear_attn(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        H: int | None = None,
        W: int | None = None,
        relpos: torch.Tensor | None = None,
        last_kv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vssd_output, vssd_state = super().non_casual_linear_attn(
            x, dt, A, B, C, D, H, W, relpos, last_kv
        )
        # Keep the exact VSSD shortcut for inference only.  ``new_modules_only``
        # deliberately leaves the inherited backbone in eval mode while still
        # optimizing the NC-M3 bridge, so checking ``self.training`` here would
        # silently remove every gradient to the new transition parameters.
        if not torch.is_grad_enabled() and self.ncm3_scale.item() == 0.0:
            return vssd_output, vssd_state
        if H is None or W is None:
            raise ValueError("NC-M3 requires explicit spatial height and width")
        ncm3_output, ncm3_state = self._trapezoidal_linear_attn(
            x, dt, A, B, C, D, H, W
        )
        scale = self.ncm3_scale.to(dtype=vssd_output.dtype)
        output = vssd_output + scale * (ncm3_output - vssd_output)
        state = vssd_state + scale * (ncm3_state - vssd_state)
        return output, state


VSSD_SMALL_CAMERA_READY: dict[str, Any] = {
    "img_size": 224,
    "patch_size": 4,
    "in_chans": 3,
    "embed_dim": 96,
    "depths": (2, 4, 15, 4),
    "num_heads": (4, 4, 8, 16),
    "mlp_ratio": 3.0,
    "drop_rate": 0.0,
    "drop_path_rate": 0.4,
    "simple_downsample": False,
    "simple_patch_embed": False,
    "ssd_expansion": 1,
    "ssd_ngroups": 1,
    "ssd_chunk_size": 256,
    "mimo_rank": 4,
    "linear_attn_duality": True,
    "ssd_positve_dA": True,
    "attn_types": ("mamba2", "mamba2", "mamba2", "standard"),
    "async_state": (12, 24, 48, 64),
    "rmt_downsample": True,
    "rmt_patch_embed": True,
    "use_cpe": True,
    "ssd_linear_norm": True,
    "exp_da": True,
    "rope": True,
}


class VSSDCheckpointBackbone(_VSSD.Backbone_VMAMBA2):
    """Pinned VSSD feature-pyramid hierarchy with swapped NC-SSD mixers."""

    def __init__(
        self,
        *,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        pretrained: str | Path | None = None,
        ncm3_scale_init: float = 0.0,
        mimo_rank: int = 4,
        adapter_init_std: float = 0.02,
        **kwargs: Any,
    ) -> None:
        original_mixer = _VSSD.Mamba2
        _VSSD.Mamba2 = VSSDCheckpointNCM3
        try:
            super().__init__(
                out_indices=out_indices,
                pretrained=None,
                mimo_rank=mimo_rank,
                adapter_init_std=adapter_init_std,
                **kwargs,
            )
        finally:
            _VSSD.Mamba2 = original_mixer

        for module in self.modules():
            if isinstance(module, VSSDCheckpointNCM3):
                module.ncm3_scale.data.fill_(ncm3_scale_init)

        if pretrained is not None:
            from vnct.utils.checkpoint import load_vssd_checkpoint

            self.checkpoint_report = load_vssd_checkpoint(self, pretrained)


def vssd_small_ncm3(
    *,
    pretrained: str | Path | None = None,
    ncm3_scale_init: float = 0.0,
    mimo_rank: int = 4,
    adapter_init_std: float = 0.02,
    **overrides: Any,
) -> VSSDCheckpointBackbone:
    """Build the ICCV 2025 camera-ready VSSD-Small BIQA backbone."""
    config = dict(VSSD_SMALL_CAMERA_READY)
    config.update(overrides)
    config["mimo_rank"] = mimo_rank
    config["adapter_init_std"] = adapter_init_std
    return VSSDCheckpointBackbone(
        pretrained=pretrained,
        ncm3_scale_init=ncm3_scale_init,
        **config,
    )


def vssd_small_original(
    *,
    pretrained: str | Path | None = None,
    out_indices: tuple[int, ...] = (0, 1, 2, 3),
    **overrides: Any,
) -> nn.Module:
    """Build the unchanged camera-ready VSSD-Small feature backbone.

    Unlike :func:`vssd_small_ncm3`, this builder does not swap any inherited
    NC-SSD mixer class. It is the structural control used to isolate the
    effect of the Mamba-3 bridge while keeping all downstream BIQA modules
    identical.
    """
    config = dict(VSSD_SMALL_CAMERA_READY)
    config.update(overrides)
    config.pop("mimo_rank", None)
    config.pop("adapter_init_std", None)
    backbone = _VSSD.Backbone_VMAMBA2(
        out_indices=out_indices,
        pretrained=None,
        **config,
    )
    if pretrained is not None:
        from vnct.utils.checkpoint import load_vssd_checkpoint

        backbone.checkpoint_report = load_vssd_checkpoint(backbone, pretrained)
    return backbone


def vssd_ncm3_debug(**overrides: Any) -> VSSDCheckpointBackbone:
    """Reduced hierarchy for CPU/CUDA integration tests."""
    config = dict(VSSD_SMALL_CAMERA_READY)
    config.update(
        embed_dim=16,
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 4, 8),
        async_state=(8, 8, 16, 16),
        drop_path_rate=0.0,
    )
    config.update(overrides)
    return VSSDCheckpointBackbone(**config)


def vssd_original_debug(**overrides: Any) -> nn.Module:
    """Reduced unchanged VSSD hierarchy for equivalence tests."""
    config = dict(
        img_size=32,
        embed_dim=16,
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 4, 8),
        async_state=(8, 8, 16, 16),
        drop_path_rate=0.0,
    )
    config.update(overrides)
    return vssd_small_original(**config)


__all__ = [
    "VSSDCheckpointBackbone",
    "VSSDCheckpointNCM3",
    "VSSD_SMALL_CAMERA_READY",
    "vssd_ncm3_debug",
    "vssd_original_debug",
    "vssd_small_original",
    "vssd_small_ncm3",
]
