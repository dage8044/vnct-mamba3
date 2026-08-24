"""Strict-enough loading of ImageNet VSSD checkpoints into BIQA backbones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointLoadReport:
    loaded_keys: int
    eligible_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def coverage(self) -> float:
        return self.loaded_keys / self.eligible_keys if self.eligible_keys else 1.0

    def __str__(self) -> str:
        return (
            f"loaded={self.loaded_keys}/{self.eligible_keys} "
            f"({self.coverage:.2%}), missing={len(self.missing_keys)}, "
            f"unexpected={len(self.unexpected_keys)}, "
            f"shape_mismatch={len(self.shape_mismatches)}"
        )


def _unwrap_checkpoint(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint).__name__}")
    for key in ("model_ema", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break
    if not all(isinstance(key, str) for key in checkpoint):
        raise TypeError("checkpoint state_dict contains a non-string key")
    return checkpoint


def _normalize_key(key: str) -> str:
    for prefix in ("module.backbone.", "module.", "backbone."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def load_vssd_checkpoint(
    model: nn.Module,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    min_coverage: float = 0.98,
) -> CheckpointLoadReport:
    """Load compatible VSSD weights and reject silent backbone shape drift.

    Classification-only ``head``/``norm`` tensors are ignored. New NC-M3
    transition and BIQA output-normalization tensors are allowed to be absent.
    Any shape mismatch among otherwise matching backbone keys is fatal.
    """
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    source = {
        _normalize_key(key): value
        for key, value in _unwrap_checkpoint(checkpoint).items()
        if isinstance(value, torch.Tensor)
    }
    target = model.state_dict()
    ignored_source = {"head.weight", "head.bias", "norm.weight", "norm.bias"}
    allowed_missing_suffixes = (
        "trap_weight",
        "trap_bias",
        "decay_weight",
        "decay_bias",
        "mimo_source_weight",
        "mimo_query_weight",
        "mimo_in",
        "mimo_out",
        "ncm3_scale",
    )

    compatible: dict[str, torch.Tensor] = {}
    shape_mismatches = []
    for key, value in source.items():
        if key in ignored_source or key not in target:
            continue
        if target[key].shape != value.shape:
            shape_mismatches.append(
                f"{key}: checkpoint{tuple(value.shape)} != model{tuple(target[key].shape)}"
            )
        else:
            compatible[key] = value
    if shape_mismatches:
        raise RuntimeError("VSSD checkpoint shape mismatch:\n" + "\n".join(shape_mismatches))

    model.load_state_dict(compatible, strict=False)
    eligible = [
        key
        for key in target
        if not key.endswith(allowed_missing_suffixes) and not key.startswith("outnorm")
    ]
    missing = tuple(key for key in eligible if key not in compatible)
    unexpected = tuple(
        key for key in source if key not in target and key not in ignored_source
    )
    report = CheckpointLoadReport(
        loaded_keys=sum(key in compatible for key in eligible),
        eligible_keys=len(eligible),
        missing_keys=missing,
        unexpected_keys=unexpected,
        shape_mismatches=(),
    )
    if report.coverage < min_coverage:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            f"VSSD checkpoint coverage {report.coverage:.2%} is below "
            f"the required {min_coverage:.2%}. Missing examples: {preview}"
        )
    return report
