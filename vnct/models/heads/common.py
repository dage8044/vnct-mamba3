"""Shared output contract for no-reference IQA prediction heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QualityHeadOutput:
    """Prediction and interpretable within-/between-stage contributions."""

    score: torch.Tensor
    stage_scores: torch.Tensor
    stage_weights: torch.Tensor
    spatial_weights: torch.Tensor


__all__ = ["QualityHeadOutput"]
