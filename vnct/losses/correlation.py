"""Differentiable batchwise correlation loss."""

from __future__ import annotations

import torch
from torch import nn


class PearsonCorrelationLoss(nn.Module):
    """Return ``1 - PLCC`` for a prediction/score batch."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.float().flatten()
        target = target.float().flatten()
        if prediction.numel() != target.numel():
            raise ValueError("prediction and target must have the same number of values")
        prediction = prediction - prediction.mean()
        target = target - target.mean()
        denominator = prediction.square().sum().sqrt() * target.square().sum().sqrt()
        correlation = (prediction * target).sum() / denominator.clamp_min(self.eps)
        return 1.0 - correlation


class LoDaPLCCLoss(nn.Module):
    """Exact PLCC objective used by the released LoDa training code."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.float()
        target = target.detach().float()
        prediction_std, prediction_mean = torch.std_mean(prediction, unbiased=False)
        prediction = (prediction - prediction_mean) / (prediction_std + self.eps)
        target_std, target_mean = torch.std_mean(target, unbiased=False)
        target = (target - target_mean) / (target_std + self.eps)
        linear_loss = torch.nn.functional.mse_loss(prediction, target) / 4.0
        correlation = torch.mean(prediction * target)
        calibrated_loss = (
            torch.nn.functional.mse_loss(correlation * prediction, target) / 4.0
        )
        return ((linear_loss + calibrated_loss) / 2.0).float()
