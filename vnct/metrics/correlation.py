"""SRCC, PLCC, KRCC, and RMSE reporting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def four_parameter_logistic(
    value: np.ndarray,
    upper: float,
    lower: float,
    midpoint: float,
    scale: float,
) -> np.ndarray:
    """VQEG-style monotonic four-parameter logistic mapping."""

    safe_scale = max(abs(float(scale)), np.finfo(np.float64).eps)
    exponent = np.clip(-(value - midpoint) / safe_scale, -60.0, 60.0)
    return lower + (upper - lower) / (1.0 + np.exp(exponent))


def fit_four_parameter_logistic(
    prediction: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Fit the reporting-only nonlinear map, falling back to raw predictions."""

    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    initial = (
        float(target.max()),
        float(target.min()),
        float(prediction.mean()),
        max(float(prediction.std()), 1e-3),
    )
    try:
        parameters, _ = curve_fit(
            four_parameter_logistic,
            prediction,
            target,
            p0=initial,
            maxfev=100_000,
        )
    except (RuntimeError, TypeError, ValueError, FloatingPointError):
        return prediction.copy()
    return four_parameter_logistic(prediction, *parameters)


def compute_iqa_metrics(
    prediction: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: prediction={prediction.shape}, target={target.shape}")
    if prediction.size < 2:
        raise ValueError("at least two samples are required to compute correlations")
    return {
        "srcc": float(stats.spearmanr(prediction, target).statistic),
        "plcc": float(stats.pearsonr(prediction, target).statistic),
        "krcc": float(stats.kendalltau(prediction, target).statistic),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }


def compute_loda_metrics(
    prediction: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Report raw SRCC and logistic-mapped PLCC/RMSE as configured by LoDa."""

    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mapped = fit_four_parameter_logistic(prediction, target)
    return {
        "srcc": float(stats.spearmanr(prediction, target).statistic),
        "plcc": float(stats.pearsonr(mapped, target).statistic),
        "krcc": float(stats.kendalltau(prediction, target).statistic),
        "rmse": float(np.sqrt(np.mean((mapped - target) ** 2))),
    }
