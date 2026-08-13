from __future__ import annotations

"""Safe, resource-light entry point for NCAT external-validation statistics."""

import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_ncat_external_validation as workflow


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation using an explicit scalar formula."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or x.shape != y.shape:
        return math.nan
    dx = x - float(np.mean(x))
    dy = y - float(np.mean(y))
    denom = math.sqrt(float(np.sum(dx * dx)) * float(np.sum(dy * dy)))
    return float(np.sum(dx * dy) / denom) if denom > 0 else math.nan


def core_metrics(measured: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if measured.shape != predicted.shape or measured.ndim != 1 or not len(measured):
        raise ValueError("measured and predicted must be non-empty one-dimensional arrays")
    if np.any(~np.isfinite(measured)) or np.any(~np.isfinite(predicted)):
        raise ValueError("measured and predicted must be finite")
    if np.any(measured <= 0):
        raise ValueError("measured rutting must be positive for MAPE")
    error = predicted - measured
    ss_tot = float(np.sum((measured - float(np.mean(measured))) ** 2))
    return {
        "n": int(len(measured)),
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": math.sqrt(float(np.mean(error ** 2))),
        "mean_bias_mm": float(np.mean(error)),
        "mape_pct": float(np.mean(np.abs(error) / measured) * 100.0),
        "r2": float(1.0 - np.sum(error ** 2) / ss_tot) if ss_tot > 0 else math.nan,
        "pearson_r": correlation(measured, predicted),
        "spearman_rho": correlation(workflow.ranks(measured), workflow.ranks(predicted)),
    }


def bootstrap_intervals(measured: np.ndarray, predicted: np.ndarray,
                        n_boot: int, seed: int) -> Dict[str, List[float]]:
    if n_boot <= 0:
        return {}
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if measured.shape != predicted.shape or measured.ndim != 1 or not len(measured):
        raise ValueError("measured and predicted must be non-empty one-dimensional arrays")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(measured), size=(n_boot, len(measured)))
    sampled_error = (predicted - measured)[idx]
    values = {
        "mae_mm": np.mean(np.abs(sampled_error), axis=1),
        "rmse_mm": np.sqrt(np.mean(sampled_error ** 2, axis=1)),
        "mean_bias_mm": np.mean(sampled_error, axis=1),
    }
    return {
        f"{key}_95pct_ci": [float(x) for x in np.percentile(value, [2.5, 97.5])]
        for key, value in values.items()
    }


workflow.correlation = correlation
workflow.core_metrics = core_metrics
workflow.bootstrap_intervals = bootstrap_intervals


if __name__ == "__main__":
    workflow.main()
