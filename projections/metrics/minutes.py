"""Minutes-specific metric helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

# Default buckets on ACTUAL minutes
# [0, 10), [10, 20), [20, 30), [30, inf)
DEFAULT_MINUTES_BUCKETS: Tuple[Tuple[float, float], ...] = (
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 30.0),
    (30.0, np.inf),
)


def bucket_name(lo: float, hi: float) -> str:
    """Return a readable name for a minutes bucket."""

    if hi == np.inf:
        return f"{int(lo)}_plus"
    return f"{int(lo)}_{int(hi)}"


def compute_mae_by_actual_minutes_bucket(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    buckets: Tuple[Tuple[float, float], ...] = DEFAULT_MINUTES_BUCKETS,
) -> Dict[str, float]:
    """Compute MAE of ``y_pred`` vs ``y_true`` within buckets defined on actual minutes."""

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"Shapes differ: y_true={y_true_arr.shape}, y_pred={y_pred_arr.shape}")

    abs_err = np.abs(y_pred_arr - y_true_arr)
    metrics: Dict[str, float] = {}

    for lo, hi in buckets:
        mask = (y_true_arr >= lo) & (y_true_arr < hi)
        name = bucket_name(lo, hi)
        if not np.any(mask):
            metrics[f"mae_{name}"] = float("nan")
            continue
        metrics[f"mae_{name}"] = float(abs_err[mask].mean())

    metrics["mae_overall"] = float(abs_err.mean())
    return metrics
