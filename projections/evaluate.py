"""Evaluation helpers for NBA minutes models."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Mapping[str, float]:
    """Compute a standard suite of regression metrics."""

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def print_metrics(metrics: Mapping[str, float]) -> None:
    """Nicely format metrics for terminal output."""

    for name, value in metrics.items():
        print(f"{name.upper():<6}: {value:0.4f}")
