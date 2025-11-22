"""Asymmetric tail scaling helpers for post-conformal calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AsymmetricK:
    """Simple container for asymmetric tail scaling factors."""

    k_low: float
    k_high: float


def compute_coverage(y_true: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> Tuple[float, float]:
    """Return (cov10, cov90) where cov10 = P(y <= q10), cov90 = P(y <= q90)."""

    cov10 = float(np.mean(y_true <= q10))
    cov90 = float(np.mean(y_true <= q90))
    return cov10, cov90


def apply_asymmetric_k(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    k: AsymmetricK,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (q10', q90') after asymmetric scaling around q50."""

    d_low = q50 - q10
    d_high = q90 - q50
    q10p = q50 - k.k_low * d_low
    q90p = q50 + k.k_high * d_high
    return q10p, q90p


def fit_global_asymmetric_k(
    df_cal: pd.DataFrame,
    *,
    minutes_col: str = "minutes",
    q10_col: str = "minutes_p10",
    q50_col: str = "minutes_p50",
    q90_col: str = "minutes_p90",
    k_low_grid: Iterable[float] | None = None,
    k_high_grid: Iterable[float] | None = None,
    target_low: float = 0.10,
    target_high: float = 0.90,
) -> AsymmetricK:
    """
    Grid search k_low, k_high on calibration set to minimize
    (cov10 - target_low)^2 + (cov90 - target_high)^2.
    """

    if k_low_grid is None:
        k_low_grid = np.arange(0.5, 3.05, 0.1)
    if k_high_grid is None:
        k_high_grid = np.arange(0.5, 1.55, 0.05)

    y = df_cal[minutes_col].to_numpy(dtype=float)
    q10 = df_cal[q10_col].to_numpy(dtype=float)
    q50 = df_cal[q50_col].to_numpy(dtype=float)
    q90 = df_cal[q90_col].to_numpy(dtype=float)

    best_loss = float("inf")
    best_k = AsymmetricK(k_low=1.0, k_high=1.0)

    for kl in k_low_grid:
        for kh in k_high_grid:
            q10p, q90p = apply_asymmetric_k(q10, q50, q90, AsymmetricK(kl, kh))
            cov10, cov90 = compute_coverage(y, q10p, q90p)
            loss = (cov10 - target_low) ** 2 + (cov90 - target_high) ** 2
            if loss < best_loss:
                best_loss = loss
                best_k = AsymmetricK(k_low=float(kl), k_high=float(kh))

    return best_k
