from __future__ import annotations

import numpy as np
import pandas as pd


def derive_rotation_priors(df_priors: pd.DataFrame) -> pd.DataFrame:
    """Derive lightweight rotation priors from minutes quantiles + minutes_prior.

    Input columns (expected):
    - minutes_prior
    - minutes_p10, minutes_p90
    - minutes_p50 (optional)

    Output columns (added; float64; clipped to [0,1]):
    - p_ge5_prior_heur: heuristic P(minutes >= 5)
    - p_eq0_prior_heur: heuristic P(minutes == 0) / DNP-ish

    Notes:
    - Deterministic, monotonic heuristic; uses *only* the above columns.
    - Coherence is enforced so p_eq0 does not wildly contradict p_ge5.
    """
    if "minutes_prior" not in df_priors.columns:
        raise ValueError("df_priors missing required column: minutes_prior")

    df = df_priors.copy()

    def as_minutes(col: str, *, default: float) -> np.ndarray:
        if col not in df.columns:
            return np.full(len(df), float(default), dtype=np.float64)
        return (
            pd.to_numeric(df[col], errors="coerce")
            .astype(np.float64)
            .fillna(float(default))
            .clip(lower=0.0)
            .to_numpy(dtype=np.float64)
        )

    m = as_minutes("minutes_prior", default=0.0)
    p10 = as_minutes("minutes_p10", default=np.nan)
    p50 = as_minutes("minutes_p50", default=np.nan)
    p90 = as_minutes("minutes_p90", default=np.nan)

    # Fill missing quantiles from minutes_prior (or minutes_p50 if present).
    p50 = np.where(np.isfinite(p50), p50, m)
    p10 = np.where(np.isfinite(p10), p10, p50)
    p90 = np.where(np.isfinite(p90), p90, p50)

    # Ensure p10 <= p90 without "fixing" p90 upward based on p50/m.
    p10 = np.maximum(p10, 0.0)
    p90 = np.maximum(p90, 0.0)
    lo = np.minimum(p10, p90)
    hi = np.maximum(p10, p90)
    p10 = lo
    p90 = hi

    is_p10_zero = p10 <= 1e-12

    # Base heuristic for P(minutes >= 5).
    p_ge5 = np.full(len(df), 0.65, dtype=np.float64)
    remaining = np.ones(len(df), dtype=bool)

    mask = remaining & (p90 < 5.0)
    p_ge5[mask] = 0.05
    remaining &= ~mask

    mask = remaining & (p90 < 7.0)
    p_ge5[mask] = 0.15
    remaining &= ~mask

    mask = remaining & (m < 5.0)
    p_ge5[mask] = 0.25
    remaining &= ~mask

    mask = remaining & is_p10_zero & (m < 8.0)
    p_ge5[mask] = 0.35
    remaining &= ~mask

    # Bumps for clearly-rotation-ish. Guard against contradicting a very low p90 (<5).
    bump_ok = p90 >= 5.0
    bump1 = bump_ok & ((m >= 10.0) | (p90 >= 12.0))
    bump2 = bump_ok & ((m >= 18.0) | (p90 >= 20.0))
    bump3 = bump_ok & ((m >= 24.0) | (p90 >= 28.0))
    p_ge5 = np.where(bump1, np.maximum(p_ge5, 0.80), p_ge5)
    p_ge5 = np.where(bump2, np.maximum(p_ge5, 0.92), p_ge5)
    p_ge5 = np.where(bump3, np.maximum(p_ge5, 0.97), p_ge5)

    # Base heuristic for P(minutes == 0).
    p0 = np.full(len(df), 0.10, dtype=np.float64)
    remaining = np.ones(len(df), dtype=bool)

    mask = remaining & (p90 < 5.0)
    p0[mask] = 0.70
    remaining &= ~mask

    mask = remaining & (p90 < 7.0)
    p0[mask] = 0.55
    remaining &= ~mask

    mask = remaining & (m < 5.0)
    p0[mask] = 0.40
    remaining &= ~mask

    mask = remaining & is_p10_zero & (m < 8.0)
    p0[mask] = 0.30
    remaining &= ~mask

    # Reduce for clearly-rotation-ish (same bump gating).
    p0 = np.where(bump1, np.minimum(p0, 0.15), p0)
    p0 = np.where(bump2, np.minimum(p0, 0.05), p0)
    p0 = np.where(bump3, np.minimum(p0, 0.02), p0)

    p_ge5 = np.clip(p_ge5, 0.0, 1.0)
    p0 = np.clip(p0, 0.0, 1.0)

    # Coherence: don't allow p0 to contradict p_ge5 wildly.
    p0_max = 1.0 - p_ge5 + 0.05
    p0 = np.minimum(p0, p0_max)
    p0 = np.clip(p0, 0.0, 1.0)

    df["p_ge5_prior_heur"] = p_ge5.astype(np.float64)
    df["p_eq0_prior_heur"] = p0.astype(np.float64)
    return df
