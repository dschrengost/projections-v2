"""Simple post-hoc calibration utilities for minutes quantiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass
class MinutesCalibrationParams:
    """Per-group scale factors for tail calibration."""

    group_keys: list[str]
    k_low: dict[tuple[Any, ...], float]
    k_high: dict[tuple[Any, ...], float]
    fallback_k_low: float = 1.0
    fallback_k_high: float = 1.0


@dataclass
class StarKHighParams:
    """Single k_high for star rows; k_low and non-star rows remain unchanged."""

    p50_threshold: float
    k_high_star: float
    group_keys: list[str] = None

    def __post_init__(self) -> None:
        if self.group_keys is None:
            self.group_keys = ["starter_flag"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "p50_threshold": float(self.p50_threshold),
            "k_high_star": float(self.k_high_star),
            "group_keys": self.group_keys,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StarKHighParams":
        return cls(
            p50_threshold=float(payload.get("p50_threshold", 32.0)),
            k_high_star=float(payload.get("k_high_star", 1.0)),
            group_keys=list(payload.get("group_keys") or ["starter_flag"]),
        )


def _compute_band_error(rate: float, band: tuple[float, float]) -> float:
    lo, hi = band
    if rate < lo:
        return (rate - lo) ** 2
    if rate > hi:
        return (rate - hi) ** 2
    return 0.0


def _star_mask(
    df: pd.DataFrame,
    *,
    p50_threshold: float,
    starter_col: str = "starter_flag",
    p50_col: str = "minutes_p50",
) -> pd.Series:
    return (df[starter_col] == 1) & (pd.to_numeric(df[p50_col], errors="coerce") >= p50_threshold)


def _compute_is_playable_mask(
    df: pd.DataFrame,
    *,
    target_col: str,
    threshold: float,
    status_col: str | None = None,
    existing_col: str | None = None,
) -> pd.Series:
    if existing_col and existing_col in df.columns:
        return pd.to_numeric(df[existing_col], errors="coerce").fillna(0).astype(bool)
    mask = pd.to_numeric(df[target_col], errors="coerce") >= threshold
    if status_col and status_col in df.columns:
        mask &= df[status_col].astype(str).str.upper() != "OUT"
    return mask


def fit_k_params(
    df: pd.DataFrame,
    group_keys: list[str],
    *,
    target_col: str = "minutes",
    minutes_p10_col: str = "minutes_p10",
    minutes_p50_col: str = "minutes_p50",
    minutes_p90_col: str = "minutes_p90",
    status_col: str | None = "status",
    is_playable_col: str = "is_playable",
    playable_minutes_threshold: float = 10.0,
    target_p10: float = 0.10,
    target_p90: float = 0.90,
    p10_band: tuple[float, float] = (0.08, 0.12),
    p90_band: tuple[float, float] = (0.87, 0.94),
    k_low_grid: np.ndarray | None = None,
    k_high_grid: np.ndarray | None = None,
    min_count: int = 200,
) -> MinutesCalibrationParams:
    """Fit per-group tail scaling factors on playable rows."""

    if k_low_grid is None:
        k_low_grid = np.linspace(0.3, 1.2, 91)  # 0.01 step
    if k_high_grid is None:
        k_high_grid = np.linspace(0.7, 2.0, 131)  # ~0.01 step

    working = df.copy()
    working[is_playable_col] = _compute_is_playable_mask(
        working,
        target_col=target_col,
        threshold=playable_minutes_threshold,
        status_col=status_col,
        existing_col=is_playable_col,
    )
    playable = working[working[is_playable_col]].copy()
    if playable.empty:
        raise ValueError("No playable rows available for calibration.")

    def _fit_band(
        y: np.ndarray,
        p50: np.ndarray,
        deltas: np.ndarray,
        grid: np.ndarray,
        band: tuple[float, float],
        *,
        is_low: bool,
    ) -> float:
        best_k = float(grid[0])
        best_err = float("inf")
        for k in grid:
            if k < 0:
                continue
            if is_low:
                q = p50 - k * deltas
                rate = float(np.mean(y < q))
            else:
                q = p50 + k * deltas
                rate = float(np.mean(y <= q))
            err = _compute_band_error(rate, band)
            if err < best_err or (err == best_err and k < best_k):
                best_err = err
                best_k = float(k)
        return best_k

    results_low: dict[tuple[Any, ...], float] = {}
    results_high: dict[tuple[Any, ...], float] = {}
    missing_groups: list[tuple[Any, ...]] = []

    grouped = playable.groupby(group_keys, dropna=False)
    for group_key, group in grouped:
        key_tuple = (group_key,) if not isinstance(group_key, tuple) else tuple(group_key)
        if len(group) < min_count:
            missing_groups.append(key_tuple)
            continue
        y = group[target_col].to_numpy(dtype=float)
        p10_raw = group[minutes_p10_col].to_numpy(dtype=float)
        p50 = group[minutes_p50_col].to_numpy(dtype=float)
        p90_raw = group[minutes_p90_col].to_numpy(dtype=float)

        left = np.maximum(p50 - p10_raw, 0.0)
        right = np.maximum(p90_raw - p50, 0.0)
        k_low_star = _fit_band(y, p50, left, k_low_grid, p10_band, is_low=True)
        k_high_star = _fit_band(y, p50, right, k_high_grid, p90_band, is_low=False)
        results_low[key_tuple] = k_low_star
        results_high[key_tuple] = k_high_star

    # Fallback/global parameters for sparse groups.
    y_all = playable[target_col].to_numpy(dtype=float)
    p10_all = playable[minutes_p10_col].to_numpy(dtype=float)
    p50_all = playable[minutes_p50_col].to_numpy(dtype=float)
    p90_all = playable[minutes_p90_col].to_numpy(dtype=float)
    fallback_left = np.maximum(p50_all - p10_all, 0.0)
    fallback_right = np.maximum(p90_all - p50_all, 0.0)
    fallback_k_low = _fit_band(y_all, p50_all, fallback_left, k_low_grid, p10_band, is_low=True)
    fallback_k_high = _fit_band(y_all, p50_all, fallback_right, k_high_grid, p90_band, is_low=False)
    for key in missing_groups:
        results_low.setdefault(key, fallback_k_low)
        results_high.setdefault(key, fallback_k_high)

    return MinutesCalibrationParams(
        group_keys=group_keys,
        k_low=results_low,
        k_high=results_high,
        fallback_k_low=fallback_k_low,
        fallback_k_high=fallback_k_high,
    )


def apply_k_params(
    df: pd.DataFrame,
    params: MinutesCalibrationParams,
    *,
    minutes_p10_col: str = "minutes_p10",
    minutes_p50_col: str = "minutes_p50",
    minutes_p90_col: str = "minutes_p90",
    out_p10_col: str = "minutes_p10_cal",
    out_p50_col: str = "minutes_p50_cal",
    out_p90_col: str = "minutes_p90_cal",
) -> pd.DataFrame:
    """Apply tail scaling to a dataframe."""

    def _lookup(key: tuple[Any, ...]) -> tuple[float, float]:
        k_low = params.k_low.get(key, params.fallback_k_low)
        k_high = params.k_high.get(key, params.fallback_k_high)
        return float(max(k_low, 0.0)), float(max(k_high, 0.0))

    working = df.copy()
    p10_raw = working[minutes_p10_col].to_numpy(dtype=float)
    p50 = working[minutes_p50_col].to_numpy(dtype=float)
    p90_raw = working[minutes_p90_col].to_numpy(dtype=float)

    k_low_array = np.empty(len(working), dtype=float)
    k_high_array = np.empty(len(working), dtype=float)
    for idx, (_, row) in enumerate(working[params.group_keys].iterrows()):
        key = tuple(row.tolist())
        k_low_array[idx], k_high_array[idx] = _lookup(key)

    left = np.maximum(p50 - p10_raw, 0.0)
    right = np.maximum(p90_raw - p50, 0.0)
    p10_cal = p50 - k_low_array * left
    p90_cal = p50 + k_high_array * right
    p10_cal = np.minimum(p10_cal, p50)
    p90_cal = np.maximum(p90_cal, p50)

    working[out_p10_col] = p10_cal
    working[out_p50_col] = p50
    working[out_p90_col] = p90_cal
    return working


def fit_star_k_high(
    df: pd.DataFrame,
    *,
    p50_threshold: float = 32.0,
    target_p90: float = 0.90,
    p90_band: tuple[float, float] = (0.87, 0.94),
    k_high_grid: np.ndarray | None = None,
    minutes_actual_col: str = "minutes",
    minutes_p50_col: str = "minutes_p50",
    minutes_p90_col: str = "minutes_p90",
    is_playable_col: str = "is_playable",
    status_col: str | None = "status",
    min_count: int = 200,
) -> StarKHighParams:
    """Fit a single k_high >= 1 for star rows; others remain uncalibrated."""

    if k_high_grid is None:
        k_high_grid = np.linspace(1.0, 1.5, 51)  # 0.01 step

    working = df.copy()
    if is_playable_col in working.columns:
        working = working[working[is_playable_col]].copy()
    if status_col and status_col in working.columns:
        working = working[working[status_col].astype(str).str.upper() != "OUT"].copy()

    star_mask = _star_mask(working, p50_threshold=p50_threshold, p50_col=minutes_p50_col)
    stars = working[star_mask].copy()
    if len(stars) < min_count:
        return StarKHighParams(p50_threshold=p50_threshold, k_high_star=1.0)

    y = stars[minutes_actual_col].to_numpy(dtype=float)
    p50 = stars[minutes_p50_col].to_numpy(dtype=float)
    p90_raw = stars[minutes_p90_col].to_numpy(dtype=float)

    def coverage(k_high: float) -> float:
        p90_cal = p50 + k_high * (p90_raw - p50)
        return float((y <= p90_cal).mean())

    best_k = float(k_high_grid[0])
    best_err = float("inf")
    for k in k_high_grid:
        if k < 1.0:
            continue
        rate = coverage(k)
        err = _compute_band_error(rate, p90_band)
        if err < best_err or (err == best_err and abs(k - 1.0) < abs(best_k - 1.0)):
            best_err = err
            best_k = float(k)

    return StarKHighParams(p50_threshold=p50_threshold, k_high_star=best_k)


def apply_star_k_high(
    df: pd.DataFrame,
    params: StarKHighParams,
    *,
    starter_col: str = "starter_flag",
    minutes_p50_col: str = "minutes_p50",
    minutes_p90_col: str = "minutes_p90",
    out_p90_col: str = "minutes_p90_star_cal",
) -> pd.DataFrame:
    """Apply star-only k_high to p90; p10 and p50 remain unchanged."""

    working = df.copy()
    p50 = working[minutes_p50_col].to_numpy(dtype=float)
    p90_raw = working[minutes_p90_col].to_numpy(dtype=float)
    star_mask = _star_mask(working, p50_threshold=params.p50_threshold, starter_col=starter_col, p50_col=minutes_p50_col)
    k_high = np.ones(len(working), dtype=float)
    k_high[star_mask.to_numpy()] = max(params.k_high_star, 1.0)
    p90_cal = p50 + k_high * (p90_raw - p50)
    p90_cal = np.maximum(p90_cal, p50)
    working[out_p90_col] = p90_cal
    return working
