"""Monitoring skeleton for Minutes V1 Quick Start."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Container for monitoring outputs."""

    overall: dict[str, float]
    rolling: pd.DataFrame


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def compute_monitoring_snapshot(
    df: pd.DataFrame,
    *,
    prediction_col: str = "p50",
    label_col: str = "minutes",
    p10_col: str = "p10_calibrated",
    p90_col: str = "p90_calibrated",
    error_threshold: float = 6.0,
    rolling_window: int = 7,
) -> MonitoringSnapshot:
    """Compute overall + rolling monitoring metrics."""

    required_cols = {prediction_col, label_col, p10_col, p90_col, "game_date", "tip_ts", "feature_as_of_ts"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for monitoring: {', '.join(sorted(missing))}")

    working = df.copy()
    working["game_date"] = pd.to_datetime(working["game_date"]).dt.normalize()
    working["tip_ts"] = _ensure_datetime(working["tip_ts"])
    working["feature_as_of_ts"] = _ensure_datetime(working["feature_as_of_ts"])

    working["error"] = working[prediction_col] - working[label_col]
    working["abs_error"] = working["error"].abs()
    working["err_gt_threshold"] = (working["abs_error"] > error_threshold).astype(float)
    working["p10_hit"] = (working[label_col] >= working[p10_col]).astype(float)
    working["p90_hit"] = (working[label_col] <= working[p90_col]).astype(float)

    freshness_minutes = (
        (working["tip_ts"] - working["feature_as_of_ts"]).dt.total_seconds() / 60.0
    )

    overall = {
        "mae": float(working["abs_error"].mean()),
        "p_gt_err_threshold": float(working["err_gt_threshold"].mean()),
        "p10_coverage": float(working["p10_hit"].mean()),
        "p90_coverage": float(working["p90_hit"].mean()),
        "freshness_minutes_mean": float(freshness_minutes.mean()),
        "freshness_minutes_max": float(freshness_minutes.max()),
    }

    daily = (
        working.groupby("game_date")[
            ["abs_error", "err_gt_threshold", "p10_hit", "p90_hit"]
        ]
        .mean()
        .rename(
            columns={
                "abs_error": "mae",
                "err_gt_threshold": "p_err_gt_threshold",
                "p10_hit": "p10_coverage",
                "p90_hit": "p90_coverage",
            }
        )
    )
    daily["rolling_mae"] = daily["mae"].rolling(window=rolling_window, min_periods=1).mean()
    return MonitoringSnapshot(overall=overall, rolling=daily.reset_index())
