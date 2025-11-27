"""Evaluation helpers for FPTS per-minute models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_BASELINE_PRIOR = "fpts_per_min_prior_10"
METRIC_KEYS: tuple[str, ...] = (
    "mae_per_min",
    "rmse_per_min",
    "smape_per_min",
    "mae_fpts",
    "rmse_fpts",
    "smape_fpts",
)

__all__ = [
    "DEFAULT_BASELINE_PRIOR",
    "METRIC_KEYS",
    "baseline_per_minute",
    "evaluate_model_vs_baseline",
    "evaluate_fpts_run",
]


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=1e-6)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 2.0)


def _metric_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minutes: np.ndarray,
    fpts_true: np.ndarray,
) -> dict[str, float]:
    fpts_pred = y_pred * minutes
    mse = mean_squared_error(y_true, y_pred)
    mse_fpts = mean_squared_error(fpts_true, fpts_pred)
    return {
        "mae_per_min": float(mean_absolute_error(y_true, y_pred)),
        "rmse_per_min": float(np.sqrt(mse)),
        "smape_per_min": _smape(y_true, y_pred),
        "mae_fpts": float(mean_absolute_error(fpts_true, fpts_pred)),
        "rmse_fpts": float(np.sqrt(mse_fpts)),
        "smape_fpts": _smape(fpts_true, fpts_pred),
    }


def baseline_per_minute(
    df: pd.DataFrame,
    *,
    prior_col: str = DEFAULT_BASELINE_PRIOR,
) -> np.ndarray:
    """Return the rolling prior column used for the baseline."""

    if prior_col not in df.columns:
        raise KeyError(f"Baseline prior column '{prior_col}' missing from dataframe.")
    baseline = pd.to_numeric(df[prior_col], errors="coerce")
    if baseline.isna().any():
        fallback = df.get("fpts_per_min_prior_5")
        if fallback is not None:
            fallback_numeric = pd.to_numeric(fallback, errors="coerce")
            baseline = baseline.fillna(fallback_numeric)
    return baseline.fillna(0.0).to_numpy(dtype=float)


def _bucket_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    starter_mask = df.get("starter_flag")
    if starter_mask is not None:
        starter_bool = starter_mask.fillna(0).astype(int) > 0
        masks["starters"] = starter_bool
        masks["bench"] = ~starter_bool

    if "minutes_p50_pred" in df.columns:
        high_minutes = df["minutes_p50_pred"] >= 28.0
        masks["high_minutes"] = high_minutes
        masks["low_minutes"] = ~high_minutes

    if {"spread_home", "home_flag"}.issubset(df.columns):
        spread = pd.to_numeric(df["spread_home"], errors="coerce")
        home_flag = df["home_flag"].fillna(0).astype(int)
        team_spread = pd.Series(
            np.where(home_flag == 1, spread, -spread), index=df.index
        )
        masks["favorites"] = team_spread <= -5.0
        masks["underdogs"] = team_spread >= 5.0

    if "teammate_out_count" in df.columns:
        heavy_injury = df["teammate_out_count"] >= 2
        masks["injury_struck"] = heavy_injury
        masks["full_strength"] = ~heavy_injury

    return masks


def _metrics_from_frame(frame: pd.DataFrame, preds: np.ndarray) -> dict[str, float]:
    y_true = frame["fpts_per_min_actual"].to_numpy(dtype=float)
    minutes = frame["actual_minutes"].to_numpy(dtype=float)
    fpts_true = frame["actual_fpts"].to_numpy(dtype=float)
    return _metric_summary(y_true, preds, minutes, fpts_true)


def _compare_masks(
    df: pd.DataFrame,
    masks: dict[str, pd.Series],
    *,
    model_preds: np.ndarray,
    baseline_preds: np.ndarray,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        subset = df.loc[mask]
        if subset.empty:
            continue
        idx = mask.to_numpy(dtype=bool)
        baseline_metrics = _metrics_from_frame(subset, baseline_preds[idx])
        model_metrics = _metrics_from_frame(subset, model_preds[idx])
        delta = {
            key: baseline_metrics[key] - model_metrics[key]
            for key in METRIC_KEYS
        }
        payload: dict[str, Any] = {
            "rows": int(len(subset)),
            "baseline": baseline_metrics,
            "model": model_metrics,
            "delta": delta,
        }
        for key in METRIC_KEYS:
            payload[f"baseline_{key}"] = baseline_metrics[key]
            payload[f"model_{key}"] = model_metrics[key]
            payload[f"delta_{key}"] = delta[key]
        results[name] = payload
    return results


def _bucket_comparisons(
    df: pd.DataFrame,
    *,
    model_preds: np.ndarray,
    baseline_preds: np.ndarray,
) -> dict[str, dict[str, Any]]:
    return _compare_masks(
        df,
        _bucket_masks(df),
        model_preds=model_preds,
        baseline_preds=baseline_preds,
    )


def evaluate_model_vs_baseline(
    df: pd.DataFrame,
    *,
    model_preds: np.ndarray,
    baseline_preds: np.ndarray,
) -> dict[str, Any]:
    """Compute baseline vs model metrics (overall + buckets)."""

    if len(df) != len(model_preds) or len(df) != len(baseline_preds):
        raise ValueError("Prediction arrays must align with dataframe rows.")

    y_true = df["fpts_per_min_actual"].to_numpy(dtype=float)
    minutes = df["actual_minutes"].to_numpy(dtype=float)
    fpts_true = df["actual_fpts"].to_numpy(dtype=float)
    baseline_metrics = _metric_summary(y_true, baseline_preds, minutes, fpts_true)
    model_metrics = _metric_summary(y_true, model_preds, minutes, fpts_true)
    delta = {
        key: baseline_metrics[key] - model_metrics[key]
        for key in METRIC_KEYS
    }
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "baseline": baseline_metrics,
        "model": model_metrics,
        "delta": delta,
        "buckets": _bucket_comparisons(
            df, model_preds=model_preds, baseline_preds=baseline_preds
        ),
    }
    for key in METRIC_KEYS:
        summary[f"baseline_{key}"] = baseline_metrics[key]
        summary[f"model_{key}"] = model_metrics[key]
        summary[f"delta_{key}"] = delta[key]
    return summary


def _range_bucket_masks(
    series: pd.Series | None,
    ranges: list[tuple[str, float | None, float | None]],
) -> dict[str, pd.Series]:
    if series is None:
        return {}
    numeric = pd.to_numeric(series, errors="coerce")
    masks: dict[str, pd.Series] = {}
    for label, lower, upper in ranges:
        mask = numeric.notna()
        if lower is not None:
            mask &= numeric >= lower
        if upper is not None:
            mask &= numeric < upper
        if mask.any():
            masks[label] = mask
    return masks


def evaluate_fpts_run(
    df: pd.DataFrame,
    *,
    model_preds: np.ndarray,
    baseline_preds: np.ndarray,
) -> dict[str, Any]:
    """Evaluate model vs baseline overall plus DFS-friendly slices."""

    if len(df) == 0:
        raise ValueError("Evaluation dataframe is empty.")

    overall = evaluate_model_vs_baseline(
        df,
        model_preds=model_preds,
        baseline_preds=baseline_preds,
    )
    results: dict[str, Any] = {"overall": overall}
    base_buckets = overall.get("buckets", {})
    if base_buckets:
        results["by_role_context"] = base_buckets

    model_series = pd.Series(model_preds, index=df.index)

    proj_raw = df.get("proj_fpts")
    proj_series = (
        pd.to_numeric(proj_raw, errors="coerce") if proj_raw is not None else None
    )
    if proj_series is None or proj_series.isna().all():
        minutes_raw = df.get("minutes_p50_pred")
        if minutes_raw is not None:
            minutes = pd.to_numeric(minutes_raw, errors="coerce").fillna(0.0)
            proj_series = model_series * minutes
        else:
            actual_raw = df.get("actual_minutes")
            actual_minutes = (
                pd.to_numeric(actual_raw, errors="coerce").fillna(0.0)
                if actual_raw is not None
                else pd.Series(0.0, index=df.index)
            )
            proj_series = model_series * actual_minutes
    proj_masks = _range_bucket_masks(
        proj_series,
        [
            ("<20 FPTS", None, 20.0),
            ("20-35 FPTS", 20.0, 35.0),
            ("35-50 FPTS", 35.0, 50.0),
            (">=50 FPTS", 50.0, None),
        ],
    )
    if proj_masks:
        results["by_proj_fpts_bucket"] = _compare_masks(
            df,
            proj_masks,
            model_preds=model_preds,
            baseline_preds=baseline_preds,
        )

    minutes_raw = df.get("minutes_p50_pred")
    minutes_series = (
        pd.to_numeric(minutes_raw, errors="coerce") if minutes_raw is not None else None
    )
    if minutes_series is None or minutes_series.isna().all():
        actual_raw = df.get("actual_minutes")
        minutes_series = (
            pd.to_numeric(actual_raw, errors="coerce")
            if actual_raw is not None
            else None
        )
    minute_masks = _range_bucket_masks(
        minutes_series,
        [
            ("<20 minutes", None, 20.0),
            ("20-28 minutes", 20.0, 28.0),
            ("28-34 minutes", 28.0, 34.0),
            (">=34 minutes", 34.0, None),
        ],
    )
    if minute_masks:
        results["by_minutes_bucket"] = _compare_masks(
            df,
            minute_masks,
            model_preds=model_preds,
            baseline_preds=baseline_preds,
        )

    injury_raw = df.get("teammate_out_count")
    injury_series = (
        pd.to_numeric(injury_raw, errors="coerce") if injury_raw is not None else None
    )
    if injury_series is not None:
        injury_masks: dict[str, pd.Series] = {}
        zeros = injury_series.fillna(0)
        none_mask = zeros == 0
        low_mask = (zeros >= 1) & (zeros <= 2)
        high_mask = zeros >= 3
        if none_mask.any():
            injury_masks["no_injuries"] = none_mask
        if low_mask.any():
            injury_masks["1-2_out"] = low_mask
        if high_mask.any():
            injury_masks["3+_out"] = high_mask
        if injury_masks:
            results["by_injury_context"] = _compare_masks(
                df,
                injury_masks,
                model_preds=model_preds,
                baseline_preds=baseline_preds,
            )

    volatility_raw = df.get("minutes_volatility_pred")
    volatility_series = (
        pd.to_numeric(volatility_raw, errors="coerce")
        if volatility_raw is not None
        else None
    )
    if volatility_series is None or volatility_series.isna().all():
        if {"minutes_p90_pred", "minutes_p10_pred"}.issubset(df.columns):
            volatility_series = (
                pd.to_numeric(df["minutes_p90_pred"], errors="coerce")
                - pd.to_numeric(df["minutes_p10_pred"], errors="coerce")
            )
    volatility_masks = _range_bucket_masks(
        volatility_series,
        [
            ("0-4 volatility", None, 4.0),
            ("4-8 volatility", 4.0, 8.0),
            (">8 volatility", 8.0, None),
        ],
    )
    if volatility_masks:
        results["by_volatility_bucket"] = _compare_masks(
            df,
            volatility_masks,
            model_preds=model_preds,
            baseline_preds=baseline_preds,
        )

    salary_series: pd.Series | None = None
    for col in ("dk_salary", "salary"):
        if col in df.columns:
            candidate = pd.to_numeric(df[col], errors="coerce")
            if not candidate.isna().all():
                salary_series = candidate
                break
    salary_masks = _range_bucket_masks(
        salary_series,
        [
            ("<4k salary", None, 4000.0),
            ("4-6k salary", 4000.0, 6000.0),
            ("6-8k salary", 6000.0, 8000.0),
            (">=8k salary", 8000.0, None),
        ],
    )
    if salary_masks:
        results["by_salary_bucket"] = _compare_masks(
            df,
            salary_masks,
            model_preds=model_preds,
            baseline_preds=baseline_preds,
        )

    return results
