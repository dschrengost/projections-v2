"""Pipeline guardrail checks for output sanity validation.

These checks run after scoring to detect obviously bad outputs before they
reach downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    warnings: list[str]
    metrics: dict[str, float]


def check_rates_output_sanity(
    df: pd.DataFrame,
    *,
    min_fga2_median: float = 0.15,
    min_top_fpts: float = 45.0,
    fpts_col: str = "fpts_mean",
    zero_tolerance: float = 0.01,
) -> GuardrailResult:
    """Check that rates predictions are plausible.

    Parameters
    ----------
    df
        DataFrame with rates predictions and optionally fpts_mean.
    min_fga2_median
        Minimum expected median for pred_fga2_per_min (default 0.15).
    min_top_fpts
        Minimum expected max fpts_mean on a slate (default 45.0).
    fpts_col
        Name of FPTS column if available.
    zero_tolerance
        Fraction of zeros allowed before warning.

    Returns
    -------
    GuardrailResult
        passed: True if all critical checks pass.
        warnings: List of warning messages.
        metrics: Dict of computed metrics for logging.
    """
    warnings = []
    metrics = {}

    # Check for NaN predictions
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if pred_cols:
        nan_count = df[pred_cols].isna().sum().sum()
        nan_rate = nan_count / (len(df) * len(pred_cols)) if len(df) > 0 else 0.0
        metrics["nan_rate"] = nan_rate
        if nan_count > 0:
            warnings.append(f"Found {nan_count} NaN values in predictions")

    # Check for zero predictions (indicator of bad feature imputation)
    if pred_cols:
        zero_counts = (df[pred_cols] == 0).sum()
        total_zeros = zero_counts.sum()
        zero_rate = total_zeros / (len(df) * len(pred_cols)) if len(df) > 0 else 0.0
        metrics["zero_rate"] = zero_rate
        if zero_rate > zero_tolerance:
            warnings.append(
                f"High zero rate in predictions: {zero_rate:.1%} (tolerance: {zero_tolerance:.1%})"
            )

    # Check fga2_per_min median
    if "pred_fga2_per_min" in df.columns:
        fga2_median = df["pred_fga2_per_min"].median()
        metrics["fga2_per_min_median"] = float(fga2_median) if pd.notna(fga2_median) else 0.0
        if pd.notna(fga2_median) and fga2_median < min_fga2_median:
            warnings.append(
                f"pred_fga2_per_min median ({fga2_median:.3f}) below threshold ({min_fga2_median})"
            )

    # Check top FPTS if available
    if fpts_col in df.columns:
        top_fpts = df[fpts_col].max()
        metrics["top_fpts"] = float(top_fpts) if pd.notna(top_fpts) else 0.0
        if pd.notna(top_fpts) and top_fpts < min_top_fpts:
            warnings.append(
                f"Top {fpts_col} ({top_fpts:.1f}) below threshold ({min_top_fpts}). "
                "This may indicate collapsed rate predictions."
            )

    # Check row count
    metrics["row_count"] = len(df)
    if len(df) == 0:
        warnings.append("Output DataFrame is empty")

    passed = len(warnings) == 0
    return GuardrailResult(passed=passed, warnings=warnings, metrics=metrics)


def check_feature_coverage(
    df: pd.DataFrame,
    *,
    expected_rows: Optional[int] = None,
    min_row_ratio: float = 0.9,
    critical_cols: Optional[list[str]] = None,
) -> GuardrailResult:
    """Check that features have adequate coverage.

    Parameters
    ----------
    df
        Feature DataFrame.
    expected_rows
        Expected number of rows (if known).
    min_row_ratio
        Minimum ratio of actual/expected rows.
    critical_cols
        Columns that should have no nulls.
    """
    warnings = []
    metrics = {"row_count": len(df)}

    if expected_rows is not None and expected_rows > 0:
        ratio = len(df) / expected_rows
        metrics["row_ratio"] = ratio
        if ratio < min_row_ratio:
            warnings.append(
                f"Row count {len(df)} is below expected {expected_rows} "
                f"(ratio: {ratio:.1%} < {min_row_ratio:.1%})"
            )

    if critical_cols:
        for col in critical_cols:
            if col not in df.columns:
                warnings.append(f"Critical column '{col}' missing from features")
            else:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    warnings.append(f"Critical column '{col}' has {null_count} null values")

    passed = len(warnings) == 0
    return GuardrailResult(passed=passed, warnings=warnings, metrics=metrics)


def check_data_freshness(
    df: pd.DataFrame,
    *,
    as_of_ts_col: str = "feature_as_of_ts",
    max_age_hours: float = 24.0,
    reference_ts: Optional[pd.Timestamp] = None,
) -> GuardrailResult:
    """Check that data is fresh enough.

    Parameters
    ----------
    df
        DataFrame with timestamp column.
    as_of_ts_col
        Name of the as-of timestamp column.
    max_age_hours
        Maximum allowed age in hours.
    reference_ts
        Reference timestamp (defaults to now UTC).
    """
    import datetime

    warnings = []
    metrics = {}

    if as_of_ts_col not in df.columns:
        return GuardrailResult(
            passed=True,
            warnings=[f"Timestamp column '{as_of_ts_col}' not present; skipping freshness check"],
            metrics={},
        )

    now = reference_ts or pd.Timestamp.now(tz=datetime.UTC)
    ts_col = pd.to_datetime(df[as_of_ts_col], utc=True, errors="coerce")
    max_ts = ts_col.max()
    min_ts = ts_col.min()

    if pd.notna(max_ts):
        age_hours = (now - max_ts).total_seconds() / 3600
        metrics["max_age_hours"] = age_hours
        metrics["newest_ts"] = max_ts.isoformat()
        if age_hours > max_age_hours:
            warnings.append(
                f"Newest data is {age_hours:.1f} hours old (max allowed: {max_age_hours})"
            )

    if pd.notna(min_ts):
        metrics["oldest_ts"] = min_ts.isoformat()

    null_count = ts_col.isna().sum()
    if null_count > 0:
        metrics["null_ts_count"] = int(null_count)
        warnings.append(f"{null_count} rows have null timestamps")

    passed = len(warnings) == 0
    return GuardrailResult(passed=passed, warnings=warnings, metrics=metrics)


__all__ = [
    "GuardrailResult",
    "check_data_freshness",
    "check_feature_coverage",
    "check_rates_output_sanity",
]
