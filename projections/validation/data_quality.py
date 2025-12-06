"""Data quality range checks for ML features.

Provides guardrails to detect when feature values fall outside expected ranges,
which can indicate data corruption, ETL bugs, or pipeline issues.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Sequence

import pandas as pd


@dataclass
class FeatureBounds:
    """Define expected value range for a single feature column."""

    column: str
    min_val: float | None = None
    max_val: float | None = None
    allow_null: bool = True


# Default bounds for minutes_v1 features
MINUTES_FEATURE_BOUNDS: list[FeatureBounds] = [
    # Target and predictions must be within NBA game bounds
    FeatureBounds("minutes", 0.0, 48.0, allow_null=True),
    FeatureBounds("p50", 0.0, 48.0),
    FeatureBounds("p10", 0.0, 48.0),
    FeatureBounds("p90", 0.0, 48.0),
    FeatureBounds("minutes_p50", 0.0, 48.0),
    FeatureBounds("minutes_p10", 0.0, 48.0),
    FeatureBounds("minutes_p90", 0.0, 48.0),
    # Rolling features
    FeatureBounds("roll_mean_3", 0.0, 48.0),
    FeatureBounds("roll_mean_5", 0.0, 48.0),
    FeatureBounds("roll_mean_10", 0.0, 48.0),
    FeatureBounds("min_last1", 0.0, 48.0),
    FeatureBounds("min_last3", 0.0, 48.0),
    FeatureBounds("min_last5", 0.0, 48.0),
    # Probabilities must be in [0, 1]
    FeatureBounds("prior_play_prob", 0.0, 1.0),
    FeatureBounds("play_prob", 0.0, 1.0),
    FeatureBounds("minutes_pred_play_prob", 0.0, 1.0),
    # Binary flags
    FeatureBounds("is_starter", 0, 1),
    FeatureBounds("starter_flag", 0, 1),
    FeatureBounds("home_flag", 0, 1),
    FeatureBounds("is_out", 0, 1),
    FeatureBounds("is_q", 0, 1),
    FeatureBounds("is_prob", 0, 1),
    FeatureBounds("ramp_flag", 0, 1),
    FeatureBounds("restriction_flag", 0, 1),
    # Reasonable bounds for rest days
    FeatureBounds("days_rest", 0, 30),
    FeatureBounds("days_since_last", 0, 365),
    # Vegas lines
    FeatureBounds("spread_home", -40.0, 40.0),
    FeatureBounds("total", 150.0, 300.0),
]


def validate_feature_ranges(
    df: pd.DataFrame,
    bounds: Sequence[FeatureBounds] | None = None,
    *,
    strict: bool = False,
) -> list[str]:
    """Check features are within expected ranges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns to validate.
    bounds : Sequence[FeatureBounds] | None
        Feature bounds to check. Defaults to MINUTES_FEATURE_BOUNDS.
    strict : bool
        If True, raise ValueError on any violation. If False, emit warning.

    Returns
    -------
    list[str]
        List of violation messages (empty if all checks pass).

    Raises
    ------
    ValueError
        If strict=True and any violations are found.
    """
    if bounds is None:
        bounds = MINUTES_FEATURE_BOUNDS

    violations: list[str] = []

    for b in bounds:
        if b.column not in df.columns:
            continue

        col = df[b.column]

        # Check minimum bound
        if b.min_val is not None:
            below_min = col < b.min_val
            below_count = int(below_min.sum())
            if below_count > 0:
                min_observed = float(col[below_min].min())
                violations.append(
                    f"{b.column}: {below_count} values below {b.min_val} "
                    f"(min observed: {min_observed:.2f})"
                )

        # Check maximum bound
        if b.max_val is not None:
            above_max = col > b.max_val
            above_count = int(above_max.sum())
            if above_count > 0:
                max_observed = float(col[above_max].max())
                violations.append(
                    f"{b.column}: {above_count} values above {b.max_val} "
                    f"(max observed: {max_observed:.2f})"
                )

        # Check unexpected nulls
        if not b.allow_null:
            null_count = int(col.isna().sum())
            if null_count > 0:
                violations.append(f"{b.column}: {null_count} unexpected null values")

    if violations:
        msg = f"Data quality violations detected: {violations}"
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, RuntimeWarning)

    return violations


__all__ = [
    "FeatureBounds",
    "MINUTES_FEATURE_BOUNDS",
    "validate_feature_ranges",
]
