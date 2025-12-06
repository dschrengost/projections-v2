"""Monitor snapshot staleness at inference time.

Ensures that injury/odds/roster snapshots are fresh enough relative to game tip
times. Stale data can lead to predictions based on outdated player availability.
"""

from __future__ import annotations

from datetime import timedelta
import warnings
from typing import Any

import pandas as pd


DEFAULT_MAX_STALENESS = timedelta(hours=6)


def check_snapshot_staleness(
    df: pd.DataFrame,
    *,
    as_of_col: str = "injury_as_of_ts",
    tip_col: str = "tip_ts",
    max_staleness: timedelta = DEFAULT_MAX_STALENESS,
    warn_only: bool = True,
) -> dict[str, Any]:
    """Check if snapshots are too stale relative to tip time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing timestamp columns.
    as_of_col : str
        Column containing the snapshot as-of timestamp.
    tip_col : str
        Column containing the game tip-off timestamp.
    max_staleness : timedelta
        Maximum allowed gap between as_of_ts and tip_ts.
    warn_only : bool
        If True, emit warning. If False, raise ValueError on stale data.

    Returns
    -------
    dict[str, Any]
        Staleness check results including max observed staleness and stale row count.

    Raises
    ------
    ValueError
        If warn_only=False and stale rows are detected.
    """
    if as_of_col not in df.columns or tip_col not in df.columns:
        return {
            "checked": False,
            "reason": f"missing columns: {as_of_col} or {tip_col}",
        }

    if df.empty:
        return {
            "checked": True,
            "max_staleness_hours": 0.0,
            "stale_rows": 0,
            "total_rows": 0,
        }

    as_of = pd.to_datetime(df[as_of_col], utc=True, errors="coerce")
    tip = pd.to_datetime(df[tip_col], utc=True, errors="coerce")

    # Only check rows where both timestamps are present
    valid_mask = as_of.notna() & tip.notna()
    if not valid_mask.any():
        return {
            "checked": True,
            "max_staleness_hours": None,
            "stale_rows": 0,
            "total_rows": len(df),
            "valid_rows": 0,
        }

    staleness = tip[valid_mask] - as_of[valid_mask]
    max_observed = staleness.max()
    max_staleness_hours = max_observed.total_seconds() / 3600 if pd.notna(max_observed) else 0.0

    stale_mask = staleness > max_staleness
    stale_rows = int(stale_mask.sum())

    result = {
        "checked": True,
        "max_staleness_hours": round(max_staleness_hours, 2),
        "stale_rows": stale_rows,
        "total_rows": len(df),
        "valid_rows": int(valid_mask.sum()),
        "threshold_hours": max_staleness.total_seconds() / 3600,
    }

    if stale_rows > 0:
        msg = (
            f"Staleness alert: {stale_rows}/{valid_mask.sum()} rows have {as_of_col} "
            f">{max_staleness} before tip (max: {max_staleness_hours:.1f}h)"
        )
        if warn_only:
            warnings.warn(msg, RuntimeWarning)
        else:
            raise ValueError(msg)

    return result


def check_all_snapshot_staleness(
    df: pd.DataFrame,
    *,
    max_staleness: timedelta = DEFAULT_MAX_STALENESS,
    warn_only: bool = True,
) -> dict[str, dict[str, Any]]:
    """Check staleness for all common snapshot columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature/snapshot columns.
    max_staleness : timedelta
        Maximum allowed staleness for all snapshot types.
    warn_only : bool
        If True, emit warnings. If False, raise on first stale column.

    Returns
    -------
    dict[str, dict[str, Any]]
        Results keyed by snapshot type (injury, odds, roster, feature).
    """
    snapshot_columns = [
        ("injury", "injury_as_of_ts"),
        ("odds", "odds_as_of_ts"),
        ("roster", "roster_as_of_ts"),
        ("feature", "feature_as_of_ts"),
    ]

    results = {}
    for name, col in snapshot_columns:
        results[name] = check_snapshot_staleness(
            df,
            as_of_col=col,
            max_staleness=max_staleness,
            warn_only=warn_only,
        )

    return results


__all__ = [
    "DEFAULT_MAX_STALENESS",
    "check_snapshot_staleness",
    "check_all_snapshot_staleness",
]
