"""Return-from-injury ramp helpers."""

from __future__ import annotations

import pandas as pd

RETURN_INT_COLUMNS: tuple[str, ...] = ("games_since_return", "days_since_return")
RETURN_FLAG_COLUMNS: tuple[str, ...] = ("restriction_flag", "ramp_flag")


def attach_return_ramp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ramp/return columns exist and use consistent dtypes."""

    enriched = df.copy()
    for column in RETURN_FLAG_COLUMNS:
        if column not in enriched:
            enriched[column] = False
        series = enriched[column]
        mask = series.isna()
        if mask.any():
            series = series.astype(object)
            series.loc[mask] = False
        enriched[column] = series.astype(bool)

    for column in RETURN_INT_COLUMNS:
        if column not in enriched:
            enriched[column] = pd.NA
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").astype("Int64")
    return enriched
