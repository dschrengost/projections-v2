"""Helpers for canonical guard/wing/big position buckets."""

from __future__ import annotations

import pandas as pd

from projections.minutes_v1.constants import ARCHETYPE_MAP

_CANONICAL = {"G": "G", "W": "W", "B": "BIG", "BIG": "BIG"}


def canonical_pos_bucket(value: str | None) -> str:
    if not value:
        return "UNK"
    text = str(value).strip().upper()
    mapped = ARCHETYPE_MAP.get(text, text)
    canonical = _CANONICAL.get(mapped, None)
    if canonical:
        return canonical
    return "UNK"


def canonical_pos_bucket_series(series: pd.Series) -> pd.Series:
    return series.apply(canonical_pos_bucket)
