from __future__ import annotations

import pandas as pd


def period_length_sec(period: int) -> int:
    """Return the length of a period in seconds (NBA regulation + OT)."""

    return 720 if int(period) <= 4 else 300


def _clock_from_period_elapsed(*, period: int, period_elapsed_sec: int) -> int:
    length = period_length_sec(period)
    return int(length - int(period_elapsed_sec))


def ensure_clock_sec_columns(stints: pd.DataFrame) -> pd.DataFrame:
    """Ensure `start_clock_sec`/`end_clock_sec` exist on a stints-like frame.

    Phase 1 `stints.parquet` should already include these, but the dataset builder
    is robust to older exports that only have period-elapsed seconds.
    """

    if {"start_clock_sec", "end_clock_sec"}.issubset(stints.columns):
        return stints
    required = {"period", "start_period_elapsed_sec", "end_period_elapsed_sec"}
    missing = required - set(stints.columns)
    if missing:
        raise ValueError(f"Cannot derive clock seconds; missing columns: {sorted(missing)}")
    out = stints.copy()
    out["start_clock_sec"] = [
        _clock_from_period_elapsed(period=int(p), period_elapsed_sec=int(s))
        for p, s in zip(out["period"].tolist(), out["start_period_elapsed_sec"].tolist())
    ]
    out["end_clock_sec"] = [
        _clock_from_period_elapsed(period=int(p), period_elapsed_sec=int(s))
        for p, s in zip(out["period"].tolist(), out["end_period_elapsed_sec"].tolist())
    ]
    return out

