"""Routing utilities for selecting minutes models by time-to-tip."""

from __future__ import annotations

import pandas as pd


def late_model_weight(
    time_to_tip_min: pd.Series,
    *,
    late_threshold_min: float = 60.0,
    blend_band_min: float = 30.0,
    missing_defaults_to_late: bool = True,
) -> pd.Series:
    """Return a [0, 1] weight for blending late vs early models.

    - `1.0` => 100% late model
    - `0.0` => 100% early model

    Weight schedule (linear blend):
      - time_to_tip <= late_threshold: weight=1
      - time_to_tip >= late_threshold + blend_band: weight=0
      - otherwise: linear interpolation between them
    """

    t = pd.to_numeric(time_to_tip_min, errors="coerce")
    # Defensive: if timestamps are inconsistent and yield negative time_to_tip,
    # treat as "at lock" for routing purposes.
    t = t.where(t >= 0, 0.0)

    if blend_band_min <= 0:
        w = (t <= float(late_threshold_min)).astype(float)
    else:
        threshold = float(late_threshold_min)
        upper = threshold + float(blend_band_min)
        w = (upper - t) / float(blend_band_min)
        w = w.clip(lower=0.0, upper=1.0)

    if missing_defaults_to_late:
        return w.where(t.notna(), 1.0)
    return w


def minutes_model_used_label(
    late_weight: pd.Series,
    *,
    eps: float = 1e-9,
    early_label: str = "early",
    late_label: str = "late",
    blend_label: str = "blend",
) -> pd.Series:
    """Convert a late-weight series into discrete model-used labels."""

    w = pd.to_numeric(late_weight, errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    out = pd.Series(blend_label, index=w.index, dtype="string")
    out.loc[w <= eps] = early_label
    out.loc[w >= 1.0 - eps] = late_label
    return out

