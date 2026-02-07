from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRACKING_NUMERIC_FEATURES: tuple[str, ...] = (
    "track_touches_per_min_szn",
    "track_sec_per_touch_szn",
    "track_pot_ast_per_min_szn",
    "track_drives_per_min_szn",
    "track_drive_fta_per_min_szn",
    "track_drive_pf_per_min_szn",
    "track_paint_touches_per_min_szn",
    "track_fta_per_drive_szn",
    "track_catch_shoot_fg3a_per_min_szn",
    "track_pull_up_fg3a_per_min_szn",
    "track_pull_up_3pa_share_szn",
)
TRACKING_ROLE_FEATURES: tuple[str, ...] = (
    "track_role_cluster",
    "track_role_is_low_minutes",
)
TRACKING_FILL_FEATURES: tuple[str, ...] = TRACKING_NUMERIC_FEATURES + TRACKING_ROLE_FEATURES

DEFAULT_TRACKING_FILL_VALUES: dict[str, float] = {
    **{col: 0.0 for col in TRACKING_NUMERIC_FEATURES},
    # Legacy live behavior used 0 for role fields.
    "track_role_cluster": 0.0,
    "track_role_is_low_minutes": 0.0,
}


def fit_tracking_fill_values(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    fill_values: dict[str, float] = {}
    active_cols = [col for col in TRACKING_FILL_FEATURES if col in feature_cols]

    for col in active_cols:
        if col not in train_df.columns:
            fill_values[col] = DEFAULT_TRACKING_FILL_VALUES[col]
            continue
        series = pd.to_numeric(train_df[col], errors="coerce")
        if col in TRACKING_ROLE_FEATURES:
            mode = series.mode(dropna=True)
            if mode.empty:
                fill_values[col] = DEFAULT_TRACKING_FILL_VALUES[col]
            else:
                fill_values[col] = float(mode.iloc[0])
            continue
        median = series.median(skipna=True)
        if pd.isna(median):
            fill_values[col] = DEFAULT_TRACKING_FILL_VALUES[col]
        else:
            fill_values[col] = float(median)
    return fill_values


def apply_tracking_fill_values(df: pd.DataFrame, fill_values: dict[str, float]) -> pd.DataFrame:
    if not fill_values:
        return df
    out = df.copy()
    for col, fill_val in fill_values.items():
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(float(fill_val))
        if col in TRACKING_ROLE_FEATURES:
            out[col] = out[col].astype(int)
    return out


def resolve_tracking_fill_values(meta: dict[str, Any], feature_cols: list[str]) -> dict[str, float]:
    preprocess = meta.get("preprocess") if isinstance(meta, dict) else None
    explicit = preprocess.get("tracking_fill_values") if isinstance(preprocess, dict) else None
    if isinstance(explicit, dict) and explicit:
        return {k: float(v) for k, v in explicit.items() if k in feature_cols}
    return {
        col: DEFAULT_TRACKING_FILL_VALUES[col]
        for col in TRACKING_FILL_FEATURES
        if col in feature_cols
    }

