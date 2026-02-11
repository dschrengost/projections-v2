"""Rest and workload cadence helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach days_since_last, b2b, 3in4, and 4in6 indicators."""

    working = df.copy()
    # Compute lags on chronological order per-player, then restore original row order.
    working["_rest_original_order"] = np.arange(len(working), dtype=int)
    working["_rest_game_date"] = pd.to_datetime(working["game_date"], errors="coerce")
    sort_cols = ["player_id", "_rest_game_date"]
    if "tip_ts" in working.columns:
        working["_rest_tip_ts"] = pd.to_datetime(working["tip_ts"], utc=True, errors="coerce")
        sort_cols.append("_rest_tip_ts")
    if "game_id" in working.columns:
        sort_cols.append("game_id")
    working.sort_values(sort_cols, kind="mergesort", inplace=True)

    grouped = working.groupby("player_id", group_keys=False, sort=False)

    prev_date = grouped["_rest_game_date"].shift(1)
    working["days_since_last"] = (working["_rest_game_date"] - prev_date).dt.days.astype("Int64")
    working["is_b2b"] = (working["days_since_last"] == 1).fillna(False).astype(int)

    third_prev = grouped["_rest_game_date"].shift(2)
    working["is_3in4"] = (
        (working["_rest_game_date"] - third_prev).dt.days <= 4
    ).fillna(False).astype(int)
    fourth_prev = grouped["_rest_game_date"].shift(3)
    working["is_4in6"] = (
        (working["_rest_game_date"] - fourth_prev).dt.days <= 6
    ).fillna(False).astype(int)

    working.sort_values("_rest_original_order", kind="mergesort", inplace=True)
    drop_cols = ["_rest_original_order", "_rest_game_date"]
    if "_rest_tip_ts" in working.columns:
        drop_cols.append("_rest_tip_ts")
    working.drop(columns=drop_cols, inplace=True)
    return working
