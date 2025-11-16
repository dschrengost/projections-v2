"""Rest and workload cadence helpers."""

from __future__ import annotations

import pandas as pd


def attach_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach days_since_last, b2b, 3in4, and 4in6 indicators."""

    working = df.copy()
    working["game_date"] = pd.to_datetime(working["game_date"])
    grouped = working.groupby("player_id", group_keys=False)

    prev_date = grouped["game_date"].shift(1)
    working["days_since_last"] = (working["game_date"] - prev_date).dt.days.astype("Int64")
    working["is_b2b"] = (working["days_since_last"] == 1).fillna(False).astype(int)

    third_prev = grouped["game_date"].shift(2)
    working["is_3in4"] = (
        (working["game_date"] - third_prev).dt.days <= 4
    ).fillna(False).astype(int)
    fourth_prev = grouped["game_date"].shift(3)
    working["is_4in6"] = (
        (working["game_date"] - fourth_prev).dt.days <= 6
    ).fillna(False).astype(int)
    return working
