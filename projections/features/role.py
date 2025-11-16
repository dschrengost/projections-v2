"""Role/starter utilization features."""

from __future__ import annotations

import pandas as pd


def attach_role_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach as-of-safe starter history features."""

    required = {"player_id", "starter_flag", "tip_ts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Role features require columns: {', '.join(sorted(missing))}")

    working = df.copy()
    working["tip_ts"] = pd.to_datetime(working["tip_ts"], utc=True)
    working.sort_values(["player_id", "tip_ts"], inplace=True)

    grouped = working.groupby("player_id", group_keys=False)
    starter_history = grouped["starter_flag"].shift(1).fillna(0.0)
    working["starter_prev_game_asof"] = starter_history

    rolling_history = grouped["starter_flag"].shift(1)
    recent_pct = (
        rolling_history.groupby(working["player_id"])
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    working["recent_start_pct_10"] = recent_pct.clip(0.0, 1.0)

    working.drop(columns=["starter_flag"], inplace=True)
    working.sort_index(inplace=True)
    return working
