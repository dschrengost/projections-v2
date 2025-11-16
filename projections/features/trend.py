"""Minutes trend and volatility features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clip_minutes(series: pd.Series) -> pd.Series:
    """Clamp minute-based features to NBA-legal bounds."""

    return series.clip(lower=0.0, upper=48.0)


def _sum_minutes_last_days(group: pd.DataFrame, days: int) -> pd.Series:
    group = group.sort_values("game_date").copy()
    results: list[float] = []
    for _, row in group.iterrows():
        cutoff = row["game_date"] - pd.Timedelta(days=days)
        mask = (group["game_date"] < row["game_date"]) & (group["game_date"] >= cutoff)
        results.append(float(group.loc[mask, "minutes"].sum()))
    return pd.Series(results, index=group.index, dtype=float)


def sum_minutes_last_days_all(df: pd.DataFrame, days: int) -> pd.Series:
    """Apply the trailing-day sum helper across the entire dataframe."""

    output = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby("player_id"):
        output.loc[group.index] = _sum_minutes_last_days(group, days)
    return output


def _rolling_iqr(values: np.ndarray) -> float:
    return float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))


def attach_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach min_last*, roll_mean_*, roll_iqr_5, z_vs_10 features."""

    working = df.copy()
    grouped = working.groupby("player_id", group_keys=False)
    shifted_minutes = grouped["minutes"].shift(1)

    working["min_last1"] = clip_minutes(shifted_minutes)
    working["min_last3"] = clip_minutes(
        shifted_minutes.groupby(working["player_id"])
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    working["min_last5"] = clip_minutes(
        shifted_minutes.groupby(working["player_id"])
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    working["sum_min_7d"] = sum_minutes_last_days_all(working, 7).fillna(0.0)

    for window in (3, 5, 10):
        working[f"roll_mean_{window}"] = clip_minutes(
            grouped["minutes"]
            .shift(1)
            .groupby(working["player_id"])
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    working["roll_iqr_5"] = (
        grouped["minutes"]
        .apply(lambda g: g.shift(1).rolling(window=5, min_periods=2).apply(_rolling_iqr, raw=True))
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    mean10 = (
        grouped["minutes"]
        .shift(1)
        .groupby(working["player_id"])
        .rolling(window=10, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    std10 = (
        grouped["minutes"]
        .shift(1)
        .groupby(working["player_id"])
        .rolling(window=10, min_periods=3)
        .std(ddof=0)
        .reset_index(level=0, drop=True)
    )
    working["z_vs_10"] = np.where(
        std10 > 0,
        (working["min_last1"] - mean10) / std10,
        0.0,
    )

    return working
