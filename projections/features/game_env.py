"""Game environment feature helpers."""

from __future__ import annotations

import pandas as pd

from projections.minutes_v1.snapshots import ensure_as_of_column


def compute_blowout_index(spread: pd.Series, total: pd.Series) -> pd.Series:
    """Heuristic blowout indicator derived from spread/total."""

    spread_component = spread.abs().fillna(0) / 10.0
    total_component = (total.fillna(total.median()) - 220.0) / 100.0
    return (spread_component + total_component).clip(lower=0).round(4)


def attach_game_environment_features(
    base_df: pd.DataFrame, odds_snapshot: pd.DataFrame
) -> pd.DataFrame:
    """Attach spread/total provenance and derived blowout index."""

    odds = ensure_as_of_column(odds_snapshot.copy())
    odds = odds.rename(columns={"home_line": "spread_home"})
    merged = base_df.merge(
        odds[["game_id", "spread_home", "total", "as_of_ts"]],
        on="game_id",
        how="left",
    )
    merged.rename(columns={"as_of_ts": "odds_as_of_ts"}, inplace=True)
    merged["blowout_index"] = compute_blowout_index(
        merged["spread_home"].fillna(0), merged["total"].fillna(220)
    )
    spread_abs = merged["spread_home"].abs()
    blowout_risk = ((spread_abs - 8.0).clip(lower=0) / 12.0).clip(upper=1.0)
    close_score = ((8.0 - spread_abs).clip(lower=0) / 8.0).clip(upper=1.0)
    merged["blowout_risk_score"] = blowout_risk.fillna(0.5)
    merged["close_game_score"] = close_score.fillna(0.5)
    return merged
