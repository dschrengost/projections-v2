"""Game environment feature helpers."""

from __future__ import annotations

import pandas as pd

from projections.minutes_v1.snapshots import ensure_as_of_column


def compute_blowout_index(spread: pd.Series, total: pd.Series) -> pd.Series:
    """Heuristic blowout indicator derived from spread/total."""

    spread_component = spread.abs().fillna(0) / 10.0
    total_component = (total.fillna(total.median()) - 220.0) / 100.0
    return (spread_component + total_component).clip(lower=0).round(4)


def _select_latest_odds_snapshot(
    base_df: pd.DataFrame,
    odds_snapshot: pd.DataFrame,
    *,
    tip_col: str = "tip_ts",
    as_of_col: str = "as_of_ts",
) -> pd.DataFrame:
    """Return at most one odds row per game_id.

    The gold pipeline reads all partitions under silver/odds_snapshot, which can
    contain duplicate game_ids (e.g. due to legacy month partitioning). If we
    merge those directly we'll duplicate player-game rows and break any
    shift-based history features (minutes trend, starter history, rest cadence).
    """

    if odds_snapshot.empty:
        return odds_snapshot.copy()

    working = ensure_as_of_column(odds_snapshot.copy(), column=as_of_col)
    working[as_of_col] = pd.to_datetime(working[as_of_col], utc=True, errors="coerce")
    working = working.dropna(subset=["game_id"])

    tip_lookup: pd.DataFrame | None = None
    if tip_col in base_df.columns and "game_id" in base_df.columns:
        tip_lookup = base_df.loc[:, ["game_id", tip_col]].drop_duplicates().copy()
        tip_lookup[tip_col] = pd.to_datetime(tip_lookup[tip_col], utc=True, errors="coerce")

    if tip_lookup is not None and not tip_lookup.empty:
        working = working.merge(tip_lookup, on="game_id", how="left")
        valid = working[as_of_col].notna() & working[tip_col].notna() & (working[as_of_col] <= working[tip_col])
        eligible = working.loc[valid].copy()
        if eligible.empty:
            return working.iloc[0:0].drop(columns=[tip_col], errors="ignore")
        idx = eligible.groupby("game_id")[as_of_col].idxmax()
        return eligible.loc[idx].drop(columns=[tip_col], errors="ignore").reset_index(drop=True)

    eligible = working.dropna(subset=[as_of_col])
    if eligible.empty:
        return working.iloc[0:0].copy()
    idx = eligible.groupby("game_id")[as_of_col].idxmax()
    return eligible.loc[idx].reset_index(drop=True)


def attach_game_environment_features(
    base_df: pd.DataFrame, odds_snapshot: pd.DataFrame
) -> pd.DataFrame:
    """Attach spread/total provenance and derived blowout index."""

    if odds_snapshot is None or odds_snapshot.empty:
        merged = base_df.copy()
        merged["spread_home"] = pd.NA
        merged["total"] = pd.NA
        merged["odds_as_of_ts"] = pd.NaT
        merged["blowout_index"] = compute_blowout_index(
            pd.Series(0, index=merged.index), pd.Series(220, index=merged.index)
        )
        merged["blowout_risk_score"] = 0.5
        merged["close_game_score"] = 0.5
        return merged

    odds = odds_snapshot.copy()
    odds = odds.rename(columns={"home_line": "spread_home"})
    odds = ensure_as_of_column(odds)
    for required in ("game_id", "spread_home", "total"):
        if required not in odds.columns:
            odds[required] = pd.NA
    odds = _select_latest_odds_snapshot(base_df, odds)
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
