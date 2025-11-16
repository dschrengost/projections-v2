"""Helpers for deriving starter labels from box score minutes."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for starter derivation: {', '.join(sorted(missing))}")


def derive_starter_flag_labels(
    df: pd.DataFrame,
    *,
    minutes_col: str = "minutes",
    game_col: str = "game_id",
    team_col: str = "team_id",
    player_col: str = "player_id",
    output_col: str = "starter_flag_label",
) -> pd.DataFrame:
    """Return a copy of ``df`` with exactly five starters per team-game."""

    required = {minutes_col, game_col, team_col, player_col}
    _ensure_columns(df, required)

    working = df.copy()
    if output_col in working.columns:
        working = working.drop(columns=[output_col])
    working[minutes_col] = pd.to_numeric(working[minutes_col], errors="coerce").fillna(0.0)

    base = working.drop_duplicates(subset=[game_col, team_col, player_col]).copy()
    group_cols = [game_col, team_col]

    def _assign(group: pd.DataFrame) -> pd.Series:
        if group[player_col].nunique() < 5:
            raise ValueError("Need at least five players per team-game to derive starters")
        ordered = group.sort_values([minutes_col, player_col], ascending=[False, True])
        mask = pd.Series(0, index=group.index, dtype=int)
        mask.loc[ordered.head(5).index] = 1
        return mask

    base[output_col] = base.groupby(group_cols, group_keys=False).apply(_assign, include_groups=False)
    sums = base.groupby(group_cols)[output_col].sum()
    if not (sums == 5).all():
        raise AssertionError("Starter derivation failed to assign exactly five starters per team-game")

    merged = working.merge(
        base[[game_col, team_col, player_col, output_col]],
        on=[game_col, team_col, player_col],
        how="left",
    )
    if merged[output_col].isna().any():
        raise AssertionError("Starter label derivation produced missing rows")
    return merged


__all__ = ["derive_starter_flag_labels"]
