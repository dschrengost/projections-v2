"""Shared feature engineering for rotation_set_minutes_v1.

These helpers are used by both:
- training dataset builder(s) under `scripts/rotation/`, and
- live feature builders used for production inference.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from projections.rotation.set_model import zfill_game_id_series

ODDS_COLS: tuple[str, str] = ("spread_home", "total")


def apply_odds_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute odds missing flags + fill missing odds with 0.0.

    This matches the logic used in `scripts/rotation/build_rotation_train_dataset_v1.py`.
    """

    for col in ODDS_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required odds column: {col}")
    out = df.copy()
    out["spread_home"] = pd.to_numeric(out["spread_home"], errors="coerce").astype("float64")
    out["total"] = pd.to_numeric(out["total"], errors="coerce").astype("float64")
    out["spread_home_missing"] = out["spread_home"].isna()
    out["total_missing"] = out["total"].isna()
    out["spread_home"] = out["spread_home"].fillna(0.0).astype("float64")
    out["total"] = out["total"].fillna(0.0).astype("float64")
    return out


def fill_numeric_missing_with_zero(
    df: pd.DataFrame,
    *,
    key_cols: Iterable[str] = ("game_id", "team_id", "player_id"),
) -> pd.DataFrame:
    """Fill missing numeric values with 0 and coerce booleans to int8.

    This is intentionally broad (all numeric columns), matching the training builder.
    """

    out = df.copy()
    key_set = set(key_cols)
    for col in out.columns:
        if col in key_set:
            continue
        series = out[col]
        if pd.api.types.is_bool_dtype(series):
            out[col] = series.fillna(False).astype("int8")
        elif pd.api.types.is_numeric_dtype(series):
            out[col] = series.fillna(0)
    return out


def propagate_team_level_columns(df: pd.DataFrame, *, cols: list[str]) -> pd.DataFrame:
    """Propagate team-level columns within each (game_id, team_id)."""

    if not cols:
        return df
    out = df.copy()
    group_cols = ["game_id", "team_id"]
    missing_group_cols = [c for c in group_cols if c not in out.columns]
    if missing_group_cols:
        raise ValueError(f"Cannot propagate team-level columns; missing keys: {missing_group_cols}")
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = out.groupby(group_cols, sort=False)[col].bfill().ffill()
    return out


def join_rotation_priors(
    df: pd.DataFrame,
    *,
    team_priors: pd.DataFrame,
    player_priors: pd.DataFrame,
) -> pd.DataFrame:
    """Join rotation_priors_v1 tables onto a minutes feature frame.

    Priors are keyed by:
      - team_priors: (game_id, team_id)
      - player_priors: (game_id, team_id, person_id) where person_id == player_id
    """

    required_keys = {"game_id", "team_id", "player_id"}
    if not required_keys.issubset(df.columns):
        raise ValueError(f"Minutes dataset missing join keys: {sorted(required_keys - set(df.columns))}")

    out = df.copy()
    out["game_id_norm"] = zfill_game_id_series(out["game_id"])
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["person_id"] = out["player_id"]

    if not team_priors.empty:
        tp = team_priors.copy()
        if "game_id_norm" not in tp.columns and "game_id" in tp.columns:
            tp["game_id_norm"] = zfill_game_id_series(tp["game_id"])
        tp["team_id"] = pd.to_numeric(tp["team_id"], errors="coerce").astype("Int64")
        tp = tp.drop_duplicates(subset=["game_id_norm", "team_id"], keep="last")
        tp_cols = ["game_id_norm", "team_id"] + [c for c in tp.columns if "prior_" in str(c).lower()]
        tp_cols = [c for i, c in enumerate(tp_cols) if c in tp.columns and c not in tp_cols[:i]]
        out = out.merge(tp.loc[:, tp_cols], on=["game_id_norm", "team_id"], how="left", suffixes=("", "_prior_team"))

    if not player_priors.empty:
        pp = player_priors.copy()
        if "game_id_norm" not in pp.columns and "game_id" in pp.columns:
            pp["game_id_norm"] = zfill_game_id_series(pp["game_id"])
        pp["team_id"] = pd.to_numeric(pp["team_id"], errors="coerce").astype("Int64")
        pp["person_id"] = pd.to_numeric(pp["person_id"], errors="coerce").astype("Int64")
        pp = pp.drop_duplicates(subset=["game_id_norm", "team_id", "person_id"], keep="last")
        pp_cols = ["game_id_norm", "team_id", "person_id"] + [c for c in pp.columns if "prior_" in str(c).lower()]
        pp_cols = [c for i, c in enumerate(pp_cols) if c in pp.columns and c not in pp_cols[:i]]
        out = out.merge(
            pp.loc[:, pp_cols],
            on=["game_id_norm", "team_id", "person_id"],
            how="left",
            suffixes=("", "_prior_player"),
        )

    prior_cols = [c for c in out.columns if "_prior_" in str(c).lower()]
    for col in prior_cols:
        if str(col).endswith("_missing") and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1).astype("int8")
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    count_cols = [
        c
        for c in out.columns
        if str(c).startswith("team_prior_n_games_") or str(c).startswith("player_prior_n_games_")
    ]
    for col in count_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int16")

    out = out.drop(columns=["game_id_norm", "person_id"])
    return out

