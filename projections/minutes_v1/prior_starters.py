"""Utilities for building starter slot priors used by the promotion calibrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from projections.minutes_v1.pos import canonical_pos_bucket_series


STARTER_FLAG_CANDIDATES: tuple[str, ...] = ("starter_flag", "starter_flag_label")
POS_COLUMN_CANDIDATES: tuple[str, ...] = ("pos_bucket", "archetype")


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing features input at {path}")
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str], *, label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Input dataframe missing required {label} column(s): {', '.join(candidates)}")


def build_starter_slot_priors(df: pd.DataFrame, *, min_minutes: float = 0.0) -> pd.DataFrame:
    """Return per-team/per-position starter minute priors and league fallbacks."""

    starter_col = _resolve_column(df, STARTER_FLAG_CANDIDATES, label="starter flag")
    pos_col = _resolve_column(df, POS_COLUMN_CANDIDATES, label="position bucket")
    required = {"team_id", "minutes", starter_col, pos_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"starter priors missing required columns: {', '.join(sorted(missing))}")

    working = df.loc[(df[starter_col] == 1) & (df["minutes"] > min_minutes)].copy()
    working["pos_bucket"] = canonical_pos_bucket_series(working[pos_col])
    agg = (
        working.groupby(["team_id", "pos_bucket"])["minutes"]
        .agg(
            starter_p25_minutes=lambda s: float(s.quantile(0.25)),
            starter_p50_minutes=lambda s: float(s.quantile(0.50)),
            starter_p75_minutes=lambda s: float(s.quantile(0.75)),
            num_starts="count",
        )
        .reset_index()
    )
    agg["scope"] = "team"

    league = (
        working.groupby(["pos_bucket"])["minutes"]
        .agg(
            starter_p25_minutes=lambda s: float(s.quantile(0.25)),
            starter_p50_minutes=lambda s: float(s.quantile(0.50)),
            starter_p75_minutes=lambda s: float(s.quantile(0.75)),
            num_starts="count",
        )
        .reset_index()
    )
    league["team_id"] = -1
    league["scope"] = "league"
    priors = pd.concat([agg, league], ignore_index=True, sort=False)
    priors["team_id"] = priors["team_id"].astype("Int64")
    return priors[
        [
            "team_id",
            "pos_bucket",
            "scope",
            "starter_p25_minutes",
            "starter_p50_minutes",
            "starter_p75_minutes",
            "num_starts",
        ]
    ]


def build_player_starter_history(df: pd.DataFrame) -> pd.DataFrame:
    starter_col = _resolve_column(df, STARTER_FLAG_CANDIDATES, label="starter flag")
    required = {"player_id", starter_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"starter history missing columns: {', '.join(sorted(missing))}")
    if "game_id" in df.columns:
        dedup = df.drop_duplicates(subset=["game_id", "player_id"])
    else:
        dedup = df
    working = dedup.loc[dedup[starter_col] == 1].copy()
    history = (
        working.groupby("player_id")[starter_col]
        .agg(starter_history_games="count")
        .reset_index()
    )
    return history


@dataclass
class StarterPriorArtifacts:
    slot_priors: pd.DataFrame
    player_history: pd.DataFrame


def load_feature_frames(inputs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in inputs:
        frames.append(_read_parquet_tree(path))
    return pd.concat(frames, ignore_index=True)
