"""Role assignment helpers for archetype delta features."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

POSITION_GROUP_MAP: dict[str, str] = {
    "PG": "G",
    "SG": "G",
    "G": "G",
    "SF": "F",
    "PF": "F",
    "F": "F",
    "PF/C": "C",
    "F/C": "C",
    "C": "C",
    "C/F": "C",
    "FC": "C",
    "G/F": "F",
    "G-F": "F",
    "F-G": "F",
    "F-C": "C",
    "C-F": "C",
}


def _normalize_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().upper()
    if not token:
        return None
    return token


def infer_position_group(value: str | None) -> str | None:
    token = _normalize_token(value)
    if token is None:
        return None
    if token in POSITION_GROUP_MAP:
        return POSITION_GROUP_MAP[token]
    for delimiter in ("/", "-", " "):
        if delimiter in token:
            for part in token.split(delimiter):
                group = infer_position_group(part)
                if group:
                    return group
    first = token[0]
    if first in {"G", "F", "C"}:
        return first
    return None


def _most_common(values: Iterable[str]) -> str | None:
    counter = Counter(value for value in values if isinstance(value, str) and value)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


@dataclass(frozen=True)
class RoleConfig:
    min_games_played: int = 10


def build_roles_table(
    labels: pd.DataFrame,
    *,
    season_label: str,
    roster_nightly: pd.DataFrame | None = None,
    config: RoleConfig | None = None,
) -> pd.DataFrame:
    cfg = config or RoleConfig()
    required = {"player_id", "season", "minutes", "starter_flag"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            "Labels dataframe missing required columns: " + ", ".join(sorted(missing))
        )

    working = labels.copy()
    working["minutes"] = pd.to_numeric(working["minutes"], errors="coerce")
    working["starter_flag"] = working["starter_flag"].fillna(0).astype(int)
    position_series = _extract_position_series(working, roster_nightly)
    working = working.assign(_position=position_series)
    stats = _aggregate_player_stats(working)
    stats["season"] = season_label
    stats["position_group"] = stats["_position"].apply(infer_position_group)
    stats.drop(columns=["_position"], inplace=True)
    stats["position_group"] = stats["position_group"].astype("string")
    stats["starter_tier"] = stats.apply(
        lambda row: _starter_tier(
            games_played=row["games_played"],
            start_rate=row["start_rate"],
            min_games=cfg.min_games_played,
        ),
        axis=1,
    )
    stats["starter_tier"] = stats["starter_tier"].astype("string")
    stats["role_key"] = np.where(
        stats["position_group"].notna(),
        stats["position_group"].astype("string") + "_" + stats["starter_tier"],
        pd.NA,
    )
    stats["role_key"] = pd.Series(stats["role_key"], dtype="string")
    ordered = stats.loc[:, ["season", "player_id", "position_group", "starter_tier", "role_key"]]
    return ordered.reset_index(drop=True)


def _starter_tier(*, games_played: int, start_rate: float, min_games: int) -> str:
    if games_played < min_games:
        return "unknown"
    if start_rate >= 0.6:
        return "starter"
    if start_rate >= 0.2:
        return "swing"
    return "bench"


def _aggregate_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("player_id", dropna=True)
    stats = grouped.agg(
        games_played=("minutes", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum()),
        games_started=("starter_flag", "sum"),
    )
    stats["games_played"] = stats["games_played"].astype(int)
    stats["games_started"] = stats["games_started"].fillna(0).astype(int)
    stats["start_rate"] = stats.apply(
        lambda row: row["games_started"] / row["games_played"] if row["games_played"] > 0 else 0.0,
        axis=1,
    )
    positions = grouped["_position"].apply(_most_common).rename("_position")
    stats = stats.join(positions, how="left")
    stats = stats.reset_index()
    stats["_position"] = stats["_position"].astype("string")
    return stats


def _extract_position_series(labels: pd.DataFrame, roster_nightly: pd.DataFrame | None) -> pd.Series:
    for column in ("primary_position", "listed_pos", "position", "pos_bucket", "archetype"):
        if column in labels.columns:
            return labels[column]
    if roster_nightly is None or roster_nightly.empty:
        return pd.Series([None] * len(labels), index=labels.index, dtype="object")
    roster = roster_nightly.copy()
    roster = roster.loc[:, [col for col in roster.columns if col in {"game_id", "player_id", "listed_pos"}]]
    roster = roster.dropna(subset=["game_id", "player_id", "listed_pos"])
    if roster.empty:
        return pd.Series([None] * len(labels), index=labels.index, dtype="object")
    roster["game_id"] = pd.to_numeric(roster["game_id"], errors="coerce").astype("Int64")
    roster["player_id"] = pd.to_numeric(roster["player_id"], errors="coerce").astype("Int64")
    roster = roster.dropna(subset=["game_id", "player_id"])
    roster = roster.drop_duplicates(subset=["game_id", "player_id"], keep="last")
    roster["listed_pos"] = roster["listed_pos"].astype(str)

    base = labels.loc[:, ["game_id", "player_id"]].copy()
    base_index = labels.index
    base["game_id"] = pd.to_numeric(base["game_id"], errors="coerce").astype("Int64")
    base["player_id"] = pd.to_numeric(base["player_id"], errors="coerce").astype("Int64")
    base["_row_id"] = np.arange(len(base))
    merged = base.merge(
        roster,
        on=["game_id", "player_id"],
        how="left",
        sort=False,
    )
    merged.sort_values("_row_id", inplace=True)
    return pd.Series(merged["listed_pos"].values, index=base_index, dtype="string")
