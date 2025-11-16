"""Computation helpers for archetype delta artifacts and runtime features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .roles import infer_position_group

_DELTA_COLUMNS: list[str] = [
    "season",
    "role_p",
    "role_t",
    "delta_per_missing",
    "delta_any",
    "n_games_present",
    "n_games_missing",
]


@dataclass(frozen=True)
class ArchetypeDeltaConfig:
    min_present_games: int = 50
    min_missing_games: int = 15
    max_abs_delta_per_missing: float = 12.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ArchetypeDeltaConfig":
        if not data:
            return cls()
        return cls(
            min_present_games=int(data.get("min_present_games", cls.min_present_games)),
            min_missing_games=int(data.get("min_missing_games", cls.min_missing_games)),
            max_abs_delta_per_missing=float(data.get("max_abs_delta_per_missing", cls.max_abs_delta_per_missing)),
        )


def load_config(path: Path | None) -> ArchetypeDeltaConfig:
    if path is None:
        return ArchetypeDeltaConfig()
    if not path.exists():
        raise FileNotFoundError(f"Archetype delta config not found at {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Archetype delta config must be a mapping")
    return ArchetypeDeltaConfig.from_mapping(data)


def build_archetype_deltas(
    labels: pd.DataFrame,
    injuries: pd.DataFrame,
    roles: pd.DataFrame,
    *,
    season_label: str,
    config: ArchetypeDeltaConfig,
) -> pd.DataFrame:
    required_label_cols = {"game_id", "team_id", "player_id", "minutes"}
    missing_labels = required_label_cols - set(labels.columns)
    if missing_labels:
        raise ValueError(
            "Labels dataframe missing required columns: " + ", ".join(sorted(missing_labels))
        )
    context = _player_role_context(labels, injuries, roles, season_label=season_label)
    context = context.dropna(subset=["role_p", "role_t"])
    context = context[context["minutes"].notna()].copy()
    if context.empty:
        return _empty_delta_frame()
    context["minutes"] = pd.to_numeric(context["minutes"], errors="coerce")
    context = context[context["minutes"].notna()]
    if context.empty:
        return _empty_delta_frame()
    grouped = context.groupby(["season", "role_p", "role_t"], as_index=False)
    records: list[dict[str, Any]] = []
    for _, group in grouped:
        present = group[group["missing_role_t_count"] == 0]
        missing = group[group["missing_role_t_count"] > 0]
        n_present = int(len(present))
        n_missing = int(len(missing))
        if n_present < config.min_present_games or n_missing < config.min_missing_games:
            continue
        mean_present = float(present["minutes"].mean()) if n_present else 0.0
        mean_missing = float(missing["minutes"].mean()) if n_missing else 0.0
        mean_missing_count = float(missing["missing_role_t_count"].mean()) if n_missing else 0.0
        delta_any = mean_missing - mean_present
        denom = mean_missing_count if mean_missing_count > 0 else 1.0
        delta_per_missing = delta_any / denom
        delta_per_missing = float(
            np.clip(delta_per_missing, -config.max_abs_delta_per_missing, config.max_abs_delta_per_missing)
        )
        records.append(
            {
                "season": group["season"].iat[0],
                "role_p": group["role_p"].iat[0],
                "role_t": group["role_t"].iat[0],
                "delta_per_missing": delta_per_missing,
                "delta_any": float(delta_any),
                "n_games_present": n_present,
                "n_games_missing": n_missing,
            }
        )
    if not records:
        return _empty_delta_frame()
    result = pd.DataFrame.from_records(records)
    result = result.loc[:, _DELTA_COLUMNS]
    return result.sort_values(["season", "role_p", "role_t"]).reset_index(drop=True)


def _empty_delta_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_DELTA_COLUMNS)


def _player_role_context(
    labels: pd.DataFrame,
    injuries: pd.DataFrame,
    roles: pd.DataFrame,
    *,
    season_label: str,
) -> pd.DataFrame:
    required_role_cols = {"season", "player_id", "role_key", "position_group"}
    missing_roles = required_role_cols - set(roles.columns)
    if missing_roles:
        raise ValueError(
            "Roles dataframe missing required columns: " + ", ".join(sorted(missing_roles))
        )
    role_lookup = roles[roles["season"] == season_label].copy()
    if role_lookup.empty:
        raise ValueError(f"No roles found for season {season_label}")
    role_lookup = (
        role_lookup.loc[:, ["player_id", "role_key", "position_group"]]
        .drop_duplicates(subset=["player_id"])
        .rename(columns={"role_key": "role_p", "position_group": "position_group_p"})
    )
    labels = labels.copy()
    labels["season"] = season_label
    labels["player_id"] = pd.to_numeric(labels["player_id"], errors="coerce").astype("Int64")
    labels["game_id"] = pd.to_numeric(labels["game_id"], errors="coerce").astype("Int64")
    labels["team_id"] = pd.to_numeric(labels["team_id"], errors="coerce").astype("Int64")
    labels = labels.dropna(subset=["player_id", "team_id", "game_id"])
    labels = labels.merge(role_lookup, on="player_id", how="left")
    labels["game_id"] = labels["game_id"].astype(int)
    labels["team_id"] = labels["team_id"].astype(int)

    if injuries is None or injuries.empty:
        raise ValueError("Injury dataset is required for archetype delta construction")
    injury_context = prepare_injury_availability(injuries)
    team_role_counts = compute_team_role_counts(injury_context, roles, season_label=season_label)
    context = labels.merge(
        team_role_counts,
        on=["game_id", "team_id"],
        how="left",
    )
    context["missing_role_t_count"] = context["missing_role_t_count"].fillna(0).astype(int)
    return context


def prepare_injury_availability(df: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "player_id", "team_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Injury features missing required columns: " + ", ".join(sorted(missing))
        )
    working = df.copy()
    working = working.dropna(subset=["game_id", "player_id", "team_id"])
    working = working.drop_duplicates(subset=["game_id", "player_id"], keep="last")
    working["game_id"] = pd.to_numeric(working["game_id"], errors="coerce").astype("Int64")
    working["player_id"] = pd.to_numeric(working["player_id"], errors="coerce").astype("Int64")
    working["team_id"] = pd.to_numeric(working["team_id"], errors="coerce").astype("Int64")
    working = working.dropna(subset=["game_id", "player_id", "team_id"])
    working["game_id"] = working["game_id"].astype(int)
    working["player_id"] = working["player_id"].astype(int)
    working["team_id"] = working["team_id"].astype(int)
    working["available_flag"] = _available_flag(working)
    return working


def _available_flag(df: pd.DataFrame) -> pd.Series:
    for column in ("available_at_tip", "available", "is_available"):
        if column in df.columns:
            return df[column].fillna(False).astype(bool)
    status_col = None
    for column in ("status_at_tip", "status"):
        if column in df.columns:
            status_col = column
            break
    if status_col is None:
        return pd.Series(True, index=df.index)
    statuses = df[status_col].fillna("AVAILABLE").astype(str).str.upper()
    return ~statuses.isin({"OUT", "O"})


def compute_team_role_counts(
    injuries: pd.DataFrame,
    roles: pd.DataFrame,
    *,
    season_label: str,
) -> pd.DataFrame:
    role_lookup = roles[roles["season"] == season_label].copy()
    if role_lookup.empty:
        raise ValueError(f"No roles found for season {season_label}")
    role_lookup = role_lookup.loc[:, ["player_id", "role_key"]]
    team_players = injuries.merge(role_lookup, on="player_id", how="left")
    team_players = team_players.dropna(subset=["role_key"])
    if team_players.empty:
        columns = [
            "game_id",
            "team_id",
            "role_t",
            "position_group_t",
            "missing_role_t_count",
        ]
        return pd.DataFrame(columns=columns)
    counts = (
        team_players.groupby(["game_id", "team_id", "role_key"], as_index=False)
        .agg(
            total_players=("player_id", "nunique"),
            available_players=("available_flag", lambda s: int(s.astype(bool).sum())),
        )
        .rename(columns={"role_key": "role_t"})
    )
    counts["missing_role_t_count"] = (counts["total_players"] - counts["available_players"]).clip(lower=0).astype(int)
    counts["position_group_t"] = counts["role_t"].apply(
        lambda value: infer_position_group(str(value).split("_", 1)[0] if isinstance(value, str) else None)
    )
    counts = counts.loc[:, ["game_id", "team_id", "role_t", "position_group_t", "missing_role_t_count"]]
    return counts


def compute_team_missing_totals(injuries: pd.DataFrame) -> pd.DataFrame:
    totals = (
        injuries.groupby(["game_id", "team_id"], as_index=False)
        .agg(
            arch_missing_total_count=(
                "available_flag",
                lambda s: int((~s.astype(bool)).sum()),
            )
        )
    )
    return totals
