from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from projections import paths
from projections.rotations.player_map import build_person_id_to_internal_id_map
from projections.rotations.rotation_predictor import canonicalize_game_id
from projections.rotations.schemas import LINEUP_COLS


REQUIRED_LABEL_COLS: tuple[str, ...] = ("game_id", "team_id", "player_id", "minutes_actual", "played_ge_1")
REQUIRED_PRIOR_COLS: tuple[str, ...] = ("game_id", "team_id", "player_id", "minutes_prior", "play_prob")


def _unique_ints_sorted(values: Iterable[Any]) -> list[int]:
    out: set[int] = set()
    for v in values:
        if v is None:
            continue
        try:
            out.add(int(v))
        except Exception:
            continue
    return sorted(out)


def build_candidate_pool_truth(labels: pd.DataFrame) -> pd.DataFrame:
    """Truth candidate set used by legacy rot_eval harness (LEAKY by design; eval-only).

    membership := minutes_actual > 0 OR played_ge_1 == True
    """
    missing = [c for c in REQUIRED_LABEL_COLS if c not in labels.columns]
    if missing:
        raise ValueError(f"labels missing required columns: {missing}")

    df = labels[list(REQUIRED_LABEL_COLS)].copy()
    df["game_id"] = df["game_id"].astype("string")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["minutes_actual"] = pd.to_numeric(df["minutes_actual"], errors="coerce").astype(np.float64).fillna(0.0)
    played = df["played_ge_1"]
    if played.dtype != bool:
        played = played.astype("boolean", copy=False).fillna(False).astype(bool)
    df["played_ge_1"] = played
    df = df.dropna(subset=["game_id", "team_id", "player_id"]).copy()
    df["team_id"] = df["team_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)

    mask = (df["minutes_actual"] > 0.0) | (df["played_ge_1"])
    out = df.loc[mask, ["game_id", "team_id", "player_id"]].copy()
    out = out.drop_duplicates().sort_values(["game_id", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)
    return out


@dataclass(frozen=True)
class PriorPoolParams:
    top_n: int
    min_minutes_prior: float
    min_play_prob: float
    min_candidates: int = 8


def build_candidate_pool_prior(
    priors: pd.DataFrame,
    *,
    top_n: int,
    min_minutes_prior: float,
    min_play_prob: float,
    min_candidates: int = 8,
) -> pd.DataFrame:
    """Build candidate pool from priors only (NO leakage).

    Selection per (game_id, team_id):
    - include top_n by minutes_prior desc (tie-break player_id asc)
    - plus anyone meeting (minutes_prior >= min_minutes_prior) OR (play_prob >= min_play_prob)
    - ensure at least min_candidates via backfill from minutes_prior rank
    """
    missing = [c for c in REQUIRED_PRIOR_COLS if c not in priors.columns]
    if missing:
        raise ValueError(f"priors missing required columns: {missing}")

    top_n = int(top_n)
    min_candidates = int(min_candidates)
    if top_n < 0:
        raise ValueError("top_n must be >= 0")
    if min_candidates <= 0:
        raise ValueError("min_candidates must be > 0")

    df = priors[list(REQUIRED_PRIOR_COLS)].copy()
    df["game_id"] = df["game_id"].astype("string").map(canonicalize_game_id)
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["minutes_prior"] = pd.to_numeric(df["minutes_prior"], errors="coerce").astype(np.float64).fillna(0.0)
    df["play_prob"] = (
        pd.to_numeric(df["play_prob"], errors="coerce").astype(np.float64).fillna(0.0).clip(0.0, 1.0)
    )
    df = df.dropna(subset=["game_id", "team_id", "player_id"]).copy()
    df["team_id"] = df["team_id"].astype(int)
    df["player_id"] = df["player_id"].astype(int)
    df = df[df["game_id"] != ""].copy()
    if df.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "player_id"])

    out_rows: list[dict[str, Any]] = []
    grouped = df.groupby(["game_id", "team_id"], sort=False)
    for (game_id, team_id), g in grouped:
        g = g.sort_values(["minutes_prior", "player_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
        ranked_ids = [int(x) for x in g["player_id"].tolist()]

        selected: set[int] = set()
        if top_n > 0:
            selected.update(ranked_ids[:top_n])

        thresh_mask = (g["minutes_prior"] >= float(min_minutes_prior)) | (g["play_prob"] >= float(min_play_prob))
        selected.update(int(x) for x in g.loc[thresh_mask, "player_id"].tolist())

        # Enforce minimum pool size deterministically via minutes_prior backfill.
        if len(selected) < min_candidates:
            for pid in ranked_ids:
                if pid in selected:
                    continue
                selected.add(pid)
                if len(selected) >= min_candidates:
                    break

        for pid in sorted(selected):
            out_rows.append({"game_id": str(game_id), "team_id": int(team_id), "player_id": int(pid)})

    out = pd.DataFrame(out_rows, columns=["game_id", "team_id", "player_id"])
    out = out.drop_duplicates().sort_values(["game_id", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)
    return out


def _infer_first_segment_lineup_player_ids(events_team_game: pd.DataFrame) -> list[int]:
    if events_team_game.empty:
        return []
    segs = events_team_game.sort_values(["segment_idx"], kind="mergesort")
    row = segs.iloc[0]
    starters = [row.get(c) for c in LINEUP_COLS]
    return _unique_ints_sorted(starters)


def build_candidate_pool_roster(
    game_id: str,
    team_id: int,
    season_start_year: int,
    *,
    data_root: Path | None = None,
    person_id_to_internal_id: dict[int, int] | None = None,
    priors_team_game: pd.DataFrame | None = None,
    events_team_game: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build candidate pool from roster_nightly (pregame-ish, NO truth leakage).

    Primary source: `silver/roster_nightly/season=YYYY/**/roster.parquet`.
    - Filters to (game_id, team_id)
    - If `active_flag` exists, keeps only active_flag==True
    - Maps NBA personId -> rot_v1 internal player_id via `build_person_id_to_internal_id_map`

    Fallback behavior (only if roster slice is missing/empty):
    - Union of `priors_team_game` (if provided; already internal ids)
    - Plus first-segment lineup ids from `events_team_game` (if provided; postgame-ish proxy)
    """
    root = (data_root or paths.get_data_root()).expanduser().resolve()
    gid = canonicalize_game_id(game_id)
    if not gid:
        raise ValueError(f"Invalid game_id: {game_id!r}")
    team_id = int(team_id)
    season_start_year = int(season_start_year)

    roster_base = root / "silver" / "roster_nightly" / f"season={season_start_year}"
    roster_df = pd.DataFrame()
    if roster_base.exists():
        try:
            dataset = ds.dataset(str(roster_base), format="parquet")
            gid_int = int(gid)
            table = dataset.to_table(
                columns=["game_id", "team_id", "player_id", "active_flag"],
                filter=(ds.field("game_id") == gid_int) & (ds.field("team_id") == team_id),
            )
            roster_df = table.to_pandas()
        except Exception:
            roster_df = pd.DataFrame()

    if not roster_df.empty and "active_flag" in roster_df.columns:
        active = roster_df["active_flag"]
        if active.dtype != bool:
            active = active.astype("boolean", copy=False).fillna(False).astype(bool)
        roster_df = roster_df.loc[active].copy()

    candidates: list[int] = []
    if not roster_df.empty:
        roster_df["player_id"] = pd.to_numeric(roster_df["player_id"], errors="coerce").astype("Int64")
        roster_df = roster_df.dropna(subset=["player_id"]).copy()
        person_ids = [int(x) for x in roster_df["player_id"].tolist()]

        if person_id_to_internal_id is None:
            mapped = build_person_id_to_internal_id_map(season_start_year=season_start_year, data_root=root)
            person_id_to_internal_id = mapped.person_id_to_internal_id

        candidates = [int(person_id_to_internal_id.get(int(pid), -1)) for pid in person_ids]
        candidates = [int(x) for x in candidates if int(x) > 0]

    if not candidates:
        fallback: set[int] = set()
        if priors_team_game is not None and not priors_team_game.empty and "player_id" in priors_team_game.columns:
            fallback.update(int(x) for x in pd.to_numeric(priors_team_game["player_id"], errors="coerce").dropna().tolist())
        if events_team_game is not None and not events_team_game.empty:
            fallback.update(_infer_first_segment_lineup_player_ids(events_team_game))
        candidates = sorted(fallback)

    out = pd.DataFrame(
        {"game_id": [gid] * len(candidates), "team_id": [team_id] * len(candidates), "player_id": candidates}
    )
    out = out.drop_duplicates().sort_values(["game_id", "team_id", "player_id"], kind="mergesort").reset_index(drop=True)
    return out

