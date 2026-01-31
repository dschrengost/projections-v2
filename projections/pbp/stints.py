from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from projections.pbp.vendor_ingest import period_length_sec


STINTS_COLS = [
    "schema_version",
    "season_id",
    "game_id",
    "stint_id",
    "period",
    "start_period_elapsed_sec",
    "end_period_elapsed_sec",
    "start_clock_sec",
    "end_clock_sec",
    "start_game_elapsed_sec",
    "end_game_elapsed_sec",
    "duration_sec",
    "away_p1",
    "away_p2",
    "away_p3",
    "away_p4",
    "away_p5",
    "home_p1",
    "home_p2",
    "home_p3",
    "home_p4",
    "home_p5",
    "away_lineup_key",
    "home_lineup_key",
]


PLAYER_STINTS_COLS = [
    "schema_version",
    "season_id",
    "game_id",
    "stint_id",
    "team_side",
    "player_id",
    "period",
    "start_game_elapsed_sec",
    "end_game_elapsed_sec",
    "duration_sec",
]


def _period_start_offset_sec(period: int) -> int:
    if period <= 0:
        return 0
    if period <= 4:
        return (period - 1) * 720
    return 4 * 720 + (period - 5) * 300


@dataclass
class BuildStintsResult:
    stints: pd.DataFrame
    player_stints: pd.DataFrame


def build_stints_from_pbp_events(
    pbp_events: pd.DataFrame,
    *,
    schema_version: str,
) -> BuildStintsResult:
    required = [
        "schema_version",
        "season_id",
        "game_id",
        "period",
        "play_id",
        "period_elapsed_sec",
        "clock_sec",
        "away_lineup_key",
        "home_lineup_key",
    ] + [f"away_p{i}" for i in range(1, 6)] + [f"home_p{i}" for i in range(1, 6)]
    missing = [c for c in required if c not in pbp_events.columns]
    if missing:
        raise ValueError(f"pbp_events missing required columns for stint build: {missing}")

    # Deterministic ordering within each game.
    # Canonical ordering: period asc, clock_sec desc, play_id asc.
    df = pbp_events.sort_values(
        ["game_id", "period", "clock_sec", "play_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # End time per row is next row's elapsed within period, else period end.
    next_period = df.groupby("game_id")["period"].shift(-1)
    next_elapsed = df.groupby("game_id")["period_elapsed_sec"].shift(-1)

    period_len = df["period"].map(period_length_sec).astype(int)
    end_elapsed = np.where(next_period == df["period"], next_elapsed, period_len).astype(float)
    start_elapsed = df["period_elapsed_sec"].astype(float)

    duration = (end_elapsed - start_elapsed).astype(int)

    # Force a stint boundary when either side's lineup changes, or when period changes.
    prev_away = df.groupby("game_id")["away_lineup_key"].shift(1)
    prev_home = df.groupby("game_id")["home_lineup_key"].shift(1)
    prev_period = df.groupby("game_id")["period"].shift(1)

    is_new_stint = (
        (prev_period.isna())
        | (df["period"] != prev_period)
        | (df["away_lineup_key"] != prev_away)
        | (df["home_lineup_key"] != prev_home)
    )

    df["_stint_seq"] = is_new_stint.groupby(df["game_id"]).cumsum().astype(int)
    df["_duration_sec"] = duration
    df["_end_period_elapsed_sec"] = end_elapsed.astype(int)

    # Aggregate rows into stints.
    group_cols = ["game_id", "_stint_seq"]
    first_cols = [
        "schema_version",
        "season_id",
        "game_id",
        "period",
        "period_elapsed_sec",
        "clock_sec",
        "away_lineup_key",
        "home_lineup_key",
    ] + [f"away_p{i}" for i in range(1, 6)] + [f"home_p{i}" for i in range(1, 6)]

    first = df.groupby(group_cols, sort=False).head(1).reset_index(drop=True)
    first = first[group_cols + [c for c in first_cols if c not in group_cols]].copy()
    last_end = df.groupby(group_cols, sort=False)["_end_period_elapsed_sec"].max().reset_index()
    total_dur = df.groupby(group_cols, sort=False)["_duration_sec"].sum().reset_index()

    stints = first.merge(last_end, on=group_cols, how="left").merge(total_dur, on=group_cols, how="left")

    stints = stints.rename(
        columns={
            "_stint_seq": "stint_id",
            "period_elapsed_sec": "start_period_elapsed_sec",
            "_end_period_elapsed_sec": "end_period_elapsed_sec",
            "clock_sec": "start_clock_sec",
            "_duration_sec": "duration_sec",
        }
    )
    stints["end_clock_sec"] = stints["period"].map(period_length_sec).astype(int) - stints["end_period_elapsed_sec"].astype(int)
    stints["start_game_elapsed_sec"] = stints["period"].map(_period_start_offset_sec).astype(int) + stints["start_period_elapsed_sec"].astype(int)
    stints["end_game_elapsed_sec"] = stints["period"].map(_period_start_offset_sec).astype(int) + stints["end_period_elapsed_sec"].astype(int)

    stints["schema_version"] = schema_version
    stints = stints.sort_values(["game_id", "stint_id"], kind="mergesort").reset_index(drop=True)

    stints_missing = [c for c in STINTS_COLS if c not in stints.columns]
    if stints_missing:
        raise ValueError(f"Internal schema bug: stints missing columns: {stints_missing}")
    stints = stints[STINTS_COLS].copy()

    # Player stints: explode wide stints -> long (10 rows per stint).
    base = stints[
        [
            "schema_version",
            "season_id",
            "game_id",
            "stint_id",
            "period",
            "start_game_elapsed_sec",
            "end_game_elapsed_sec",
            "duration_sec",
        ]
    ].copy()

    home_long = base.join(stints[[f"home_p{i}" for i in range(1, 6)]])
    home_long = home_long.melt(
        id_vars=list(base.columns),
        value_vars=[f"home_p{i}" for i in range(1, 6)],
        var_name="slot",
        value_name="player_id",
    )
    home_long["team_side"] = "home"

    away_long = base.join(stints[[f"away_p{i}" for i in range(1, 6)]])
    away_long = away_long.melt(
        id_vars=list(base.columns),
        value_vars=[f"away_p{i}" for i in range(1, 6)],
        var_name="slot",
        value_name="player_id",
    )
    away_long["team_side"] = "away"

    player_stints = pd.concat([home_long, away_long], ignore_index=True)
    player_stints["player_id"] = player_stints["player_id"].astype(np.int64)
    player_stints = player_stints.drop(columns=["slot"]).sort_values(
        ["game_id", "stint_id", "team_side", "player_id"],
        kind="mergesort",
    )

    missing_ps = [c for c in PLAYER_STINTS_COLS if c not in player_stints.columns]
    if missing_ps:
        raise ValueError(f"Internal schema bug: player_stints missing columns: {missing_ps}")
    player_stints = player_stints[PLAYER_STINTS_COLS].reset_index(drop=True)

    return BuildStintsResult(stints=stints, player_stints=player_stints)


def assert_stints_have_required_lineups(stints: pd.DataFrame) -> None:
    home_cols = [f"home_p{i}" for i in range(1, 6)]
    away_cols = [f"away_p{i}" for i in range(1, 6)]
    if stints[home_cols + away_cols].isna().any().any():
        raise ValueError("stints has null player_id values in on-court lineups")
