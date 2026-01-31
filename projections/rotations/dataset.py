from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from projections.rotations.schemas import LINEUP_COLS, ROTATION_EVENTS_COLS, ROTATION_LABELS_COLS
from projections.rotations.utils_time import ensure_clock_sec_columns


def _coerce_game_id_int(game_id: pd.Series) -> pd.Series:
    out = pd.to_numeric(game_id, errors="coerce").astype("Int64")
    if out.isna().any():
        bad = game_id[out.isna()].head(10).tolist()
        raise ValueError(f"Invalid game_id values (cannot coerce to int): {bad}")
    return out


def _validate_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "season", "home_team_id", "away_team_id"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule missing required columns: {sorted(missing)}")
    out = schedule.copy()
    out["game_id_int"] = _coerce_game_id_int(out["game_id"])
    out["home_team_id"] = pd.to_numeric(out["home_team_id"], errors="coerce").astype("Int64")
    out["away_team_id"] = pd.to_numeric(out["away_team_id"], errors="coerce").astype("Int64")
    if out["home_team_id"].isna().any() or out["away_team_id"].isna().any():
        raise ValueError("schedule contains missing/invalid home_team_id/away_team_id")
    out["season_id"] = out["season"].astype("string")
    return out[["game_id_int", "season_id", "home_team_id", "away_team_id"]].drop_duplicates(
        subset=["game_id_int"],
        keep="last",
    )


def _attach_schedule(stints_like: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    schedule_lookup = _validate_schedule(schedule)
    work = stints_like.copy()
    # Phase 1 bundles include a nullable `season_id` column; schedule join is authoritative.
    if "season_id" in work.columns:
        work = work.drop(columns=["season_id"])
    work["game_id"] = work["game_id"].astype("string")
    work["game_id_int"] = _coerce_game_id_int(work["game_id"])
    merged = work.merge(schedule_lookup, on="game_id_int", how="left", validate="many_to_one")
    missing = merged[merged["home_team_id"].isna() | merged["away_team_id"].isna()]["game_id"].unique().tolist()
    if missing:
        raise ValueError(f"Missing schedule rows for {len(missing)} games (examples={missing[:10]})")
    return merged


def _ensure_unique_lineup_ids(df: pd.DataFrame, cols: list[str], *, context: str) -> None:
    if df[cols].isna().any().any():
        bad = df.loc[df[cols].isna().any(axis=1), ["game_id"] + cols].head(5).to_dict(orient="records")
        raise ValueError(f"{context}: null lineup ids present (examples={bad})")
    values = df[cols].astype("int64").to_numpy()
    unique_counts = pd.DataFrame(values).nunique(axis=1).to_numpy()
    if (unique_counts != 5).any():
        idx = int((unique_counts != 5).nonzero()[0][0])
        example = df.iloc[idx][["game_id", "period"] + cols].to_dict()
        raise ValueError(f"{context}: non-unique lineup ids found (example={example})")


@dataclass(frozen=True)
class RotationDataset:
    rotation_events: pd.DataFrame
    rotation_labels: pd.DataFrame


def build_rotation_events(
    stints: pd.DataFrame,
    *,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Build canonical rotation event stream from Phase 1 stints."""

    stints = ensure_clock_sec_columns(stints)
    required = {"game_id", "stint_id", "period", "start_clock_sec", "end_clock_sec", "duration_sec"} | {
        f"home_p{i}" for i in range(1, 6)
    } | {f"away_p{i}" for i in range(1, 6)}
    missing = required - set(stints.columns)
    if missing:
        raise ValueError(f"stints missing required columns: {sorted(missing)}")

    merged = _attach_schedule(stints, schedule)

    home_cols = [f"home_p{i}" for i in range(1, 6)]
    away_cols = [f"away_p{i}" for i in range(1, 6)]
    _ensure_unique_lineup_ids(merged, home_cols, context="stints.home")
    _ensure_unique_lineup_ids(merged, away_cols, context="stints.away")

    base_cols = [
        "season_id",
        "game_id",
        "stint_id",
        "period",
        "start_clock_sec",
        "end_clock_sec",
        "duration_sec",
        "home_team_id",
        "away_team_id",
    ]

    base = merged[base_cols + home_cols + away_cols].copy()

    home = base[base_cols + home_cols].copy()
    home = home.rename(columns={c: f"lineup_p{i+1}" for i, c in enumerate(home_cols)})
    home["team_id"] = home["home_team_id"].astype("int64")
    home["opponent_team_id"] = home["away_team_id"].astype("int64")
    home["is_home"] = True

    away = base[base_cols + away_cols].copy()
    away = away.rename(columns={c: f"lineup_p{i+1}" for i, c in enumerate(away_cols)})
    away["team_id"] = away["away_team_id"].astype("int64")
    away["opponent_team_id"] = away["home_team_id"].astype("int64")
    away["is_home"] = False

    out = pd.concat([home, away], ignore_index=True)
    out["raw_ref"] = out["stint_id"].astype("int64").astype(str)

    out["segment_idx"] = 0
    out = out.sort_values(
        ["team_id", "game_id", "period", "start_clock_sec", "stint_id"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["segment_idx"] = out.groupby(["team_id", "game_id"], sort=False).cumcount().astype("int64")

    keep = list(ROTATION_EVENTS_COLS)
    missing = [c for c in keep if c not in out.columns]
    if missing:
        raise ValueError(f"Internal bug: rotation_events missing columns: {missing}")

    # Enforce deterministic ordering and column order.
    out = out.sort_values(
        ["team_id", "game_id", "period", "start_clock_sec", "segment_idx"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out = out.loc[:, keep].copy()

    out["season_id"] = out["season_id"].astype("string")
    out["game_id"] = out["game_id"].astype("string")
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("int64")
    out["opponent_team_id"] = pd.to_numeric(out["opponent_team_id"], errors="coerce").astype("int64")
    out["is_home"] = out["is_home"].astype(bool)
    out["segment_idx"] = pd.to_numeric(out["segment_idx"], errors="coerce").astype("int64")
    out["period"] = pd.to_numeric(out["period"], errors="coerce").astype("int64")
    out["start_clock_sec"] = pd.to_numeric(out["start_clock_sec"], errors="coerce").astype("int64")
    out["end_clock_sec"] = pd.to_numeric(out["end_clock_sec"], errors="coerce").astype("int64")
    out["duration_sec"] = pd.to_numeric(out["duration_sec"], errors="coerce").astype("int64")
    for c in LINEUP_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("int64")
    out["raw_ref"] = out["raw_ref"].astype("string")

    return out


def _infer_starters_from_events(rotation_events: pd.DataFrame) -> pd.DataFrame:
    required = {"team_id", "game_id", "segment_idx", "duration_sec"} | set(LINEUP_COLS)
    missing = required - set(rotation_events.columns)
    if missing:
        raise ValueError(f"rotation_events missing required columns for starter inference: {sorted(missing)}")

    work = rotation_events.loc[:, ["team_id", "game_id", "segment_idx", "duration_sec", *LINEUP_COLS]].copy()
    work = work.sort_values(["team_id", "game_id", "segment_idx"], kind="mergesort")
    work = work[work["duration_sec"] > 0].copy()
    if work.empty:
        return pd.DataFrame(columns=["team_id", "game_id", "player_id"])

    first = work.groupby(["team_id", "game_id"], sort=False).head(1)
    starter_rows: list[pd.DataFrame] = []
    for c in LINEUP_COLS:
        starter_rows.append(first[["team_id", "game_id"]].assign(player_id=first[c].astype("int64")))
    starters = pd.concat(starter_rows, ignore_index=True).drop_duplicates(subset=["team_id", "game_id", "player_id"])
    return starters


def build_rotation_labels(
    player_stints: pd.DataFrame,
    *,
    schedule: pd.DataFrame,
    rotation_events: pd.DataFrame,
) -> pd.DataFrame:
    """Build membership + regime labels from Phase 1 player stints."""

    required = {"game_id", "team_side", "player_id", "duration_sec"}
    missing = required - set(player_stints.columns)
    if missing:
        raise ValueError(f"player_stints missing required columns: {sorted(missing)}")

    merged = _attach_schedule(player_stints, schedule)
    side = merged["team_side"].astype("string").str.lower()
    is_home = side == "home"
    is_away = side == "away"
    if (~(is_home | is_away)).any():
        bad = merged.loc[~(is_home | is_away), "team_side"].dropna().unique().tolist()[:10]
        raise ValueError(f"Unexpected team_side values (expected home/away): {bad}")

    merged = merged.copy()
    merged["team_id"] = merged["home_team_id"].where(is_home, merged["away_team_id"]).astype("int64")

    durations = pd.to_numeric(merged["duration_sec"], errors="coerce").astype("int64")
    merged = merged.assign(duration_sec=durations)

    grouped = (
        merged.groupby(["game_id", "team_id", "player_id"], sort=False)["duration_sec"]
        .sum()
        .reset_index()
    )
    grouped["minutes_actual"] = grouped["duration_sec"].astype("float64") / 60.0
    grouped["played_ge_1"] = grouped["minutes_actual"] >= 1.0
    grouped["played_ge_5"] = grouped["minutes_actual"] >= 5.0

    starters = _infer_starters_from_events(rotation_events)
    grouped = grouped.merge(
        starters.assign(starter_actual=True),
        on=["team_id", "game_id", "player_id"],
        how="left",
        validate="many_to_one",
    )
    grouped["starter_actual"] = grouped["starter_actual"].eq(True)

    rotation_counts = (
        grouped.groupby(["team_id", "game_id"], sort=False)["played_ge_5"]
        .sum()
        .reset_index(name="_rotation_ge_5_count")
    )

    def _bucket(count: int) -> str:
        if int(count) <= 8:
            return "tight"
        if int(count) <= 10:
            return "normal"
        return "deep"

    rotation_counts["regime_label"] = rotation_counts["_rotation_ge_5_count"].map(_bucket).astype("string")
    grouped = grouped.merge(
        rotation_counts[["team_id", "game_id", "regime_label"]],
        on=["team_id", "game_id"],
        how="left",
        validate="many_to_one",
    )

    out = grouped.loc[:, list(ROTATION_LABELS_COLS)].copy()
    out["game_id"] = out["game_id"].astype("string")
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("int64")
    out["minutes_actual"] = pd.to_numeric(out["minutes_actual"], errors="coerce").astype("float64")
    out["played_ge_1"] = out["played_ge_1"].astype(bool)
    out["played_ge_5"] = out["played_ge_5"].astype(bool)
    out["starter_actual"] = out["starter_actual"].astype(bool)
    out["regime_label"] = out["regime_label"].astype("string")

    return out


def build_rotation_dataset(
    *,
    stints: pd.DataFrame,
    player_stints: pd.DataFrame,
    schedule: pd.DataFrame,
) -> RotationDataset:
    rotation_events = build_rotation_events(stints, schedule=schedule)
    rotation_labels = build_rotation_labels(player_stints, schedule=schedule, rotation_events=rotation_events)
    return RotationDataset(rotation_events=rotation_events, rotation_labels=rotation_labels)
