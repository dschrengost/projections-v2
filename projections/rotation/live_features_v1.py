"""Live feature builder for `rotation_set_minutes_v1`.

This builder is designed to match the training dataset transforms exactly where
possible, reusing shared helpers extracted from
`scripts/rotation/build_rotation_train_dataset_v1.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from projections.features.dnp_history import (
    DNP_HISTORY_FEATURE_COLUMNS,
    DNPHistoryConfig,
    compute_dnp_history_features_for_live,
    derive_roster_active_pre_tip,
)
from projections.rotation.rotation_set_minutes_features_v1 import (
    apply_odds_missing_flags,
    fill_numeric_missing_with_zero,
    join_rotation_priors,
    propagate_team_level_columns,
)

KEY_COLS: tuple[str, str, str] = ("game_id", "team_id", "player_id")
OPPONENT_TEAM_ID_COL = "opponent_team_id"


class RotationLiveFeaturesError(RuntimeError):
    """Raised when live feature construction fails."""


@dataclass(frozen=True)
class RotationSetMinutesV1LiveFeaturesResult:
    features: pd.DataFrame
    dropped_extra_columns: list[str]


@dataclass(frozen=True)
class RotationSetMinutesV1FeatureSpec:
    model_dir: Path
    feature_columns: list[str]

    @classmethod
    def load(cls, model_dir: Path) -> "RotationSetMinutesV1FeatureSpec":
        model_dir = Path(model_dir).expanduser().resolve()
        path = model_dir / "feature_columns.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        cols = payload.get("columns")
        if not isinstance(cols, list) or not cols:
            raise RotationLiveFeaturesError(f"Invalid feature_columns.json at {path}")
        return cls(model_dir=model_dir, feature_columns=[str(c) for c in cols])


def _season_for_date(day: pd.Timestamp) -> int:
    # Match the convention used elsewhere in the repo (Aug–Jul season boundary).
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _required_prior_windows(feature_columns: Iterable[str]) -> set[int]:
    windows: set[int] = set()
    for col in feature_columns:
        text = str(col)
        if "_prior_" not in text:
            continue
        suffix = text.rsplit("_prior_", 1)[-1]
        suffix = suffix.removesuffix("_missing")
        try:
            windows.add(int(suffix))
        except ValueError:
            continue
    return windows


def load_rotation_priors_for_game_ids(
    data_root: Path,
    *,
    season: int,
    game_ids: Iterable[str],
    game_date: str | None = None,
    require_all_game_ids: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load rotation_priors_v1 partitions for the requested game_ids."""

    data_root = Path(data_root).expanduser().resolve()
    team_root = data_root / "silver" / "rotation_priors_v1" / "team_game_priors" / f"season={int(season)}"
    player_root = data_root / "silver" / "rotation_priors_v1" / "player_game_priors" / f"season={int(season)}"

    game_ids_norm = [str(gid).zfill(10) for gid in game_ids if str(gid).strip()]

    def _read_many(root: Path) -> tuple[pd.DataFrame, list[str]]:
        frames: list[pd.DataFrame] = []
        missing: list[str] = []
        for gid in game_ids_norm:
            path = root / f"game_id={gid}.parquet"
            if not path.exists():
                missing.append(gid)
                continue
            frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame(), missing
        return pd.concat(frames, ignore_index=True), missing

    team_df, missing_team = _read_many(team_root)
    player_df, missing_player = _read_many(player_root)

    if require_all_game_ids and (missing_team or missing_player):
        date_clause = f" date={game_date}" if game_date else ""
        expected_team = [str(team_root / f"game_id={gid}.parquet") for gid in missing_team]
        expected_player = [str(player_root / f"game_id={gid}.parquet") for gid in missing_player]
        raise RotationLiveFeaturesError(
            "rotation_priors_v1 missing required priors partitions"
            f"{date_clause} season={int(season)} "
            f"missing_team_game_ids={missing_team} missing_player_game_ids={missing_player} "
            f"expected_team_paths={expected_team} expected_player_paths={expected_player}"
        )

    return team_df, player_df


def load_latest_rotation_priors_by_entity(
    data_root: Path,
    *,
    season: int,
    team_ids: Iterable[int] | None = None,
    player_ids: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the most recent rotation_priors_v1 for each team/player.

    This is the correct approach for live inference: we want the latest available
    priors for each entity, not priors keyed by today's (non-existent) game_ids.

    Returns:
        (team_priors, player_priors) DataFrames with one row per team/player
        containing their most recent prior values.
    """
    data_root = Path(data_root).expanduser().resolve()
    team_root = data_root / "silver" / "rotation_priors_v1" / "team_game_priors" / f"season={int(season)}"
    player_root = data_root / "silver" / "rotation_priors_v1" / "player_game_priors" / f"season={int(season)}"

    def _load_all_and_get_latest(root: Path, entity_col: str, filter_ids: set[int] | None) -> pd.DataFrame:
        if not root.exists():
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for path in root.glob("game_id=*.parquet"):
            try:
                df = pd.read_parquet(path)
                frames.append(df)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
        combined[entity_col] = pd.to_numeric(combined[entity_col], errors="coerce").astype("Int64")

        # Filter to requested entities if provided
        if filter_ids:
            combined = combined[combined[entity_col].isin(filter_ids)].copy()

        if combined.empty:
            return pd.DataFrame()

        # Get the most recent row per entity
        combined = combined.sort_values("game_date", ascending=False)
        latest = combined.groupby(entity_col, dropna=False).first().reset_index()

        return latest

    team_filter = set(int(t) for t in team_ids) if team_ids else None
    player_filter = set(int(p) for p in player_ids) if player_ids else None

    team_priors = _load_all_and_get_latest(team_root, "team_id", team_filter)
    player_priors = _load_all_and_get_latest(player_root, "person_id", player_filter)

    return team_priors, player_priors


@dataclass(frozen=True)
class RotationPriorsLoadResult:
    """Result of loading rotation priors for live features."""

    team_priors: pd.DataFrame
    player_priors: pd.DataFrame
    teams_found: int
    teams_missing: int
    players_found: int
    players_missing: int
    used_latest_fallback: bool
    warning_message: str | None = None


def load_rotation_priors_for_live_inference(
    data_root: Path,
    *,
    season: int,
    game_date: str | None = None,
    game_ids: Iterable[str],
    team_ids: Iterable[int],
    player_ids: Iterable[int],
    allow_priors_fallback: bool = False,
) -> RotationPriorsLoadResult:
    """Load rotation priors for live inference.

    By default, this requires priors partitions for all requested game_ids.
    If `allow_priors_fallback=True`, missing game_id partitions are filled using
    the latest available priors per entity (team/player), stamped onto the
    missing game_ids.

    This fallback is an emergency stopgap and must be explicitly enabled.
    """

    game_ids_list = list(game_ids)
    team_ids_set = set(int(t) for t in team_ids)
    player_ids_set = set(int(p) for p in player_ids)

    game_ids_norm = [str(gid).zfill(10) for gid in game_ids_list if str(gid).strip()]
    data_root = Path(data_root).expanduser().resolve()
    team_root = data_root / "silver" / "rotation_priors_v1" / "team_game_priors" / f"season={int(season)}"
    player_root = data_root / "silver" / "rotation_priors_v1" / "player_game_priors" / f"season={int(season)}"

    missing_team_game_ids = [gid for gid in game_ids_norm if not (team_root / f"game_id={gid}.parquet").exists()]
    missing_player_game_ids = [gid for gid in game_ids_norm if not (player_root / f"game_id={gid}.parquet").exists()]

    team_priors, player_priors = load_rotation_priors_for_game_ids(
        data_root,
        season=season,
        game_ids=game_ids_norm,
        require_all_game_ids=False,
    )

    if (missing_team_game_ids or missing_player_game_ids) and not allow_priors_fallback:
        date_clause = f" date={game_date}" if game_date else ""
        expected_team = [str(team_root / f"game_id={gid}.parquet") for gid in missing_team_game_ids]
        expected_player = [str(player_root / f"game_id={gid}.parquet") for gid in missing_player_game_ids]
        raise RotationLiveFeaturesError(
            "rotation_priors_v1 missing required priors partitions "
            f"{date_clause} season={int(season)} missing_team_game_ids={missing_team_game_ids} "
            f"missing_player_game_ids={missing_player_game_ids} "
            f"expected_team_paths={expected_team} expected_player_paths={expected_player}"
        )

    # Check coverage
    if not team_priors.empty:
        team_priors["team_id"] = pd.to_numeric(team_priors["team_id"], errors="coerce").astype("Int64")
        teams_in_priors = set(team_priors["team_id"].dropna().astype(int).tolist())
    else:
        teams_in_priors = set()

    if not player_priors.empty:
        player_priors["person_id"] = pd.to_numeric(player_priors["person_id"], errors="coerce").astype("Int64")
        players_in_priors = set(player_priors["person_id"].dropna().astype(int).tolist())
    else:
        players_in_priors = set()

    teams_found = len(teams_in_priors & team_ids_set)
    teams_missing = len(team_ids_set - teams_in_priors)
    players_found = len(players_in_priors & player_ids_set)
    players_missing = len(player_ids_set - players_in_priors)

    # If we have good coverage and no missing game_id partitions, return as-is.
    if teams_missing == 0 and players_missing == 0 and not missing_team_game_ids and not missing_player_game_ids:
        return RotationPriorsLoadResult(
            team_priors=team_priors,
            player_priors=player_priors,
            teams_found=teams_found,
            teams_missing=0,
            players_found=players_found,
            players_missing=0,
            used_latest_fallback=False,
        )

    # Fallback: load latest priors for missing entities and stamp onto missing game_ids.
    date_clause = f" date={game_date}" if game_date else ""
    warning_msg = (
        f"Priors missing for game_ids={sorted(set(missing_team_game_ids + missing_player_game_ids))} "
        f"(teams_missing={teams_missing}, players_missing={players_missing})."
        f"{date_clause} "
        "Falling back to latest available priors and stamping onto missing game_ids."
    )

    latest_team, latest_player = load_latest_rotation_priors_by_entity(
        data_root,
        season=season,
        team_ids=sorted(team_ids_set) if missing_team_game_ids else None,
        player_ids=sorted(player_ids_set) if missing_player_game_ids else None,
    )

    def _stamp(df: pd.DataFrame, *, game_ids_to_stamp: list[str]) -> pd.DataFrame:
        if df.empty or not game_ids_to_stamp:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for gid in game_ids_to_stamp:
            copy = df.copy()
            copy["game_id"] = gid
            copy["game_id_norm"] = gid
            frames.append(copy)
        return pd.concat(frames, ignore_index=True)

    # Merge fallback priors: ensure the missing game_id partitions are represented.
    if missing_team_game_ids:
        stamped_team = _stamp(latest_team, game_ids_to_stamp=missing_team_game_ids)
        if not stamped_team.empty:
            team_priors = pd.concat([team_priors, stamped_team], ignore_index=True) if not team_priors.empty else stamped_team
    if missing_player_game_ids:
        stamped_player = _stamp(latest_player, game_ids_to_stamp=missing_player_game_ids)
        if not stamped_player.empty:
            player_priors = (
                pd.concat([player_priors, stamped_player], ignore_index=True) if not player_priors.empty else stamped_player
            )

    # Recalculate coverage
    if not team_priors.empty:
        team_priors["team_id"] = pd.to_numeric(team_priors["team_id"], errors="coerce").astype("Int64")
        teams_in_priors = set(team_priors["team_id"].dropna().astype(int).tolist())
    else:
        teams_in_priors = set()

    if not player_priors.empty:
        player_priors["person_id"] = pd.to_numeric(player_priors["person_id"], errors="coerce").astype("Int64")
        players_in_priors = set(player_priors["person_id"].dropna().astype(int).tolist())
    else:
        players_in_priors = set()

    final_teams_found = len(teams_in_priors & team_ids_set)
    final_teams_missing = len(team_ids_set - teams_in_priors)
    final_players_found = len(players_in_priors & player_ids_set)
    final_players_missing = len(player_ids_set - players_in_priors)

    return RotationPriorsLoadResult(
        team_priors=team_priors,
        player_priors=player_priors,
        teams_found=final_teams_found,
        teams_missing=final_teams_missing,
        players_found=final_players_found,
        players_missing=final_players_missing,
        used_latest_fallback=True,
        warning_message=warning_msg,
    )


def compute_minutes_features_row_missing(
    df: pd.DataFrame,
    *,
    base_feature_cols: list[str],
    threshold: float = 0.5,
) -> pd.Series:
    """Heuristic row-level "feature row missing" indicator for live inference.

    Training builds this flag during the labels→features join. In live inference,
    we approximate it using the fraction of missing values across core non-prior
    model features.
    """

    if df.empty:
        return pd.Series([], dtype="int8")
    cols = [c for c in base_feature_cols if c in df.columns]
    if not cols:
        return pd.Series(0, index=df.index, dtype="int8")
    missing_rate = df[cols].isna().mean(axis=1)
    return (missing_rate >= float(threshold)).astype("int8")


def _dtype_for_feature(col: str) -> str:
    if col.endswith("_missing"):
        return "int8"
    if col.startswith(("team_prior_n_games_", "player_prior_n_games_")):
        return "int16"
    if col in {
        "home_flag",
        "restriction_flag",
        "ramp_flag",
        "is_out",
        "is_q",
        "is_prob",
        "injury_snapshot_missing",
        "available_B",
        "available_G",
        "available_W",
        "depth_same_pos_active",
        "is_projected_starter",
        "is_confirmed_starter",
        "same_archetype_overlap",
        "days_since_last",
        "is_b2b",
        "is_3in4",
        "is_4in6",
        "minutes_features_row_missing",
        "spread_home_missing",
        "total_missing",
    }:
        return "int8"
    return "float64"


def enforce_feature_dtypes(df: pd.DataFrame, *, feature_columns: list[str]) -> pd.DataFrame:
    """Coerce feature columns into stable int/float dtypes (training-consistent)."""

    out = df.copy()
    for col in feature_columns:
        if col not in out.columns:
            continue
        dtype = _dtype_for_feature(col)
        if dtype.startswith("int"):
            fill = 1 if col.endswith("_missing") else 0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill).astype(dtype)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(dtype)
    return out


def validate_and_project_features(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    keep_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Validate required feature columns and project onto an ordered slice."""

    keep = [c for c in keep_cols if c in df.columns]
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise RotationLiveFeaturesError(f"Missing required features (n={len(missing)}): {missing}")

    allowed = set(KEY_COLS) | set(keep) | set(feature_columns)
    extra = [c for c in df.columns if c not in allowed]
    if extra:
        raise RotationLiveFeaturesError(f"Unexpected extra columns present (n={len(extra)}): {extra}")

    ordered = [*KEY_COLS, *keep, *feature_columns]
    return df.loc[:, ordered].copy()


def build_rotation_set_minutes_v1_features(
    minutes_features: pd.DataFrame,
    *,
    team_priors: pd.DataFrame,
    player_priors: pd.DataFrame,
    feature_columns: list[str],
    historical_features: pd.DataFrame | None = None,
) -> RotationSetMinutesV1LiveFeaturesResult:
    """Build a live feature frame for the rotation_set_minutes_v1 model.

    Args:
        minutes_features: Current game features (today's slate)
        team_priors: Rotation priors at team level
        player_priors: Rotation priors at player level
        feature_columns: Required feature columns for the model
        historical_features: Optional historical features with realized minutes
            for computing DNP history features. If provided, must have columns:
            game_id, team_id, player_id, game_date, is_out, minutes
    """

    missing_keys = [c for c in KEY_COLS if c not in minutes_features.columns]
    if missing_keys:
        raise RotationLiveFeaturesError(f"Minutes features missing required keys: {missing_keys}")

    work = minutes_features.copy()
    # Ensure a deterministic key dtype and clean rows.
    work["game_id"] = pd.to_numeric(work["game_id"], errors="coerce").astype("Int64")
    work["team_id"] = pd.to_numeric(work["team_id"], errors="coerce").astype("Int64")
    work["player_id"] = pd.to_numeric(work["player_id"], errors="coerce").astype("Int64")
    work = work.dropna(subset=list(KEY_COLS)).copy()

    work = propagate_team_level_columns(
        work,
        cols=[
            "home_team_id",
            "away_team_id",
            OPPONENT_TEAM_ID_COL,
            "home_flag",
            "spread_home",
            "total",
            "odds_as_of_ts",
        ],
    )
    work = apply_odds_missing_flags(work)

    # Compute minutes_features_row_missing before global numeric fill.
    base_non_prior = [
        c
        for c in feature_columns
        if "_prior_" not in c and not c.endswith("_missing") and c not in {"minutes_features_row_missing"}
    ]
    work["minutes_features_row_missing"] = compute_minutes_features_row_missing(work, base_feature_cols=base_non_prior)

    work = fill_numeric_missing_with_zero(work)
    work = join_rotation_priors(work, team_priors=team_priors, player_priors=player_priors)

    # Compute DNP history features if historical data is available and features are needed
    dnp_feature_cols = [c for c in DNP_HISTORY_FEATURE_COLUMNS if c in feature_columns]
    if dnp_feature_cols:
        if historical_features is not None and not historical_features.empty:
            # Compute DNP history features from historical data
            work = compute_dnp_history_features_for_live(
                work,
                historical_features,
                config=DNPHistoryConfig(),
                game_date_col="game_date",
                player_id_col="player_id",
                team_id_col="team_id",
                is_out_col="is_out",
                minutes_col="minutes",
            )
        else:
            # Fill with default values if no historical data
            work["roster_active_pre_tip"] = derive_roster_active_pre_tip(work, is_out_col="is_out")
            config = DNPHistoryConfig()
            work["games_since_last_roster_active"] = config.games_since_cap
            work["never_roster_active_before"] = 1
            work["consecutive_active_dnp"] = 0
            work["active_but_dnp_rate_last10"] = config.alpha / (config.alpha + config.beta)
            work["inactive_streak_len"] = 0

    # Validate feature list + keep opponent_team_id for embeddings.
    # Also keep minutes_features_row_missing for guardrail logic even if not a model feature.
    keep_cols = [OPPONENT_TEAM_ID_COL] if OPPONENT_TEAM_ID_COL in work.columns else []
    if "minutes_features_row_missing" in work.columns and "minutes_features_row_missing" not in feature_columns:
        keep_cols.append("minutes_features_row_missing")
    work = enforce_feature_dtypes(work, feature_columns=feature_columns)

    # For the live builder output we allow extra columns in `minutes_features` but do not export them.
    export_allowed = set(KEY_COLS) | set(keep_cols) | set(feature_columns)
    extra = [c for c in work.columns if c not in export_allowed]
    if extra:
        work = work.drop(columns=extra)

    missing = [c for c in feature_columns if c not in work.columns]
    if missing:
        raise RotationLiveFeaturesError(f"Missing required features after build (n={len(missing)}): {missing}")

    ordered = [*KEY_COLS, *keep_cols, *feature_columns]
    return RotationSetMinutesV1LiveFeaturesResult(
        features=work.loc[:, ordered].copy(),
        dropped_extra_columns=extra,
    )
