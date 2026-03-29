"""Live feature builder for `rotation_set_minutes_v1`.

This builder is designed to match the training dataset transforms exactly where
possible, reusing shared helpers extracted from
`scripts/rotation/build_rotation_train_dataset_v1.py`.
"""

from __future__ import annotations

import json
import logging
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
    SAME_POS_CONTEXT_BUCKETS,
    ROTATION_SET_DERIVED_FEATURES,
    add_rotation_set_derived_features,
    apply_odds_missing_flags,
    fill_numeric_missing_with_zero,
    join_rotation_priors,
    propagate_team_level_columns,
)

KEY_COLS: tuple[str, str, str] = ("game_id", "team_id", "player_id")
OPPONENT_TEAM_ID_COL = "opponent_team_id"
_MAX_REASONABLE_PRIORS_PARTITION_ROWS = 5000
_logger = logging.getLogger(__name__)

_PLAYER_PRIOR_REFRESH_BASE_COLS: tuple[str, ...] = (
    "minutes_from_stints",
    "started_proxy",
    "fg2_pct",
    "fg3_pct",
    "ft_pct",
    "efg_pct",
    "fg2a_per_min",
    "fg3a_per_min",
    "fta_per_min",
    "three_pa_share",
)
_PLAYER_PRIOR_REFRESH_STD_COLS: tuple[str, ...] = ("minutes_from_stints",)
_TEAM_PRIOR_REFRESH_BASE_COLS: tuple[str, ...] = (
    "depth_6",
    "depth_10",
    "depth_14",
    "effective_n",
    "bench_conc_top1",
    "bench_conc_top2",
    "starter_pool_minutes",
    "bench_pool_minutes",
    "team_total_minutes_from_stints",
    "bench_share",
    "starter_share",
    "depth_gap_10_6",
    "fg2_pct_allowed",
    "fg3_pct_allowed",
    "fta_rate_allowed",
    "efg_pct_allowed",
    "three_pa_share_allowed",
)
_TEAM_PRIOR_REFRESH_STD_COLS: tuple[str, ...] = (
    "depth_6",
    "depth_10",
    "depth_14",
    "effective_n",
    "bench_conc_top1",
    "bench_conc_top2",
    "starter_pool_minutes",
    "bench_pool_minutes",
    "team_total_minutes_from_stints",
    "bench_share",
    "starter_share",
    "depth_gap_10_6",
)


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


def _is_regular_season_game_id(game_id: str) -> bool:
    """Check if a game_id is a regular season game (prefix 002).

    Game ID prefixes:
    - 002: Regular season
    - 003: Playoffs / NBA Cup knockout rounds
    - 004: Preseason
    - 005: Play-in tournament
    - 006: All-Star Weekend

    All-Star and NBA Cup knockout games use special team IDs that don't match
    regular season team IDs, which breaks the team_id-based priors join.
    """
    gid = str(game_id).zfill(10)
    return gid.startswith("002")


def _update_player_priors_with_latest_labels(
    player_priors: pd.DataFrame,
    data_root: Path,
    season: int,
) -> pd.DataFrame:
    """Update player priors to include the game they were computed for.

    The priors in rotation_priors_v1 are "pre-game" - they show what we knew
    BEFORE each game. For live inference, we need "post-game" priors that include
    the results from the latest game.

    This function loads box score labels and updates the rolling prior columns
    (e.g., started_proxy_rate_prior_5, minutes_from_stints_prior_5) to include
    the latest game's results.
    """
    if player_priors.empty:
        return player_priors

    # Load box score labels
    labels_path = data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    if not labels_path.exists():
        return player_priors

    try:
        labels = pd.read_parquet(labels_path)
    except Exception:
        return player_priors

    counts_path = data_root / "gold" / "labels_boxscore_counts" / "labels_boxscore_counts.parquet"
    counts_labels = pd.DataFrame()
    if counts_path.exists():
        try:
            counts_labels = pd.read_parquet(
                counts_path,
                columns=["player_id", "game_date", "fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "minutes"],
            )
        except Exception:
            counts_labels = pd.DataFrame()

    labels["player_id"] = pd.to_numeric(labels["player_id"], errors="coerce").astype("Int64")
    labels["game_date"] = pd.to_datetime(labels["game_date"], errors="coerce")
    if "minutes" in labels.columns:
        labels["minutes"] = pd.to_numeric(labels["minutes"], errors="coerce")
    if "starter_flag_label" in labels.columns:
        labels["starter_flag_label"] = pd.to_numeric(
            labels["starter_flag_label"], errors="coerce"
        )
    if not counts_labels.empty:
        counts_labels["player_id"] = pd.to_numeric(counts_labels["player_id"], errors="coerce").astype("Int64")
        counts_labels["game_date"] = pd.to_datetime(counts_labels["game_date"], errors="coerce")
        for col in ("fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "minutes"):
            counts_labels[col] = pd.to_numeric(counts_labels[col], errors="coerce").fillna(0.0)
        fga = counts_labels["fga2"] + counts_labels["fga3"]
        counts_labels["fg2_pct"] = (counts_labels["fg2m"] / counts_labels["fga2"].replace(0.0, pd.NA)).fillna(0.0)
        counts_labels["fg3_pct"] = (counts_labels["fg3m"] / counts_labels["fga3"].replace(0.0, pd.NA)).fillna(0.0)
        counts_labels["ft_pct"] = (counts_labels["ftm"] / counts_labels["fta"].replace(0.0, pd.NA)).fillna(0.0)
        counts_labels["efg_pct"] = ((counts_labels["fg2m"] + 1.5 * counts_labels["fg3m"]) / fga.replace(0.0, pd.NA)).fillna(0.0)
        counts_labels["fg2a_per_min"] = (counts_labels["fga2"] / counts_labels["minutes"].replace(0.0, pd.NA)).fillna(0.0)
        counts_labels["fg3a_per_min"] = (counts_labels["fga3"] / counts_labels["minutes"].replace(0.0, pd.NA)).fillna(0.0)
        counts_labels["fta_per_min"] = (counts_labels["fta"] / counts_labels["minutes"].replace(0.0, pd.NA)).fillna(0.0)
        counts_labels["three_pa_share"] = (counts_labels["fga3"] / fga.replace(0.0, pd.NA)).fillna(0.0)

    # For each player, get their latest game from priors and check if labels
    # have that game's results
    updated = player_priors.copy()
    updated["person_id"] = pd.to_numeric(updated["person_id"], errors="coerce").astype("Int64")

    # Get the game_id that each prior was computed for
    if "game_id" not in updated.columns:
        return player_priors

    windows: tuple[int, ...] = (5, 10, 20)
    for idx, row in updated.iterrows():
        person_id = row["person_id"]
        prior_game_id = row.get("game_id") or row.get("game_id_norm")
        if pd.isna(person_id) or pd.isna(prior_game_id):
            continue

        # Get this player's labels, sorted by date descending
        player_labels = labels[labels["player_id"] == int(person_id)].copy()
        if player_labels.empty:
            continue
        player_counts = counts_labels[counts_labels["player_id"] == int(person_id)].copy() if not counts_labels.empty else pd.DataFrame()

        player_labels = player_labels.sort_values("game_date", ascending=False)
        if not player_counts.empty:
            player_counts = player_counts.sort_values("game_date", ascending=False)

        for window in windows:
            last_n = player_labels.head(window)
            if last_n.empty:
                continue

            n_games = int(len(last_n))
            start_col = f"started_proxy_rate_prior_{window}"
            start_missing_col = f"{start_col}_missing"
            minutes_col = f"minutes_from_stints_prior_{window}"
            minutes_std_col = f"minutes_from_stints_std_prior_{window}"
            minutes_missing_col = f"{minutes_col}_missing"
            n_games_col = f"player_prior_n_games_{window}"
            source_date_col = f"player_prior_source_max_game_date_{window}"

            if "starter_flag_label" in last_n.columns and start_col in updated.columns:
                new_start_rate = pd.to_numeric(
                    last_n["starter_flag_label"], errors="coerce"
                ).mean()
                if not pd.isna(new_start_rate):
                    updated.at[idx, start_col] = float(new_start_rate)
                    if start_missing_col in updated.columns:
                        updated.at[idx, start_missing_col] = 0

            if "minutes" in last_n.columns and minutes_col in updated.columns:
                minutes_series = pd.to_numeric(last_n["minutes"], errors="coerce").dropna()
                if not minutes_series.empty:
                    updated.at[idx, minutes_col] = float(minutes_series.mean())
                    if minutes_std_col in updated.columns:
                        # Priors std columns are population-style rolling std.
                        updated.at[idx, minutes_std_col] = float(
                            minutes_series.std(ddof=0)
                        )
                    if minutes_missing_col in updated.columns:
                        updated.at[idx, minutes_missing_col] = 0

            if not player_counts.empty:
                last_n_counts = player_counts.head(window)
                for col in _PLAYER_PRIOR_REFRESH_BASE_COLS:
                    if col in {"minutes_from_stints", "started_proxy"}:
                        continue
                    prior_col = f"{col}_prior_{window}"
                    if prior_col not in updated.columns or col not in last_n_counts.columns:
                        continue
                    series = pd.to_numeric(last_n_counts[col], errors="coerce").fillna(0.0)
                    updated.at[idx, prior_col] = float(series.mean())
                    miss_col = f"{prior_col}_missing"
                    if miss_col in updated.columns:
                        updated.at[idx, miss_col] = 0

            if n_games_col in updated.columns:
                updated.at[idx, n_games_col] = n_games

            if source_date_col in updated.columns:
                latest_date = pd.to_datetime(last_n["game_date"], errors="coerce").max()
                if pd.notna(latest_date):
                    updated.at[idx, source_date_col] = latest_date

    return updated


def _refresh_latest_player_priors_from_history(player_history: pd.DataFrame) -> pd.DataFrame:
    """Refresh latest player priors from stored observed history rows."""

    if player_history.empty:
        return player_history

    required = {"person_id", "game_date", "minutes_from_stints", "started_proxy"}
    if not required.issubset(player_history.columns):
        return pd.DataFrame()

    history = player_history.copy()
    history["person_id"] = pd.to_numeric(history["person_id"], errors="coerce").astype("Int64")
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    history["minutes_from_stints"] = pd.to_numeric(
        history["minutes_from_stints"], errors="coerce"
    ).fillna(0.0)
    history["started_proxy"] = pd.to_numeric(
        history["started_proxy"], errors="coerce"
    ).fillna(0.0)
    for col in _PLAYER_PRIOR_REFRESH_BASE_COLS:
        if col in {"minutes_from_stints", "started_proxy"}:
            continue
        if col not in history.columns:
            history[col] = 0.0
        history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0.0)
    history["ctx_same_pos_bucket"] = (
        history.get("ctx_same_pos_bucket", pd.Series(pd.NA, index=history.index))
        .astype("string")
        .fillna("unknown")
        .str.lower()
        .str.strip()
    )
    history = history.dropna(subset=["person_id", "game_date"]).copy()
    if history.empty:
        return pd.DataFrame()

    refreshed_rows: list[pd.Series] = []
    windows: tuple[int, ...] = (5, 10, 20)
    player_sort_cols = [c for c in ("person_id", "game_date", "game_id") if c in history.columns]
    for _, player_hist in history.sort_values(
        player_sort_cols,
        kind="mergesort",
    ).groupby("person_id", sort=False):
        latest = player_hist.iloc[-1].copy()
        for window in windows:
            last_n = player_hist.tail(window).copy()
            n_games = int(len(last_n))
            latest[f"minutes_from_stints_prior_{window}"] = float(last_n["minutes_from_stints"].mean())
            latest[f"minutes_from_stints_std_prior_{window}"] = float(last_n["minutes_from_stints"].std(ddof=0))
            latest[f"started_proxy_rate_prior_{window}"] = float(last_n["started_proxy"].mean())
            for col in _PLAYER_PRIOR_REFRESH_BASE_COLS:
                if col in {"minutes_from_stints", "started_proxy"} or col not in last_n.columns:
                    continue
                latest[f"{col}_prior_{window}"] = float(pd.to_numeric(last_n[col], errors="coerce").fillna(0.0).mean())
                latest[f"{col}_prior_{window}_missing"] = 0
            for col in _PLAYER_PRIOR_REFRESH_STD_COLS:
                if col == "minutes_from_stints" or col not in last_n.columns:
                    continue
                latest[f"{col}_std_prior_{window}"] = float(pd.to_numeric(last_n[col], errors="coerce").fillna(0.0).std(ddof=0))
                latest[f"{col}_std_prior_{window}_missing"] = 0
            latest[f"player_prior_n_games_{window}"] = n_games
            latest[f"player_prior_source_max_game_date_{window}"] = pd.to_datetime(
                last_n["game_date"], errors="coerce"
            ).max()
            for bucket in SAME_POS_CONTEXT_BUCKETS:
                matched = last_n.loc[last_n["ctx_same_pos_bucket"] == bucket].copy()
                latest[f"ctx_same_pos_{bucket}_prior_n_games_{window}"] = int(len(matched))
                latest[f"ctx_same_pos_{bucket}_prior_source_max_game_date_{window}"] = (
                    pd.to_datetime(matched["game_date"], errors="coerce").max()
                    if not matched.empty
                    else pd.NaT
                )
                latest[f"minutes_from_stints_ctx_same_pos_{bucket}_prior_{window}"] = float(
                    pd.to_numeric(matched["minutes_from_stints"], errors="coerce")
                    .fillna(0.0)
                    .mean()
                ) if not matched.empty else 0.0
                latest[f"started_proxy_rate_ctx_same_pos_{bucket}_prior_{window}"] = float(
                    pd.to_numeric(matched["started_proxy"], errors="coerce")
                    .fillna(0.0)
                    .mean()
                ) if not matched.empty else 0.0
        refreshed_rows.append(latest)

    if not refreshed_rows:
        return pd.DataFrame()

    return pd.DataFrame(refreshed_rows).reset_index(drop=True)


def _refresh_latest_team_priors_from_history(team_history: pd.DataFrame) -> pd.DataFrame:
    """Refresh latest team priors from stored observed team history rows."""

    if team_history.empty:
        return team_history

    required = {"team_id", "game_date"}
    if not required.issubset(team_history.columns):
        return pd.DataFrame()

    history = team_history.copy()
    history["team_id"] = pd.to_numeric(history["team_id"], errors="coerce").astype("Int64")
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    available_cols = [c for c in _TEAM_PRIOR_REFRESH_BASE_COLS if c in history.columns]
    if not available_cols:
        return pd.DataFrame()
    for col in available_cols:
        history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0.0)
    if "team_ot_flag" in history.columns:
        history["team_ot_flag"] = pd.to_numeric(history["team_ot_flag"], errors="coerce").fillna(0.0)
    history = history.dropna(subset=["team_id", "game_date"]).copy()
    if history.empty:
        return pd.DataFrame()

    refreshed_rows: list[pd.Series] = []
    windows: tuple[int, ...] = (5, 10, 20)
    team_sort_cols = [c for c in ("team_id", "game_date", "game_id") if c in history.columns]
    for _, team_hist in history.sort_values(
        team_sort_cols,
        kind="mergesort",
    ).groupby("team_id", sort=False):
        latest = team_hist.iloc[-1].copy()
        for window in windows:
            last_n = team_hist.tail(window).copy()
            n_games = int(len(last_n))
            latest[f"team_prior_n_games_{window}"] = n_games
            latest[f"team_prior_source_max_game_date_{window}"] = pd.to_datetime(
                last_n["game_date"], errors="coerce"
            ).max()
            for col in available_cols:
                latest[f"{col}_prior_{window}"] = float(pd.to_numeric(last_n[col], errors="coerce").fillna(0.0).mean())
                latest[f"{col}_prior_{window}_missing"] = 0
            for col in _TEAM_PRIOR_REFRESH_STD_COLS:
                if col not in last_n.columns:
                    continue
                latest[f"{col}_std_prior_{window}"] = float(pd.to_numeric(last_n[col], errors="coerce").fillna(0.0).std(ddof=0))
                latest[f"{col}_std_prior_{window}_missing"] = 0
            if "team_ot_flag" in last_n.columns:
                latest[f"team_ot_rate_prior_{window}"] = float(
                    pd.to_numeric(last_n["team_ot_flag"], errors="coerce").fillna(0.0).mean()
                )
                latest[f"team_ot_rate_prior_{window}_missing"] = 0
        refreshed_rows.append(latest)

    if not refreshed_rows:
        return pd.DataFrame()

    return pd.DataFrame(refreshed_rows).reset_index(drop=True)


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

    IMPORTANT: Excludes All-Star and special games (non-002 prefix) because they
    use special team IDs that don't match regular season teams. If the latest
    game for a player is an All-Star game, the priors would have the wrong team_id
    and fail to join with the minutes features.

    After loading, updates player priors with box score labels to include the
    results from the latest game (since priors are "pre-game" snapshots).

    Returns:
        (team_priors, player_priors) DataFrames with one row per team/player
        containing their most recent prior values.
    """
    data_root = Path(data_root).expanduser().resolve()
    team_root = data_root / "silver" / "rotation_priors_v1" / "team_game_priors" / f"season={int(season)}"
    player_root = data_root / "silver" / "rotation_priors_v1" / "player_game_priors" / f"season={int(season)}"

    def _load_all_history(root: Path, entity_col: str, filter_ids: set[int] | None) -> pd.DataFrame:
        if not root.exists():
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        skipped_paths: list[str] = []
        for path in sorted(root.glob("game_id=*.parquet")):
            # Skip non-regular-season games (All-Star, playoffs, etc.)
            # These use special team IDs that break the team_id-based join
            game_id_str = path.name.replace("game_id=", "").replace(".parquet", "")
            if not _is_regular_season_game_id(game_id_str):
                continue
            try:
                df = pd.read_parquet(path)
            except Exception:
                skipped_paths.append(str(path))
                continue

            # Defensive guard for transient/corrupt partitions while writer jobs run.
            required_cols = {entity_col, "game_date"}
            if not required_cols.issubset(set(df.columns)):
                skipped_paths.append(str(path))
                continue

            try:
                nrows = int(len(df))
            except Exception:
                skipped_paths.append(str(path))
                continue

            if nrows < 0 or nrows > _MAX_REASONABLE_PRIORS_PARTITION_ROWS:
                skipped_paths.append(str(path))
                continue

            try:
                if any(len(df[col]) != nrows for col in df.columns):
                    skipped_paths.append(str(path))
                    continue
            except Exception:
                skipped_paths.append(str(path))
                continue

            if filter_ids:
                try:
                    entity_vals = pd.to_numeric(df[entity_col], errors="coerce")
                    df = df.loc[entity_vals.isin(filter_ids)].copy()
                except Exception:
                    skipped_paths.append(str(path))
                    continue
                if df.empty:
                    continue

            frames.append(df)

        if skipped_paths:
            _logger.warning(
                "Skipped %d priors partitions under %s due to read/schema/shape validation failures.",
                len(skipped_paths),
                root,
            )

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True, copy=False)
        combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
        combined[entity_col] = pd.to_numeric(combined[entity_col], errors="coerce").astype("Int64")

        if filter_ids:
            combined = combined[combined[entity_col].isin(filter_ids)].copy()

        if combined.empty:
            return pd.DataFrame()

        return combined

    team_filter = set(int(t) for t in team_ids) if team_ids else None
    player_filter = set(int(p) for p in player_ids) if player_ids else None

    team_history = _load_all_history(team_root, "team_id", team_filter)
    if not team_history.empty:
        team_sort_cols = [c for c in ("team_id", "game_date", "game_id") if c in team_history.columns]
        team_history = team_history.sort_values(team_sort_cols, kind="mergesort", na_position="last")
        refreshed_team = _refresh_latest_team_priors_from_history(team_history)
        if refreshed_team.empty:
            team_priors = team_history.sort_values(
                "game_date", ascending=False, kind="stable", na_position="last"
            ).drop_duplicates(subset=["team_id"], keep="first").reset_index(drop=True)
        else:
            team_priors = refreshed_team
    else:
        team_priors = pd.DataFrame()

    player_history = _load_all_history(player_root, "person_id", player_filter)
    if not player_history.empty:
        player_sort_cols = [c for c in ("person_id", "game_date", "game_id") if c in player_history.columns]
        player_history = player_history.sort_values(
            player_sort_cols,
            kind="mergesort",
            na_position="last",
        )
        refreshed = _refresh_latest_player_priors_from_history(player_history)
        if refreshed.empty:
            latest_player = player_history.sort_values(
                "game_date", ascending=False, kind="stable", na_position="last"
            ).drop_duplicates(subset=["person_id"], keep="first").reset_index(drop=True)
            player_priors = _update_player_priors_with_latest_labels(latest_player, data_root, season)
        else:
            player_priors = refreshed
    else:
        player_priors = pd.DataFrame()

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

    all_team_partitions_missing = bool(game_ids_norm) and len(missing_team_game_ids) == len(game_ids_norm)
    all_player_partitions_missing = bool(game_ids_norm) and len(missing_player_game_ids) == len(game_ids_norm)

    def _coverage_counts(tp: pd.DataFrame, pp: pd.DataFrame) -> tuple[int, int, int, int]:
        if not tp.empty:
            tp = tp.copy()
            tp["team_id"] = pd.to_numeric(tp["team_id"], errors="coerce").astype("Int64")
            teams_in_priors = set(tp["team_id"].dropna().astype(int).tolist())
        else:
            teams_in_priors = set()

        if not pp.empty:
            pp = pp.copy()
            pp["person_id"] = pd.to_numeric(pp["person_id"], errors="coerce").astype("Int64")
            players_in_priors = set(pp["person_id"].dropna().astype(int).tolist())
        else:
            players_in_priors = set()

        teams_found_local = len(teams_in_priors & team_ids_set)
        teams_missing_local = len(team_ids_set - teams_in_priors)
        players_found_local = len(players_in_priors & player_ids_set)
        players_missing_local = len(player_ids_set - players_in_priors)
        return teams_found_local, teams_missing_local, players_found_local, players_missing_local

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

    # Live slates frequently do not have same-day game_id priors partitions yet.
    # In fallback mode, short-circuit directly to latest-by-entity priors.
    if allow_priors_fallback and all_team_partitions_missing and all_player_partitions_missing:
        latest_team, latest_player = load_latest_rotation_priors_by_entity(
            data_root,
            season=season,
            team_ids=sorted(team_ids_set) if team_ids_set else None,
            player_ids=sorted(player_ids_set) if player_ids_set else None,
        )
        stamped_team = _stamp(latest_team, game_ids_to_stamp=game_ids_norm)
        stamped_player = _stamp(latest_player, game_ids_to_stamp=game_ids_norm)
        teams_found, teams_missing, players_found, players_missing = _coverage_counts(
            stamped_team, stamped_player
        )
        date_clause = f" date={game_date}" if game_date else ""
        warning_msg = (
            "No game-id priors partitions found for requested slate game_ids "
            f"(n={len(game_ids_norm)}).{date_clause} "
            f"Using latest available priors by entity (teams_missing={teams_missing}, "
            f"players_missing={players_missing})."
        )
        return RotationPriorsLoadResult(
            team_priors=stamped_team,
            player_priors=stamped_player,
            teams_found=teams_found,
            teams_missing=teams_missing,
            players_found=players_found,
            players_missing=players_missing,
            used_latest_fallback=True,
            warning_message=warning_msg,
        )

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
    teams_found, teams_missing, players_found, players_missing = _coverage_counts(
        team_priors, player_priors
    )

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
    final_teams_found, final_teams_missing, final_players_found, final_players_missing = _coverage_counts(
        team_priors, player_priors
    )

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

    # Backward-compat: some historical minutes snapshots may not carry
    # context-missing indicators that newer rotation-set models expect.
    # Derive them from raw context columns before global numeric fill.
    if "team_ctx_missing" in feature_columns and "team_ctx_missing" not in work.columns:
        team_ctx_cols = [c for c in ("team_pace_szn", "team_off_rtg_szn", "team_def_rtg_szn") if c in work.columns]
        if team_ctx_cols:
            work["team_ctx_missing"] = work[team_ctx_cols].isna().all(axis=1).astype("int8")
        else:
            work["team_ctx_missing"] = 1

    if "opp_ctx_missing" in feature_columns and "opp_ctx_missing" not in work.columns:
        opp_ctx_cols = [c for c in ("opp_pace_szn", "opp_def_rtg_szn") if c in work.columns]
        if opp_ctx_cols:
            work["opp_ctx_missing"] = work[opp_ctx_cols].isna().all(axis=1).astype("int8")
        else:
            work["opp_ctx_missing"] = 1

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

    # Compute derived features (team counts, vacancy features, etc.) if needed by the model.
    derived_needed = [c for c in ROTATION_SET_DERIVED_FEATURES if c in feature_columns and c not in work.columns]
    if derived_needed:
        work = add_rotation_set_derived_features(work, feature_columns=feature_columns)

    # ---------------------------------------------------------------------------
    # Action Network props features (an_*)
    # ---------------------------------------------------------------------------
    # Some historical `features_minutes_v1` snapshots predate the Action props
    # attachment step and will be missing these columns entirely. Newer
    # rotation-set models expect them as numeric features; when absent we fill
    # defaults matching `projections.features.action_props`.
    if any(str(c).startswith("an_") for c in feature_columns):
        try:
            from projections.features.action_props import ACTION_MARKET_FEATURE_COLUMNS, _market_feature_defaults

            defaults = _market_feature_defaults()
            for col in ACTION_MARKET_FEATURE_COLUMNS:
                if col not in feature_columns:
                    continue
                default = defaults.get(col, 0.0)
                if col not in work.columns:
                    work[col] = default
                else:
                    work[col] = work[col].fillna(default)
        except Exception:  # noqa: BLE001
            # Best-effort only: if import fails for any reason, leave columns as-is
            # and let the missing-features gate below raise a loud error.
            pass

    # ---------------------------------------------------------------------------
    # Prop-implied minutes features (an_implied_minutes, etc.)
    # ---------------------------------------------------------------------------
    # Newer rotation-set models may include prop-derived implied minutes features.
    # These are derived from Action props + historical boxscores, so older snapshots
    # (or best-effort runs where the attachment fails) may not carry them. Fill
    # conservative defaults that indicate "missing" rather than "0 implied minutes".
    implied_cols = {
        "an_pra_per_min_prior": 1.0,
        "an_pra_prior_minutes_sum": 0.0,
        "an_pra_prior_games": 0,
        "an_implied_minutes": 0.0,
        "an_has_implied_minutes": 0,
        "an_implied_minutes_missing": 1,
    }
    if any(col in feature_columns for col in implied_cols):
        for col, default in implied_cols.items():
            if col not in feature_columns:
                continue
            if col not in work.columns:
                work[col] = default
            else:
                work[col] = work[col].fillna(default)

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

    # ---------------------------------------------------------------------------
    # Sanity check: warn if starter features are degenerate (would cause flattening)
    # This is a soft check - log warnings but do not fail production.
    # ---------------------------------------------------------------------------
    import logging
    logger = logging.getLogger(__name__)
    
    for team_id, grp in work.groupby("team_id", sort=False):
        n_players = len(grp)
        if n_players == 0:
            continue
        
        # Check recent_start_pct_10: should NOT be all zeros for a team-game
        if "recent_start_pct_10" in grp.columns:
            rsp = pd.to_numeric(grp["recent_start_pct_10"], errors="coerce").fillna(0)
            if n_players > 5 and (rsp == 0).sum() >= n_players * 0.9:
                logger.warning(
                    f"[rotation_features] team_id={team_id}: recent_start_pct_10 is ~100%% zeros "
                    f"({int((rsp==0).sum())}/{n_players}). Model may flatten minutes distribution."
                )
        
        # Check is_projected_starter: sum should be ~5 (one lineup)
        if "is_projected_starter" in grp.columns:
            proj_sum = grp["is_projected_starter"].sum()
            if proj_sum == 0:
                logger.warning(
                    f"[rotation_features] team_id={team_id}: is_projected_starter sum=0. "
                    "Model has no starter signal."
                )
    
    ordered = [*KEY_COLS, *keep_cols, *feature_columns]
    return RotationSetMinutesV1LiveFeaturesResult(
        features=work.loc[:, ordered].copy(),
        dropped_extra_columns=extra,
    )
