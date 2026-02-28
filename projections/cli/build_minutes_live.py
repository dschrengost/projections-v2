"""Build same-day Minutes V1 feature slices for live inference.

Required feature provenance (9 share-model features):
- team_pace_szn, team_off_rtg_szn, team_def_rtg_szn:
  Source: gold/rates_training_base/season={season}/game_date={prior_date}/
  Join key: team_id
  
- opp_pace_szn, opp_def_rtg_szn:
  Source: Same as team context, renamed columns
  Join key: opponent_team_id
  
- vac_min_szn, vac_min_{guard,wing,big}_szn:
  Computed from injuries_snapshot + roster_nightly + historical labels
  Join key: (game_id, team_id)
  Semantic: Season-to-date minutes played by players currently OUT
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import typer
from unidecode import unidecode

from projections import paths

from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest, write_ids_csv
from projections.minutes_v1.features import MinutesFeatureBuilder
from projections.minutes_v1.pos import canonical_pos_bucket_series
from projections.minutes_v1.schemas import (
    BOX_SCORE_LABELS_SCHEMA,
    FEATURES_MINUTES_V1_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)
from projections.minutes_v1.snapshots import select_injury_snapshot
from projections.minutes_v1.starter_flags import (
    StarterFlagResult,
    derive_starter_flag_label,
    normalize_starter_signals,
)
from projections.labels import derive_starter_flag_labels
from projections.etl import storage as bronze_storage
from projections.pipeline.status import JobStatus, write_status
from projections.features.action_props import (
    ACTION_MARKET_FEATURE_COLUMNS,
    attach_action_props_features,
    load_action_props_feature_snapshots_for_date_live,
)
from scrapers.nba_players import NbaPlayersScraper, PlayerProfile

UTC = timezone.utc
DEFAULT_DATA_ROOT = paths.get_data_root()
DEFAULT_OUTPUT_ROOT = paths.data_path("live", "features_minutes_v1")
LIVE_SOURCE_NAME = "live_inference_roster"
FEATURE_FILENAME = "features.parquet"
SUMMARY_FILENAME = "summary.json"
IDS_FILENAME = "ids.csv"
LATEST_POINTER = "latest_run.json"
ACTIVE_ROSTER_FILENAME = "active_roster.parquet"
INACTIVE_PLAYERS_FILENAME = "inactive_players.csv"
RATES_TRAINING_BASE_FILENAME = "rates_training_base.parquet"


def _normalize_name_for_matching(name: str) -> str:
    return unidecode(str(name)).strip().lower()

VACANCY_FEATURE_COLUMNS: tuple[str, ...] = (
    "vac_min_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
)
TEAM_CONTEXT_COLUMNS: tuple[str, ...] = (
    "team_pace_szn",
    "team_off_rtg_szn",
    "team_def_rtg_szn",
)
OPP_CONTEXT_COLUMNS: tuple[str, ...] = (
    "opp_pace_szn",
    "opp_def_rtg_szn",
)

# Required features for share model - builder must produce these
REQUIRED_MINUTES_FEATURES: frozenset[str] = frozenset({
    # Team/opponent context (from rates_training_base)
    "team_pace_szn", "team_off_rtg_szn", "team_def_rtg_szn",
    "opp_pace_szn", "opp_def_rtg_szn",
    # Vacancy features (computed from injuries + historical labels)
    "vac_min_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn",
    # Trend features (from historical labels)
    "roll_mean_5", "roll_mean_10", "min_last3", "min_last5",
})

_OUT_LIKE_STATUS_VALUES: set[str] = {"OUT", "O", "Q", "QUESTIONABLE", "DOUBTFUL", "D", "INACTIVE"}

app = typer.Typer(help=__doc__)


def _verify_required_features(
    df: pd.DataFrame,
    run_id: str,
    warnings: list[str],
) -> None:
    """Verify that all required features are present in the output DataFrame.
    
    Args:
        df: The features DataFrame to verify
        run_id: Run identifier for logging
        warnings: List to append warning messages to
        
    Raises:
        RuntimeError: If PROJECTIONS_REQUIRE_ALL_FEATURES=1 and features are missing
    """
    missing = REQUIRED_MINUTES_FEATURES - set(df.columns)
    if missing:
        sorted_missing = sorted(missing)
        msg = f"[{run_id}] Missing {len(missing)} required features: {sorted_missing}"
        warnings.append(msg)
        typer.echo(f"[build-minutes-live] WARNING: {msg}", err=True)
        
        if os.environ.get("PROJECTIONS_REQUIRE_ALL_FEATURES") == "1":
            raise RuntimeError(msg)
    else:
        typer.echo(f"[build-minutes-live] Required features verified ({len(REQUIRED_MINUTES_FEATURES)}/{len(REQUIRED_MINUTES_FEATURES)})", err=True)


def _nan_rate(df: pd.DataFrame, cols: list[str]) -> float | None:
    present = [col for col in cols if col in df.columns]
    if not present or df.empty:
        return 0.0
    return float(df[present].isna().mean().mean())


def _normalize_day(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _season_start_from_day(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _season_label(season_start: int) -> str:
    return f"{season_start}-{(season_start + 1) % 100:02d}"


def _normalize_run_timestamp(value: datetime | None) -> pd.Timestamp:
    if value is None:
        base = datetime.now(tz=UTC)
    else:
        base = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        base = base.astimezone(UTC)
    return pd.Timestamp(base)


def _format_run_id(run_ts: pd.Timestamp) -> str:
    return run_ts.strftime("%Y%m%dT%H%M%SZ")


def _ensure_run_output_dir(root: Path, day: pd.Timestamp, run_id: str) -> tuple[Path, Path]:
    day_dir = root / day.strftime("%Y-%m-%d")
    run_dir = day_dir / f"run={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return day_dir, run_dir


def _write_latest_pointer(day_dir: Path, *, run_id: str, run_as_of_ts: pd.Timestamp) -> None:
    import os

    if os.environ.get("PROJECTIONS_SKIP_POINTER_WRITES", "").strip().lower() in {"1", "true", "yes"}:
        return

    from projections.pipeline import writer_guard

    writer_guard.assert_can_write_pointers(purpose=f"build_minutes_live promote {day_dir}")
    pointer = day_dir / LATEST_POINTER
    payload = {"run_id": run_id, "run_as_of_ts": run_as_of_ts.isoformat()}
    pointer.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet input at {path}")
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files discovered under {path}")
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _read_parquet_if_exists(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_parquet(path)


def _load_table(default_dir: Path, override: Path | None) -> pd.DataFrame:
    target = override or default_dir
    return _read_parquet_tree(target)


def _load_injuries_bronze_window(
    *,
    data_root: Path,
    season_value: int,
    target_day: pd.Timestamp,
    days_before: int = 1,
    days_after: int = 1,
) -> pd.DataFrame:
    """Load bronze injuries_raw across a small day window around the target slate date."""
    frames: list[pd.DataFrame] = []
    for offset in range(-days_before, days_after + 1):
        day = (target_day + pd.Timedelta(days=offset)).date()
        day_frame = bronze_storage.read_bronze_day(
            "injuries_raw",
            data_root,
            season_value,
            day,
            include_runs=False,
            prefer_history=True,
        )
        if day_frame.empty:
            continue
        frames.append(day_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _filter_by_game_ids(df: pd.DataFrame, game_ids: Iterable[int]) -> pd.DataFrame:
    if df.empty or "game_id" not in df.columns:
        return df.copy()
    normalized = pd.Series(game_ids, dtype="Int64").dropna().astype(int).tolist()
    if not normalized:
        return df.iloc[0:0].copy()
    return df[pd.to_numeric(df["game_id"], errors="coerce").astype("Int64").isin(normalized)].copy()


def _player_hist_minutes_szn(
    labels: pd.DataFrame,
    *,
    target_day: pd.Timestamp,
    player_ids: set[int],
) -> pd.DataFrame:
    if labels.empty or not player_ids:
        return pd.DataFrame(columns=["player_id", "hist_minutes_szn"])
    required = {"player_id", "game_date", "minutes"}
    if not required.issubset(labels.columns):
        return pd.DataFrame(columns=["player_id", "hist_minutes_szn"])

    working = labels.loc[:, ["player_id", "game_date", "minutes"]].copy()
    working["game_date"] = pd.to_datetime(working["game_date"]).dt.normalize()
    working = working.loc[working["game_date"] < target_day].copy()
    if working.empty:
        return pd.DataFrame(columns=["player_id", "hist_minutes_szn"])

    working["player_id"] = pd.to_numeric(working["player_id"], errors="coerce").astype("Int64")
    working = working.dropna(subset=["player_id"])
    if working.empty:
        return pd.DataFrame(columns=["player_id", "hist_minutes_szn"])

    working = working.loc[working["player_id"].astype(int).isin(player_ids)].copy()
    if working.empty:
        return pd.DataFrame(columns=["player_id", "hist_minutes_szn"])

    working["minutes"] = pd.to_numeric(working["minutes"], errors="coerce").fillna(0.0).astype(float)
    grouped = (
        working.groupby("player_id", as_index=False)["minutes"]
        .sum()
        .rename(columns={"minutes": "hist_minutes_szn"})
    )
    grouped["hist_minutes_szn"] = grouped["hist_minutes_szn"].astype(float)
    return grouped


def _compute_vacancy_features(
    *,
    injuries_snapshot: pd.DataFrame,
    roster_nightly: pd.DataFrame,
    labels_source: pd.DataFrame,
    target_day: pd.Timestamp,
    warnings: list[str],
) -> pd.DataFrame:
    """Compute vacancy features for the live slate (team-level, joined to each player row)."""

    if injuries_snapshot.empty:
        return pd.DataFrame(columns=["game_id", "team_id", *VACANCY_FEATURE_COLUMNS])

    required = {"game_id", "player_id", "status"}
    if not required.issubset(injuries_snapshot.columns):
        missing = sorted(required - set(injuries_snapshot.columns))
        warnings.append(
            f"vacancy: injuries_snapshot missing required columns {missing}; leaving vacancy features at defaults."
        )
        return pd.DataFrame(columns=["game_id", "team_id", *VACANCY_FEATURE_COLUMNS])

    injuries = injuries_snapshot.copy()
    injuries["game_id"] = pd.to_numeric(injuries["game_id"], errors="coerce").astype("Int64")
    injuries["player_id"] = pd.to_numeric(injuries["player_id"], errors="coerce").astype("Int64")
    injuries = injuries.dropna(subset=["game_id", "player_id"])
    if injuries.empty:
        return pd.DataFrame(columns=["game_id", "team_id", *VACANCY_FEATURE_COLUMNS])

    injuries["as_of_ts"] = pd.to_datetime(injuries.get("as_of_ts"), utc=True, errors="coerce")
    injuries = (
        injuries.sort_values(["game_id", "player_id", "as_of_ts"], kind="mergesort")
        .groupby(["game_id", "player_id"], as_index=False)
        .tail(1)
    )

    status = injuries["status"].astype(str).str.upper().str.strip()
    injuries = injuries.loc[status.isin(_OUT_LIKE_STATUS_VALUES)].copy()
    if injuries.empty:
        return pd.DataFrame(columns=["game_id", "team_id", *VACANCY_FEATURE_COLUMNS])

    roster_cols = [col for col in ("game_id", "player_id", "team_id", "listed_pos") if col in roster_nightly.columns]
    roster_map = roster_nightly.loc[:, roster_cols].dropna(subset=["game_id", "player_id"]).copy() if roster_cols else pd.DataFrame()
    if not roster_map.empty:
        roster_map["game_id"] = pd.to_numeric(roster_map["game_id"], errors="coerce").astype("Int64")
        roster_map["player_id"] = pd.to_numeric(roster_map["player_id"], errors="coerce").astype("Int64")
        if "team_id" in roster_map.columns:
            roster_map["team_id"] = pd.to_numeric(roster_map["team_id"], errors="coerce").astype("Int64")
        roster_map = roster_map.dropna(subset=["game_id", "player_id"]).drop_duplicates(
            subset=["game_id", "player_id"], keep="last"
        )
        injuries = injuries.merge(
            roster_map,
            on=["game_id", "player_id"],
            how="left",
            suffixes=("", "_roster"),
        )
        if "team_id" in injuries.columns and "team_id_roster" in injuries.columns:
            injuries["team_id"] = injuries["team_id"].fillna(injuries["team_id_roster"])
        elif "team_id_roster" in injuries.columns and "team_id" not in injuries.columns:
            injuries = injuries.rename(columns={"team_id_roster": "team_id"})

        if "listed_pos" in injuries.columns and "listed_pos_roster" in injuries.columns:
            injuries["listed_pos"] = injuries["listed_pos"].fillna(injuries["listed_pos_roster"])
        elif "listed_pos_roster" in injuries.columns and "listed_pos" not in injuries.columns:
            injuries = injuries.rename(columns={"listed_pos_roster": "listed_pos"})

        injuries.drop(columns=["team_id_roster", "listed_pos_roster"], inplace=True, errors="ignore")

    if "team_id" not in injuries.columns:
        warnings.append("vacancy: missing team_id mapping; leaving vacancy features at defaults.")
        return pd.DataFrame(columns=["game_id", "team_id", *VACANCY_FEATURE_COLUMNS])

    injuries["team_id"] = pd.to_numeric(injuries["team_id"], errors="coerce").astype("Int64")
    injuries = injuries.dropna(subset=["team_id"])
    if injuries.empty:
        return pd.DataFrame(columns=["game_id", "team_id", *VACANCY_FEATURE_COLUMNS])

    player_ids = set(injuries["player_id"].dropna().astype(int).tolist())
    player_hist = _player_hist_minutes_szn(labels_source, target_day=target_day, player_ids=player_ids)
    injuries = injuries.merge(player_hist, on="player_id", how="left")
    injuries["hist_minutes_szn"] = injuries["hist_minutes_szn"].fillna(0.0)

    pos_bucket = canonical_pos_bucket_series(injuries.get("listed_pos", pd.Series("UNK", index=injuries.index)))
    injuries["pos_bucket"] = pos_bucket
    injuries["hist_minutes_guard_szn"] = injuries["hist_minutes_szn"].where(injuries["pos_bucket"] == "G", 0.0)
    injuries["hist_minutes_wing_szn"] = injuries["hist_minutes_szn"].where(injuries["pos_bucket"] == "W", 0.0)
    injuries["hist_minutes_big_szn"] = injuries["hist_minutes_szn"].where(injuries["pos_bucket"] == "BIG", 0.0)

    grouped = injuries.groupby(["game_id", "team_id"], as_index=False).agg(
        vac_min_szn=("hist_minutes_szn", "sum"),
        vac_min_guard_szn=("hist_minutes_guard_szn", "sum"),
        vac_min_wing_szn=("hist_minutes_wing_szn", "sum"),
        vac_min_big_szn=("hist_minutes_big_szn", "sum"),
    )
    for col in VACANCY_FEATURE_COLUMNS:
        grouped[col] = pd.to_numeric(grouped.get(col), errors="coerce").fillna(0.0).astype(float)
    return grouped


def _load_team_context_from_rates_training_base(
    *,
    data_root: Path,
    season_value: int,
    target_day: pd.Timestamp,
    team_ids: set[int],
    warnings: list[str],
    max_days_back: int = 14,
) -> pd.DataFrame:
    if not team_ids:
        return pd.DataFrame(columns=["team_id", *TEAM_CONTEXT_COLUMNS])

    remaining = set(team_ids)
    frames: list[pd.DataFrame] = []
    any_candidates = False

    def _scan_season(season: int, horizon_days: int) -> None:
        nonlocal any_candidates, remaining, frames
        if not remaining:
            return
        root = data_root / "gold" / "rates_training_base" / f"season={season}"
        if not root.exists():
            return

        candidates: list[tuple[pd.Timestamp, Path]] = []
        for day_dir in root.glob("game_date=*"):
            try:
                day_value = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except Exception:  # noqa: BLE001
                continue
            if day_value >= target_day:
                continue
            pq_path = day_dir / RATES_TRAINING_BASE_FILENAME
            if pq_path.exists():
                candidates.append((day_value, pq_path))

        if not candidates:
            return
        any_candidates = True
        candidates.sort(key=lambda pair: pair[0], reverse=True)

        for idx, (_day, pq_path) in enumerate(candidates):
            if not remaining:
                break
            if horizon_days > 0 and idx >= int(horizon_days):
                break
            try:
                df = pd.read_parquet(pq_path, columns=["team_id", *TEAM_CONTEXT_COLUMNS])
            except Exception:  # noqa: BLE001
                continue

            if df.empty:
                continue
            df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["team_id"])
            if df.empty:
                continue
            df = df.loc[df["team_id"].astype(int).isin(remaining)].copy()
            if df.empty:
                continue

            grouped = df.groupby("team_id", as_index=False)[list(TEAM_CONTEXT_COLUMNS)].mean()
            frames.append(grouped)
            found = set(grouped["team_id"].dropna().astype(int).tolist())
            remaining -= found

    # Primary scan: current season, bounded by max_days_back.
    _scan_season(int(season_value), int(max_days_back))
    # Fallback scan: prior season for still-missing teams (use full history).
    if remaining:
        _scan_season(int(season_value) - 1, 0)

    if not any_candidates:
        warnings.append("team-context: no prior rates_training_base partitions available")
        return pd.DataFrame(columns=["team_id", *TEAM_CONTEXT_COLUMNS])

    if not frames:
        warnings.append("team-context: failed to load any usable rates_training_base partitions")
        return pd.DataFrame(columns=["team_id", *TEAM_CONTEXT_COLUMNS])

    if remaining:
        warnings.append(
            f"team-context: partial coverage; missing {len(remaining)} team(s) after lookback/fallback."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["team_id"]).drop_duplicates(subset=["team_id"], keep="first")
    for col in TEAM_CONTEXT_COLUMNS:
        combined[col] = pd.to_numeric(combined.get(col), errors="coerce")
    return combined.reset_index(drop=True)


def _player_profiles_to_frame(players: List[PlayerProfile]) -> pd.DataFrame:
    if not players:
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_slug",
                "first_name",
                "last_name",
                "team_id",
                "team_slug",
                "team_abbreviation",
                "team_name",
                "jersey_number",
                "position",
                "height",
                "weight",
                "country",
                "roster_status",
            ]
        )
    records = [
        {
            "player_id": profile.person_id,
            "player_slug": profile.player_slug,
            "first_name": profile.first_name,
            "last_name": profile.last_name,
            "team_id": profile.team_id,
            "team_slug": profile.team_slug,
            "team_abbreviation": profile.team_abbreviation,
            "team_name": profile.team_name,
            "jersey_number": profile.jersey_number,
            "position": profile.position,
            "height": profile.height,
            "weight": profile.weight,
            "country": profile.country,
            "roster_status": profile.roster_status,
        }
        for profile in players
    ]
    return pd.DataFrame.from_records(records)


def _active_roster_pairs(roster_df: pd.DataFrame) -> set[Tuple[int, int]]:
    if roster_df.empty:
        return set()
    working = roster_df.dropna(subset=["team_id", "player_id"]).copy()
    if working.empty:
        return set()
    working["team_id"] = pd.to_numeric(working["team_id"], errors="coerce")
    working["player_id"] = pd.to_numeric(working["player_id"], errors="coerce")
    working = working.dropna(subset=["team_id", "player_id"])
    if working.empty:
        return set()
    return {
        (int(row.team_id), int(row.player_id))
        for row in working.itertuples(index=False)
    }


def _minutes_between(later: pd.Timestamp, earlier: pd.Timestamp | None) -> float | None:
    if earlier is None or pd.isna(earlier):
        return None
    delta = later - earlier
    return round(delta.total_seconds() / 60.0, 2)


def _load_label_sources(
    *,
    data_root: Path,
    season_value: int,
    override_path: Path | None,
    warnings: list[str],
) -> tuple[pd.DataFrame, str]:
    """Load label sources preferring gold daily labels, falling back to legacy."""

    if override_path:
        labels = _read_parquet_tree(override_path)
        return labels, str(override_path)

    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    gold_dir = data_root / "gold" / "labels_minutes_v1" / f"season={season_value}"
    legacy_path = data_root / "labels" / f"season={season_value}" / "boxscore_labels.parquet"

    if gold_dir.exists():
        try:
            frames.append(_read_parquet_tree(gold_dir))
            sources.append(str(gold_dir))
        except FileNotFoundError:
            warnings.append(f"Gold label directory {gold_dir} is empty; falling back to legacy labels.")
    if legacy_path.exists():
        frames.append(pd.read_parquet(legacy_path))
        sources.append(str(legacy_path))
    if not frames:
        raise FileNotFoundError(
            f"No label sources found. Expected gold labels at {gold_dir} or legacy labels at {legacy_path}."
        )

    labels = pd.concat(frames, ignore_index=True, sort=False)
    if "label_frozen_ts" in labels.columns:
        labels["label_frozen_ts"] = pd.to_datetime(labels["label_frozen_ts"], utc=True, errors="coerce")
    else:
        labels["label_frozen_ts"] = pd.NaT
    labels.sort_values(
        ["game_id", "team_id", "player_id", "label_frozen_ts"],
        inplace=True,
        kind="mergesort",
    )
    labels = labels.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")
    return labels, " + ".join(sources)


def _load_label_history(
    labels: pd.DataFrame,
    *,
    target_day: pd.Timestamp,
    history_days: int | None,
    run_as_of_ts: pd.Timestamp,
    label_source: str,
) -> pd.DataFrame:
    labels = labels.copy()
    minutes_col = labels.get("minutes")
    if minutes_col is not None and minutes_col.dtype == object:
        parsed = pd.to_timedelta(minutes_col, errors="coerce")
        labels["minutes"] = (parsed.dt.total_seconds() / 60.0).astype("Float64")
    if "minutes" in labels.columns:
        labels["minutes"] = pd.to_numeric(labels["minutes"], errors="coerce")

    # Older label snapshots may be missing newer required fields; backfill sensible defaults
    # so schema enforcement passes for live builds.
    if "starter_flag_label" not in labels.columns:
        starter_series = labels.get("starter_flag")
        starter_bool = starter_series.astype("boolean", copy=False).fillna(False) if starter_series is not None else pd.Series(
            False, index=labels.index, dtype="boolean"
        )
        labels["starter_flag_label"] = starter_bool.astype("Int64")
    if "label_frozen_ts" not in labels.columns:
        labels["label_frozen_ts"] = pd.NaT

    labels = enforce_schema(labels, BOX_SCORE_LABELS_SCHEMA, allow_missing_optional=True)
    labels["game_date"] = pd.to_datetime(labels["game_date"]).dt.normalize()
    mask = labels["game_date"] < target_day
    if history_days is not None and history_days > 0:
        cutoff = target_day - pd.Timedelta(days=history_days)
        mask &= labels["game_date"] >= cutoff
    if "label_frozen_ts" in labels.columns:
        frozen = pd.to_datetime(labels["label_frozen_ts"], utc=True, errors="coerce")
        mask &= frozen.isna() | (frozen <= run_as_of_ts) | (labels["game_date"] < target_day)
        labels["label_frozen_ts"] = frozen
    history = labels.loc[mask].copy()
    # Drop rows with missing minutes to avoid NaNs in trend/roll features; warn if many get dropped.
    if "minutes" in history.columns:
        before = len(history)
        history = history.dropna(subset=["minutes"])
        dropped = before - len(history)
        if dropped > 0:
            typer.echo(
                f"[live] warning: dropped {dropped} label rows with NaN minutes from history ({before} -> {len(history)}).",
                err=True,
            )

    # Historical label sources have occasionally carried unreliable starter flags (e.g. all ones).
    # Always derive starters from *boxscore minutes* for history to ensure exactly 5 starters
    # per team-game and prevent starter history features from becoming constant.
    try:
        history = derive_starter_flag_labels(
            history,
            minutes_col="minutes",
            game_col="game_id",
            team_col="team_id",
            player_col="player_id",
            output_col="starter_flag_label",
        )
        history["starter_flag"] = history["starter_flag_label"]
    except Exception as exc:
        typer.echo(
            f"[live] warning: failed to derive starter_flag_label from minutes in history ({exc}); "
            "using starter flags from the label source as-is.",
            err=True,
        )
    # If history is empty, this can legitimately happen on the first slate date of a season.
    # In that case, continue with an empty history so live features can still be built.
    #
    # We still fail closed when labels exist prior to target_day but filtering removed them,
    # to avoid silently flattening history-driven features mid-season.
    if history.empty:
        min_day = labels["game_date"].min() if "game_date" in labels.columns else None
        if min_day is None or pd.isna(min_day) or pd.Timestamp(min_day) >= target_day:
            typer.echo(
                f"[minutes-live] warning: no historical label rows found before {target_day.date()} "
                f"(labels={label_source}); continuing with empty history.",
                err=True,
            )
            return history
        raise RuntimeError(
            f"No historical label rows found before {target_day.date()} (labels={label_source})."
        )
    return history


def _build_live_labels(
    roster_slice: pd.DataFrame,
    *,
    target_day: pd.Timestamp,
    season_label: str,
) -> pd.DataFrame:
    if roster_slice.empty:
        raise RuntimeError("Roster snapshot slice for target date is empty.")

    working = roster_slice.copy()
    working = working.dropna(subset=["game_id", "player_id", "team_id"])
    working["game_date"] = pd.to_datetime(working["game_date"]).dt.normalize()
    working = working[working["game_date"] == target_day]
    if working.empty:
        raise RuntimeError("Roster snapshot does not include any rows for the target date.")

    if "as_of_ts" in working.columns:
        working["as_of_ts"] = pd.to_datetime(working["as_of_ts"], utc=True, errors="coerce")
        working = working.sort_values(["game_id", "team_id", "player_id", "as_of_ts"])
    else:
        working = working.sort_values(["game_id", "team_id", "player_id"])
    working = working.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")
    working = normalize_starter_signals(working)
    starter_result: StarterFlagResult = derive_starter_flag_label(
        working,
        group_cols=("game_id", "team_id"),
    )
    if starter_result.overflow:
        sample = starter_result.overflow[:10]
        warnings_msg = (
            "Starter overflow detected while building live labels "
            f"(game/team requested starters > 5). sample={sample}. "
            "Applying capped top-5 selection."
        )
        typer.echo(f"[minutes-live] WARNING: {warnings_msg}", err=True)
    starter_series = starter_result.values.reindex(working.index).fillna(0).astype("Int64")
    timestamp = pd.Timestamp.now(tz=UTC)

    live_df = pd.DataFrame(
        {
            "game_id": working["game_id"].astype("Int64"),
            "player_id": working["player_id"].astype("Int64"),
            "team_id": working["team_id"].astype("Int64"),
            "player_name": working.get("player_name"),
            "season": season_label,
            "game_date": working["game_date"],
            "minutes": pd.Series(pd.NA, index=working.index, dtype="Float64"),
            "starter_flag": starter_series,
            "starter_flag_label": starter_series,
            "source": LIVE_SOURCE_NAME,
            "label_frozen_ts": timestamp,
        }
    )
    live_df = enforce_schema(live_df, BOX_SCORE_LABELS_SCHEMA, allow_missing_optional=True)
    return live_df


def _select_roster_slice(
    roster_df: pd.DataFrame,
    *,
    target_day: pd.Timestamp,
    run_as_of_ts: pd.Timestamp,
    fallback_days: int,
    max_age_hours: int,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    working = roster_df.copy()
    working["game_date"] = pd.to_datetime(working["game_date"]).dt.normalize()
    working["as_of_ts"] = pd.to_datetime(working.get("as_of_ts"), utc=True, errors="coerce")
    same_day = working[working["game_date"] == target_day].copy()
    source_day: pd.Timestamp | None = target_day if not same_day.empty else None
    if same_day.empty and fallback_days > 0:
        window_start = target_day - pd.Timedelta(days=fallback_days)
        window_mask = (working["game_date"] <= target_day) & (working["game_date"] >= window_start)
        window = working.loc[window_mask].copy()
        if not window.empty:
            source_day = window["game_date"].max()
            same_day = window[window["game_date"] == source_day].copy()
            same_day["game_date"] = target_day

    if same_day.empty:
        return pd.DataFrame(columns=working.columns), None, None

    snapshot_ts = pd.to_datetime(same_day["as_of_ts"], utc=True, errors="coerce").dropna()
    latest_snapshot = snapshot_ts.max() if not snapshot_ts.empty else None
    if latest_snapshot is None:
        raise RuntimeError("Roster snapshot rows are missing as_of_ts timestamps.")

    age_minutes = _minutes_between(run_as_of_ts, latest_snapshot)
    age_hours = None if age_minutes is None else age_minutes / 60.0
    if age_hours is not None and age_hours > max_age_hours:
        raise RuntimeError(
            f"Roster snapshot is {age_hours:.1f}h old relative to run_as_of_ts; exceeds {max_age_hours}h limit."
        )
    return same_day, source_day, latest_snapshot


def _per_game_tip_lookup(schedule_df: pd.DataFrame) -> dict[int, pd.Timestamp]:
    if schedule_df.empty or "tip_ts" not in schedule_df.columns:
        return {}
    tips = pd.to_datetime(schedule_df["tip_ts"], utc=True, errors="coerce")
    ids = pd.to_numeric(schedule_df["game_id"], errors="coerce").astype("Int64")
    return {
        int(game_id): tip_ts
        for game_id, tip_ts in zip(ids.tolist(), tips.tolist())
        if game_id is not None and tip_ts is not None
    }


def _filter_snapshot_by_asof(
    df: pd.DataFrame,
    *,
    time_col: str,
    run_as_of_ts: pd.Timestamp,
    tip_lookup: dict[int, pd.Timestamp],
    dataset_name: str,
    warnings: list[str],
    backfill_mode: bool = False,
) -> pd.DataFrame:
    if df.empty or time_col not in df.columns or "game_id" not in df.columns:
        return df

    working = df.copy()
    working["game_id"] = pd.to_numeric(working["game_id"], errors="coerce").astype("Int64")
    working[time_col] = pd.to_datetime(working[time_col], utc=True, errors="coerce")

    # Backfill mode: ignore run_as_of ceiling, but still enforce anti-leak via tip_ts cutoff.
    if backfill_mode:
        tip_ts = working["game_id"].map(tip_lookup)
        limit_ts = tip_ts.fillna(run_as_of_ts)
        allowed = working[time_col].isna() | (working[time_col] <= limit_ts)
        filtered = working.loc[allowed].copy()
        dropped = len(working) - len(filtered)
        if dropped > 0:
            warnings.append(
                f"[backfill-mode] {dataset_name}: dropped {dropped} rows with snapshot_ts after tip_ts."
            )
        group_cols = ["game_id", "player_id"] if "player_id" in working.columns else ["game_id"]
        latest = (
            filtered.sort_values(time_col)
            .groupby(group_cols, as_index=False)
            .tail(1)
        )
        warnings.append(f"[backfill-mode] {dataset_name}: using tip-relative latest snapshot per game.")
        return latest

    # For roster, keep the latest snapshot per player/game, but respect run/tip cutoffs.
    if dataset_name == "roster_nightly":
        tip_ts = working["game_id"].map(tip_lookup)
        limit_ts = tip_ts.fillna(run_as_of_ts)
        allowed = working[time_col].isna() | (working[time_col] <= run_as_of_ts)
        allowed &= working[time_col].isna() | (working[time_col] <= limit_ts)
        filtered = working.loc[allowed].copy()
        dropped = len(working) - len(filtered)
        if dropped > 0:
            warnings.append(
                f"{dataset_name}: dropped {dropped} rows with snapshot_ts beyond run/tip bounds."
            )
        latest = (
            filtered.sort_values(time_col)
            .groupby(["game_id", "player_id"], as_index=False)
            .tail(1)
        )
        return latest

    tip_ts = working["game_id"].map(tip_lookup)
    limit_ts = tip_ts.fillna(run_as_of_ts)
    allowed = working[time_col].isna() | (working[time_col] <= run_as_of_ts)
    allowed &= working[time_col].isna() | (working[time_col] <= limit_ts)
    filtered = working.loc[allowed].copy()
    dropped = len(working) - len(filtered)
    if dropped > 0:
        warnings.append(
            f"{dataset_name}: dropped {dropped} rows with snapshot_ts beyond run/tip bounds."
        )
    return filtered


def _snapshot_stats(df: pd.DataFrame, *, time_col: str, run_as_of_ts: pd.Timestamp) -> dict | None:
    if df.empty or time_col not in df.columns:
        return None
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    latest = ts.max()
    return {
        "latest_ts": latest.isoformat(),
        "age_minutes": _minutes_between(run_as_of_ts, latest),
    }


def _compute_injury_diagnostics(
    injuries_raw: pd.DataFrame,
    injuries_slice: pd.DataFrame,
    live_features: pd.DataFrame,
    *,
    tip_lookup: dict[int, pd.Timestamp],
    source: str,
) -> dict:
    """Compute per-game injury resolution diagnostics for summary.json.
    
    Returns:
        Dict with overall stats and per-game breakdown for debugging missing injury_as_of_ts.
    """
    diagnostics: dict = {
        "source": source,  # "bronze" or "silver"
        "raw_rows": len(injuries_raw),
        "filtered_rows": len(injuries_slice),
        "per_game": [],
    }
    
    # Get unique game_ids from live features
    if "game_id" not in live_features.columns:
        return diagnostics
        
    game_ids = pd.to_numeric(live_features["game_id"], errors="coerce").dropna().unique()
    
    # Compute overall missing rate
    if "injury_as_of_ts" in live_features.columns:
        missing_rate = float(live_features["injury_as_of_ts"].isna().mean())
        diagnostics["injury_as_of_ts_missing_rate"] = round(missing_rate, 3)
    
    # Per-game breakdown
    for game_id in sorted(game_ids):
        game_id_int = int(game_id)
        game_info: dict = {"game_id": game_id_int}
        
        # Tip time for this game
        tip_ts = tip_lookup.get(game_id_int)
        if tip_ts:
            game_info["tip_ts"] = tip_ts.isoformat()
        
        # Raw injuries for this game
        raw_game = injuries_raw[injuries_raw["game_id"] == game_id_int] if not injuries_raw.empty else pd.DataFrame()
        game_info["raw_rows"] = len(raw_game)
        
        if not raw_game.empty and "as_of_ts" in raw_game.columns:
            raw_ts = pd.to_datetime(raw_game["as_of_ts"], utc=True, errors="coerce").dropna()
            if not raw_ts.empty:
                game_info["raw_latest_ts"] = raw_ts.max().isoformat()
        
        # Filtered injuries for this game
        filtered_game = injuries_slice[injuries_slice["game_id"] == game_id_int] if not injuries_slice.empty else pd.DataFrame()
        game_info["filtered_rows"] = len(filtered_game)
        
        if not filtered_game.empty and "as_of_ts" in filtered_game.columns:
            filtered_ts = pd.to_datetime(filtered_game["as_of_ts"], utc=True, errors="coerce").dropna()
            if not filtered_ts.empty:
                game_info["selected_ts"] = filtered_ts.max().isoformat()
        
        # Features for this game
        features_game = live_features[live_features["game_id"] == game_id_int]
        game_info["feature_rows"] = len(features_game)
        
        if "injury_as_of_ts" in features_game.columns:
            missing = features_game["injury_as_of_ts"].isna().sum()
            game_info["injury_as_of_ts_missing"] = int(missing)
        
        diagnostics["per_game"].append(game_info)
    
    return diagnostics


def _write_summary(
    path: Path,
    *,
    date: pd.Timestamp,
    run_as_of_ts: pd.Timestamp,
    rows: int,
    games: Iterable[int],
    roster_meta: dict,
    snapshot_meta: dict,
    active_roster_meta: dict | None,
    active_validation: dict | None,
    warnings: list[str],
) -> None:
    summary = {
        "date": date.date().isoformat(),
        "run_as_of_ts": run_as_of_ts.isoformat(),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "rows": rows,
        "games": sorted({str(int(gid)) for gid in games}),
        "roster": roster_meta,
        "snapshots": snapshot_meta,
        "active_roster": active_roster_meta,
        "active_validation": active_validation,
        "warnings": warnings,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_minutes_live_logic(
    date: datetime = typer.Option(..., "--date", help="Target slate date (YYYY-MM-DD)."),
    run_as_of_ts: datetime | None = typer.Option(
        None,
        "--run-as-of-ts",
        help="Timestamp representing the information state for this run. Defaults to now (UTC).",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        help="Root containing data partitions (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    out_root: Path = typer.Option(
        DEFAULT_OUTPUT_ROOT,
        "--out-root",
        help="Directory where live features will be written (per-day subfolders).",
    ),
    labels_path: Path | None = typer.Option(
        None,
        "--labels-path",
        help=(
            "Optional explicit boxscore labels parquet. Defaults to "
            "<data_root>/labels/season=YYYY/boxscore_labels.parquet."
        ),
    ),
    schedule_path: Path | None = typer.Option(None, help="Optional override for schedule parquet directory."),
    injuries_path: Path | None = typer.Option(None, help="Optional override for injuries_snapshot parquet."),
    odds_path: Path | None = typer.Option(None, help="Optional override for odds_snapshot parquet."),
    roster_path: Path | None = typer.Option(None, help="Optional override for roster_nightly parquet."),
    roles_path: Path | None = typer.Option(
        None,
        "--roles-path",
        help="Optional override for minutes roles parquet (season partition).",
    ),
    archetype_path: Path | None = typer.Option(
        None,
        "--archetype-path",
        help="Optional override for archetype deltas parquet (season partition).",
    ),
    coach_path: Path | None = typer.Option(None, help="Optional CSV override for coach_tenure metadata."),
    history_days: int | None = typer.Option(
        None,
        "--history-days",
        min=1,
        help="Optional rolling history window (in days) for label context. Defaults to full season.",
    ),
    season_start: int | None = typer.Option(
        None,
        "--season-start",
        help="Season start year override (e.g., 2024 for 2024-25). Defaults based on --date.",
    ),
    roster_fallback_days: int = typer.Option(
        0,
        "--roster-fallback-days",
        min=0,
        help="Allow using the most recent roster snapshot within this many days before --date when same-day data is missing.",
    ),
    roster_max_age_hours: int = typer.Option(
        18,
        "--roster-max-age-hours",
        min=1,
        help="Maximum allowed age (in hours) of the roster snapshot relative to run_as_of_ts.",
    ),
    validate_active_roster: bool = typer.Option(
        True,
        "--validate-active-roster/--skip-active-roster",
        help="Fetch NBA.com active roster snapshot and compare against live players.",
    ),
    enforce_active_roster: bool = typer.Option(
        False,
        "--enforce-active-roster",
        help="Drop players that are not present on the NBA.com active roster snapshot.",
    ),
    lock_buffer_minutes: int = typer.Option(
        0,
        "--lock-buffer-minutes",
        min=0,
        help="Skip games whose tip_ts is more than this many minutes before run_as_of_ts (avoid re-scoring locked games).",
    ),
    scraper_timeout: float = typer.Option(
        10.0,
        "--scraper-timeout",
        help="HTTP timeout (seconds) for NBA.com roster scraping.",
    ),
    backfill_mode: bool = typer.Option(
        False,
        "--backfill-mode",
        help=(
            "Enable backfill-friendly settings for historical runs. "
            "Uses tip-relative injury selection (ignores run_as_of_ts ceiling), "
            "enables roster fallback, skips active roster validation, and relaxes age checks."
        ),
    ),
    allow_rotowire_props_fallback: bool = typer.Option(
        False,
        "--allow-rotowire-props-fallback/--no-allow-rotowire-props-fallback",
        help=(
            "Deprecated. Live props now resolve from Rotowire bronze props "
            "converted into the same action-props feature schema."
        ),
    ),
    run_id_override: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run ID override. If not provided, derived from run_as_of_ts.",
    ),
) -> None:
    target_day = _normalize_day(date)
    run_ts = _normalize_run_timestamp(run_as_of_ts)
    run_id = run_id_override if run_id_override else _format_run_id(run_ts)
    season_value = season_start or _season_start_from_day(target_day)
    season_label = _season_label(season_value)
    warnings: list[str] = []
    active_roster_df: pd.DataFrame | None = None
    active_roster_summary: dict | None = None
    active_pairs_set: set[Tuple[int, int]] = set()
    inactive_details: pd.DataFrame | None = None

    # Apply backfill mode defaults
    if backfill_mode:
        if roster_fallback_days == 0:
            roster_fallback_days = 7
        if roster_max_age_hours == 18:
            roster_max_age_hours = 720  # 30 days
        validate_active_roster = False
        warnings.append(
            "[backfill-mode] Using tip-relative injury selection, roster fallback, and relaxed age checks."
        )

    if validate_active_roster:
        players_scraper = NbaPlayersScraper(timeout=scraper_timeout)
        try:
            player_profiles = players_scraper.fetch_players(active_only=True)
        except Exception as exc:  # pragma: no cover - network guarded
            warnings.append(f"Active roster scrape failed: {exc}")
        else:
            active_roster_df = _player_profiles_to_frame(player_profiles)
            if active_roster_df.empty:
                warnings.append("Active roster scrape returned zero rows.")
            else:
                active_roster_summary = {
                    "players": int(len(active_roster_df)),
                    "teams": int(active_roster_df["team_id"].nunique(dropna=True)),
                }
                active_pairs_set = _active_roster_pairs(active_roster_df)

    labels_source_df, label_source = _load_label_sources(
        data_root=data_root,
        season_value=season_value,
        override_path=labels_path,
        warnings=warnings,
    )
    labels_frame = _load_label_history(
        labels_source_df,
        target_day=target_day,
        history_days=history_days,
        run_as_of_ts=run_ts,
        label_source=label_source,
    )

    schedule_default = data_root / "silver" / "schedule" / f"season={season_value}"
    injuries_default = data_root / "silver" / "injuries_snapshot" / f"season={season_value}"
    odds_default = data_root / "silver" / "odds_snapshot" / f"season={season_value}"
    roster_default = data_root / "silver" / "roster_nightly" / f"season={season_value}"
    roles_default = data_root / "gold" / "minutes_roles" / f"season={season_value}" / "roles.parquet"
    archetype_default = (
        data_root
        / "gold"
        / "features_minutes_v1"
        / f"season={season_value}"
        / "archetype_deltas.parquet"
    )

    schedule_df = _load_table(schedule_default, schedule_path)
    
    # Prefer bronze injuries_raw for live builds as well so we retain multiple snapshots per day.
    # Read a small day window to capture the latest pre-tip update per game/player even when
    # ingestion/report dates and game tips cross midnight/timezone boundaries.
    if injuries_path is None:
        injuries_df = _load_injuries_bronze_window(
            data_root=data_root,
            season_value=season_value,
            target_day=target_day,
            days_before=1,
            days_after=1,
        )
        injuries_source = "bronze"
        if not injuries_df.empty:
            tag = "[backfill-mode]" if backfill_mode else "[live]"
            warnings.append(
                f"{tag} Loaded injuries_raw from bronze day-window "
                f"{(target_day - pd.Timedelta(days=1)).date().isoformat()}.."
                f"{(target_day + pd.Timedelta(days=1)).date().isoformat()}."
            )
        else:
            # Fall back to silver if bronze partitions are missing (keeps pipeline unblocked).
            injuries_df = _load_table(injuries_default, injuries_path)
            injuries_source = "silver"
            warnings.append(
                f"[live] warning: bronze injuries_raw empty for day={target_day.date().isoformat()}; "
                "falling back to silver injuries_snapshot."
            )
    else:
        injuries_df = _load_table(injuries_default, injuries_path)
        injuries_source = "override"

    injuries_raw_row_count = len(injuries_df)
    if injuries_source == "bronze" and not injuries_df.empty and {"game_id", "player_id"}.issubset(injuries_df.columns):
        tip_frame = schedule_df.loc[:, ["game_id", "tip_ts"]].drop_duplicates().copy()
        tip_frame["tip_ts"] = pd.to_datetime(tip_frame["tip_ts"], utc=True, errors="coerce")
        injuries_merged = injuries_df.merge(tip_frame, on="game_id", how="left")
        if {"status", "restriction_flag", "ramp_flag"}.issubset(injuries_merged.columns):
            injuries_df = select_injury_snapshot(injuries_merged)
        else:
            warnings.append(
                "[live] bronze injuries_raw missing required columns for snapshot selection; "
                "continuing with raw rows."
            )
    else:
        injuries_raw_row_count = len(injuries_df)

    odds_df = _load_table(odds_default, odds_path)
    roster_df = _load_table(roster_default, roster_path)

    slate_game_ids: set[int] = set()
    if {"game_id", "game_date"}.issubset(schedule_df.columns):
        schedule_days = pd.to_datetime(schedule_df["game_date"], errors="coerce").dt.normalize()
        slate_rows = schedule_df.loc[schedule_days == target_day, ["game_id"]].copy()
        if not slate_rows.empty:
            slate_game_ids = set(
                pd.to_numeric(slate_rows["game_id"], errors="coerce").dropna().astype(int).tolist()
            )

    # Live mode: drop NBA.com lineup signals and let Rotowire provide starter truth.
    # Backfill mode: preserve historical lineup fields from roster_nightly as fallback,
    # because Rotowire partitions are often unavailable historically.
    if not roster_df.empty:
        roster_df = roster_df.copy()
        if not backfill_mode:
            for column in ("lineup_role", "lineup_status", "lineup_roster_status"):
                if column in roster_df.columns:
                    roster_df[column] = pd.NA
            if "lineup_timestamp" in roster_df.columns:
                roster_df["lineup_timestamp"] = pd.NaT
            if "is_projected_starter" in roster_df.columns:
                roster_df["is_projected_starter"] = False
            if "is_confirmed_starter" in roster_df.columns:
                roster_df["is_confirmed_starter"] = False
        else:
            typer.echo(
                "[minutes-live] Backfill mode: preserving roster_nightly lineup/starter fields when Rotowire is missing."
            )

    # Load Rotowire lineups for lineup-status updates.
    # Rotowire is prioritized over NBA.com because it typically updates faster.
    rotowire_confirmed_names: set[str] = set()  # Players with confirmed_starter role
    rotowire_projected_names: set[str] = set()  # Players with projected_starter role
    rotowire_out_names: set[str] = set()  # Players explicitly marked out
    rotowire_lineups_path = data_root / "silver" / "rotowire_lineups" / f"date={target_day.date()}" / "lineups.parquet"
    if rotowire_lineups_path.exists():
        try:
            rotowire_df = pd.read_parquet(rotowire_lineups_path)
            if not rotowire_df.empty and "lineup_role" in rotowire_df.columns:
                # Carry through starters and explicit outs. These are the
                # lineup-state signals that should affect live availability.
                tracked_roles = rotowire_df["lineup_role"].isin(["confirmed_starter", "projected_starter", "out"])
                rotowire_status = rotowire_df[tracked_roles].copy()

                if not rotowire_status.empty:
                    # Keep a per-player scrape timestamp so downstream lineup_available
                    # contract can treat Rotowire-provided starter rows as lineup-present.
                    name_norm_series = (
                        rotowire_status["player_name"]
                        .astype(str)
                        .map(_normalize_name_for_matching)
                    )
                    if "ingested_ts" in rotowire_status.columns:
                        ingested_ts = pd.to_datetime(rotowire_status["ingested_ts"], utc=True, errors="coerce")
                    else:
                        ingested_ts = pd.Series(pd.NaT, index=rotowire_status.index, dtype="datetime64[ns, UTC]")
                    starter_ts_by_name = (
                        pd.DataFrame({"name_norm": name_norm_series, "ingested_ts": ingested_ts})
                        .dropna(subset=["name_norm"])
                        .groupby("name_norm", sort=False)["ingested_ts"]
                        .max()
                    )

                    # Separate confirmed/projected starters from explicit outs.
                    confirmed_mask = rotowire_status["lineup_role"] == "confirmed_starter"
                    projected_mask = rotowire_status["lineup_role"] == "projected_starter"
                    out_mask = rotowire_status["lineup_role"] == "out"
                    rotowire_confirmed_names = set(
                        rotowire_status.loc[confirmed_mask, "player_name"]
                        .astype(str)
                        .map(_normalize_name_for_matching)
                        .unique()
                    )
                    rotowire_projected_names = set(
                        rotowire_status.loc[projected_mask, "player_name"]
                        .astype(str)
                        .map(_normalize_name_for_matching)
                        .unique()
                    )
                    rotowire_out_names = set(
                        rotowire_status.loc[out_mask, "player_name"]
                        .astype(str)
                        .map(_normalize_name_for_matching)
                        .unique()
                    )
                    typer.echo(
                        f"[minutes-live] Loaded {len(rotowire_status)} lineup rows from Rotowire "
                        f"({len(rotowire_confirmed_names)} confirmed, {len(rotowire_projected_names)} projected, "
                        f"{len(rotowire_out_names)} out)."
                    )

                    # Apply lineup status in roster_df based on Rotowire.
                    if not roster_df.empty and "player_name" in roster_df.columns:
                        roster_df = roster_df.copy()
                        name_normalized = roster_df["player_name"].map(_normalize_name_for_matching)
                        tracked_names = rotowire_confirmed_names | rotowire_projected_names | rotowire_out_names
                        rotowire_match = name_normalized.isin(tracked_names)
                        slate_mask = pd.Series(True, index=roster_df.index, dtype=bool)
                        if slate_game_ids and "game_id" in roster_df.columns:
                            roster_game_ids = pd.to_numeric(roster_df["game_id"], errors="coerce").astype("Int64")
                            slate_mask = roster_game_ids.isin(list(slate_game_ids)).fillna(False)
                        elif "game_date" in roster_df.columns:
                            roster_days = pd.to_datetime(roster_df["game_date"], errors="coerce").dt.normalize()
                            slate_mask = (roster_days == target_day).fillna(False)

                        # Respect anti-leak semantics: only consume Rotowire rows that
                        # were ingested by tip time for the player's game.
                        eligible = rotowire_match & slate_mask
                        rotowire_ts = pd.to_datetime(
                            name_normalized.map(starter_ts_by_name),
                            utc=True,
                            errors="coerce",
                        )
                        if "tip_ts" in roster_df.columns:
                            tip_ts = pd.to_datetime(roster_df["tip_ts"], utc=True, errors="coerce")
                            eligible = eligible & (rotowire_ts.isna() | (rotowire_ts <= tip_ts))
                        else:
                            eligible = eligible & (rotowire_ts.isna() | (rotowire_ts <= run_ts))

                        # Avoid conflicts: do not mark players as starters if our other feeds already
                        # consider them inactive/out for the slate.
                        if "active_flag" in roster_df.columns:
                            starter_eligible = eligible & roster_df["active_flag"].fillna(False).astype(bool)
                        else:
                            starter_eligible = eligible.copy()
                        if "lineup_role" in roster_df.columns:
                            role_norm = (
                                roster_df["lineup_role"]
                                .astype("string", copy=False)
                                .str.strip()
                                .str.lower()
                                .fillna("")
                            )
                            starter_eligible = starter_eligible & ~role_norm.eq("out")

                        if eligible.any():
                            # Initialize columns if missing
                            if "is_confirmed_starter" not in roster_df.columns:
                                roster_df["is_confirmed_starter"] = False
                            if "is_projected_starter" not in roster_df.columns:
                                roster_df["is_projected_starter"] = False
                            if "lineup_role" not in roster_df.columns:
                                roster_df["lineup_role"] = pd.NA
                            if "lineup_timestamp" not in roster_df.columns:
                                roster_df["lineup_timestamp"] = pd.NaT

                            out_eligible = eligible & name_normalized.isin(rotowire_out_names)
                            projected_eligible = starter_eligible & name_normalized.isin(rotowire_projected_names)
                            confirmed_eligible = starter_eligible & name_normalized.isin(rotowire_confirmed_names)
                            any_starter_eligible = projected_eligible | confirmed_eligible

                            # Explicit Rotowire out rows should clear starter flags.
                            if out_eligible.any():
                                roster_df.loc[out_eligible, "is_projected_starter"] = False
                                roster_df.loc[out_eligible, "is_confirmed_starter"] = False
                                roster_df.loc[out_eligible, "lineup_role"] = "out"

                            # Upgrade is_projected_starter for all eligible starters (confirmed or projected)
                            roster_df.loc[any_starter_eligible, "is_projected_starter"] = True

                            # Upgrade is_confirmed_starter only for confirmed starters
                            roster_df.loc[confirmed_eligible, "is_confirmed_starter"] = True
                            roster_df.loc[projected_eligible, "lineup_role"] = "projected_starter"
                            roster_df.loc[confirmed_eligible, "lineup_role"] = "confirmed_starter"

                            # Stamp lineup_timestamp for all eligible Rotowire rows
                            # so lineup_available contract remains consistent.
                            rotowire_ts = rotowire_ts.fillna(run_ts)
                            roster_df.loc[eligible, "lineup_timestamp"] = rotowire_ts.loc[eligible].values

                            projected_count = int(any_starter_eligible.sum())
                            confirmed_count = int(confirmed_eligible.sum())
                            out_count = int(out_eligible.sum())
                            typer.echo(
                                f"[minutes-live] Applied Rotowire lineup states to {int(eligible.sum())} players "
                                f"({projected_count} starters, {confirmed_count} confirmed, {out_count} out)."
                            )
            else:
                typer.echo("[minutes-live] Rotowire lineups file exists but is empty or missing lineup_role column.")
        except Exception as exc:
            warnings.append(f"Failed to load Rotowire lineups: {exc}")
            typer.echo(f"[minutes-live] Warning: Failed to load Rotowire lineups: {exc}")

    # Enforce <=5 starters per game/team in roster rows used for the current slate.
    if not roster_df.empty and {"game_id", "team_id", "is_projected_starter", "is_confirmed_starter"}.issubset(roster_df.columns):
        roster_df = roster_df.copy()
        roster_game_ids = pd.to_numeric(roster_df["game_id"], errors="coerce").astype("Int64")
        if slate_game_ids:
            starter_scope = roster_game_ids.isin(list(slate_game_ids)).fillna(False)
        elif "game_date" in roster_df.columns:
            roster_days = pd.to_datetime(roster_df["game_date"], errors="coerce").dt.normalize()
            starter_scope = (roster_days == target_day).fillna(False)
        else:
            starter_scope = pd.Series(False, index=roster_df.index, dtype=bool)

        scoped = roster_df.loc[starter_scope].copy()
        if not scoped.empty:
            # Deduplicate roster snapshots so overflow diagnostics reflect player counts,
            # not raw polling rows.
            if "as_of_ts" in scoped.columns:
                scoped["as_of_ts"] = pd.to_datetime(scoped["as_of_ts"], utc=True, errors="coerce")
                scoped = scoped.sort_values(["game_id", "team_id", "player_id", "as_of_ts"], kind="mergesort")
            else:
                scoped = scoped.sort_values(["game_id", "team_id", "player_id"], kind="mergesort")
            scoped = scoped.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")

            scoped = normalize_starter_signals(scoped)
            cap_result = derive_starter_flag_label(
                scoped,
                prefer_sources=("is_confirmed_starter", "is_projected_starter"),
                group_cols=("game_id", "team_id"),
                max_starters=5,
            )
            keep = cap_result.values.reindex(scoped.index).fillna(0).astype("Int64").eq(1)
            confirmed_prev = scoped["is_confirmed_starter"].astype("boolean", copy=False).fillna(False)
            projected_new = keep.astype(bool)
            confirmed_new = (keep & confirmed_prev).astype(bool)

            roster_df.loc[scoped.index, "is_projected_starter"] = projected_new.values
            roster_df.loc[scoped.index, "is_confirmed_starter"] = confirmed_new.values
            if "lineup_role" in roster_df.columns:
                role = pd.Series(pd.NA, index=scoped.index, dtype="object")
                role.loc[projected_new] = "projected_starter"
                role.loc[confirmed_new] = "confirmed_starter"
                roster_df.loc[scoped.index, "lineup_role"] = role.values

            if cap_result.overflow:
                sample = cap_result.overflow[:10]
                warnings.append(
                    "Starter overflow detected before live-label creation "
                    f"(requested starters > 5). sample={sample}. Applying capped top-5 selection."
                )
                typer.echo(
                    "[minutes-live] WARNING: Starter overflow detected before live-label creation "
                    f"(requested starters > 5). sample={sample}. Applying capped top-5 selection.",
                    err=True,
                )

    roster_slice, roster_source_day, roster_snapshot_ts = _select_roster_slice(
        roster_df,
        target_day=target_day,
        run_as_of_ts=run_ts,
        fallback_days=roster_fallback_days,
        max_age_hours=roster_max_age_hours,
    )
    if roster_slice.empty:
        raise RuntimeError(
            f"Roster snapshot does not include rows for {target_day.date()} and no fallback within {roster_fallback_days} day(s) was found."
        )
    roster_slice = normalize_starter_signals(roster_slice)
    if roster_source_day is not None and roster_source_day != target_day:
        warnings.append(
            f"Roster fallback: using snapshot from {roster_source_day.date()} for {target_day.date()} (max {roster_fallback_days}d)."
        )
        typer.echo(
            f"[minutes-live] Using roster snapshot from {roster_source_day.date()} for {target_day.date()} (fallback {roster_fallback_days}d)."
        )
    live_labels = _build_live_labels(roster_slice, target_day=target_day, season_label=season_label)

    # History is always retained; live labels may be pruned by lock gating.
    history_labels = labels_frame.copy()
    live_labels_working = live_labels.copy()

    all_game_ids = pd.to_numeric(history_labels["game_id"], errors="coerce").dropna().astype(int).unique().tolist()
    live_game_ids = pd.to_numeric(live_labels_working["game_id"], errors="coerce").dropna().astype(int).unique().tolist()
    schedule_slice = _filter_by_game_ids(schedule_df, all_game_ids + live_game_ids)
    if schedule_slice.empty:
        raise RuntimeError("Schedule slice is empty after filtering by requested game_ids.")
    schedule_for_builder = schedule_slice.copy()

    allowed_live_ids = live_game_ids
    schedule_live = schedule_slice.copy()
    if lock_buffer_minutes > 0:
        tips = pd.to_datetime(schedule_live["tip_ts"], utc=True, errors="coerce")
        cutoff = run_ts - pd.Timedelta(minutes=lock_buffer_minutes)
        allowed_mask = tips.isna() | (tips >= cutoff)
        locked_games = schedule_live.loc[~allowed_mask, "game_id"].dropna().unique().tolist()
        schedule_live = schedule_live.loc[allowed_mask].copy()
        if schedule_live.empty:
            raise RuntimeError("All games are past the lock cutoff; nothing to score.")
        if locked_games:
            warnings.append(
                f"[lock-guard] Skipping {len(locked_games)} game(s) with tip_ts before {cutoff.isoformat()}."
            )
        else:
            warnings.append("[lock-guard] No games skipped; cutoff not triggered.")
        allowed_live_ids = pd.to_numeric(schedule_live["game_id"], errors="coerce").dropna().astype(int).unique().tolist()
        live_labels_working = live_labels_working[live_labels_working["game_id"].isin(allowed_live_ids)].copy()

    combined_labels = pd.concat([history_labels, live_labels_working], ignore_index=True, sort=False)
    all_game_ids = pd.to_numeric(combined_labels["game_id"], errors="coerce").dropna().astype(int).unique().tolist()
    if not all_game_ids:
        raise RuntimeError("No game_ids available after combining historical labels and live stubs.")

    tip_lookup = _per_game_tip_lookup(schedule_live)

    injuries_slice = _filter_snapshot_by_asof(
        _filter_by_game_ids(injuries_df, allowed_live_ids),
        time_col="as_of_ts",
        run_as_of_ts=run_ts,
        tip_lookup=tip_lookup,
        dataset_name="injuries_snapshot",
        warnings=warnings,
        backfill_mode=backfill_mode,
    )
    if injuries_slice.empty:
        latest_inj_ts = pd.to_datetime(injuries_df.get("as_of_ts"), utc=True, errors="coerce")
        latest_ts_str = latest_inj_ts.max().isoformat() if not latest_inj_ts.dropna().empty else "NA"
        # Continue with warning (vacancy-sensitive features may degrade for this slate).
        warn_msg = (
            f"Injury snapshot is empty after as-of filtering. "
            f"run_as_of_ts={run_ts.isoformat()} latest_injury_as_of_ts={latest_ts_str}. "
            "Continuing with empty injury data; vacancy features may be affected."
        )
        warnings.append(warn_msg)
        typer.echo(f"[minutes-live] WARNING: {warn_msg}")
    odds_slice = _filter_snapshot_by_asof(
        _filter_by_game_ids(odds_df, allowed_live_ids),
        time_col="as_of_ts",
        run_as_of_ts=run_ts,
        tip_lookup=tip_lookup,
        dataset_name="odds_snapshot",
        warnings=warnings,
        backfill_mode=backfill_mode,
    )
    roster_builder_slice = _filter_snapshot_by_asof(
        _filter_by_game_ids(roster_df.copy(), allowed_live_ids),
        time_col="as_of_ts",
        run_as_of_ts=run_ts,
        tip_lookup=tip_lookup,
        dataset_name="roster_nightly",
        warnings=warnings,
        backfill_mode=backfill_mode,
    )

    coach_df = None
    coach_file = coach_path or (data_root / "static" / "coach_tenure.csv")
    if coach_file.exists():
        coach_df = pd.read_csv(coach_file)

    roles_df = _read_parquet_if_exists(roles_path or roles_default)
    if roles_path and roles_df is None:
        warnings.append(f"Roles parquet not found at {roles_path}; archetype features disabled.")
    archetype_deltas_df = _read_parquet_if_exists(archetype_path or archetype_default)
    if archetype_path and archetype_deltas_df is None:
        warnings.append(
            f"Archetype deltas parquet not found at {archetype_path}; archetype features disabled."
        )

    builder = MinutesFeatureBuilder(
        schedule=schedule_for_builder,
        injuries_snapshot=injuries_slice,
        odds_snapshot=odds_slice,
        roster_nightly=roster_builder_slice,
        coach_tenure=coach_df,
        archetype_roles=roles_df,
        archetype_deltas=archetype_deltas_df,
    )
    raw_features = builder.build(combined_labels)
    if "starter_flag" not in raw_features.columns and "starter_flag_label" in raw_features.columns:
        raw_features["starter_flag"] = raw_features["starter_flag_label"]
    if "starter_flag" not in raw_features.columns and {"game_id", "player_id", "starter_flag"}.issubset(combined_labels.columns):
        label_flags = (
            combined_labels.loc[:, ["game_id", "player_id", "starter_flag"]]
            .dropna(subset=["game_id", "player_id"])
            .drop_duplicates(subset=["game_id", "player_id"], keep="last")
        )
        raw_features = raw_features.merge(label_flags, on=["game_id", "player_id"], how="left")
    deduped = deduplicate_latest(raw_features, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    aligned = enforce_schema(deduped, FEATURES_MINUTES_V1_SCHEMA, allow_missing_optional=True)
    validate_with_pandera(aligned, FEATURES_MINUTES_V1_SCHEMA)

    aligned["game_date"] = pd.to_datetime(aligned["game_date"]).dt.normalize()
    live_slice = aligned[aligned["game_date"] == target_day].copy()
    if live_slice.empty:
        raise RuntimeError(f"No feature rows produced for {target_day.date()}.")
    live_slice.sort_values(["game_id", "player_id"], inplace=True)
    # Guard against duplicate rows per player-game (multiple snapshots); keep latest feature_as_of_ts.
    live_slice = deduplicate_latest(live_slice, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    live_slice = live_slice.drop_duplicates(subset=list(KEY_COLUMNS), keep="last").copy()

    # Recompute core trend features using history only (prior to target_day).
    trend_cols = [
        "min_last1",
        "min_last3",
        "min_last5",
        "sum_min_7d",
        "roll_mean_3",
        "roll_mean_5",
        "roll_mean_10",
        "roll_iqr_5",
        "z_vs_10",
    ]
    try:
        history_work = history_labels.copy()
        history_work["game_date"] = pd.to_datetime(history_work["game_date"]).dt.normalize()
        history_work.sort_values(["player_id", "game_date"], inplace=True)
        latest_by_player: list[dict[str, object]] = []
        cutoff_7d = target_day - pd.Timedelta(days=7)
        for pid, group in history_work.groupby("player_id"):
            minutes = pd.to_numeric(group["minutes"], errors="coerce")
            dates = pd.to_datetime(group["game_date"]).dt.normalize()
            if minutes.empty:
                continue
            
            # Filter out 0-minute games (DNP/injury) for rolling averages
            # These should not count toward a player's baseline minutes expectation
            played_mask = minutes > 0
            played_minutes = minutes[played_mask]
            
            # Use last played game for min_last1 (not DNP games)
            last_minutes = played_minutes.iloc[-1] if not played_minutes.empty else 0.0
            
            # Rolling averages over games actually played
            last3 = played_minutes.tail(3).mean() if len(played_minutes) >= 1 else pd.NA
            last5 = played_minutes.tail(5).mean() if len(played_minutes) >= 1 else pd.NA
            mean3 = last3
            mean5 = last5
            mean10 = played_minutes.tail(10).mean() if len(played_minutes) >= 1 else pd.NA
            iqr5 = played_minutes.tail(5).quantile(0.75) - played_minutes.tail(5).quantile(0.25) if len(played_minutes.tail(5)) >= 2 else 0.0
            
            # sum_min_7d uses all games in window (including DNPs, as it's schedule-aware)
            recent_window = minutes[dates >= cutoff_7d]
            sum7 = float(recent_window.sum()) if not recent_window.empty else 0.0
            
            # z-score over played games
            last10_played = played_minutes.tail(10)
            mu10 = last10_played.mean() if not last10_played.empty else 0.0
            std10 = last10_played.std(ddof=0) if not last10_played.empty else 0.0
            z10 = float((last_minutes - mu10) / std10) if std10 and std10 > 0 else 0.0
            latest_by_player.append(
                {
                    "player_id": pid,
                    "min_last1": float(last_minutes),
                    "min_last3": float(last3) if pd.notna(last3) else pd.NA,
                    "min_last5": float(last5) if pd.notna(last5) else pd.NA,
                    "sum_min_7d": float(sum7),
                    "roll_mean_3": float(mean3) if pd.notna(mean3) else pd.NA,
                    "roll_mean_5": float(mean5) if pd.notna(mean5) else pd.NA,
                    "roll_mean_10": float(mean10) if pd.notna(mean10) else pd.NA,
                    "roll_iqr_5": float(iqr5) if pd.notna(iqr5) else 0.0,
                    "z_vs_10": z10,
                }
            )
        trend_frame = pd.DataFrame(latest_by_player)
        if not trend_frame.empty:
            live_slice = live_slice.merge(trend_frame, on="player_id", how="left", suffixes=("", "_recomp"))
            for col in trend_cols:
                recomputed = f"{col}_recomp"
                if recomputed in live_slice.columns:
                    live_slice[col] = live_slice[recomputed].combine_first(live_slice.get(col))
                    live_slice.drop(columns=[recomputed], inplace=True)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"trend recompute failed: {exc}")

    # ---------------------------------------------------------------------------
    # Recompute rest/schedule features from historical labels
    # These require per-player game history which the builder doesn't have in live mode
    # ---------------------------------------------------------------------------
    try:
        history_work = history_labels.copy()
        history_work["game_date"] = pd.to_datetime(history_work["game_date"]).dt.normalize()
        history_work = history_work[history_work["game_date"] < target_day].copy()
        if not history_work.empty:
            history_work["player_id"] = pd.to_numeric(history_work["player_id"], errors="coerce").astype("Int64")
            history_work = history_work.dropna(subset=["player_id"])
            history_work.sort_values(["player_id", "game_date"], inplace=True)

            rest_features: list[dict[str, object]] = []
            for pid, group in history_work.groupby("player_id"):
                dates = pd.to_datetime(group["game_date"]).dt.normalize().sort_values()
                if dates.empty:
                    continue

                last_game_date = dates.iloc[-1]
                days_since = (target_day - last_game_date).days

                # Count games in last N days for is_3in4, is_4in6
                games_in_4d = int((dates >= (target_day - pd.Timedelta(days=4))).sum())
                games_in_6d = int((dates >= (target_day - pd.Timedelta(days=6))).sum())

                rest_features.append({
                    "player_id": pid,
                    "days_since_last_recomp": float(days_since) if days_since >= 0 else 0.0,
                    "is_b2b_recomp": int(days_since == 1),
                    "is_3in4_recomp": int(games_in_4d >= 2),  # 3rd game in 4 days = already played 2
                    "is_4in6_recomp": int(games_in_6d >= 3),  # 4th game in 6 days = already played 3
                })

            if rest_features:
                rest_frame = pd.DataFrame(rest_features)
                rest_frame["player_id"] = pd.to_numeric(rest_frame["player_id"], errors="coerce").astype("Int64")
                live_slice = live_slice.merge(rest_frame, on="player_id", how="left")

                for col in ["days_since_last", "is_b2b", "is_3in4", "is_4in6"]:
                    recomp_col = f"{col}_recomp"
                    if recomp_col in live_slice.columns:
                        live_slice[col] = (
                            pd.to_numeric(live_slice[recomp_col], errors="coerce")
                            .combine_first(pd.to_numeric(live_slice.get(col), errors="coerce"))
                            .fillna(0.0)
                        )
                        live_slice.drop(columns=[recomp_col], inplace=True)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"rest feature recompute failed: {exc}")

    # ---------------------------------------------------------------------------
    # Recompute recency features (recent_start_pct_10) from historical starter flags
    # NOTE: Use starter_flag_label as primary source; starter_flag may be corrupt (all 1s)
    # ---------------------------------------------------------------------------
    try:
        history_work = history_labels.copy()
        history_work["game_date"] = pd.to_datetime(history_work["game_date"]).dt.normalize()
        history_work = history_work[history_work["game_date"] < target_day].copy()
        
        # Determine the correct starter flag column to use
        # Priority: starter_flag_label (ground truth) > starter_flag (may be corrupt)
        starter_col = None
        if "starter_flag_label" in history_work.columns:
            # Check if starter_flag_label has valid variance
            sfl = pd.to_numeric(history_work["starter_flag_label"], errors="coerce").fillna(0)
            if sfl.std() > 0.01:  # Has variance (not all same value)
                starter_col = "starter_flag_label"
        
        if starter_col is None and "starter_flag" in history_work.columns:
            sf = pd.to_numeric(history_work["starter_flag"], errors="coerce").fillna(0)
            # Check if starter_flag is corrupt (all 1s or all 0s)
            if sf.std() > 0.01:  # Has variance
                starter_col = "starter_flag"
            elif sf.mean() > 0.95:  # All 1s - corrupt!
                typer.echo(
                    f"[minutes-live] WARNING: starter_flag is corrupt (all 1s, mean={sf.mean():.3f}). "
                    "recent_start_pct_10 will use starter_flag_label fallback or default to 0."
                )
                warnings.append("starter_flag corrupt (all 1s); falling back to starter_flag_label")
                # Try starter_flag_label even if variance check failed
                if "starter_flag_label" in history_work.columns:
                    starter_col = "starter_flag_label"
        
        if not history_work.empty and starter_col is not None:
            history_work["player_id"] = pd.to_numeric(history_work["player_id"], errors="coerce").astype("Int64")
            history_work["_starter_val"] = pd.to_numeric(history_work[starter_col], errors="coerce").fillna(0)
            history_work = history_work.dropna(subset=["player_id"])
            history_work.sort_values(["player_id", "game_date"], inplace=True)

            recency_features: list[dict[str, object]] = []
            for pid, group in history_work.groupby("player_id"):
                last_10 = group.tail(10)
                start_pct = float(last_10["_starter_val"].mean()) if len(last_10) > 0 else 0.0
                recency_features.append({
                    "player_id": pid,
                    "recent_start_pct_10_recomp": start_pct,
                })

            if recency_features:
                recency_frame = pd.DataFrame(recency_features)
                recency_frame["player_id"] = pd.to_numeric(recency_frame["player_id"], errors="coerce").astype("Int64")
                live_slice = live_slice.merge(recency_frame, on="player_id", how="left")

                if "recent_start_pct_10_recomp" in live_slice.columns:
                    live_slice["recent_start_pct_10"] = (
                        pd.to_numeric(live_slice["recent_start_pct_10_recomp"], errors="coerce")
                        .combine_first(pd.to_numeric(live_slice.get("recent_start_pct_10"), errors="coerce"))
                        .fillna(0.0)
                        .clip(0.0, 1.0)
                    )
                    live_slice.drop(columns=["recent_start_pct_10_recomp"], inplace=True)
                    
                    # Diagnostic: log distribution of recent_start_pct_10
                    rsp = live_slice["recent_start_pct_10"]
                    nonzero_count = int((rsp > 0).sum())
                    typer.echo(
                        f"[minutes-live] recent_start_pct_10 recomputed from {starter_col}: "
                        f"nonzero={nonzero_count}/{len(rsp)}, mean={rsp.mean():.3f}"
                    )
        else:
            if starter_col is None:
                warnings.append("No valid starter flag column found for recent_start_pct_10 recompute")
                typer.echo("[minutes-live] WARNING: No valid starter flag column found; recent_start_pct_10 will be 0")
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"recency feature recompute failed: {exc}")

    # Reinstate starter signals from roster slice if the builder dropped them.
    starter_cols = ["is_projected_starter", "is_confirmed_starter"]
    if not roster_slice.empty and set(starter_cols).issubset(roster_slice.columns):
        starter_hint = roster_slice[["game_id", "player_id"] + starter_cols].copy()
        starter_hint = starter_hint.drop_duplicates(subset=["game_id", "player_id"], keep="last")
        for col in starter_cols:
            starter_hint[col] = starter_hint[col].astype("boolean", copy=False)
        live_slice = live_slice.merge(
            starter_hint,
            on=["game_id", "player_id"],
            how="left",
            suffixes=("", "_roster"),
        )
        for col in starter_cols:
            roster_col = f"{col}_roster"
            base = live_slice[col] if col in live_slice.columns else pd.Series(False, index=live_slice.index)
            roster_vals = live_slice[roster_col] if roster_col in live_slice.columns else pd.Series(False, index=live_slice.index)
            live_slice[col] = base.fillna(False) | roster_vals.fillna(False)
            if roster_col in live_slice.columns:
                live_slice.drop(columns=[roster_col], inplace=True)

    vacancy_features = _compute_vacancy_features(
        injuries_snapshot=injuries_slice,
        roster_nightly=roster_builder_slice,
        labels_source=labels_source_df,
        target_day=target_day,
        warnings=warnings,
    )
    if not vacancy_features.empty:
        live_slice = live_slice.merge(
            vacancy_features,
            on=["game_id", "team_id"],
            how="left",
            suffixes=("", "_vac"),
        )
        for col in VACANCY_FEATURE_COLUMNS:
            merged_col = f"{col}_vac"
            if merged_col in live_slice.columns:
                live_slice[col] = (
                    pd.to_numeric(live_slice[merged_col], errors="coerce")
                    .combine_first(pd.to_numeric(live_slice[col], errors="coerce"))
                    .fillna(0.0)
                    .astype(float)
                )
                live_slice.drop(columns=[merged_col], inplace=True)
            else:
                live_slice[col] = pd.to_numeric(live_slice[col], errors="coerce").fillna(0.0).astype(float)
    else:
        for col in VACANCY_FEATURE_COLUMNS:
            live_slice[col] = pd.to_numeric(live_slice[col], errors="coerce").fillna(0.0).astype(float)

    # ---------------------------------------------------------------------------
    # Normalize vacancy features to match training distribution
    # Training was built from Oct 2023 - Oct 2024 data (~10-15 games into each season)
    # Live inference may be later in the season with higher cumulative minutes
    # Scale factor = training_mean / live_mean to bring values into training range
    # ---------------------------------------------------------------------------
    TRAINING_VACANCY_MEAN = 183.0  # From rotation_train_v1_20260103 config
    TRAINING_VACANCY_MAX = 2400.0  # Reasonable max from training (99th percentile)
    for col in VACANCY_FEATURE_COLUMNS:
        if col in live_slice.columns:
            raw_values = pd.to_numeric(live_slice[col], errors="coerce").fillna(0.0)
            live_mean = float(raw_values.mean()) if len(raw_values) > 0 else 0.0
            if live_mean > 0 and live_mean > TRAINING_VACANCY_MEAN * 2:
                # Apply scaling only if live values are significantly higher than training
                scale_factor = TRAINING_VACANCY_MEAN / live_mean
                # Cap at training max to avoid extreme values
                live_slice[col] = (raw_values * scale_factor).clip(upper=TRAINING_VACANCY_MAX).astype(float)
            else:
                live_slice[col] = raw_values.clip(upper=TRAINING_VACANCY_MAX).astype(float)
    team_ids = set(pd.to_numeric(live_slice["team_id"], errors="coerce").dropna().astype(int).tolist())
    opponent_ids = set(
        pd.to_numeric(live_slice["opponent_team_id"], errors="coerce").dropna().astype(int).tolist()
    )
    context_team_ids = team_ids | opponent_ids
    # Backfills can hit sparse rates_training_base coverage for late-season slates.
    # Use a longer lookback horizon there to avoid collapsing context to means.
    team_context_max_days_back = 180 if backfill_mode else 14
    team_context = _load_team_context_from_rates_training_base(
        data_root=data_root,
        season_value=season_value,
        target_day=target_day,
        team_ids=context_team_ids,
        warnings=warnings,
        max_days_back=team_context_max_days_back,
    )
    if not team_context.empty:
        live_slice = live_slice.merge(
            team_context,
            on="team_id",
            how="left",
            suffixes=("", "_ctx"),
        )
        for col in TEAM_CONTEXT_COLUMNS:
            merged_col = f"{col}_ctx"
            if merged_col in live_slice.columns:
                live_slice[col] = pd.to_numeric(live_slice[merged_col], errors="coerce").combine_first(
                    pd.to_numeric(live_slice[col], errors="coerce")
                )
                live_slice.drop(columns=[merged_col], inplace=True)

        opp_ctx = team_context.rename(
            columns={
                "team_id": "opponent_team_id",
                "team_pace_szn": "opp_pace_szn",
                "team_def_rtg_szn": "opp_def_rtg_szn",
            }
        )
        opp_cols = ["opponent_team_id", *OPP_CONTEXT_COLUMNS]
        opp_cols = [col for col in opp_cols if col in opp_ctx.columns]
        if len(opp_cols) > 1:
            live_slice = live_slice.merge(
                opp_ctx[opp_cols],
                on="opponent_team_id",
                how="left",
                suffixes=("", "_oppctx"),
            )
            for col in OPP_CONTEXT_COLUMNS:
                merged_col = f"{col}_oppctx"
                if merged_col in live_slice.columns:
                    live_slice[col] = pd.to_numeric(live_slice[merged_col], errors="coerce").combine_first(
                        pd.to_numeric(live_slice[col], errors="coerce")
                    )
                    live_slice.drop(columns=[merged_col], inplace=True)

    # Fill any remaining NaNs with conservative defaults (mean or 100.0).
    for col in (*TEAM_CONTEXT_COLUMNS, *OPP_CONTEXT_COLUMNS):
        values = pd.to_numeric(live_slice[col], errors="coerce")
        mean_val = float(values.mean(skipna=True)) if not values.dropna().empty else 100.0
        live_slice[col] = values.fillna(mean_val).astype(float)

    action_props_snapshot_rows = 0
    action_props_matched_rows = 0
    action_props_snapshots = pd.DataFrame()
    action_props_source = "none"
    expected_props_teams = {
        str(team).strip().upper()
        for team in live_slice.get("team_tricode", pd.Series(dtype="object")).dropna().tolist()
        if str(team).strip()
    }
    rotowire_props_root = data_root / "bronze" / "props"
    if rotowire_props_root.exists():
        try:
            snapshot_frames: list[pd.DataFrame] = []
            source_modes: list[str] = []
            day_snapshots, day_source = load_action_props_feature_snapshots_for_date_live(
                action_props_dir=rotowire_props_root,
                game_date=target_day,
                allow_rotowire_fallback=allow_rotowire_props_fallback,
                rotowire_props_root=rotowire_props_root,
                expected_team_tricodes=expected_props_teams,
            )
            if not day_snapshots.empty:
                snapshot_frames.append(day_snapshots)
            if day_source != "none":
                source_modes.append(day_source)

            next_day_snapshots, next_day_source = load_action_props_feature_snapshots_for_date_live(
                action_props_dir=rotowire_props_root,
                game_date=target_day + pd.Timedelta(days=1),
                allow_rotowire_fallback=allow_rotowire_props_fallback,
                rotowire_props_root=rotowire_props_root,
                expected_team_tricodes=expected_props_teams,
            )
            if not next_day_snapshots.empty:
                snapshot_frames.append(next_day_snapshots)
            if next_day_source != "none":
                source_modes.append(next_day_source)

            action_props_snapshots = (
                pd.concat(snapshot_frames, ignore_index=True)
                if snapshot_frames
                else pd.DataFrame()
            )
            action_props_snapshot_rows = int(len(action_props_snapshots))
            action_props_source = "+".join(sorted(set(source_modes))) if source_modes else "none"
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Live props load failed: {exc}")

    live_slice = attach_action_props_features(
        live_slice,
        action_props_snapshots,
        strict_asof=True,
        as_of_col="feature_as_of_ts",
        tip_col="tip_ts",
        game_date_offsets=(0, -1),
        clamp_late_asof_to_game_date=True,
    )
    if "an_has_any_props" in live_slice.columns:
        action_props_matched_rows = int(
            pd.to_numeric(live_slice["an_has_any_props"], errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .sum()
        )
    if action_props_source == "none":
        msg = "Live props unavailable: no Rotowire-derived snapshots were found for the slate."
        warnings.append(msg)
        typer.echo(f"[minutes-live] WARNING: {msg}")
    elif action_props_matched_rows == 0:
        msg = (
            "Live props loaded from Rotowire, but zero rows matched the current live slice."
        )
        warnings.append(msg)
        typer.echo(f"[minutes-live] WARNING: {msg}")
    typer.echo(
        f"[minutes-live] Action props: source={action_props_source}, snapshots={action_props_snapshot_rows}, "
        f"matched_rows={action_props_matched_rows}, total_rows={len(live_slice)}"
    )

    prop_implied_minutes_diag: dict[str, object] = {
        "enabled": True,
        "season": int(season_value),
        "history_start": None,
        "history_end": None,
        "lookback_days": 365,
        "history_rows": 0,
        "prior_rows": 0,
        "matched_rows": 0,
        "coverage_rate": 0.0,
        "players_with_props": 0,
    }
    try:
        from projections.features.prop_implied_minutes import (
            attach_prop_implied_minutes,
            compute_player_pra_priors_asof,
            load_fpts_training_base_history_multi_season,
        )

        prior_end = (target_day - pd.Timedelta(days=1)).normalize()
        lookback_days = 365
        prior_start = (prior_end - pd.Timedelta(days=lookback_days)).normalize()
        prop_implied_minutes_diag["history_start"] = prior_start.date().isoformat()
        prop_implied_minutes_diag["history_end"] = prior_end.date().isoformat()
        prop_implied_minutes_diag["lookback_days"] = int(lookback_days)

        player_ids: list[int] | None = None
        if action_props_matched_rows > 0 and "player_id" in live_slice.columns:
            has_props = pd.to_numeric(live_slice.get("an_has_any_props", 0), errors="coerce").fillna(0.0).gt(0.0)
            pid = pd.to_numeric(live_slice.loc[has_props, "player_id"], errors="coerce").dropna()
            if not pid.empty:
                player_ids = pid.astype(int).unique().tolist()
        prop_implied_minutes_diag["players_with_props"] = len(player_ids or [])

        history = load_fpts_training_base_history_multi_season(
            data_root=data_root,
            start=prior_start,
            end=prior_end,
            player_ids=player_ids,
        )
        prop_implied_minutes_diag["history_rows"] = int(len(history))

        priors = compute_player_pra_priors_asof(history)
        prop_implied_minutes_diag["prior_rows"] = int(len(priors))

        live_slice = attach_prop_implied_minutes(
            live_slice,
            priors=priors,
            join_keys=("player_id",),
        )
        if "an_implied_minutes" in live_slice.columns:
            matched = (
                pd.to_numeric(live_slice["an_implied_minutes"], errors="coerce")
                .fillna(0.0)
                .gt(0.0)
                .sum()
            )
            prop_implied_minutes_diag["matched_rows"] = int(matched)
            prop_implied_minutes_diag["coverage_rate"] = (
                round(float(matched) / float(action_props_matched_rows), 4)
                if action_props_matched_rows > 0
                else 0.0
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Prop implied minutes attach failed: {exc}")
        # Ensure downstream parity by providing defaults when attachment fails.
        for col, default in (
            ("an_pra_per_min_prior", 1.0),
            ("an_pra_prior_minutes_sum", 0.0),
            ("an_pra_prior_games", 0),
            ("an_implied_minutes", 0.0),
            ("an_has_implied_minutes", 0),
            ("an_implied_minutes_missing", 1),
        ):
            if col not in live_slice.columns:
                live_slice[col] = default
        prop_implied_minutes_diag["enabled"] = False

    active_validation: dict | None = None
    if active_roster_df is not None and not active_roster_df.empty and active_pairs_set:
        team_series = pd.to_numeric(live_slice["team_id"], errors="coerce")
        player_series = pd.to_numeric(live_slice["player_id"], errors="coerce")
        invalid_mask: List[bool] = []
        for team_val, player_val in zip(team_series.tolist(), player_series.tolist()):
            if pd.isna(team_val) or pd.isna(player_val):
                invalid_mask.append(False)
                continue
            pair = (int(team_val), int(player_val))
            invalid_mask.append(pair not in active_pairs_set)
        if invalid_mask:
            mismatch_count = int(sum(invalid_mask))
            if mismatch_count:
                inactive_details = live_slice.loc[
                    invalid_mask, ["game_id", "team_id", "player_id", "player_name"]
                ].copy()
                warnings.append(
                    f"Detected {mismatch_count} live rows not present on NBA.com active roster snapshot."
                )
                if enforce_active_roster:
                    live_slice = live_slice.loc[~pd.Series(invalid_mask).to_numpy()].copy()
            active_validation = {
                "mismatches": mismatch_count,
                "enforced": bool(enforce_active_roster),
                "dropped_rows": mismatch_count if enforce_active_roster else 0,
            }

    day_dir, run_dir = _ensure_run_output_dir(out_root, target_day, run_id)
    feature_path = run_dir / FEATURE_FILENAME
    ids_path = run_dir / IDS_FILENAME
    
    # Verify required features are present
    _verify_required_features(live_slice, run_id, warnings)
    
    live_slice.to_parquet(feature_path, index=False)
    write_ids_csv(live_slice, ids_path)
    if active_roster_df is not None and not active_roster_df.empty:
        active_roster_df.to_parquet(run_dir / ACTIVE_ROSTER_FILENAME, index=False)
    if inactive_details is not None and not inactive_details.empty:
        inactive_details.to_csv(run_dir / INACTIVE_PLAYERS_FILENAME, index=False)

    roster_meta = {
        "source_date": roster_source_day.date().isoformat() if roster_source_day is not None else None,
        "snapshot_ts": roster_snapshot_ts.isoformat() if roster_snapshot_ts is not None else None,
        "snapshot_age_minutes": _minutes_between(run_ts, roster_snapshot_ts) if roster_snapshot_ts is not None else None,
    }
    snapshot_meta = {
        "injuries": _snapshot_stats(injuries_slice, time_col="as_of_ts", run_as_of_ts=run_ts),
        "injuries_source": injuries_source,
        "injuries_raw_rows": injuries_raw_row_count,
        "injuries_filtered_rows": len(injuries_slice),
        "odds": _snapshot_stats(odds_slice, time_col="as_of_ts", run_as_of_ts=run_ts),
        "roster": _snapshot_stats(roster_builder_slice, time_col="as_of_ts", run_as_of_ts=run_ts),
        "action_props": {
            "source_dir": str(rotowire_props_root),
            "source": action_props_source,
            "allow_rotowire_fallback": bool(allow_rotowire_props_fallback),
            "snapshot_rows": action_props_snapshot_rows,
            "matched_rows": action_props_matched_rows,
            "coverage_rate": (
                round(float(action_props_matched_rows) / float(len(live_slice)), 4)
                if len(live_slice) > 0
                else 0.0
            ),
            "feature_columns_present": [
                col for col in ACTION_MARKET_FEATURE_COLUMNS if col in live_slice.columns
            ],
        },
        "prop_implied_minutes": {
            **prop_implied_minutes_diag,
            "feature_columns_present": [
                col
                for col in (
                    "an_pra_per_min_prior",
                    "an_pra_prior_minutes_sum",
                    "an_pra_prior_games",
                    "an_implied_minutes",
                    "an_has_implied_minutes",
                    "an_implied_minutes_missing",
                )
                if col in live_slice.columns
            ],
        },
    }

    summary_path = run_dir / SUMMARY_FILENAME
    _write_summary(
        summary_path,
        date=target_day,
        run_as_of_ts=run_ts,
        rows=len(live_slice),
        games=live_slice["game_id"],
        roster_meta=roster_meta,
        snapshot_meta=snapshot_meta,
        active_roster_meta=active_roster_summary,
        active_validation=active_validation,
        warnings=warnings,
    )
    _write_latest_pointer(day_dir, run_id=run_id, run_as_of_ts=run_ts)

    typer.echo(f"[minutes-live] run={run_id} wrote {len(live_slice):,} rows to {feature_path}")
    return len(live_slice), _nan_rate(live_slice, ["minutes_p50", "minutes_p90", "proj_minutes"])


@app.command()
def main(
    date: datetime = typer.Option(..., "--date", help="Target slate date (YYYY-MM-DD)."),
    run_as_of_ts: datetime | None = typer.Option(
        None,
        "--run-as-of-ts",
        help="Timestamp representing the information state for this run. Defaults to now (UTC).",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        help="Root containing data partitions (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    out_root: Path = typer.Option(
        DEFAULT_OUTPUT_ROOT,
        "--out-root",
        help="Directory where live features will be written (per-day subfolders).",
    ),
    labels_path: Path | None = typer.Option(
        None,
        "--labels-path",
        help=(
            "Optional explicit boxscore labels parquet. Defaults to "
            "<data_root>/labels/season=YYYY/boxscore_labels.parquet."
        ),
    ),
    schedule_path: Path | None = typer.Option(None, help="Optional override for schedule parquet directory."),
    injuries_path: Path | None = typer.Option(None, help="Optional override for injuries_snapshot parquet."),
    odds_path: Path | None = typer.Option(None, help="Optional override for odds_snapshot parquet."),
    roster_path: Path | None = typer.Option(None, help="Optional override for roster_nightly parquet."),
    roles_path: Path | None = typer.Option(
        None,
        "--roles-path",
        help="Optional override for minutes roles parquet (season partition).",
    ),
    archetype_path: Path | None = typer.Option(
        None,
        "--archetype-path",
        help="Optional override for archetype deltas parquet (season partition).",
    ),
    coach_path: Path | None = typer.Option(None, help="Optional CSV override for coach_tenure metadata."),
    history_days: int | None = typer.Option(
        None,
        "--history-days",
        min=1,
        help="Optional rolling history window (in days) for label context. Defaults to full season.",
    ),
    season_start: int | None = typer.Option(
        None,
        "--season-start",
        help="Season start year override (e.g., 2024 for 2024-25). Defaults based on --date.",
    ),
    roster_fallback_days: int = typer.Option(
        0,
        "--roster-fallback-days",
        min=0,
        help="Allow using the most recent roster snapshot within this many days before --date when same-day data is missing.",
    ),
    roster_max_age_hours: int = typer.Option(
        18,
        "--roster-max-age-hours",
        min=1,
        help="Maximum allowed age (in hours) of the roster snapshot relative to run_as_of_ts.",
    ),
    validate_active_roster: bool = typer.Option(
        True,
        "--validate-active-roster/--skip-active-roster",
        help="Fetch NBA.com active roster snapshot and compare against live players.",
    ),
    enforce_active_roster: bool = typer.Option(
        False,
        "--enforce-active-roster",
        help="Drop players that are not present on the NBA.com active roster snapshot.",
    ),
    lock_buffer_minutes: int = typer.Option(
        0,
        "--lock-buffer-minutes",
        min=0,
        help="Skip games whose tip_ts is more than this many minutes before run_as_of_ts (avoid re-scoring locked games).",
    ),
    scraper_timeout: float = typer.Option(
        10.0,
        "--scraper-timeout",
        help="HTTP timeout (seconds) for NBA.com roster scraping.",
    ),
    backfill_mode: bool = typer.Option(
        False,
        "--backfill-mode",
        help=(
            "Enable backfill-friendly settings for historical runs. "
            "Uses tip-relative injury selection (ignores run_as_of_ts ceiling), "
            "enables roster fallback, skips active roster validation, and relaxes age checks."
        ),
    ),
    allow_rotowire_props_fallback: bool = typer.Option(
        False,
        "--allow-rotowire-props-fallback/--no-allow-rotowire-props-fallback",
        help=(
            "Deprecated. Live props now resolve from Rotowire bronze props "
            "converted into the same action-props feature schema."
        ),
    ),
    run_id_override: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run ID override. If not provided, derived from run_as_of_ts.",
    ),
) -> None:
    target_day = _normalize_day(date)
    run_ts = _normalize_run_timestamp(run_as_of_ts)
    target_date = target_day.date().isoformat()
    run_ts_iso = run_ts.isoformat()

    rows_written = 0
    nan_rate = None
    try:
        rows_written, nan_rate = _build_minutes_live_logic(
            date=target_day.to_pydatetime(),
            run_as_of_ts=run_ts.to_pydatetime(),
            data_root=data_root,
            out_root=out_root,
            labels_path=labels_path,
            schedule_path=schedule_path,
            injuries_path=injuries_path,
            odds_path=odds_path,
            roster_path=roster_path,
            roles_path=roles_path,
            archetype_path=archetype_path,
            coach_path=coach_path,
            history_days=history_days,
            season_start=season_start,
            roster_fallback_days=roster_fallback_days,
            roster_max_age_hours=roster_max_age_hours,
            validate_active_roster=validate_active_roster,
            enforce_active_roster=enforce_active_roster,
            lock_buffer_minutes=lock_buffer_minutes,
            scraper_timeout=scraper_timeout,
            backfill_mode=backfill_mode,
            allow_rotowire_props_fallback=allow_rotowire_props_fallback,
            run_id_override=run_id_override,
        )
        write_status(
            JobStatus(
                job_name="build_minutes_live",
                stage="gold",
                target_date=target_date,
                run_ts=run_ts_iso,
                status="success",
                rows_written=rows_written,
                expected_rows=rows_written,
                nan_rate_key_cols=nan_rate,
            )
        )
    except Exception as exc:  # noqa: BLE001
        write_status(
            JobStatus(
                job_name="build_minutes_live",
                stage="gold",
                target_date=target_date,
                run_ts=run_ts_iso,
                status="error",
                rows_written=rows_written,
                expected_rows=None,
                message=str(exc),
            )
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    app()
