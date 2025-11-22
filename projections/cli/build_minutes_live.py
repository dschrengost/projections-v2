"""Build same-day Minutes V1 feature slices for live inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import typer

from projections import paths

from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest, write_ids_csv
from projections.minutes_v1.features import MinutesFeatureBuilder
from projections.minutes_v1.schemas import (
    BOX_SCORE_LABELS_SCHEMA,
    FEATURES_MINUTES_V1_SCHEMA,
    enforce_schema,
    validate_with_pandera,
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

app = typer.Typer(help=__doc__)


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


def _filter_by_game_ids(df: pd.DataFrame, game_ids: Iterable[int]) -> pd.DataFrame:
    if df.empty or "game_id" not in df.columns:
        return df.copy()
    normalized = pd.Series(game_ids, dtype="Int64").dropna().astype(int).tolist()
    if not normalized:
        return df.iloc[0:0].copy()
    return df[pd.to_numeric(df["game_id"], errors="coerce").astype("Int64").isin(normalized)].copy()


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
    # If history still empty, bail explicitly to avoid silent flat features.
    if history.empty:
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

    starter_col = working.get("starter_flag")
    projected = working.get("is_projected_starter")
    confirmed = working.get("is_confirmed_starter")
    lineup_status = working.get("lineup_status")

    starter_series = pd.Series(0, index=working.index, dtype=int)
    if starter_col is not None:
        starter_series |= starter_col.astype("boolean", copy=False).fillna(False).astype(int)

    # Heuristic boost: if no starters marked yet, incorporate projected/confirmed/expected flags.
    if starter_series.sum() == 0:
        if projected is not None:
            starter_series |= projected.astype("boolean", copy=False).fillna(False).astype(int)
        if confirmed is not None:
            starter_series |= confirmed.astype("boolean", copy=False).fillna(False).astype(int)
        if lineup_status is not None:
            starter_series |= lineup_status.str.lower().eq("expected").fillna(False).astype(int)
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
            "starter_flag": starter_series.astype("Int64"),
            "starter_flag_label": starter_series.astype("Int64"),
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
) -> pd.DataFrame:
    if df.empty or time_col not in df.columns or "game_id" not in df.columns:
        return df

    working = df.copy()
    working["game_id"] = pd.to_numeric(working["game_id"], errors="coerce").astype("Int64")
    working[time_col] = pd.to_datetime(working[time_col], utc=True, errors="coerce")

    # For roster, keep the latest snapshot per game_id (no gating); as-of gating is handled downstream via feature_as_of_ts.
    if dataset_name == "roster_nightly":
        latest = (
            working.sort_values(time_col)
            .groupby("game_id", as_index=False)
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
) -> None:
    target_day = _normalize_day(date)
    run_ts = _normalize_run_timestamp(run_as_of_ts)
    run_id = _format_run_id(run_ts)
    season_value = season_start or _season_start_from_day(target_day)
    season_label = _season_label(season_value)
    warnings: list[str] = []
    active_roster_df: pd.DataFrame | None = None
    active_roster_summary: dict | None = None
    active_pairs_set: set[Tuple[int, int]] = set()
    inactive_details: pd.DataFrame | None = None

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
    injuries_df = _load_table(injuries_default, injuries_path)
    odds_df = _load_table(odds_default, odds_path)
    roster_df = _load_table(roster_default, roster_path)

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
    # Normalize starter flags from lineup_status for downstream merges.
    if "lineup_status" in roster_slice.columns:
        status_norm = roster_slice["lineup_status"].astype(str).str.lower()
        if "is_projected_starter" in roster_slice.columns:
            roster_slice["is_projected_starter"] = (
                roster_slice["is_projected_starter"].astype("boolean", copy=False).fillna(False)
                | status_norm.isin(["expected", "confirmed"])
            )
        if "is_confirmed_starter" in roster_slice.columns:
            roster_slice["is_confirmed_starter"] = (
                roster_slice["is_confirmed_starter"].astype("boolean", copy=False).fillna(False)
                | status_norm.eq("confirmed")
            )
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
    )
    if injuries_slice.empty:
        latest_inj_ts = pd.to_datetime(injuries_df.get("as_of_ts"), utc=True, errors="coerce")
        latest_ts_str = latest_inj_ts.max().isoformat() if not latest_inj_ts.dropna().empty else "NA"
        raise RuntimeError(
            "Injury snapshot is empty after as-of filtering; refusing to score. "
            f"run_as_of_ts={run_ts.isoformat()} latest_injury_as_of_ts={latest_ts_str}. "
            "Run build_minutes_live after the injury scrape completes or pass a run_as_of_ts at/after the snapshot time."
    )
    odds_slice = _filter_snapshot_by_asof(
        _filter_by_game_ids(odds_df, allowed_live_ids),
        time_col="as_of_ts",
        run_as_of_ts=run_ts,
        tip_lookup=tip_lookup,
        dataset_name="odds_snapshot",
        warnings=warnings,
    )
    roster_builder_slice = _filter_snapshot_by_asof(
        _filter_by_game_ids(roster_df.copy(), allowed_live_ids),
        time_col="as_of_ts",
        run_as_of_ts=run_ts,
        tip_lookup=tip_lookup,
        dataset_name="roster_nightly",
        warnings=warnings,
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
    deduped = deduplicate_latest(raw_features, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    aligned = enforce_schema(deduped, FEATURES_MINUTES_V1_SCHEMA)
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
            last_minutes = minutes.iloc[-1]
            last3 = minutes.tail(3).mean()
            last5 = minutes.tail(5).mean()
            mean3 = last3
            mean5 = last5
            mean10 = minutes.tail(10).mean()
            iqr5 = minutes.tail(5).quantile(0.75) - minutes.tail(5).quantile(0.25) if len(minutes.tail(5)) >= 2 else 0.0
            recent_window = minutes[dates >= cutoff_7d]
            sum7 = float(recent_window.sum()) if not recent_window.empty else 0.0
            last10 = minutes.tail(10)
            mu10 = last10.mean()
            std10 = last10.std(ddof=0)
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
        "odds": _snapshot_stats(odds_slice, time_col="as_of_ts", run_as_of_ts=run_ts),
        "roster": _snapshot_stats(roster_builder_slice, time_col="as_of_ts", run_as_of_ts=run_ts),
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


if __name__ == "__main__":  # pragma: no cover
    app()
