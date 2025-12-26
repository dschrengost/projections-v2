#!/usr/bin/env python
"""Backfill gold slates from boxscore labels for historical training data.

This script regenerates slates using boxscore labels as the roster source,
bypassing the corrupted roster_nightly data. This ensures slates have the
correct historical player lists.

Usage:
    uv run python scripts/backfill_slates_from_labels.py \
        --start 2023-10-01 \
        --end 2025-06-30 \
        --season 2023 \
        --force
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections import paths
from projections.etl import storage as bronze_storage
from projections.etl.storage import iter_days
from projections.minutes_v1.features import MinutesFeatureBuilder
from projections.minutes_v1.schemas import (
    BOX_SCORE_LABELS_SCHEMA,
    FEATURES_MINUTES_V1_SCHEMA,
    SLATE_FEATURES_MINUTES_V1_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)
from projections.minutes_v1.snapshots import select_latest_before

UTC = timezone.utc

app = typer.Typer(help=__doc__)


def _read_parquet_tree(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _season_label(season_start: int) -> str:
    return f"{season_start}-{(season_start + 1) % 100:02d}"


def _git_rev_parse_head() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S603, S607
            or None
        )
    except Exception:  # noqa: BLE001
        return None


def _max_ts(df: pd.DataFrame, col: str) -> str | None:
    if df.empty or col not in df.columns:
        return None
    ts = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return None
    return ts.max().isoformat()


def _load_labels_for_season(data_root: Path, season: int) -> pd.DataFrame:
    """Load all boxscore labels for a season."""
    labels_path = data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing boxscore labels at {labels_path}")
    labels = pd.read_parquet(labels_path)
    if "label_frozen_ts" in labels.columns:
        labels["label_frozen_ts"] = pd.to_datetime(labels["label_frozen_ts"], utc=True, errors="coerce")
    else:
        labels["label_frozen_ts"] = pd.NaT

    minutes_col = labels.get("minutes")
    if minutes_col is not None and minutes_col.dtype == object:
        parsed = pd.to_timedelta(minutes_col, errors="coerce")
        labels["minutes"] = (parsed.dt.total_seconds() / 60.0).astype("Float64")
    if "minutes" in labels.columns:
        labels["minutes"] = pd.to_numeric(labels["minutes"], errors="coerce")

    if "starter_flag_label" not in labels.columns:
        starter_series = labels.get("starter_flag")
        starter_bool = (
            starter_series.astype("boolean", copy=False).fillna(False)
            if starter_series is not None
            else pd.Series(False, index=labels.index, dtype="boolean")
        )
        labels["starter_flag_label"] = starter_bool.astype("Int64")

    labels = enforce_schema(labels, BOX_SCORE_LABELS_SCHEMA, allow_missing_optional=True)
    labels["game_date"] = pd.to_datetime(labels["game_date"]).dt.normalize()
    labels.sort_values(
        ["game_id", "team_id", "player_id", "label_frozen_ts"],
        inplace=True,
        kind="mergesort",
    )
    labels = labels.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")
    return labels


def _load_schedule_for_builder(data_root: Path, season: int, game_ids: list[int]) -> pd.DataFrame:
    """Load schedule data for feature builder."""
    schedule_root = data_root / "silver" / "schedule" / f"season={season}"
    schedule_df = _read_parquet_tree(schedule_root)
    if schedule_df.empty:
        raise FileNotFoundError(f"Missing schedule partitions under {schedule_root}")
    schedule_df["game_id"] = pd.to_numeric(schedule_df["game_id"], errors="coerce").astype("Int64")
    schedule_df["tip_ts"] = pd.to_datetime(schedule_df["tip_ts"], utc=True, errors="coerce")
    if "game_date" in schedule_df.columns:
        schedule_df["game_date"] = pd.to_datetime(schedule_df["game_date"], errors="coerce").dt.normalize()
    needed = set(int(gid) for gid in game_ids)
    return schedule_df.loc[schedule_df["game_id"].isin(needed)].copy()


def _build_live_labels_from_boxscore(
    labels_for_game: pd.DataFrame,
    *,
    game_id: int,
    game_date: pd.Timestamp,
    season_label: str,
    frozen_at: pd.Timestamp,
) -> pd.DataFrame:
    """Build live labels scaffold from boxscore labels (clears outcome columns)."""
    if labels_for_game.empty:
        raise RuntimeError("Boxscore labels are empty; cannot build live labels.")

    working = labels_for_game.copy()
    working = working.dropna(subset=["game_id", "player_id", "team_id"])
    working["game_id"] = pd.to_numeric(working["game_id"], errors="coerce").astype("Int64")
    working["player_id"] = pd.to_numeric(working["player_id"], errors="coerce").astype("Int64")
    working["team_id"] = pd.to_numeric(working["team_id"], errors="coerce").astype("Int64")
    working["game_date"] = pd.to_datetime(working.get("game_date"), errors="coerce").dt.normalize()
    working = working[(working["game_id"] == int(game_id)) & (working["game_date"] == game_date)]
    if working.empty:
        raise RuntimeError(f"Boxscore labels have no rows for game_id={game_id} date={game_date.date()}")

    working = working.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")
    live_df = pd.DataFrame(
        {
            "game_id": working["game_id"].astype("Int64"),
            "player_id": working["player_id"].astype("Int64"),
            "team_id": working["team_id"].astype("Int64"),
            "player_name": working.get("player_name"),
            "season": season_label,
            "game_date": working["game_date"],
            "minutes": pd.Series(pd.NA, index=working.index, dtype="Float64"),
            "starter_flag": pd.Series(0, index=working.index, dtype="Int64"),
            "starter_flag_label": pd.Series(0, index=working.index, dtype="Int64"),
            "source": "backfill_boxscore_labels",
            "label_frozen_ts": frozen_at,
        }
    )
    return enforce_schema(live_df, BOX_SCORE_LABELS_SCHEMA, allow_missing_optional=True)


def _freeze_game_from_labels(
    *,
    game_id: int,
    game_date: pd.Timestamp,
    season: int,
    all_labels: pd.DataFrame,
    data_root: Path,
    out_root: Path,
    force: bool,
    history_days: int | None,
) -> tuple[Path, Path] | None:
    """Freeze a single game slate using boxscore labels as roster source."""
    snapshot_type = "pretip"
    frozen_at = pd.Timestamp.now(tz="UTC")
    season_label_str = _season_label(season)

    out_path = out_root / f"season={season}" / f"game_date={game_date.date().isoformat()}" / f"game_id={game_id}"
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_path = out_path / f"{snapshot_type}.parquet"
    manifest_path = out_path / f"manifest.{snapshot_type}.json"

    if parquet_path.exists() and not force:
        return None  # Skip existing

    # Get labels for this game
    target_day = game_date.normalize()
    labels_for_game = all_labels[
        (all_labels["game_id"] == game_id) & (all_labels["game_date"] == target_day)
    ].copy()

    if labels_for_game.empty:
        return None  # No labels for this game

    # Get tip_ts from schedule
    schedule_root = data_root / "silver" / "schedule" / f"season={season}"
    schedule_df = _read_parquet_tree(schedule_root)
    schedule_df["game_id"] = pd.to_numeric(schedule_df["game_id"], errors="coerce").astype("Int64")
    schedule_df["tip_ts"] = pd.to_datetime(schedule_df["tip_ts"], utc=True, errors="coerce")
    game_schedule = schedule_df[schedule_df["game_id"] == game_id]
    
    if game_schedule.empty:
        return None  # No schedule entry

    tip_ts = game_schedule["tip_ts"].iloc[0]
    if pd.isna(tip_ts):
        return None
    snapshot_ts = tip_ts  # pretip = at tip time

    # Load historical labels (for feature building)
    history = all_labels[all_labels["game_date"] < target_day].copy()
    if history_days is not None and history_days > 0:
        cutoff = target_day - pd.Timedelta(days=history_days)
        history = history[history["game_date"] >= cutoff].copy()
    if "minutes" in history.columns:
        history = history.dropna(subset=["minutes"])
    if history.empty:
        return None  # Need history for features

    # Build live labels from boxscore (clears outcome columns)
    try:
        live_labels = _build_live_labels_from_boxscore(
            labels_for_game,
            game_id=game_id,
            game_date=target_day,
            season_label=season_label_str,
            frozen_at=frozen_at,
        )
    except RuntimeError:
        return None

    combined_labels = pd.concat([history, live_labels], ignore_index=True, sort=False)

    # Load odds (if available)
    odds_raw = bronze_storage.read_bronze_day(
        "odds_raw",
        data_root,
        season,
        game_date.date(),
        include_runs=True,
        prefer_history=True,
    )
    odds_raw = odds_raw[odds_raw.get("game_id") == game_id].copy() if not odds_raw.empty else odds_raw
    odds_at_ts = (
        select_latest_before(
            odds_raw,
            snapshot_ts,
            group_cols=["game_id"],
            as_of_col="as_of_ts",
            ingested_col="ingested_ts",
        )
        if not odds_raw.empty
        else odds_raw
    )

    # Load injuries (if available)
    injuries_frames: list[pd.DataFrame] = []
    for day in (game_date.date() - timedelta(days=1), game_date.date()):
        frame = bronze_storage.read_bronze_day(
            "injuries_raw",
            data_root,
            season,
            day,
            include_runs=False,
            prefer_history=True,
        )
        if not frame.empty:
            injuries_frames.append(frame)
    injuries_raw = pd.concat(injuries_frames, ignore_index=True) if injuries_frames else pd.DataFrame()
    injuries_raw = (
        injuries_raw[injuries_raw.get("game_id") == game_id].copy()
        if not injuries_raw.empty
        else injuries_raw
    )
    injuries_at_ts = (
        select_latest_before(
            injuries_raw,
            snapshot_ts,
            group_cols=["game_id", "player_id"],
            as_of_col="as_of_ts",
            ingested_col="ingested_ts",
        )
        if not injuries_raw.empty
        else injuries_raw
    )

    # Load schedule for builder
    needed_ids = (
        pd.to_numeric(combined_labels["game_id"], errors="coerce").dropna().astype(int).unique().tolist()
    )
    schedule_for_builder = _load_schedule_for_builder(data_root, season, needed_ids)
    if schedule_for_builder.empty:
        return None

    # Load roles and archetype deltas (if available)
    roles_path = data_root / "gold" / "minutes_roles" / f"season={season}" / "roles.parquet"
    archetype_path = data_root / "gold" / "features_minutes_v1" / f"season={season}" / "archetype_deltas.parquet"
    roles_df = pd.read_parquet(roles_path) if roles_path.exists() else None
    archetype_deltas_df = pd.read_parquet(archetype_path) if archetype_path.exists() else None

    # Build features
    builder = MinutesFeatureBuilder(
        schedule=schedule_for_builder,
        injuries_snapshot=injuries_at_ts,
        odds_snapshot=odds_at_ts,
        roster_nightly=pd.DataFrame(),  # Empty! Force boxscore path
        archetype_roles=roles_df,
        archetype_deltas=archetype_deltas_df,
    )
    features = builder.build(combined_labels)
    frozen_features = features[features.get("game_id") == game_id].copy()
    if frozen_features.empty:
        return None

    frozen_features = enforce_schema(frozen_features, FEATURES_MINUTES_V1_SCHEMA, allow_missing_optional=True)
    validate_with_pandera(frozen_features, FEATURES_MINUTES_V1_SCHEMA)

    frozen_features["snapshot_type"] = snapshot_type
    frozen_features["snapshot_ts"] = pd.to_datetime(snapshot_ts, utc=True)
    frozen_features["frozen_at"] = frozen_at
    frozen_features = enforce_schema(frozen_features, SLATE_FEATURES_MINUTES_V1_SCHEMA, allow_missing_optional=True)
    validate_with_pandera(frozen_features, SLATE_FEATURES_MINUTES_V1_SCHEMA)

    _atomic_write_parquet(frozen_features, parquet_path)

    manifest = {
        "game_id": game_id,
        "season": season,
        "game_date": game_date.date().isoformat(),
        "tip_ts": pd.Timestamp(tip_ts).isoformat(),
        "snapshot_type": snapshot_type,
        "snapshot_ts": pd.Timestamp(snapshot_ts).isoformat(),
        "frozen_at": frozen_at.isoformat(),
        "git_sha": _git_rev_parse_head(),
        "row_count": int(len(frozen_features)),
        "source": "backfill_from_boxscore_labels",
        "inputs": {
            "history_label_rows": int(len(history)),
            "live_label_rows": int(len(live_labels)),
            "injuries_raw_max_as_of_ts": _max_ts(injuries_at_ts, "as_of_ts"),
            "odds_raw_max_as_of_ts": _max_ts(odds_at_ts, "as_of_ts"),
        },
    }
    _atomic_write_json(manifest_path, manifest)

    return parquet_path, manifest_path


def _process_game_worker(args: tuple) -> tuple:
    """Process a single game - designed for multiprocessing."""
    game_id, game_date_str, season_val, data_root_str, out_root_str, force_val, history_days_val = args
    try:
        # Parse game_date from ISO string
        game_date = pd.Timestamp(game_date_str)
        # Reload labels in subprocess to avoid pickle issues
        all_labels_local = _load_labels_for_season(Path(data_root_str), season_val)
        result = _freeze_game_from_labels(
            game_id=game_id,
            game_date=game_date,
            season=season_val,
            all_labels=all_labels_local,
            data_root=Path(data_root_str),
            out_root=Path(out_root_str),
            force=force_val,
            history_days=history_days_val,
        )
        if result is None:
            return ("skipped", game_id)
        else:
            return ("processed", game_id)
    except Exception as exc:
        return ("failed", game_id, str(exc))


@app.command()
def main(
    start_date: datetime = typer.Option(..., "--start", help="Start date (inclusive)."),
    end_date: datetime = typer.Option(..., "--end", help="End date (inclusive)."),
    season: int = typer.Option(..., "--season", help="Season year (e.g., 2023 for 2023-24)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data directory.",
    ),
    out_root: Path | None = typer.Option(
        None,
        "--out-root",
        help="Optional override for gold/slates root.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing slates."),
    history_days: int | None = typer.Option(
        90, "--history-days", help="Rolling history window for features."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="List games without processing."),
    workers: int = typer.Option(8, "--workers", help="Number of parallel workers."),
) -> None:
    """Backfill gold slates from boxscore labels."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    data_root = data_root.resolve()
    out_root = (out_root or (data_root / "gold" / "slates")).resolve()
    start_day = pd.Timestamp(start_date).normalize()
    end_day = pd.Timestamp(end_date).normalize()

    typer.echo(f"[backfill] Loading labels for season={season}...")
    all_labels = _load_labels_for_season(data_root, season)
    typer.echo(f"[backfill] Loaded {len(all_labels):,} label rows")

    # Get unique games in date range
    mask = (all_labels["game_date"] >= start_day) & (all_labels["game_date"] <= end_day)
    games_df = all_labels.loc[mask, ["game_id", "game_date"]].drop_duplicates()
    games = [(int(r.game_id), pd.Timestamp(r.game_date)) for r in games_df.itertuples()]
    typer.echo(f"[backfill] Found {len(games)} games in date range")

    if dry_run:
        for game_id, game_date in games[:20]:
            typer.echo(f"  {game_date.date()} game_id={game_id}")
        if len(games) > 20:
            typer.echo(f"  ... and {len(games) - 20} more")
        return

    processed = 0
    skipped = 0
    failed = 0

    typer.echo(f"[backfill] Using {workers} parallel workers...")

    # Prepare args for each game - convert Timestamp to ISO string for pickling
    args_list = [
        (game_id, game_date.isoformat(), season, str(data_root), str(out_root), force, history_days)
        for game_id, game_date in games
    ]

    # Use multiprocessing with spawn to avoid fork issues
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = {executor.submit(_process_game_worker, args): args[0] for args in args_list}
        
        for future in as_completed(futures):
            game_id = futures[future]
            try:
                result = future.result()
                if result[0] == "processed":
                    processed += 1
                elif result[0] == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    typer.echo(f"[backfill] FAILED game_id={result[1]}: {result[2]}", err=True)
                
                total = processed + skipped + failed
                if total % 50 == 0:
                    typer.echo(f"[backfill] progress: {total}/{len(games)} (processed={processed} skipped={skipped} failed={failed})")
            except Exception as exc:
                failed += 1
                typer.echo(f"[backfill] FAILED game_id={game_id}: {exc}", err=True)

    typer.echo(f"[backfill] DONE: processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    app()

