"""Backfill historical Minutes V1 features for training."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import typer


from projections import paths
from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest
from projections.minutes_v1.features import MinutesFeatureBuilder
from projections.minutes_v1.schemas import (
    BOX_SCORE_LABELS_SCHEMA,
    FEATURES_MINUTES_V1_SCHEMA,
    enforce_schema,
    validate_with_pandera,
)

# Re-use logic from build_minutes_live where possible, but adapted for batch processing
from projections.cli.build_minutes_live import (
    _normalize_day,
    _season_start_from_day,
    _season_label,
    _read_parquet_tree,
    _read_parquet_if_exists,
    _load_table,
    _filter_by_game_ids,
    _load_label_history,
    _per_game_tip_lookup,
    _filter_snapshot_by_asof,
)

UTC = timezone.utc
DEFAULT_DATA_ROOT = paths.get_data_root()
DEFAULT_OUTPUT_ROOT = paths.data_path("gold", "features_minutes_v1")

app = typer.Typer(help=__doc__)


def _iter_days(start: datetime, end: datetime) -> Iterable[datetime]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _get_simulated_run_ts(target_day: pd.Timestamp, hour_utc: int = 16) -> pd.Timestamp:
    """Simulate a run timestamp (e.g., 11:00 AM ET = 16:00 UTC)."""
    return target_day.replace(hour=hour_utc, minute=0, second=0, microsecond=0, tzinfo=UTC)


@app.command()
def main(
    start_date: datetime = typer.Option(..., "--start", help="Start date (YYYY-MM-DD)."),
    end_date: datetime = typer.Option(..., "--end", help="End date (YYYY-MM-DD)."),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        help="Root containing data partitions.",
    ),
    out_root: Path = typer.Option(
        DEFAULT_OUTPUT_ROOT,
        "--out-root",
        help="Directory where gold features will be written (season partitions).",
    ),
    hour_utc: int = typer.Option(
        16,
        "--hour-utc",
        help="Hour (UTC) to simulate the run at (default 16 = 11am ET).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing season partitions.",
    ),
) -> None:
    start_ts = _normalize_day(start_date)
    end_ts = _normalize_day(end_date)
    
    # Group days by season to handle loading appropriate season-partitioned data
    days_by_season = {}
    for day in _iter_days(start_ts, end_ts):
        season = _season_start_from_day(day)
        if season not in days_by_season:
            days_by_season[season] = []
        days_by_season[season].append(day)

    for season_start, days in days_by_season.items():
        season_label = _season_label(season_start)
        typer.echo(f"Processing {len(days)} days for season {season_label}...")
        
        # Load season-level datasets once per season to avoid thrashing IO
        labels_path = data_root / "labels" / f"season={season_start}" / "boxscore_labels.parquet"
        if not labels_path.exists():
            typer.echo(f"Skipping season {season_label}: labels not found at {labels_path}")
            continue
            
        # Load full history for labels (we need this to get game_ids for the day)
        # Note: _load_label_history filters by date, so we might need to load it all or be clever.
        # Let's load the whole parquet for the season, it's usually small enough.
        full_labels = pd.read_parquet(labels_path)
        
        # Fix minutes column if it's in ISO duration format
        minutes_col = full_labels.get("minutes")
        if minutes_col is not None and minutes_col.dtype == object:
            # Check if it looks like a string duration
            if minutes_col.dropna().astype(str).str.startswith("PT").any():
                parsed = pd.to_timedelta(minutes_col, errors="coerce")
                full_labels["minutes"] = (parsed.dt.total_seconds() / 60.0).astype("Float64")

        full_labels["game_date"] = pd.to_datetime(full_labels["game_date"]).dt.normalize()
        
        # Pre-load other season datasets
        schedule_path = data_root / "silver" / "schedule" / f"season={season_start}"
        injuries_path = data_root / "silver" / "injuries_snapshot" / f"season={season_start}"
        odds_path = data_root / "silver" / "odds_snapshot" / f"season={season_start}"
        roster_path = data_root / "silver" / "roster_nightly" / f"season={season_start}"
        roles_path = data_root / "gold" / "minutes_roles" / f"season={season_start}" / "roles.parquet"
        archetype_path = data_root / "gold" / "features_minutes_v1" / f"season={season_start}" / "archetype_deltas.parquet"
        
        schedule_df = _load_table(schedule_path, None)
        injuries_df = _load_table(injuries_path, None)
        odds_df = _load_table(odds_path, None)
        roster_df = _load_table(roster_path, None)
        roles_df = _read_parquet_if_exists(roles_path)
        archetype_deltas_df = _read_parquet_if_exists(archetype_path)
        
        coach_file = data_root / "static" / "coach_tenure.csv"
        coach_df = pd.read_csv(coach_file) if coach_file.exists() else None

        season_features = []
        
        for target_day in days:
            run_ts = _get_simulated_run_ts(target_day, hour_utc)
            
            # 1. Get labels/games for this day
            day_labels = full_labels[full_labels["game_date"] == target_day].copy()
            if day_labels.empty:
                continue
                
            # Ensure schema compliance for labels
            # (Logic adapted from _load_label_history)
            if "starter_flag_label" not in day_labels.columns:
                starter_series = day_labels.get("starter_flag")
                starter_bool = starter_series.astype("boolean", copy=False).fillna(False) if starter_series is not None else pd.Series(
                    False, index=day_labels.index, dtype="boolean"
                )
                day_labels["starter_flag_label"] = starter_bool.astype("Int64")
            if "label_frozen_ts" not in day_labels.columns:
                day_labels["label_frozen_ts"] = pd.NaT
            
            day_labels = enforce_schema(day_labels, BOX_SCORE_LABELS_SCHEMA, allow_missing_optional=True)
            
            all_game_ids = day_labels["game_id"].dropna().astype(int).unique().tolist()
            if not all_game_ids:
                continue

            # 2. Filter Schedule
            schedule_slice = _filter_by_game_ids(schedule_df, all_game_ids)
            if schedule_slice.empty:
                continue
            tip_lookup = _per_game_tip_lookup(schedule_slice)

            # 3. Filter Snapshots (Time Travel!)
            # This is the critical part: we filter snapshots to only include data known at run_ts
            warnings = []
            injuries_slice = _filter_snapshot_by_asof(
                _filter_by_game_ids(injuries_df, all_game_ids),
                time_col="as_of_ts",
                run_as_of_ts=run_ts,
                tip_lookup=tip_lookup,
                dataset_name="injuries_snapshot",
                warnings=warnings,
            )
            
            odds_slice = _filter_snapshot_by_asof(
                _filter_by_game_ids(odds_df, all_game_ids),
                time_col="as_of_ts",
                run_as_of_ts=run_ts,
                tip_lookup=tip_lookup,
                dataset_name="odds_snapshot",
                warnings=warnings,
            )
            
            # Roster logic is a bit more complex in live pipeline (fallback days etc)
            # For backfill, we'll use a simplified version of _select_roster_slice logic
            # We just want the latest snapshot before run_ts
            roster_working = _filter_by_game_ids(roster_df, all_game_ids)
            roster_slice = _filter_snapshot_by_asof(
                roster_working,
                time_col="as_of_ts",
                run_as_of_ts=run_ts,
                tip_lookup=tip_lookup,
                dataset_name="roster_nightly",
                warnings=warnings,
            )

            # 4. Build Features
            builder = MinutesFeatureBuilder(
                schedule=schedule_slice,
                injuries_snapshot=injuries_slice,
                odds_snapshot=odds_slice,
                roster_nightly=roster_slice,
                coach_tenure=coach_df,
                archetype_roles=roles_df,
                archetype_deltas=archetype_deltas_df,
            )
            
            try:
                raw_features = builder.build(day_labels)
                deduped = deduplicate_latest(raw_features, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
                aligned = enforce_schema(deduped, FEATURES_MINUTES_V1_SCHEMA)
                # We skip pandera validation for speed in backfill, or keep it if safety > speed
                # validate_with_pandera(aligned, FEATURES_MINUTES_V1_SCHEMA)
                
                season_features.append(aligned)
            except Exception as e:
                typer.echo(f"Error building features for {target_day.date()}: {e}", err=True)
                continue

        # 5. Write Season Partition
        if season_features:
            combined = pd.concat(season_features, ignore_index=True)
            combined = combined.sort_values(["game_date", "game_id", "player_id"])
            
            out_dir = out_root / f"season={season_start}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "features.parquet"
            
            if out_path.exists() and not force:
                typer.echo(f"Output exists at {out_path}, skipping (use --force to overwrite)")
            else:
                combined.to_parquet(out_path, index=False)
                typer.echo(f"Wrote {len(combined)} rows to {out_path}")
        else:
            typer.echo(f"No features generated for season {season_label}")

if __name__ == "__main__":
    app()
