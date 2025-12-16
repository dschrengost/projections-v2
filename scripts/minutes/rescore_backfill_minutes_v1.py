"""
Rescore minutes predictions for historical dates using the CURRENT production model.

This orchestrator:
1. Validates snapshot availability (injuries, roster) for each date
2. Rebuilds features using the current feature builder (if missing)
3. Scores using the current production minutes model
4. Writes to gold/projections_minutes_v1 with consistent run_id

Usage:
    # Dry run - see what would be done
    uv run python -m scripts.minutes.rescore_backfill_minutes_v1 \
        --data-root /home/daniel/projections-data \
        --start-date 2024-12-01 \
        --end-date 2024-12-20 \
        --dry-run

    # Execute for a date range
    uv run python -m scripts.minutes.rescore_backfill_minutes_v1 \
        --data-root /home/daniel/projections-data \
        --start-date 2022-10-20 \
        --end-date 2025-02-01 \
        --force
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.cli import score_minutes_v1
from projections.paths import data_path

app = typer.Typer(add_completion=False, help=__doc__)


@dataclass
class DateAudit:
    """Audit result for a single date."""
    game_date: date
    num_games: int
    injuries_covered: int
    roster_covered: int
    can_process: bool
    skip_reason: str | None = None


def _season_from_date(d: date) -> int:
    """NBA season start year (Aug-Jul)."""
    return d.year if d.month >= 8 else d.year - 1


def _iter_days(start: date, end: date):
    """Iterate over days in [start, end] inclusive."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _load_schedule(data_root: Path, start: date, end: date) -> pd.DataFrame:
    """Load schedule for the date range."""
    schedule_root = data_root / "silver" / "schedule"
    frames: list[pd.DataFrame] = []
    for season_dir in schedule_root.glob("season=*"):
        for month_dir in season_dir.glob("month=*"):
            path = month_dir / "schedule.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["game_id", "game_date", "tip_ts"])
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    start_ts = pd.Timestamp(start).date()
    end_ts = pd.Timestamp(end).date()
    df = df[(df["game_date"] >= start_ts) & (df["game_date"] <= end_ts)]
    return df


def _load_silver_data(data_root: Path, dataset: str, season: int) -> pd.DataFrame:
    """Load all silver data for a dataset and season."""
    root = data_root / "silver" / dataset / f"season={season}"
    if not root.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for month_dir in root.glob("month=*"):
        for pq_file in month_dir.glob("*.parquet"):
            try:
                frames.append(pd.read_parquet(pq_file))
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _audit_dates(
    data_root: Path,
    start: date,
    end: date,
) -> list[DateAudit]:
    """Audit snapshot availability for each date."""
    # Load schedule
    schedule = _load_schedule(data_root, start, end)
    if schedule.empty:
        return []

    # Group games by date
    games_by_date: dict[date, set[int]] = {}
    for _, row in schedule.iterrows():
        d = row["game_date"]
        gid = int(row["game_id"])
        if d not in games_by_date:
            games_by_date[d] = set()
        games_by_date[d].add(gid)

    # Pre-load silver datasets per season
    seasons = sorted(set(_season_from_date(d) for d in _iter_days(start, end)))
    silver_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for season in seasons:
        for dataset in ["injuries_snapshot", "roster_nightly"]:
            silver_cache[(dataset, season)] = _load_silver_data(data_root, dataset, season)

    results: list[DateAudit] = []
    for day in _iter_days(start, end):
        game_ids = games_by_date.get(day, set())
        num_games = len(game_ids)

        if num_games == 0:
            continue

        season = _season_from_date(day)
        injuries_df = silver_cache.get(("injuries_snapshot", season), pd.DataFrame())
        roster_df = silver_cache.get(("roster_nightly", season), pd.DataFrame())

        # Count games with coverage
        injuries_covered = 0
        roster_covered = 0

        if not injuries_df.empty and "game_id" in injuries_df.columns:
            injury_games = set(pd.to_numeric(injuries_df["game_id"], errors="coerce").dropna().astype(int).unique())
            injuries_covered = len(game_ids & injury_games)

        if not roster_df.empty and "game_id" in roster_df.columns:
            roster_games = set(pd.to_numeric(roster_df["game_id"], errors="coerce").dropna().astype(int).unique())
            roster_covered = len(game_ids & roster_games)

        # Determine if we can process
        can_process = injuries_covered == num_games and roster_covered == num_games
        skip_reason = None
        if injuries_covered < num_games:
            skip_reason = f"missing injuries ({injuries_covered}/{num_games})"
        elif roster_covered < num_games:
            skip_reason = f"missing roster ({roster_covered}/{num_games})"

        results.append(DateAudit(
            game_date=day,
            num_games=num_games,
            injuries_covered=injuries_covered,
            roster_covered=roster_covered,
            can_process=can_process,
            skip_reason=skip_reason,
        ))

    return results


def _existing_rows(path: Path) -> int:
    """Count rows in existing parquet file."""
    if not path.exists():
        return 0
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return 0


def _remove_existing(day: date, out_root: Path) -> int:
    """Remove any existing outputs for a day; return rows removed."""
    total_rows = 0
    for candidate_dir in [out_root / day.isoformat(), out_root / f"game_date={day.isoformat()}"]:
        parquet_path = candidate_dir / score_minutes_v1.OUTPUT_FILENAME
        rows = _existing_rows(parquet_path)
        total_rows += rows
        if parquet_path.exists():
            parquet_path.unlink(missing_ok=True)
        if candidate_dir.exists():
            try:
                candidate_dir.rmdir()
            except OSError:
                pass
    return total_rows


def _mirror_partition(day_df: pd.DataFrame, out_root: Path, day: date) -> None:
    """Ensure a game_date=YYYY-MM-DD partition exists."""
    partition_dir = out_root / f"game_date={day.isoformat()}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    partition_path = partition_dir / score_minutes_v1.OUTPUT_FILENAME
    day_df.to_parquet(partition_path, index=False)


def _get_features_path_for_date(features_root: Path, day: date) -> Path | None:
    """
    Get the path to features for a specific date.
    
    Returns:
        Path to feature file, or None if not found.
        Checks both seasonal partitions and run-based live features.
    """
    # First check seasonal partition
    season = _season_from_date(day)
    seasonal_path = features_root / f"season={season}" / f"month={day.month:02d}" / "features.parquet"
    if seasonal_path.exists():
        try:
            df = pd.read_parquet(seasonal_path, columns=["game_date"])
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
            if (df["game_date"] == day).any():
                return seasonal_path
        except Exception:
            pass
    
    # Check run-based live features (from build_minutes_live)
    day_dir = features_root / day.isoformat()
    if day_dir.exists():
        # Check for latest_run.json to find the run ID
        latest_pointer = day_dir / "latest_run.json"
        if latest_pointer.exists():
            try:
                import json
                payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
                run_id = payload.get("run_id")
                if run_id:
                    run_path = day_dir / f"run={run_id}" / "features.parquet"
                    if run_path.exists():
                        return run_path
            except Exception:
                pass
        # Fall back to scanning for runs
        runs = sorted(day_dir.glob("run=*/features.parquet"))
        if runs:
            return runs[-1]  # Most recent run
    
    return None


def _build_features_for_date(
    data_root: Path,
    day: date,
    schedule_df: pd.DataFrame,
) -> Path | None:
    """
    Build features for a date using build_minutes_live in backfill mode.
    
    Returns path to built features file, or None if build failed.
    """
    # Get tip_ts from schedule to compute run_as_of_ts
    day_games = schedule_df[schedule_df["game_date"] == day]
    if day_games.empty:
        return False
    
    # Use max tip_ts + 1 day as run_as_of_ts (ensure we have post-game data)
    tip_ts = pd.to_datetime(day_games["tip_ts"], utc=True, errors="coerce")
    max_tip = tip_ts.max()
    if pd.isna(max_tip):
        # If no tip_ts, use end of day
        run_as_of_ts = f"{day.isoformat()}T23:59:59"
    else:
        # Format without timezone for CLI compatibility
        run_as_of_ts = (max_tip + pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
    
    # Build features using CLI (via subprocess for isolation)
    cmd = [
        sys.executable, "-m", "projections.cli.build_minutes_live",
        "--date", day.isoformat(),
        "--data-root", str(data_root),
        "--out-root", str(data_root / "gold" / "features_minutes_v1"),
        "--run-as-of-ts", run_as_of_ts,
        "--backfill-mode",
        "--skip-active-roster",
    ]
    
    typer.echo(f"    Building features with: build_minutes_live --date {day} --run-as-of-ts {run_as_of_ts}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        if result.returncode != 0:
            typer.echo(f"    Feature build failed: {result.stderr[:500]}", err=True)
            return None
        # Return path to built features
        features_root = data_root / "gold" / "features_minutes_v1"
        return _get_features_path_for_date(features_root, day)
    except subprocess.TimeoutExpired:
        typer.echo(f"    Feature build timed out after 300s", err=True)
        return None
    except Exception as exc:
        typer.echo(f"    Feature build error: {exc}", err=True)
        return None


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)."),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)."),
    data_root: Optional[Path] = typer.Option(
        None, help="Root containing data (defaults to PROJECTIONS_DATA_ROOT or ./data)."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Only print what would be done without executing.",
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Recompute even when a parquet already exists for a date.",
    ),
    skip_missing_injuries: bool = typer.Option(
        True,
        "--skip-missing-injuries/--no-skip-missing-injuries",
        help="Skip dates with missing injury snapshots (default True).",
    ),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    out_root = root / "gold" / "projections_minutes_v1"
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve production bundle and log run_id
    bundle, resolved_bundle_dir, model_meta, model_run_id = score_minutes_v1._resolve_bundle_artifacts(
        None, score_minutes_v1.DEFAULT_BUNDLE_CONFIG
    )
    typer.echo(f"[minutes_backfill] using production model run_id: {model_run_id}")
    typer.echo(f"[minutes_backfill] bundle_dir: {resolved_bundle_dir}")

    # Audit all dates
    typer.echo(f"[minutes_backfill] auditing dates {start} to {end}...")
    audits = _audit_dates(root, start, end)
    if not audits:
        typer.echo("[minutes_backfill] no dates with games found in range")
        raise typer.Exit()

    # Categorize dates
    to_process: list[DateAudit] = []
    to_skip: list[DateAudit] = []
    already_done: list[date] = []

    for audit in audits:
        if not audit.can_process and skip_missing_injuries:
            to_skip.append(audit)
            continue

        # Check if already exists
        parquet_path = out_root / audit.game_date.isoformat() / score_minutes_v1.OUTPUT_FILENAME
        mirror_path = out_root / f"game_date={audit.game_date.isoformat()}" / score_minutes_v1.OUTPUT_FILENAME
        existing = _existing_rows(parquet_path) or _existing_rows(mirror_path)

        if existing > 0 and not force:
            already_done.append(audit.game_date)
            continue

        to_process.append(audit)

    # Summary
    typer.echo(f"[minutes_backfill] total dates with games: {len(audits)}")
    typer.echo(f"[minutes_backfill] to process: {len(to_process)}")
    typer.echo(f"[minutes_backfill] already done (skipping): {len(already_done)}")
    typer.echo(f"[minutes_backfill] skipped (missing snapshots): {len(to_skip)}")

    if to_skip:
        typer.echo("\n[minutes_backfill] SKIPPED dates:")
        for audit in to_skip[:10]:
            typer.echo(f"  {audit.game_date}: {audit.skip_reason}")
        if len(to_skip) > 10:
            typer.echo(f"  ... and {len(to_skip) - 10} more")

    if dry_run:
        typer.echo("\n[minutes_backfill] DRY RUN - dates that would be processed:")
        for audit in to_process[:20]:
            typer.echo(f"  {audit.game_date}: {audit.num_games} games")
        if len(to_process) > 20:
            typer.echo(f"  ... and {len(to_process) - 20} more")
        typer.echo(f"\n[minutes_backfill] Would score {len(to_process)} dates with run_id={model_run_id}")
        raise typer.Exit()

    if not to_process:
        typer.echo("[minutes_backfill] nothing to do")
        raise typer.Exit()

    # Process dates
    features_root = root / "gold" / "features_minutes_v1"
    injuries_root = root / "bronze" / "injuries_raw"
    schedule_root = root / "silver" / "schedule"
    
    # Load schedule for feature building
    schedule_df = _load_schedule(root, start, end)

    scored = 0
    failed = 0
    features_built = 0
    results: list[dict] = []

    for i, audit in enumerate(to_process, 1):
        day = audit.game_date
        typer.echo(f"\n[{i}/{len(to_process)}] Scoring {day} ({audit.num_games} games)...")

        if force:
            removed = _remove_existing(day, out_root)
            if removed:
                typer.echo(f"  removed existing ({removed} rows)")

        # Get features path for this date
        features_path = _get_features_path_for_date(features_root, day)
        
        # Build features if missing
        if features_path is None:
            typer.echo(f"  features missing for {day}, attempting to build...")
            features_path = _build_features_for_date(root, day, schedule_df)
            if features_path is not None:
                typer.echo(f"  features built: {features_path}")
                features_built += 1
            else:
                results.append({"date": day, "status": "FAILED", "error": "feature build failed"})
                typer.echo(f"  FAILED: could not build features", err=True)
                failed += 1
                continue

        try:
            score_minutes_v1.score_minutes_range_to_parquet(
                day,
                day,
                features_root=features_root,
                features_path=features_path,
                bundle_dir=resolved_bundle_dir,
                bundle_config=score_minutes_v1.DEFAULT_BUNDLE_CONFIG,
                artifact_root=out_root,
                injuries_root=injuries_root,
                schedule_root=schedule_root,
                limit_rows=None,
                mode="historical",
                run_id=None,
                live_features_root=score_minutes_v1.DEFAULT_LIVE_FEATURES_ROOT,
                minutes_output="both",
                starter_priors_path=score_minutes_v1.DEFAULT_STARTER_PRIORS,
                starter_history_path=score_minutes_v1.DEFAULT_STARTER_HISTORY,
                promotion_config=score_minutes_v1.DEFAULT_PROMOTION_CONFIG,
                promotion_prior_enabled=True,
                promotion_prior_debug=False,
                reconcile_team_minutes="none",
                reconcile_config=score_minutes_v1.DEFAULT_RECONCILE_CONFIG,
                reconcile_debug=False,
                prediction_logs_root=score_minutes_v1.DEFAULT_PREDICTION_LOGS_ROOT,
                disable_play_prob=False,
                target_dates={day},
                debug_describe=False,
            )

            # Verify output
            day_path = out_root / day.isoformat() / score_minutes_v1.OUTPUT_FILENAME
            if not day_path.exists():
                alt_path = out_root / f"game_date={day.isoformat()}" / score_minutes_v1.OUTPUT_FILENAME
                day_path = alt_path if alt_path.exists() else day_path

            if day_path.exists():
                day_df = pd.read_parquet(day_path)
                _mirror_partition(day_df, out_root, day)

                # Stats
                n_rows = len(day_df)
                p50_notna = day_df["minutes_p50"].notna().mean() * 100 if "minutes_p50" in day_df.columns else 0
                play_prob_notna = day_df["play_prob"].notna().mean() * 100 if "play_prob" in day_df.columns else 0

                results.append({
                    "date": day,
                    "status": "OK",
                    "rows": n_rows,
                    "minutes_p50_%": f"{p50_notna:.0f}%",
                    "play_prob_%": f"{play_prob_notna:.0f}%",
                    "run_id": model_run_id,
                })
                typer.echo(f"  OK: {n_rows} rows, minutes_p50={p50_notna:.0f}%, play_prob={play_prob_notna:.0f}%")
                scored += 1
            else:
                results.append({"date": day, "status": "MISSING", "rows": 0})
                typer.echo("  MISSING: output file not created", err=True)
                failed += 1

        except Exception as exc:
            results.append({"date": day, "status": "FAILED", "error": str(exc)[:50]})
            typer.echo(f"  FAILED: {exc}", err=True)
            failed += 1

    # Final summary
    typer.echo("\n" + "=" * 60)
    typer.echo("[minutes_backfill] SUMMARY")
    typer.echo("=" * 60)
    typer.echo(f"Model run_id: {model_run_id}")
    typer.echo(f"Dates processed: {scored + failed}")
    typer.echo(f"  Succeeded: {scored}")
    typer.echo(f"  Failed: {failed}")
    typer.echo(f"  Features built: {features_built}")
    typer.echo(f"  Skipped (missing data): {len(to_skip)}")
    typer.echo(f"  Skipped (already done): {len(already_done)}")

    if failed > 0:
        typer.echo("\nFailed dates:")
        for r in results:
            if r.get("status") in ("FAILED", "MISSING"):
                typer.echo(f"  {r['date']}: {r.get('error', r['status'])}")


if __name__ == "__main__":
    app()
