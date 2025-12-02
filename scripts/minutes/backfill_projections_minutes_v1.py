"""
Backfill minutes_v1 projections into gold/projections_minutes_v1 using the live scoring pipeline.

Example (2023-24 regular season):
    uv run python -m scripts.minutes.backfill_projections_minutes_v1 \
      --data-root   /home/daniel/projections-data \
      --start-date  2023-10-01 \
      --end-date    2024-06-30 \
      --season-type "Regular Season"

Force a single-day rescore with debug (overwrites existing outputs):
    uv run python -m scripts.minutes.backfill_projections_minutes_v1 \
      --data-root /home/daniel/projections-data \
      --start-date 2023-10-24 \
      --end-date   2023-10-24 \
      --overwrite-existing

Skip known feature-desert dates (from find_feature_desert_dates):
    uv run python -m scripts.minutes.backfill_projections_minutes_v1 \
      --data-root /home/daniel/projections-data \
      --start-date 2022-10-01 \
      --end-date   2024-06-30 \
      --desert-csv /home/daniel/projections-data/artifacts/minutes_v1/feature_deserts.csv \
      --skip-desert
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from projections.cli import score_minutes_v1
from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _iter_days(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _existing_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return 0


def _remove_existing(day: date, out_root: Path) -> int:
    """Remove any existing outputs for a day; return rows removed if known."""

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
                # Directory not empty or other files present; leave it.
                pass
    return total_rows


def _mirror_partition(day_df: pd.DataFrame, out_root: Path, day: date) -> None:
    """Ensure a game_date=YYYY-MM-DD partition exists for downstream readers."""

    partition_dir = out_root / f"game_date={day.isoformat()}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    partition_path = partition_dir / score_minutes_v1.OUTPUT_FILENAME
    day_df.to_parquet(partition_path, index=False)


def _copy_daily_minutes_into_gold(day: date, out_root: Path) -> Optional[pd.DataFrame]:
    """Fallback: mirror an existing daily minutes run into gold when scoring fails."""

    daily_dir = score_minutes_v1.DEFAULT_DAILY_ROOT / day.isoformat()
    if not daily_dir.exists():
        return None

    run_id: str | None = None
    latest_pointer = daily_dir / score_minutes_v1.LATEST_POINTER
    if latest_pointer.exists():
        try:
            payload = json.loads(latest_pointer.read_text(encoding="utf-8"))
            run_id = payload.get("run_id")
        except json.JSONDecodeError:
            run_id = None
    if run_id is None:
        runs = sorted(daily_dir.glob("run=*"))
        if runs:
            run_id = runs[-1].name.split("=", 1)[1]
    if run_id is None:
        return None

    minutes_path = daily_dir / f"run={run_id}" / score_minutes_v1.OUTPUT_FILENAME
    if not minutes_path.exists():
        return None

    df = pd.read_parquet(minutes_path)
    _mirror_partition(df, out_root, day)
    return df


@app.command()
def main(
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD (inclusive)."),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD (inclusive)."),
    data_root: Optional[Path] = typer.Option(
        None, help="Root containing data (defaults to PROJECTIONS_DATA_ROOT or ./data)."
    ),
    season_type: str = typer.Option(
        "Regular Season", help="Season type label (pass-through for logging only)."
    ),
    overwrite_existing: bool = typer.Option(
        False,
        "--overwrite-existing/--no-overwrite-existing",
        help="Recompute even when a parquet already exists for a date.",
    ),
    bundle_run_id: Optional[str] = typer.Option(
        None,
        help="Optional minutes_lgbm run id (falls back to production bundle if omitted).",
    ),
    debug_describe: bool = typer.Option(
        False,
        "--debug-describe/--no-debug-describe",
        help="Print raw vs reconciled minutes_p50 describe() during scoring.",
    ),
    desert_csv: Optional[Path] = typer.Option(
        None,
        "--desert-csv",
        help="CSV of feature-desert dates (from find_feature_desert_dates); skip them when provided.",
    ),
    skip_desert: bool = typer.Option(
        True,
        "--skip-desert/--no-skip-desert",
        help="When a desert CSV is provided, skip dates flagged as desert/partial desert (default True).",
    ),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    out_root = root / "gold" / "projections_minutes_v1"
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve bundle (production by default).
    bundle_dir: Path | None = None
    if bundle_run_id:
        candidate = Path(bundle_run_id)
        if not candidate.is_absolute():
            candidate = Path("artifacts/minutes_lgbm") / bundle_run_id
        bundle_dir = candidate
    bundle, resolved_bundle_dir, model_meta, model_run_id = score_minutes_v1._resolve_bundle_artifacts(
        bundle_dir, score_minutes_v1.DEFAULT_BUNDLE_CONFIG
    )

    features_root = root / "gold" / "features_minutes_v1"
    injuries_root = root / "bronze" / "injuries_raw"
    schedule_root = root / "silver" / "schedule"
    desert_dates: set[date] = set()
    if desert_csv:
        df_desert = pd.read_csv(desert_csv)
        if "game_date" in df_desert.columns:
            df_desert["game_date"] = pd.to_datetime(df_desert["game_date"]).dt.date
            desert_col = df_desert.get("is_desert")
            partial_col = df_desert.get("is_partial_desert")
            mask = (
                (desert_col.fillna(False).astype(bool) if desert_col is not None else False)
                | (partial_col.fillna(False).astype(bool) if partial_col is not None else False)
            )
            desert_dates = set(df_desert.loc[mask, "game_date"].tolist())
            # Only dates explicitly listed are skipped; ranges beyond the CSV contents are still scored.

    target_days: list[date] = []
    skipped_existing = 0
    overwritten = 0
    skipped_desert = 0
    for day in _iter_days(start, end):
        if desert_dates and day in desert_dates and skip_desert:
            typer.echo(f"[backfill] {day}: marked as feature-desert; skipping minutes scoring.")
            skipped_desert += 1
            continue
        parquet_path = out_root / day.isoformat() / score_minutes_v1.OUTPUT_FILENAME
        mirror_path = out_root / f"game_date={day.isoformat()}" / score_minutes_v1.OUTPUT_FILENAME
        existing = _existing_rows(parquet_path) or _existing_rows(mirror_path)
        if existing > 0 and not overwrite_existing:
            typer.echo(f"[backfill] {day}: skipping existing ({existing} rows)")
            skipped_existing += 1
            continue
        if existing > 0 and overwrite_existing:
            removed_rows = _remove_existing(day, out_root)
            typer.echo(f"[backfill] {day}: removing existing outputs (rows~{removed_rows}) due to overwrite_existing=True")
            overwritten += 1
        target_days.append(day)

    if not target_days:
        typer.echo("[backfill] nothing to do; all dates already present.")
        raise typer.Exit()

    min_day, max_day = min(target_days), max(target_days)
    typer.echo(
        f"[backfill] scoring {min_day} to {max_day} season_type='{season_type}' "
        f"using bundle={resolved_bundle_dir} (run_id={model_run_id}) into {out_root} "
        f"overwrite_existing={overwrite_existing} debug_describe={debug_describe}"
    )

    written_frames: list[pd.DataFrame] = []
    scored_dates = 0
    fallback_dates = 0
    desert_dates = 0
    for day in target_days:
        try:
            score_minutes_v1.score_minutes_range_to_parquet(
                day,
                day,
                features_root=features_root,
                features_path=None,
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
                debug_describe=debug_describe if debug_describe is not None else (len(target_days) == 1),
            )
            scored_dates += 1
        except Exception as exc:  # noqa: BLE001
            typer.echo(
                f"[backfill] {day}: scoring failed ({exc}); attempting fallback copy from daily artifacts.",
                err=True,
            )
            fallback_df = _copy_daily_minutes_into_gold(day, out_root)
            if fallback_df is not None:
                typer.echo(
                    f"[backfill] {day}: mirrored minutes from {score_minutes_v1.DEFAULT_DAILY_ROOT / day.isoformat()} into gold."
                )
                written_frames.append(fallback_df)
                fallback_dates += 1
                continue
            typer.echo(f"[backfill] {day}: no daily artifacts found; marking as missing.", err=True)
            desert_dates += 1
            continue

        day_path = out_root / day.isoformat() / score_minutes_v1.OUTPUT_FILENAME
        if not day_path.exists():
            alt_path = out_root / f"game_date={day.isoformat()}" / score_minutes_v1.OUTPUT_FILENAME
            day_path = alt_path if alt_path.exists() else day_path
        if not day_path.exists():
            fallback_df = _copy_daily_minutes_into_gold(day, out_root)
            if fallback_df is not None:
                typer.echo(
                    f"[backfill] {day}: scoring succeeded but file missing; mirrored minutes from daily artifacts."
                )
                written_frames.append(fallback_df)
                fallback_dates += 1
                continue
            typer.echo(
                f"[backfill] {day}: scoring succeeded but no output file and no fallback available; marking as missing.",
                err=True,
            )
            desert_dates += 1
            continue

        day_df = pd.read_parquet(day_path)
        _mirror_partition(day_df, out_root, day)
        written_frames.append(day_df)

    typer.echo(
        f"[backfill] complete. dates_scored={len(target_days)} written={len(written_frames)} "
        f"skipped_existing={skipped_existing} skipped_desert={skipped_desert} overwritten={overwritten} "
        f"scored_dates={scored_dates} mirrored_fallback={fallback_dates} missing={desert_dates}"
    )
    if written_frames:
        combined = pd.concat(written_frames, ignore_index=True)
        if not combined.empty:
            stats = combined[["minutes_p50", "play_prob"]].describe()
            typer.echo("[backfill] minutes_p50/play_prob stats for written dates:")
            typer.echo(stats.to_string())


if __name__ == "__main__":
    app()
