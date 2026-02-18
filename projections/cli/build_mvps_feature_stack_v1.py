"""Build the full MVPS feature stack for one or more dates.

This is a convenience orchestrator around:
1. `projections.cli.build_minutes_live`
2. `projections.cli.build_rotation_set_minutes_features_v1`
3. `projections.cli.build_mvps_participation_features_v1`

It ensures all three layers are built with the same run id per date.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import typer

from projections import paths

app = typer.Typer(help=__doc__)


def _parse_day(value: str) -> date:
    return pd.Timestamp(value).date()


def _season_from_day(day: date) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _iter_days(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _has_games_for_day(data_root: Path, day: date) -> bool | None:
    season = _season_from_day(day)
    schedule_path = data_root / "silver" / "schedule" / f"season={season}" / f"month={day.month:02d}" / "schedule.parquet"
    if not schedule_path.exists():
        return None
    try:
        df = pd.read_parquet(schedule_path, columns=["game_date"])
    except Exception:
        return None
    if df.empty:
        return False
    day_series = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    return bool((day_series == day).any())


def _max_tip_ts_for_day(data_root: Path, day: date) -> pd.Timestamp | None:
    season = _season_from_day(day)
    schedule_path = data_root / "silver" / "schedule" / f"season={season}" / f"month={day.month:02d}" / "schedule.parquet"
    if not schedule_path.exists():
        return None
    try:
        df = pd.read_parquet(schedule_path, columns=["game_date", "tip_ts"])
    except Exception:
        return None
    if df.empty:
        return None
    game_day = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    day_df = df.loc[game_day == day].copy()
    if day_df.empty or "tip_ts" not in day_df.columns:
        return None
    tips = pd.to_datetime(day_df["tip_ts"], utc=True, errors="coerce").dropna()
    if tips.empty:
        return None
    return pd.Timestamp(tips.max())


def _run_cmd(cmd: list[str], *, dry_run: bool, env_overrides: dict[str, str] | None = None) -> None:
    typer.echo(f"[mvps-stack] $ {shlex.join(cmd)}")
    if dry_run:
        return
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    subprocess.run(cmd, check=True, env=env)


@app.command()
def main(
    start_date: str = typer.Option(..., "--start-date", help="Start date (inclusive), YYYY-MM-DD."),
    end_date: str | None = typer.Option(None, "--end-date", help="End date (inclusive), YYYY-MM-DD."),
    run_id: str | None = typer.Option(None, "--run-id", help="Shared run id for all outputs. Defaults to UTC timestamp."),
    data_root: Path = typer.Option(paths.get_data_root(), "--data-root", help="PROJECTIONS_DATA_ROOT override."),
    backfill_mode: bool = typer.Option(True, "--backfill-mode/--live-mode", help="Use historical-safe minutes feature settings."),
    skip_active_roster: bool = typer.Option(
        True,
        "--skip-active-roster/--validate-active-roster",
        help="Skip NBA.com active-roster validation (recommended for backfills).",
    ),
    run_as_of_ts: str | None = typer.Option(
        None,
        "--run-as-of-ts",
        help=(
            "Override run_as_of_ts passed to build_minutes_live for every date. "
            "If omitted and --backfill-mode is set, a per-date historical timestamp is used."
        ),
    ),
    backfill_run_asof_offset_hours: int = typer.Option(
        24,
        "--backfill-run-asof-offset-hours",
        help="When --backfill-mode and --run-as-of-ts is not set, use (max_tip_ts + this many hours).",
    ),
    lock_buffer_minutes: int = typer.Option(
        0,
        "--lock-buffer-minutes",
        min=0,
        help="Skip games tipped more than this many minutes before run_as_of_ts (forwarded to build_minutes_live).",
    ),
    roster_fallback_days: int | None = typer.Option(
        None,
        "--roster-fallback-days",
        min=0,
        help="Optional pass-through override to build_minutes_live --roster-fallback-days.",
    ),
    roster_max_age_hours: int | None = typer.Option(
        None,
        "--roster-max-age-hours",
        min=1,
        help="Optional pass-through override to build_minutes_live --roster-max-age-hours.",
    ),
    rotation_model_dir: Path | None = typer.Option(
        None,
        "--rotation-model-dir",
        help="Optional model dir for build_rotation_set_minutes_features_v1 --model-dir.",
    ),
    model_feature_columns: Path | None = typer.Option(
        None,
        "--model-feature-columns",
        help="Optional path to MVPS participation feature_columns.json.",
    ),
    allow_unsafe_pointer_writes: bool = typer.Option(
        False,
        "--allow-unsafe-pointer-writes",
        help="Allow latest_run.json pointer writes outside Prefect writer-guard.",
    ),
    skip_pointer_writes: bool = typer.Option(
        True,
        "--skip-pointer-writes/--write-pointers",
        help="Set PROJECTIONS_SKIP_POINTER_WRITES=1 for subprocesses (recommended for backfills).",
    ),
    refresh_rotation_priors: bool = typer.Option(
        False,
        "--refresh-rotation-priors",
        help="Run scripts/rotation/build_rotation_priors_v1.py once before date loop.",
    ),
    rotation_priors_overwrite: bool = typer.Option(
        False,
        "--rotation-priors-overwrite",
        help="When refreshing priors, pass --overwrite.",
    ),
    skip_no_games: bool = typer.Option(
        True,
        "--skip-no-games/--include-no-games",
        help="Skip dates with zero scheduled games when schedule data exists.",
    ),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error",
        help="Continue remaining dates if one date fails.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without executing."),
) -> None:
    root = Path(data_root).expanduser().resolve()
    start = _parse_day(start_date)
    end = _parse_day(end_date) if end_date else start
    if end < start:
        raise typer.BadParameter("--end-date must be on/after --start-date.")

    resolved_run_id = run_id or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    days = list(_iter_days(start, end))
    typer.echo(
        f"[mvps-stack] dates={start.isoformat()}..{end.isoformat()} n_days={len(days)} run_id={resolved_run_id} root={root}"
    )

    if refresh_rotation_priors:
        priors_cmd = [sys.executable, "scripts/rotation/build_rotation_priors_v1.py", "--data-root", str(root)]
        if rotation_priors_overwrite:
            priors_cmd.append("--overwrite")
        priors_env: dict[str, str] = {}
        if skip_pointer_writes:
            priors_env["PROJECTIONS_SKIP_POINTER_WRITES"] = "1"
        elif allow_unsafe_pointer_writes:
            priors_env["PROJECTIONS_ALLOW_UNSAFE_POINTER_WRITES"] = "1"
        _run_cmd(priors_cmd, dry_run=dry_run, env_overrides=priors_env)

    processed = 0
    skipped = 0
    failed = 0
    for day in days:
        has_games = _has_games_for_day(root, day)
        if skip_no_games and has_games is False:
            typer.echo(f"[mvps-stack] skip date={day.isoformat()} reason=no_games")
            skipped += 1
            continue

        day_str = day.isoformat()
        typer.echo(f"[mvps-stack] building date={day_str}")
        try:
            cmd_env: dict[str, str] = {}
            if skip_pointer_writes:
                cmd_env["PROJECTIONS_SKIP_POINTER_WRITES"] = "1"
            elif allow_unsafe_pointer_writes:
                cmd_env["PROJECTIONS_ALLOW_UNSAFE_POINTER_WRITES"] = "1"

            resolved_run_asof: str | None = run_as_of_ts
            if resolved_run_asof is None and backfill_mode:
                tip_max = _max_tip_ts_for_day(root, day)
                if tip_max is not None:
                    resolved_run_asof = (tip_max + pd.Timedelta(hours=int(backfill_run_asof_offset_hours))).strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    )
                else:
                    fallback_ts = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1)
                    resolved_run_asof = fallback_ts.strftime("%Y-%m-%dT%H:%M:%S")

            minutes_cmd = [
                sys.executable,
                "-m",
                "projections.cli.build_minutes_live",
                "--date",
                day_str,
                "--run-id",
                resolved_run_id,
                "--data-root",
                str(root),
                "--lock-buffer-minutes",
                str(int(lock_buffer_minutes)),
            ]
            if resolved_run_asof is not None:
                minutes_cmd.extend(["--run-as-of-ts", resolved_run_asof])
            if backfill_mode:
                minutes_cmd.append("--backfill-mode")
            if skip_active_roster:
                minutes_cmd.append("--skip-active-roster")
            if roster_fallback_days is not None:
                minutes_cmd.extend(["--roster-fallback-days", str(int(roster_fallback_days))])
            if roster_max_age_hours is not None:
                minutes_cmd.extend(["--roster-max-age-hours", str(int(roster_max_age_hours))])
            _run_cmd(minutes_cmd, dry_run=dry_run, env_overrides=cmd_env)

            rotation_cmd = [
                sys.executable,
                "-m",
                "projections.cli.build_rotation_set_minutes_features_v1",
                "--date",
                day_str,
                "--run-id",
                resolved_run_id,
                "--data-root",
                str(root),
            ]
            if rotation_model_dir is not None:
                rotation_cmd.extend(["--model-dir", str(Path(rotation_model_dir).expanduser().resolve())])
            _run_cmd(rotation_cmd, dry_run=dry_run, env_overrides=cmd_env)

            part_cmd = [
                sys.executable,
                "-m",
                "projections.cli.build_mvps_participation_features_v1",
                "--date",
                day_str,
                "--run-id",
                resolved_run_id,
                "--data-root",
                str(root),
            ]
            if model_feature_columns is not None:
                part_cmd.extend(["--model-feature-columns", str(Path(model_feature_columns).expanduser().resolve())])
            if allow_unsafe_pointer_writes:
                part_cmd.append("--allow-unsafe-pointer-writes")
            _run_cmd(part_cmd, dry_run=dry_run, env_overrides=cmd_env)

            processed += 1
        except subprocess.CalledProcessError as exc:
            failed += 1
            typer.echo(f"[mvps-stack] FAILED date={day_str} returncode={exc.returncode}", err=True)
            if not continue_on_error:
                raise

    typer.echo(f"[mvps-stack] done processed={processed} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    app()
