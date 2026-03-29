"""Prefect flow to build rotation_v1 and rotation_priors_v1.

This runs after the daily gamerotation scrape to refresh silver rotation data
and derived priors used by the rotation overlay minutes model.
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from prefect import flow, get_run_logger, task

from projections import paths

PROJECT_ROOT = paths.get_project_root()
TRACKING_REQUIRED_MEASURE_TYPES: tuple[str, ...] = (
    "Possessions",
    "Passing",
    "Drives",
    "CatchShoot",
    "PullUpShot",
)


def _safe_get_run_logger():
    try:
        return get_run_logger()
    except Exception:  # noqa: BLE001
        return logging.getLogger("rotation-priors-update")


def _run_python_script(
    script_rel_path: str,
    args: list[str],
    *,
    data_root: Path,
    timeout_s: int,
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    cmd = [sys.executable, str(PROJECT_ROOT / script_rel_path), *args]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{script_rel_path} failed exit_code={result.returncode}")


def _default_pbp_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _default_pbp_season_id() -> str:
    # NBA season rolls in August.
    now_et = datetime.now(tz=ZoneInfo("America/New_York"))
    start_year = int(now_et.year) if int(now_et.month) >= 8 else int(now_et.year) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _default_tracking_target_date() -> date:
    now_et = datetime.now(tz=ZoneInfo("America/New_York")).date()
    return now_et - timedelta(days=1)


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _season_start_date_from_id(season_id: str) -> date:
    start_year, _ = _season_parts(season_id)
    return date(start_year, 10, 1)


def _collect_boxscore_dates(*, data_root: Path, season_id: str, as_of_date: date) -> set[date]:
    start_year, end_year = _season_parts(season_id)
    out: set[date] = set()
    for season_year in (start_year, end_year):
        season_dir = data_root / "bronze" / "boxscores_raw" / f"season={season_year}"
        if not season_dir.exists():
            continue
        for date_dir in season_dir.glob("date=*"):
            try:
                day = _parse_iso_date(date_dir.name.split("=", 1)[1])
            except Exception:  # noqa: BLE001
                continue
            if day > as_of_date:
                continue
            if (date_dir / "boxscores_raw.parquet").exists():
                out.add(day)
    return out


def _collect_tracking_dates(
    *,
    data_root: Path,
    season_id: str,
    as_of_date: date,
) -> tuple[set[date], dict[date, set[str]]]:
    season_dir = (
        data_root
        / "bronze"
        / "nba"
        / "tracking"
        / f"season={season_id}"
        / "season_type=Regular+Season"
    )
    dates: set[date] = set()
    measures_by_date: dict[date, set[str]] = {}
    if not season_dir.exists():
        return dates, measures_by_date
    for day_dir in season_dir.glob("game_date=*"):
        try:
            day = _parse_iso_date(day_dir.name.split("=", 1)[1])
        except Exception:  # noqa: BLE001
            continue
        if day > as_of_date:
            continue
        measures: set[str] = set()
        for measure_dir in day_dir.glob("pt_measure_type=*"):
            measure = measure_dir.name.split("=", 1)[-1]
            has_rows = any(measure_dir.glob("*.parquet"))
            if has_rows:
                measures.add(measure)
        if measures:
            dates.add(day)
            measures_by_date[day] = measures
    return dates, measures_by_date


def _season_parts(season_id: str) -> tuple[int, int]:
    text = str(season_id).strip()
    if "-" not in text:
        raise ValueError(f"Invalid season_id={season_id!r}; expected format YYYY-YY (e.g. 2025-26).")
    start_s, end_s = text.split("-", 1)
    start = int(start_s)
    end_two = int(end_s)
    end = 2000 + end_two if end_two < 100 else end_two
    if end < start:
        end += 100
    return start, end


def _resolve_pbp_input_glob(*, data_root: Path, season_id: str, explicit_glob: str | None) -> str:
    if explicit_glob:
        return explicit_glob

    start, end = _season_parts(season_id)
    season_token = f"{start}_{str(end)[-2:]}"
    logs_dir = (
        data_root
        / "bronze"
        / "pbp_vendor"
        / f"season_{season_token}"
        / f"{start}-{end}_NBA_PbP_Logs"
    )
    nested = str(logs_dir / "*" / "*.csv")
    flat = str(logs_dir / "*.csv")
    if glob.glob(nested):
        return nested
    if glob.glob(flat):
        return flat
    # Default to nested season subdir pattern for first-run daily fetches.
    return nested


@task(name="pbp-vendor-ingest", retries=1, retry_delay_seconds=300)
def pbp_vendor_ingest_task(
    *,
    data_root: Path,
    season_id: str,
    run_id: str,
    input_glob: str,
    resume: bool,
    overwrite: bool,
    skip_bad_games: bool,
    limit_games: int | None,
) -> Path:
    artifacts_root = data_root / "artifacts" / "pbp_v1"
    args: list[str] = [
        "--input-glob",
        input_glob,
        "--artifacts-root",
        str(artifacts_root),
        "--season-id",
        season_id,
        "--run-id",
        run_id,
    ]
    if resume:
        args.append("--resume")
    if overwrite:
        args.append("--overwrite")
    if skip_bad_games:
        args.append("--skip-bad-games")
    if limit_games is not None:
        args.extend(["--limit-games", str(int(limit_games))])
    _run_python_script(
        "projections/cli/pbp_vendor_ingest.py",
        args,
        data_root=data_root,
        timeout_s=7200,
    )
    return artifacts_root / run_id


@task(name="pbp-vendor-fetch-daily-zip", retries=2, retry_delay_seconds=300)
def pbp_vendor_fetch_daily_zip_task(
    *,
    data_root: Path,
    season_id: str,
    vendor_daily_url: str,
    timeout_seconds: int,
) -> None:
    args: list[str] = [
        "--season-id",
        season_id,
        "--url",
        vendor_daily_url,
        "--data-root",
        str(data_root),
        "--timeout-seconds",
        str(int(timeout_seconds)),
    ]
    _run_python_script(
        "projections/cli/pbp_vendor_fetch_daily_zip.py",
        args,
        data_root=data_root,
        timeout_s=max(int(timeout_seconds), 60) + 600,
    )


@task(name="pbp-build-stints", retries=1, retry_delay_seconds=180)
def pbp_build_stints_task(*, data_root: Path, bundle_dir: Path, overwrite: bool) -> None:
    args = [str(bundle_dir)]
    if overwrite:
        args.append("--overwrite")
    _run_python_script(
        "projections/cli/pbp_build_stints.py",
        args,
        data_root=data_root,
        timeout_s=3600,
    )


@task(name="pbp-qa", retries=1, retry_delay_seconds=120)
def pbp_qa_task(*, data_root: Path, bundle_dir: Path, allow_failures: bool) -> None:
    args = [str(bundle_dir)]
    if allow_failures:
        args.append("--allow-failures")
    _run_python_script(
        "projections/cli/pbp_qa.py",
        args,
        data_root=data_root,
        timeout_s=1800,
    )


@task(name="pbp-publish", retries=1, retry_delay_seconds=120)
def pbp_publish_task(*, data_root: Path, bundle_dir: Path, force: bool) -> None:
    args = [
        str(bundle_dir),
        "--artifacts-root",
        str(data_root / "artifacts" / "pbp_v1"),
    ]
    if force:
        args.append("--force")
    _run_python_script(
        "projections/cli/pbp_publish.py",
        args,
        data_root=data_root,
        timeout_s=900,
    )


@task(name="tracking-backfill", retries=1, retry_delay_seconds=300)
def tracking_backfill_task(
    *,
    data_root: Path,
    season_id: str,
    target_date: date,
    lookback_days: int,
    max_retries: int,
    sleep_seconds: float,
    timeout_seconds: float,
) -> None:
    window_days = max(1, int(lookback_days))
    end_day = target_date
    start_day = target_date - timedelta(days=window_days - 1)
    args: list[str] = [
        "backfill",
        "--season",
        season_id,
        "--start-date",
        start_day.isoformat(),
        "--end-date",
        end_day.isoformat(),
        "--data-root",
        str(data_root),
        "--max-retries",
        str(max(1, int(max_retries))),
        "--sleep-seconds",
        str(max(0.0, float(sleep_seconds))),
        "--timeout",
        str(max(1.0, float(timeout_seconds))),
    ]
    per_request_timeout = max(1.0, float(timeout_seconds))
    request_count_est = window_days * 9
    hard_timeout = int(max(900, per_request_timeout * request_count_est + 600))
    _run_python_script(
        "scripts/tracking/scrape_tracking_raw.py",
        args,
        data_root=data_root,
        timeout_s=hard_timeout,
    )


@task(name="tracking-build-roles", retries=1, retry_delay_seconds=300)
def tracking_build_roles_task(
    *,
    data_root: Path,
    season_start_date: date,
    end_date: date,
    overwrite_existing: bool,
) -> None:
    args: list[str] = [
        "--start-date",
        season_start_date.isoformat(),
        "--end-date",
        end_date.isoformat(),
        "--data-root",
        str(data_root),
    ]
    if overwrite_existing:
        args.append("--overwrite-existing")
    else:
        args.append("--no-overwrite-existing")
    _run_python_script(
        "scripts/tracking/build_tracking_roles.py",
        args,
        data_root=data_root,
        timeout_s=3600,
    )


@task(name="tracking-coverage-guard")
def tracking_coverage_guard_task(
    *,
    data_root: Path,
    season_id: str,
    as_of_date: date,
    strict: bool,
    max_lag_days: int,
    recent_window_days: int,
    max_recent_missing_dates: int,
    required_measures: list[str] | None,
    max_recent_measure_gap_dates: int,
) -> dict[str, str]:
    logger = _safe_get_run_logger()
    box_dates = _collect_boxscore_dates(
        data_root=data_root,
        season_id=season_id,
        as_of_date=as_of_date,
    )
    tracking_dates, measures_by_date = _collect_tracking_dates(
        data_root=data_root,
        season_id=season_id,
        as_of_date=as_of_date,
    )
    required = set(required_measures or TRACKING_REQUIRED_MEASURE_TYPES)

    if not box_dates:
        msg = (
            "[tracking-coverage] no boxscore dates were found as reference; "
            "skipping strict checks."
        )
        logger.warning(msg)
        return {
            "status": "no-boxscores",
            "message": msg,
            "latest_boxscore_date": "",
            "latest_tracking_date": "",
            "lag_days": "",
            "recent_missing_dates": "",
            "recent_measure_gap_dates": "",
        }

    latest_box = max(box_dates)
    latest_tracking = max(tracking_dates) if tracking_dates else None
    lag_days = (latest_box - latest_tracking).days if latest_tracking is not None else 10_000

    window = max(1, int(recent_window_days))
    recent_start = latest_box - timedelta(days=window - 1)
    recent_box_dates = {d for d in box_dates if d >= recent_start}
    recent_missing_dates = sorted(d for d in recent_box_dates if d not in tracking_dates)

    recent_measure_gap_dates: list[date] = []
    for d in sorted(recent_box_dates):
        present = measures_by_date.get(d, set())
        if d in recent_missing_dates:
            recent_measure_gap_dates.append(d)
            continue
        if required - present:
            recent_measure_gap_dates.append(d)

    logger.info(
        "[tracking-coverage] season=%s as_of=%s latest_box=%s latest_tracking=%s lag_days=%s recent_missing=%s recent_measure_gaps=%s required=%s",
        season_id,
        as_of_date.isoformat(),
        latest_box.isoformat(),
        latest_tracking.isoformat() if latest_tracking is not None else "",
        lag_days,
        len(recent_missing_dates),
        len(recent_measure_gap_dates),
        sorted(required),
    )

    errors: list[str] = []
    if latest_tracking is None:
        errors.append("tracking has no partitions as of guard date")
    if lag_days > int(max_lag_days):
        errors.append(
            f"tracking lag_days={lag_days} exceeds max_lag_days={int(max_lag_days)} "
            f"(latest_box={latest_box.isoformat()} latest_tracking={latest_tracking.isoformat() if latest_tracking else 'none'})"
        )
    if len(recent_missing_dates) > int(max_recent_missing_dates):
        preview = ",".join(d.isoformat() for d in recent_missing_dates[-10:])
        errors.append(
            f"recent_missing_dates={len(recent_missing_dates)} exceeds max_recent_missing_dates={int(max_recent_missing_dates)} "
            f"(tail={preview})"
        )
    if len(recent_measure_gap_dates) > int(max_recent_measure_gap_dates):
        preview = ",".join(d.isoformat() for d in recent_measure_gap_dates[-10:])
        errors.append(
            f"recent_measure_gap_dates={len(recent_measure_gap_dates)} exceeds max_recent_measure_gap_dates={int(max_recent_measure_gap_dates)} "
            f"(tail={preview})"
        )

    if errors and strict:
        raise RuntimeError("[tracking-coverage] " + " | ".join(errors))
    if errors:
        logger.warning("[tracking-coverage] non-strict warnings: %s", " | ".join(errors))

    return {
        "status": "ok" if not errors else "warn",
        "message": "" if not errors else " | ".join(errors),
        "latest_boxscore_date": latest_box.isoformat(),
        "latest_tracking_date": latest_tracking.isoformat() if latest_tracking is not None else "",
        "lag_days": str(lag_days),
        "recent_missing_dates": str(len(recent_missing_dates)),
        "recent_measure_gap_dates": str(len(recent_measure_gap_dates)),
    }


@task(name="build-rotation-v1", retries=1, retry_delay_seconds=120)
def build_rotation_v1_task(
    *, data_root: Path, overwrite: bool, retry_quarantined: bool
) -> None:
    args: list[str] = []
    if overwrite:
        args.append("--overwrite")
    if retry_quarantined:
        args.append("--retry-quarantined")
    _run_python_script(
        "scripts/rotation/build_rotation_dataset_v1.py",
        args,
        data_root=data_root,
        timeout_s=3600,
    )


@task(name="build-rotation-priors-v1", retries=1, retry_delay_seconds=120)
def build_rotation_priors_v1_task(
    *,
    data_root: Path,
    priors_windows: list[int] | None,
    clean_priors: bool,
) -> None:
    args: list[str] = []
    if priors_windows:
        for w in priors_windows:
            args.extend(["--window", str(int(w))])
    if clean_priors:
        args.append("--clean")
        args.append("--overwrite")
    _run_python_script(
        "scripts/rotation/build_rotation_priors_v1.py",
        args,
        data_root=data_root,
        timeout_s=3600,
    )


@flow(name="rotation-priors-update", log_prints=True)
def rotation_priors_update_flow(
    *,
    run_pbp_ingest: bool = True,
    pbp_fetch_daily_zip: bool = True,
    pbp_vendor_daily_url: str | None = None,
    pbp_fetch_timeout_seconds: int = 900,
    pbp_run_id: str | None = None,
    pbp_season_id: str | None = None,
    pbp_input_glob: str | None = None,
    pbp_resume: bool = False,
    pbp_overwrite: bool = False,
    pbp_skip_bad_games: bool = True,
    pbp_limit_games: int | None = None,
    pbp_allow_qa_failures: bool = False,
    pbp_publish_force: bool = False,
    run_tracking_ingest: bool = True,
    tracking_target_date: str | None = None,
    tracking_lookback_days: int = 3,
    tracking_max_retries: int = 3,
    tracking_sleep_seconds: float = 0.5,
    tracking_timeout_seconds: float = 20.0,
    tracking_build_roles: bool = True,
    tracking_roles_overwrite_existing: bool = True,
    tracking_guard_enabled: bool = True,
    tracking_guard_strict: bool = True,
    tracking_guard_max_lag_days: int = 3,
    tracking_guard_recent_window_days: int = 14,
    tracking_guard_max_recent_missing_dates: int = 2,
    tracking_guard_required_measures: list[str] | None = None,
    tracking_guard_max_recent_measure_gap_dates: int = 2,
    overwrite_rotation_v1: bool = False,
    retry_quarantined: bool = True,
    priors_windows: list[int] | None = None,
    clean_priors: bool = False,
) -> dict[str, str]:
    logger = _safe_get_run_logger()
    data_root = paths.get_data_root()
    season_id = pbp_season_id or _default_pbp_season_id()
    pbp_run = pbp_run_id or _default_pbp_run_id()
    tracking_day = (
        _parse_iso_date(tracking_target_date)
        if tracking_target_date
        else _default_tracking_target_date()
    )
    tracking_season_start = _season_start_date_from_id(season_id)
    resolved_input_glob = _resolve_pbp_input_glob(
        data_root=data_root,
        season_id=season_id,
        explicit_glob=pbp_input_glob,
    )
    logger.info(
        "[rotation-priors-update] data_root=%s run_pbp_ingest=%s pbp_fetch_daily_zip=%s pbp_run_id=%s pbp_season_id=%s pbp_input_glob=%s "
        "pbp_allow_qa_failures=%s pbp_publish_force=%s run_tracking_ingest=%s tracking_target_date=%s tracking_lookback_days=%s "
        "tracking_build_roles=%s tracking_guard_enabled=%s tracking_guard_strict=%s overwrite_rotation_v1=%s retry_quarantined=%s "
        "clean_priors=%s priors_windows=%s",
        data_root,
        run_pbp_ingest,
        pbp_fetch_daily_zip,
        pbp_run,
        season_id,
        resolved_input_glob,
        pbp_allow_qa_failures,
        pbp_publish_force,
        run_tracking_ingest,
        tracking_day.isoformat(),
        tracking_lookback_days,
        tracking_build_roles,
        tracking_guard_enabled,
        tracking_guard_strict,
        overwrite_rotation_v1,
        retry_quarantined,
        clean_priors,
        priors_windows,
    )

    pbp_bundle_dir: Path | None = None
    if run_pbp_ingest:
        resolved_vendor_url = str(
            pbp_vendor_daily_url or os.environ.get("PBP_VENDOR_DAILY_URL") or ""
        ).strip()
        should_fetch_daily_zip = bool(pbp_fetch_daily_zip)
        if should_fetch_daily_zip and not resolved_vendor_url:
            # Fail-open: keep the priors pipeline moving by ingesting existing local files.
            logger.warning(
                "[rotation-priors-update] pbp_fetch_daily_zip requested but no vendor URL was configured; "
                "skipping zip fetch and continuing with existing pbp_input_glob=%s",
                resolved_input_glob,
            )
            should_fetch_daily_zip = False
        if should_fetch_daily_zip:
            pbp_vendor_fetch_daily_zip_task(
                data_root=data_root,
                season_id=season_id,
                vendor_daily_url=resolved_vendor_url,
                timeout_seconds=int(pbp_fetch_timeout_seconds),
            )
        pbp_bundle_dir = pbp_vendor_ingest_task(
            data_root=data_root,
            season_id=season_id,
            run_id=pbp_run,
            input_glob=resolved_input_glob,
            resume=pbp_resume,
            overwrite=pbp_overwrite,
            skip_bad_games=pbp_skip_bad_games,
            limit_games=pbp_limit_games,
        )
        pbp_build_stints_task(
            data_root=data_root,
            bundle_dir=pbp_bundle_dir,
            overwrite=pbp_overwrite,
        )
        pbp_qa_task(
            data_root=data_root,
            bundle_dir=pbp_bundle_dir,
            allow_failures=pbp_allow_qa_failures,
        )
        publish_force = bool(pbp_publish_force or pbp_allow_qa_failures)
        if pbp_allow_qa_failures and not pbp_publish_force:
            logger.info(
                "[rotation-priors-update] pbp_allow_qa_failures=True -> enabling forced publish."
            )
        pbp_publish_task(
            data_root=data_root,
            bundle_dir=pbp_bundle_dir,
            force=publish_force,
        )

    tracking_guard_result: dict[str, str] = {}
    if run_tracking_ingest:
        tracking_backfill_task(
            data_root=data_root,
            season_id=season_id,
            target_date=tracking_day,
            lookback_days=tracking_lookback_days,
            max_retries=tracking_max_retries,
            sleep_seconds=tracking_sleep_seconds,
            timeout_seconds=tracking_timeout_seconds,
        )
    if tracking_build_roles:
        tracking_build_roles_task(
            data_root=data_root,
            season_start_date=tracking_season_start,
            end_date=tracking_day,
            overwrite_existing=tracking_roles_overwrite_existing,
        )
    if tracking_guard_enabled:
        tracking_guard_result = tracking_coverage_guard_task(
            data_root=data_root,
            season_id=season_id,
            as_of_date=tracking_day,
            strict=tracking_guard_strict,
            max_lag_days=tracking_guard_max_lag_days,
            recent_window_days=tracking_guard_recent_window_days,
            max_recent_missing_dates=tracking_guard_max_recent_missing_dates,
            required_measures=tracking_guard_required_measures,
            max_recent_measure_gap_dates=tracking_guard_max_recent_measure_gap_dates,
        )

    build_rotation_v1_task(
        data_root=data_root,
        overwrite=overwrite_rotation_v1,
        retry_quarantined=retry_quarantined,
    )
    build_rotation_priors_v1_task(
        data_root=data_root, priors_windows=priors_windows, clean_priors=clean_priors
    )
    return {
        "status": "ok",
        "run_pbp_ingest": str(run_pbp_ingest),
        "pbp_fetch_daily_zip": str(pbp_fetch_daily_zip),
        "pbp_run_id": pbp_run if run_pbp_ingest else "",
        "pbp_bundle_dir": str(pbp_bundle_dir) if pbp_bundle_dir is not None else "",
        "run_tracking_ingest": str(run_tracking_ingest),
        "tracking_target_date": tracking_day.isoformat(),
        "tracking_guard_status": tracking_guard_result.get("status", ""),
        "tracking_guard_message": tracking_guard_result.get("message", ""),
    }
