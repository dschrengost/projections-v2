"""Prefect flow to build rotation_v1 and rotation_priors_v1.

This runs after the daily gamerotation scrape to refresh silver rotation data
and derived priors used by the rotation overlay minutes model.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from prefect import flow, get_run_logger, task

from projections import paths

PROJECT_ROOT = paths.get_project_root()


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
    overwrite_rotation_v1: bool = False,
    retry_quarantined: bool = True,
    priors_windows: list[int] | None = None,
    clean_priors: bool = False,
) -> dict[str, str]:
    logger = get_run_logger()
    data_root = paths.get_data_root()
    season_id = pbp_season_id or _default_pbp_season_id()
    pbp_run = pbp_run_id or _default_pbp_run_id()
    resolved_input_glob = _resolve_pbp_input_glob(
        data_root=data_root,
        season_id=season_id,
        explicit_glob=pbp_input_glob,
    )
    logger.info(
        "[rotation-priors-update] data_root=%s run_pbp_ingest=%s pbp_fetch_daily_zip=%s pbp_run_id=%s pbp_season_id=%s pbp_input_glob=%s "
        "pbp_allow_qa_failures=%s pbp_publish_force=%s overwrite_rotation_v1=%s retry_quarantined=%s clean_priors=%s priors_windows=%s",
        data_root,
        run_pbp_ingest,
        pbp_fetch_daily_zip,
        pbp_run,
        season_id,
        resolved_input_glob,
        pbp_allow_qa_failures,
        pbp_publish_force,
        overwrite_rotation_v1,
        retry_quarantined,
        clean_priors,
        priors_windows,
    )

    pbp_bundle_dir: Path | None = None
    if run_pbp_ingest:
        resolved_vendor_url = str(pbp_vendor_daily_url or os.environ.get("PBP_VENDOR_DAILY_URL") or "").strip()
        if pbp_fetch_daily_zip:
            if not resolved_vendor_url:
                raise ValueError(
                    "pbp_fetch_daily_zip=True requires pbp_vendor_daily_url parameter or env PBP_VENDOR_DAILY_URL."
                )
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
    }
