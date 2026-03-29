"""Prefect flow to scrape NBA Stats GameRotation into bronze storage.

This deployment exists so Prefect is the only scheduler/orchestrator for the
rotation data pipeline (systemd should only run Prefect workers).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from prefect import flow, get_run_logger, task

from projections import paths

PROJECT_ROOT = paths.get_project_root()
ET_TZ = ZoneInfo("America/New_York")


def _run_gamerotation_scrape(
    *,
    start_date: str,
    end_date: str,
    data_root: Path,
    overwrite: bool,
    timeout_s: float,
    max_failure_rate: float | None,
    min_success_coverage: float | None,
    subprocess_timeout_s: float | None,
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "backfill_gamerotation_bronze.py"),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--timeout",
        str(float(timeout_s)),
    ]
    if max_failure_rate is not None:
        cmd.extend(["--max-failure-rate", str(float(max_failure_rate))])
    if min_success_coverage is not None:
        cmd.extend(["--min-success-coverage", str(float(min_success_coverage))])
    if overwrite:
        cmd.append("--overwrite")

    # The scrape iterates many game_ids per date and each game can retry on
    # upstream 5xx/timeout responses; keep subprocess timeout decoupled from
    # per-request timeout to avoid premature flow failures.
    if subprocess_timeout_s is None:
        effective_subprocess_timeout = max(600.0, float(timeout_s) * 40.0)
    else:
        raw_timeout = float(subprocess_timeout_s)
        effective_subprocess_timeout = None if raw_timeout <= 0 else raw_timeout

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=effective_subprocess_timeout,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"[gamerotation] failed exit_code={result.returncode}")


@task(name="gamerotation-scrape", retries=2, retry_delay_seconds=120)
def gamerotation_scrape_task(
    *,
    game_date: str | None,
    start_date: str | None,
    end_date: str | None,
    overwrite: bool,
    lookback_days: int,
    timeout_s: float,
    max_failure_rate: float | None,
    min_success_coverage: float | None,
    subprocess_timeout_s: float | None,
) -> dict[str, str]:
    logger = get_run_logger()
    data_root = paths.get_data_root()

    if game_date is not None:
        start_date = game_date
        end_date = game_date
    elif start_date is None and end_date is None:
        # Daily default: rolling lookback ending yesterday in ET.
        end = datetime.now(tz=ET_TZ).date() - timedelta(days=1)
        window_days = max(1, int(lookback_days))
        start = end - timedelta(days=window_days - 1)
        start_date = start.isoformat()
        end_date = end.isoformat()
    elif start_date is None or end_date is None:
        # If only one side is provided, default the other to yesterday in ET.
        iso = (datetime.now(tz=ET_TZ).date() - timedelta(days=1)).isoformat()
        start_date = start_date or iso
        end_date = end_date or iso

    logger.info(
        "[gamerotation] start_date=%s end_date=%s overwrite=%s lookback_days=%s data_root=%s timeout_s=%.1f "
        "max_failure_rate=%s min_success_coverage=%s subprocess_timeout_s=%s",
        start_date,
        end_date,
        overwrite,
        int(lookback_days),
        data_root,
        float(timeout_s),
        "disabled" if max_failure_rate is None else f"{float(max_failure_rate):.2f}",
        "disabled"
        if min_success_coverage is None
        else f"{float(min_success_coverage):.2f}",
        "auto"
        if subprocess_timeout_s is None
        else (
            "disabled"
            if float(subprocess_timeout_s) <= 0
            else f"{float(subprocess_timeout_s):.1f}"
        ),
    )
    _run_gamerotation_scrape(
        start_date=str(start_date),
        end_date=str(end_date),
        data_root=data_root,
        overwrite=overwrite,
        timeout_s=float(timeout_s),
        max_failure_rate=max_failure_rate,
        min_success_coverage=min_success_coverage,
        subprocess_timeout_s=subprocess_timeout_s,
    )
    return {"start_date": str(start_date), "end_date": str(end_date)}


@flow(name="gamerotation-scrape", log_prints=True)
def gamerotation_scrape_flow(
    *,
    game_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    overwrite: bool = False,
    write_bronze_copy: bool = True,
    lookback_days: int = 14,
    timeout_s: float = 20.0,
    max_failure_rate: float | None = 0.5,
    min_success_coverage: float | None = 1.0,
    subprocess_timeout_s: float | None = 900.0,
    max_retries: int = 3,
) -> dict[str, str]:
    # Kept for backwards-compatible deployment params (write_bronze_copy/max_retries are unused).
    _ = write_bronze_copy, max_retries
    return gamerotation_scrape_task(
        game_date=game_date,
        start_date=start_date,
        end_date=end_date,
        overwrite=overwrite,
        lookback_days=lookback_days,
        timeout_s=timeout_s,
        max_failure_rate=max_failure_rate,
        min_success_coverage=min_success_coverage,
        subprocess_timeout_s=subprocess_timeout_s,
    )
