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
    if overwrite:
        cmd.append("--overwrite")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=max(60.0, float(timeout_s) + 60.0),
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
    timeout_s: float,
) -> dict[str, str]:
    logger = get_run_logger()
    data_root = paths.get_data_root()

    if game_date is not None:
        start_date = game_date
        end_date = game_date
    elif start_date is None or end_date is None:
        # Daily default: yesterday in ET.
        iso = (datetime.now(tz=ET_TZ).date() - timedelta(days=1)).isoformat()
        start_date = start_date or iso
        end_date = end_date or iso

    logger.info(
        "[gamerotation] start_date=%s end_date=%s overwrite=%s data_root=%s timeout_s=%.1f",
        start_date,
        end_date,
        overwrite,
        data_root,
        float(timeout_s),
    )
    _run_gamerotation_scrape(
        start_date=str(start_date),
        end_date=str(end_date),
        data_root=data_root,
        overwrite=overwrite,
        timeout_s=float(timeout_s),
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
    timeout_s: float = 20.0,
    max_retries: int = 3,
) -> dict[str, str]:
    # Kept for backwards-compatible deployment params (write_bronze_copy/max_retries are unused).
    _ = write_bronze_copy, max_retries
    return gamerotation_scrape_task(
        game_date=game_date,
        start_date=start_date,
        end_date=end_date,
        overwrite=overwrite,
        timeout_s=timeout_s,
    )
