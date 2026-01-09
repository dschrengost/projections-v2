"""Prefect flow to scrape NBA boxscores for evaluation and labels.

This exists to keep Prefect as the only scheduler/orchestrator for ETL jobs
that feed the production ecosystem (systemd should only run Prefect workers).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from prefect import flow, get_run_logger, task

from projections import paths


PROJECT_ROOT = paths.get_project_root()
ET_TZ = ZoneInfo("America/New_York")
_DEFAULT_UV_PATH = Path("/home/daniel/.local/bin/uv")


def _uv_bin() -> str:
    env_uv = os.environ.get("UV_BIN")
    if env_uv:
        if Path(env_uv).exists():
            return env_uv
        raise FileNotFoundError(f"UV_BIN={env_uv} specified but file does not exist")

    if _DEFAULT_UV_PATH.exists():
        return str(_DEFAULT_UV_PATH)

    which_uv = shutil.which("uv")
    if which_uv:
        return which_uv

    raise FileNotFoundError(
        "Could not find 'uv' executable. Either:\n"
        "  1. Set UV_BIN=/path/to/uv environment variable, or\n"
        "  2. Install uv and ensure it's in PATH, or\n"
        f"  3. Install uv to {_DEFAULT_UV_PATH}"
    )


def _resolve_season_for_date(day: datetime.date) -> int:
    # Season label convention in the repo ETL is start year (e.g. 2025 for 2025-26).
    # Approximation: NBA season starts in October.
    return day.year if day.month >= 10 else day.year - 1


def _run_boxscores_etl(
    *, start: str, end: str, season: int, data_root: Path, timeout_s: int
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    uv_bin = _uv_bin()
    cmd = [
        uv_bin,
        "run",
        "python",
        "-m",
        "projections.etl.boxscores",
        "--start",
        start,
        "--end",
        end,
        "--season",
        str(season),
        "--data-root",
        str(data_root),
    ]
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
        raise RuntimeError(f"[boxscores-etl] failed exit_code={result.returncode}")


@task(name="boxscores-etl", retries=2, retry_delay_seconds=120)
def boxscores_etl_task(*, game_date: str | None, data_root: Path) -> dict[str, str]:
    logger = get_run_logger()

    if game_date is None:
        # Match old ops behavior: run in ET and scrape yesterday.
        game_date = (datetime.now(tz=ET_TZ).date() - timedelta(days=1)).isoformat()
    season = _resolve_season_for_date(datetime.fromisoformat(game_date).date())

    logger.info(
        f"[boxscores-etl] game_date={game_date} season={season} data_root={data_root}"
    )
    _run_boxscores_etl(
        start=game_date,
        end=game_date,
        season=season,
        data_root=data_root,
        timeout_s=900,
    )
    return {"game_date": game_date, "season": str(season)}


@flow(name="boxscores-etl", log_prints=True)
def boxscores_etl_flow(*, game_date: str | None = None) -> dict[str, str]:
    data_root = paths.get_data_root()
    return boxscores_etl_task(game_date=game_date, data_root=data_root)
