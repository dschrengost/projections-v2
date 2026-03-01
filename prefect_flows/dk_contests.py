"""Prefect flows for DraftKings NBA GPP contest scraping and ETL.

Replaces the cron-based shell scripts from the dkcontests repo.  Two flows:

1. dk_contests_morning_flow (6:00 AM ET daily)
   - Scrape today's contest list
   - Download yesterday's contest results
   - Scrape yesterday's payout curves
   - ETL load yesterday into DuckDB

2. dk_contests_prelock_flow (3:30 PM ET daily)
   - Ingest pre-lock player pools
   - Scrape today's payout curves
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from prefect import flow, get_run_logger, task

from projections import paths


PROJECT_ROOT = paths.get_project_root()
ET_TZ = ZoneInfo("America/New_York")
_DEFAULT_UV_PATH = Path("/home/daniel/.local/bin/uv")

SCRAPERS_DIR = PROJECT_ROOT / "scrapers" / "dk_contests"


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


def _run_scraper(
    *, module: str, args: list[str], data_root: Path, timeout_s: int, label: str
) -> None:
    """Run a dk_contests scraper module as a subprocess."""
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    uv_bin = _uv_bin()
    cmd = [uv_bin, "run", "python", str(SCRAPERS_DIR / module), *args]
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
        raise RuntimeError(f"[dk-contests][{label}] failed exit_code={result.returncode}")


def _bronze_data_root(data_root: Path) -> str:
    """Return the bronze data root for DK contests."""
    return str(data_root / "bronze" / "dk_contests" / "nba_gpp_data")


def _db_path(data_root: Path) -> str:
    """Return the DuckDB path for DK contests."""
    return str(data_root / "bronze" / "dk_contests" / "analytics" / "contests.duckdb")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(name="dk-scrape-contest-list", retries=2, retry_delay_seconds=120)
def scrape_contest_list_task(*, target_date: str, data_root: Path) -> dict[str, str]:
    """Scrape today's NBA GPP contest list from DraftKings."""
    logger = get_run_logger()
    logger.info("[dk-contests][scrape-list] date=%s data_root=%s", target_date, data_root)
    _run_scraper(
        module="nba_gpp_scraper.py",
        args=["--date", target_date, "--output-dir", _bronze_data_root(data_root)],
        data_root=data_root,
        timeout_s=300,
        label="scrape-list",
    )
    return {"date": target_date, "step": "scrape-list"}


@task(name="dk-download-results", retries=2, retry_delay_seconds=120)
def download_results_task(*, target_date: str, data_root: Path) -> dict[str, str]:
    """Download contest results for a given date."""
    logger = get_run_logger()
    logger.info("[dk-contests][download-results] date=%s data_root=%s", target_date, data_root)
    _run_scraper(
        module="download_contest_results.py",
        args=[
            "--date", target_date,
            "--data-root", _bronze_data_root(data_root),
            "--auth-browser",
            "--headless",
        ],
        data_root=data_root,
        timeout_s=1800,
        label="download-results",
    )
    return {"date": target_date, "step": "download-results"}


@task(name="dk-scrape-payouts", retries=2, retry_delay_seconds=120)
def scrape_payouts_task(*, target_date: str, data_root: Path) -> dict[str, str]:
    """Scrape payout curves for contests on a given date."""
    logger = get_run_logger()
    logger.info("[dk-contests][scrape-payouts] date=%s data_root=%s", target_date, data_root)
    _run_scraper(
        module="payouts_scraper.py",
        args=[
            "--date", target_date,
            "--data-root", _bronze_data_root(data_root),
            "--db", _db_path(data_root),
        ],
        data_root=data_root,
        timeout_s=600,
        label="scrape-payouts",
    )
    return {"date": target_date, "step": "scrape-payouts"}


@task(name="dk-ingest-players", retries=2, retry_delay_seconds=120)
def ingest_players_task(*, target_date: str, data_root: Path) -> dict[str, str]:
    """Ingest pre-lock player pools for a given date."""
    logger = get_run_logger()
    logger.info("[dk-contests][ingest-players] date=%s data_root=%s", target_date, data_root)
    _run_scraper(
        module="prelock.py",
        args=[
            "--date", target_date,
            "--data-root", _bronze_data_root(data_root),
            "--db", _db_path(data_root),
        ],
        data_root=data_root,
        timeout_s=600,
        label="ingest-players",
    )
    return {"date": target_date, "step": "ingest-players"}


@task(name="dk-etl-load", retries=2, retry_delay_seconds=120)
def etl_load_task(*, target_date: str, data_root: Path) -> dict[str, str]:
    """Load contest data into DuckDB for a given date."""
    logger = get_run_logger()
    db = _db_path(data_root)
    logger.info("[dk-contests][etl-load] date=%s db=%s data_root=%s", target_date, db, data_root)
    # Init schema first
    _run_scraper(
        module="etl_duckdb.py",
        args=[
            "--init",
            "--data-root", _bronze_data_root(data_root),
            "--db", db,
        ],
        data_root=data_root,
        timeout_s=120,
        label="etl-init",
    )
    # Load the date
    _run_scraper(
        module="etl_duckdb.py",
        args=[
            "--date", target_date,
            "--data-root", _bronze_data_root(data_root),
            "--db", db,
        ],
        data_root=data_root,
        timeout_s=600,
        label="etl-load",
    )
    return {"date": target_date, "step": "etl-load"}


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------


@flow(name="dk-contests-morning", log_prints=True)
def dk_contests_morning_flow(
    *, target_date: str | None = None, yesterday: str | None = None
) -> dict[str, str]:
    """Morning flow: scrape today's list, download+process yesterday's results.

    Parameters
    ----------
    target_date : str | None
        Date for the contest list scrape (defaults to today ET).
    yesterday : str | None
        Date for results download / payouts / ETL (defaults to yesterday ET).
    """
    data_root = paths.get_data_root()
    logger = get_run_logger()

    now_et = datetime.now(tz=ET_TZ)
    if target_date is None:
        target_date = now_et.date().isoformat()
    if yesterday is None:
        yesterday = (now_et.date() - timedelta(days=1)).isoformat()

    logger.info(
        "[dk-contests-morning] target_date=%s yesterday=%s data_root=%s",
        target_date, yesterday, data_root,
    )

    # 1. Scrape today's contest list
    scrape_contest_list_task(target_date=target_date, data_root=data_root)

    # 2. Download yesterday's results
    download_results_task(target_date=yesterday, data_root=data_root)

    # 3. Scrape yesterday's payouts (backfill)
    scrape_payouts_task(target_date=yesterday, data_root=data_root)

    # 4. ETL load yesterday
    etl_load_task(target_date=yesterday, data_root=data_root)

    return {"target_date": target_date, "yesterday": yesterday}


@flow(name="dk-contests-prelock", log_prints=True)
def dk_contests_prelock_flow(*, target_date: str | None = None) -> dict[str, str]:
    """Pre-lock flow: ingest player pools and scrape payouts for today.

    Parameters
    ----------
    target_date : str | None
        Date to process (defaults to today ET).
    """
    data_root = paths.get_data_root()
    logger = get_run_logger()

    if target_date is None:
        target_date = datetime.now(tz=ET_TZ).date().isoformat()

    logger.info(
        "[dk-contests-prelock] target_date=%s data_root=%s", target_date, data_root
    )

    # 1. Ingest pre-lock player pools
    ingest_players_task(target_date=target_date, data_root=data_root)

    # 2. Scrape today's payouts
    scrape_payouts_task(target_date=target_date, data_root=data_root)

    return {"target_date": target_date}
