"""Prefect flow to refresh gold/rates_training_base partitions."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import pandas as pd
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


def _latest_rates_training_base_date(data_root: Path) -> pd.Timestamp | None:
    base = data_root / "gold" / "rates_training_base"
    if not base.exists():
        return None
    latest: pd.Timestamp | None = None
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            parquet_path = day_dir / "rates_training_base.parquet"
            if not parquet_path.exists():
                continue
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if latest is None or day > latest:
                latest = day
    return latest


def _latest_minutes_labels_date(data_root: Path) -> pd.Timestamp | None:
    base = data_root / "gold" / "labels_minutes_v1"
    if not base.exists():
        return None
    latest: pd.Timestamp | None = None
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            parquet_path = day_dir / "labels.parquet"
            if not parquet_path.exists():
                continue
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if latest is None or day > latest:
                latest = day
    return latest


def _run_rates_training_base_build(
    *,
    data_root: Path,
    start_date: str,
    end_date: str,
    timeout_s: int,
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    cmd = [
        _uv_bin(),
        "run",
        "python",
        "-m",
        "scripts.rates.build_training_base",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
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
        raise RuntimeError(f"scripts.rates.build_training_base failed with exit_code={result.returncode}")


@task(name="refresh-rates-training-base", retries=1, retry_delay_seconds=120)
def refresh_rates_training_base_task(
    *,
    data_root: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, str]:
    logger = get_run_logger()

    if end_date is None:
        target_end = pd.Timestamp((datetime.now(tz=ET_TZ).date() - timedelta(days=1))).normalize()
    else:
        target_end = pd.Timestamp(end_date).normalize()
    labels_max = _latest_minutes_labels_date(data_root)
    if labels_max is None:
        logger.info("[rates-training-base-refresh] no-op (labels_minutes_v1 missing)")
        return {
            "status": "noop_missing_labels",
            "start_date": "none",
            "end_date": str(target_end.date()),
        }
    effective_end = min(target_end, labels_max)

    if start_date is None:
        latest_existing = _latest_rates_training_base_date(data_root)
        if latest_existing is None:
            target_start = effective_end
        else:
            target_start = latest_existing + pd.Timedelta(days=1)
    else:
        target_start = pd.Timestamp(start_date).normalize()

    if target_start > effective_end:
        logger.info(
            "[rates-training-base-refresh] no-op (start %s > effective_end %s; labels_max=%s)",
            target_start.date(),
            effective_end.date(),
            labels_max.date(),
        )
        return {
            "status": "noop",
            "start_date": str(target_start.date()),
            "end_date": str(effective_end.date()),
            "labels_max_date": str(labels_max.date()),
        }

    logger.info(
        "[rates-training-base-refresh] building start=%s end=%s data_root=%s (labels_max=%s)",
        target_start.date(),
        effective_end.date(),
        data_root,
        labels_max.date(),
    )
    _run_rates_training_base_build(
        data_root=data_root,
        start_date=str(target_start.date()),
        end_date=str(effective_end.date()),
        timeout_s=60 * 60,
    )
    latest_after = _latest_rates_training_base_date(data_root)
    return {
        "status": "ok",
        "start_date": str(target_start.date()),
        "end_date": str(effective_end.date()),
        "labels_max_date": str(labels_max.date()),
        "latest_partition_after": str(latest_after.date()) if latest_after is not None else "none",
    }


@flow(name="rates-training-base-refresh", log_prints=True)
def rates_training_base_refresh_flow(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, str]:
    data_root = paths.get_data_root()
    logger = get_run_logger()
    logger.info("[rates-training-base-refresh] data_root=%s", data_root)
    return refresh_rates_training_base_task(
        data_root=data_root,
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":  # pragma: no cover
    rates_training_base_refresh_flow()
