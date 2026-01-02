"""Prefect flow for nightly NBA evaluation.

This runs accuracy evaluation against yesterday's results and ensures boxscores
exist first, so Prefect is the only scheduler/orchestrator for the job.
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


def _resolve_season_for_date(day: datetime.date) -> int:
    return day.year if day.month >= 10 else day.year - 1


def _run_boxscores_etl(*, game_date: str, season: int, data_root: Path, timeout_s: int) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    # Use 'uv run python' instead of sys.executable to ensure correct SSL/network stack
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "projections.etl.boxscores",
        "--start",
        game_date,
        "--end",
        game_date,
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
        raise RuntimeError(f"[nightly-eval][boxscores-etl] failed exit_code={result.returncode}")


def _run_analyze_accuracy(*, output_path: Path, data_root: Path, timeout_s: int) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    # Use 'uv run python' instead of sys.executable to ensure correct SSL/network stack
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "scripts.analyze_accuracy",
        "--output",
        str(output_path),
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
        raise RuntimeError(f"[nightly-eval] analyze_accuracy failed exit_code={result.returncode}")


@task(name="ensure-boxscores", retries=2, retry_delay_seconds=120)
def ensure_boxscores_task(*, game_date: str | None, data_root: Path) -> dict[str, str]:
    logger = get_run_logger()
    if game_date is None:
        game_date = (datetime.now(tz=ET_TZ).date() - timedelta(days=1)).isoformat()
    season = _resolve_season_for_date(datetime.fromisoformat(game_date).date())
    logger.info(f"[nightly-eval][inputs] boxscores_game_date={game_date} season={season} data_root={data_root}")
    _run_boxscores_etl(game_date=game_date, season=season, data_root=data_root, timeout_s=900)
    return {"game_date": game_date, "season": str(season)}


@task(name="analyze-accuracy", retries=1, retry_delay_seconds=60)
def analyze_accuracy_task(*, output_path: Path | None, data_root: Path) -> dict[str, str]:
    logger = get_run_logger()
    if output_path is None:
        output_path = data_root / "reports" / "eval_latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[nightly-eval][outputs] output_path={output_path}")
    _run_analyze_accuracy(output_path=output_path, data_root=data_root, timeout_s=900)
    return {"output_path": str(output_path)}


@flow(name="nightly-eval", log_prints=True)
def nightly_eval_flow(*, game_date: str | None = None, output_path: str | None = None) -> dict[str, str]:
    data_root = paths.get_data_root()
    logger = get_run_logger()
    logger.info(f"[nightly-eval] data_root={data_root}")

    ensure_result = ensure_boxscores_task(game_date=game_date, data_root=data_root)
    output_result = analyze_accuracy_task(
        output_path=Path(output_path) if output_path is not None else None,
        data_root=data_root,
    )
    return {**ensure_result, **output_result}

