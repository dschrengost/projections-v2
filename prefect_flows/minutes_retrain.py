"""Prefect flow for weekly Minutes V1 recency retraining + head-to-head eval."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from prefect import flow, get_run_logger, task

from projections import paths


PROJECT_ROOT = paths.get_project_root()
DEFAULT_MINUTES_CONFIG = PROJECT_ROOT / "config/minutes_current_run.json"
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


def _run_python_module(
    module: str,
    args: list[str],
    *,
    data_root: Path,
    timeout_s: int,
) -> None:
    env = os.environ.copy()
    env["PROJECTIONS_DATA_ROOT"] = str(data_root)
    cmd = [_uv_bin(), "run", "python", "-m", module, *args]
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
        raise RuntimeError(f"{module} failed with exit_code={result.returncode}")


def _default_retrain_run_id() -> str:
    return datetime.now(tz=UTC).strftime("minutes_v1_recency_h35_%Y%m%dT%H%M%SZ")


def _resolve_current_bundle_from_config(config_path: Path) -> Path:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw = payload.get("bundle_dir")
    if not raw:
        raise RuntimeError(f"Missing bundle_dir in {config_path}")
    bundle_dir = Path(raw)
    if not bundle_dir.is_absolute():
        bundle_dir = (PROJECT_ROOT / bundle_dir).resolve()
    return bundle_dir


@task(name="minutes-retrain", retries=0)
def retrain_minutes_task(
    *,
    run_id: str | None,
    data_root: Path,
    season: int,
    train_start_date: str,
    train_end_date: str,
    cal_start_date: str,
    cal_end_date: str,
    half_life_days: float,
    train_random_state: int,
    allow_guard_failure: bool,
) -> dict[str, str]:
    logger = get_run_logger()
    effective_run_id = run_id.strip() if run_id else _default_retrain_run_id()
    logger.info(
        "[minutes-retrain] run_id=%s season=%s train=[%s..%s] cal=[%s..%s] half_life=%.2f allow_guard_failure=%s",
        effective_run_id,
        season,
        train_start_date,
        train_end_date,
        cal_start_date,
        cal_end_date,
        half_life_days,
        allow_guard_failure,
    )

    args = [
        "run",
        "--run-id",
        effective_run_id,
        "--data-root",
        str(data_root),
        "--season",
        str(season),
        "--train-start-date",
        train_start_date,
        "--train-end-date",
        train_end_date,
        "--cal-start-date",
        cal_start_date,
        "--cal-end-date",
        cal_end_date,
        "--half-life-days",
        str(half_life_days),
        "--train-random-state",
        str(train_random_state),
    ]
    if allow_guard_failure:
        args.append("--allow-guard-failure")

    _run_python_module(
        "projections.cli.retrain_minutes_v1_recency",
        args,
        data_root=data_root,
        timeout_s=60 * 60 * 2,
    )

    bundle_dir = data_root / "artifacts" / "minutes_lgbm" / effective_run_id
    if not (bundle_dir / "lgbm_quantiles.joblib").exists():
        raise RuntimeError(f"Retrain completed but bundle artifact missing under {bundle_dir}")
    return {
        "run_id": effective_run_id,
        "bundle_dir": str(bundle_dir),
    }


@task(name="minutes-head-to-head-eval", retries=0)
def evaluate_candidate_task(
    *,
    retrain_run_id: str,
    retrain_bundle_dir: str,
    data_root: Path,
    season: int,
    current_bundle_dir: str | None,
    eval_run_id: str | None,
) -> dict[str, str]:
    logger = get_run_logger()
    current_bundle = (
        Path(current_bundle_dir).expanduser().resolve()
        if current_bundle_dir
        else _resolve_current_bundle_from_config(DEFAULT_MINUTES_CONFIG)
    )
    retrain_bundle = Path(retrain_bundle_dir).expanduser().resolve()
    effective_eval_run_id = eval_run_id.strip() if eval_run_id else f"minutes_head_to_head_{retrain_run_id}"
    eval_report_path = data_root / "reports" / f"{effective_eval_run_id}.md"

    logger.info(
        "[minutes-eval] eval_run_id=%s current=%s retrain=%s",
        effective_eval_run_id,
        current_bundle,
        retrain_bundle,
    )
    args = [
        "--eval-run-id",
        effective_eval_run_id,
        "--current-bundle",
        str(current_bundle),
        "--retrain-bundle",
        str(retrain_bundle),
        "--data-root",
        str(data_root),
        "--season",
        str(season),
        "--report-path",
        str(eval_report_path),
    ]
    _run_python_module(
        "projections.cli.eval_minutes_bundles",
        args,
        data_root=data_root,
        timeout_s=60 * 45,
    )

    eval_root = data_root / "artifacts" / "minutes_eval_runs" / effective_eval_run_id
    summary_path = eval_root / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Head-to-head eval completed but summary missing: {summary_path}")
    return {
        "eval_run_id": effective_eval_run_id,
        "summary_path": str(summary_path),
        "report_path": str(eval_report_path),
        "current_bundle_dir": str(current_bundle),
        "retrain_bundle_dir": str(retrain_bundle),
    }


@flow(name="minutes-retrain-pipeline", log_prints=True)
def minutes_retrain_flow(
    *,
    run_id: str | None = None,
    eval_run_id: str | None = None,
    season: int = 2025,
    train_start_date: str = "2025-02-01",
    train_end_date: str = "2026-01-31",
    cal_start_date: str = "2026-02-01",
    cal_end_date: str = "2026-02-05",
    half_life_days: float = 35.0,
    train_random_state: int = 42,
    allow_guard_failure: bool = True,
    run_head_to_head_eval: bool = True,
    current_bundle_dir: str | None = None,
) -> dict[str, Any]:
    """Retrain a minutes candidate bundle and optionally run head-to-head eval."""

    logger = get_run_logger()
    data_root = paths.get_data_root()
    logger.info("[minutes-retrain-flow] data_root=%s", data_root)

    retrain_result = retrain_minutes_task(
        run_id=run_id,
        data_root=data_root,
        season=season,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        cal_start_date=cal_start_date,
        cal_end_date=cal_end_date,
        half_life_days=half_life_days,
        train_random_state=train_random_state,
        allow_guard_failure=allow_guard_failure,
    )

    result: dict[str, Any] = {"retrain": retrain_result}
    if run_head_to_head_eval:
        eval_result = evaluate_candidate_task(
            retrain_run_id=retrain_result["run_id"],
            retrain_bundle_dir=retrain_result["bundle_dir"],
            data_root=data_root,
            season=season,
            current_bundle_dir=current_bundle_dir,
            eval_run_id=eval_run_id,
        )
        result["eval"] = eval_result
    else:
        logger.info("[minutes-retrain-flow] run_head_to_head_eval=False; skipping evaluation stage")
    return result


if __name__ == "__main__":  # pragma: no cover
    minutes_retrain_flow()
