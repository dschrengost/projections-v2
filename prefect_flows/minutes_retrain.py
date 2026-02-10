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

import pandas as pd
from prefect import flow, get_run_logger, task

from projections import model_selectors, paths


PROJECT_ROOT = paths.get_project_root()
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


def _normalize_date(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _labels_date_bounds(*, data_root: Path, season: int) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    label_path = data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    if not label_path.exists():
        return None, None
    labels = pd.read_parquet(label_path, columns=["game_date"])
    if labels.empty or "game_date" not in labels.columns:
        return None, None
    game_dates = pd.to_datetime(labels["game_date"], errors="coerce").dropna().dt.normalize()
    if game_dates.empty:
        return None, None
    return game_dates.min(), game_dates.max()


def _resolve_training_window(
    *,
    labels_max_date: pd.Timestamp,
    labels_min_date: pd.Timestamp | None,
    train_window_days: int,
    cal_window_days: int,
) -> dict[str, str]:
    if train_window_days <= 0:
        raise ValueError("train_window_days must be > 0")
    if cal_window_days <= 0:
        raise ValueError("cal_window_days must be > 0")

    cal_end_date = pd.Timestamp(labels_max_date).normalize()
    cal_start_date = cal_end_date - pd.Timedelta(days=cal_window_days - 1)
    train_end_date = cal_start_date - pd.Timedelta(days=1)
    train_start_date = train_end_date - pd.Timedelta(days=train_window_days - 1)

    if labels_min_date is not None and cal_start_date < labels_min_date:
        cal_start_date = labels_min_date.normalize()
        train_end_date = cal_start_date - pd.Timedelta(days=1)
    if labels_min_date is not None and train_start_date < labels_min_date:
        train_start_date = labels_min_date.normalize()

    if train_start_date > train_end_date:
        raise RuntimeError(
            "invalid rolling window after clamp: "
            f"train_start={train_start_date.date()} train_end={train_end_date.date()} "
            f"cal_start={cal_start_date.date()} cal_end={cal_end_date.date()}"
        )

    return {
        "train_start_date": str(train_start_date.date()),
        "train_end_date": str(train_end_date.date()),
        "cal_start_date": str(cal_start_date.date()),
        "cal_end_date": str(cal_end_date.date()),
    }


def _resolve_requested_windows(
    *,
    labels_max_date: pd.Timestamp | None,
    labels_min_date: pd.Timestamp | None,
    train_start_date: str | None,
    train_end_date: str | None,
    cal_start_date: str | None,
    cal_end_date: str | None,
    train_window_days: int,
    cal_window_days: int,
) -> dict[str, str]:
    explicit_dates = [train_start_date, train_end_date, cal_start_date, cal_end_date]
    any_explicit = any(v is not None for v in explicit_dates)
    all_explicit = all(v is not None for v in explicit_dates)
    if any_explicit and not all_explicit:
        raise ValueError(
            "Provide either all explicit train/cal dates or none. "
            "Missing one or more of train_start_date/train_end_date/cal_start_date/cal_end_date."
        )
    if all_explicit:
        return {
            "train_start_date": str(_normalize_date(train_start_date).date()),
            "train_end_date": str(_normalize_date(train_end_date).date()),
            "cal_start_date": str(_normalize_date(cal_start_date).date()),
            "cal_end_date": str(_normalize_date(cal_end_date).date()),
        }
    if labels_max_date is None:
        raise RuntimeError("Cannot resolve rolling windows: labels max date is missing.")
    return _resolve_training_window(
        labels_max_date=labels_max_date,
        labels_min_date=labels_min_date,
        train_window_days=train_window_days,
        cal_window_days=cal_window_days,
    )


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
    config_path = model_selectors.active_minutes_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    current_bundle = (
        Path(current_bundle_dir).expanduser().resolve()
        if current_bundle_dir
        else _resolve_current_bundle_from_config(config_path)
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
        "selector_path": str(config_path),
        "current_bundle_dir": str(current_bundle),
        "retrain_bundle_dir": str(retrain_bundle),
    }


@flow(name="minutes-retrain-pipeline", log_prints=True)
def minutes_retrain_flow(
    *,
    run_id: str | None = None,
    eval_run_id: str | None = None,
    season: int = 2025,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    cal_start_date: str | None = None,
    cal_end_date: str | None = None,
    train_window_days: int = 120,
    cal_window_days: int = 14,
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
    labels_min_date, labels_max_date = _labels_date_bounds(data_root=data_root, season=season)
    if labels_max_date is None:
        raise RuntimeError(f"No labels available to retrain for season={season} under {data_root / 'labels'}")
    windows = _resolve_requested_windows(
        labels_max_date=labels_max_date,
        labels_min_date=labels_min_date,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        cal_start_date=cal_start_date,
        cal_end_date=cal_end_date,
        train_window_days=train_window_days,
        cal_window_days=cal_window_days,
    )
    logger.info(
        "[minutes-retrain-flow] windows train=[%s..%s] cal=[%s..%s] "
        "(labels_min=%s labels_max=%s train_window_days=%s cal_window_days=%s)",
        windows["train_start_date"],
        windows["train_end_date"],
        windows["cal_start_date"],
        windows["cal_end_date"],
        labels_min_date.date() if labels_min_date is not None else "none",
        labels_max_date.date(),
        train_window_days,
        cal_window_days,
    )

    retrain_result = retrain_minutes_task(
        run_id=run_id,
        data_root=data_root,
        season=season,
        train_start_date=windows["train_start_date"],
        train_end_date=windows["train_end_date"],
        cal_start_date=windows["cal_start_date"],
        cal_end_date=windows["cal_end_date"],
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
