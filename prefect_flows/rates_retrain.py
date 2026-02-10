"""Prefect flows for automated Rates v1 retraining and calibration monitoring."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from prefect import flow, get_run_logger, task

from projections import model_selectors, paths
from projections.rates_v1.loader import load_rates_bundle
from projections.rates_v1.score import predict_rates
from scripts.rates.train_rates_v1 import (
    TARGETS,
    TARGET_LABEL_MAP,
    _load_training_base,
    _prepare_features,
)


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
    return datetime.now(tz=UTC).strftime("rates_v1_stage5_recency_h75_%Y%m%d_%H%M%S")


def _iter_partition_dates(base: Path, *, day_prefix: str, filename: str) -> list[pd.Timestamp]:
    if not base.exists():
        return []
    out: list[pd.Timestamp] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob(f"{day_prefix}=*"):
            parquet_path = day_dir / filename
            if not parquet_path.exists():
                continue
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            out.append(day)
    return sorted(out)


def _labels_date_bounds(data_root: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    base = data_root / "gold" / "labels_minutes_v1"
    days = _iter_partition_dates(base, day_prefix="game_date", filename="labels.parquet")
    if not days:
        return None, None
    return days[0], days[-1]


def _read_current_rates_run_id(config_path: Path) -> str:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    if not run_id:
        raise RuntimeError(f"Missing run_id in {config_path}")
    return str(run_id)


def _read_selector_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Selector config must be a JSON object: {path}")
    return payload


def _resolve_training_window(
    *,
    labels_max_date: pd.Timestamp,
    labels_min_date: pd.Timestamp | None,
    train_window_days: int,
    cal_window_days: int,
    val_window_days: int,
) -> dict[str, str]:
    if train_window_days <= 0:
        raise ValueError("train_window_days must be > 0")
    if cal_window_days <= 0:
        raise ValueError("cal_window_days must be > 0")
    if val_window_days <= 0:
        raise ValueError("val_window_days must be > 0")

    end_date = pd.Timestamp(labels_max_date).normalize()
    cal_end_date = end_date - pd.Timedelta(days=val_window_days - 1)
    train_end_date = cal_end_date - pd.Timedelta(days=cal_window_days)
    train_start_date = train_end_date - pd.Timedelta(days=train_window_days)

    if labels_min_date is not None and train_start_date < labels_min_date:
        train_start_date = labels_min_date.normalize()

    if train_start_date >= train_end_date:
        raise RuntimeError(
            f"invalid window after clamp: train_start={train_start_date.date()} train_end={train_end_date.date()}"
        )

    return {
        "start_date": str(train_start_date.date()),
        "end_date": str(end_date.date()),
        "train_end_date": str(train_end_date.date()),
        "cal_end_date": str(cal_end_date.date()),
    }


def _should_run_biweekly(*, run_day: date, anchor_day: date) -> bool:
    if run_day < anchor_day:
        return False
    delta_days = (run_day - anchor_day).days
    return (delta_days // 7) % 2 == 0


def _default_eval_slices(*, train_end_date: str, cal_end_date: str, end_date: str) -> dict[str, tuple[str, str]]:
    train_end = pd.Timestamp(train_end_date).normalize()
    cal_end = pd.Timestamp(cal_end_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    normal_start = train_end
    normal_end = max(train_end, cal_end)
    chaos_end = end
    chaos_start = max(train_end, end - pd.Timedelta(days=8))
    return {
        "normal_pre_deadline": (str(normal_start.date()), str(normal_end.date())),
        "chaos_deadline": (str(chaos_start.date()), str(chaos_end.date())),
    }


def evaluate_head_to_head_from_rates_base(
    *,
    data_root: Path,
    current_run_id: str,
    retrain_run_id: str,
    slices: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    min_slice_start = min(pd.Timestamp(v[0]).normalize() for v in slices.values())
    max_slice_end = max(pd.Timestamp(v[1]).normalize() for v in slices.values())
    df = _load_training_base(data_root, start=min_slice_start, end=max_slice_end)
    df = _prepare_features(
        df,
        use_predicted_minutes=True,
        fallback_minutes_with_actual=True,
        use_tracking_features=True,
    )

    bundle_current = load_rates_bundle(current_run_id, base_artifacts_root=data_root)
    bundle_retrain = load_rates_bundle(retrain_run_id, base_artifacts_root=data_root)
    union_features = sorted(set(bundle_current.feature_cols) | set(bundle_retrain.feature_cols))
    for col in union_features:
        if col not in df.columns:
            df[col] = np.nan

    result: dict[str, Any] = {
        "current_run_id": current_run_id,
        "retrain_run_id": retrain_run_id,
        "slices": {},
    }
    for slice_name, (start_s, end_s) in slices.items():
        start = pd.Timestamp(start_s).normalize()
        end = pd.Timestamp(end_s).normalize()
        slice_df = df[(df["game_date"] >= start) & (df["game_date"] <= end)].copy()
        rows_raw = int(len(slice_df))
        if rows_raw == 0:
            result["slices"][slice_name] = {
                "summary": {"start": start_s, "end": end_s, "rows_raw": 0, "rows_scored": 0},
                "per_target": {},
            }
            continue

        preds_current = predict_rates(slice_df, bundle_current)
        preds_retrain = predict_rates(slice_df, bundle_retrain)
        per_target: dict[str, Any] = {}
        avg_mae_current_vals: list[float] = []
        avg_mae_retrain_vals: list[float] = []

        for target in TARGETS:
            label_col = TARGET_LABEL_MAP.get(target, target)
            if label_col not in slice_df.columns:
                continue
            mask = pd.to_numeric(slice_df[label_col], errors="coerce").notna().to_numpy()
            n_rows = int(mask.sum())
            if n_rows == 0:
                continue

            y_true = pd.to_numeric(slice_df.loc[mask, label_col], errors="coerce").to_numpy(dtype=float)
            cur_col = target if target in preds_current.columns else f"{target}_mean"
            ret_col = target if target in preds_retrain.columns else f"{target}_mean"
            if cur_col not in preds_current.columns or ret_col not in preds_retrain.columns:
                continue

            p_cur = pd.to_numeric(preds_current.loc[mask, cur_col], errors="coerce").to_numpy(dtype=float)
            p_ret = pd.to_numeric(preds_retrain.loc[mask, ret_col], errors="coerce").to_numpy(dtype=float)

            cur_err = p_cur - y_true
            ret_err = p_ret - y_true
            cur_mae = float(np.abs(cur_err).mean())
            ret_mae = float(np.abs(ret_err).mean())
            cur_rmse = float(np.sqrt(np.mean(cur_err**2)))
            ret_rmse = float(np.sqrt(np.mean(ret_err**2)))

            avg_mae_current_vals.append(cur_mae)
            avg_mae_retrain_vals.append(ret_mae)
            per_target[target] = {
                "label_col": label_col,
                "current": {"mae": cur_mae, "rmse": cur_rmse, "n": n_rows},
                "retrain": {"mae": ret_mae, "rmse": ret_rmse, "n": n_rows},
                "delta_mae_retrain_minus_current": float(ret_mae - cur_mae),
                "delta_rmse_retrain_minus_current": float(ret_rmse - cur_rmse),
            }

        avg_mae_current = float(np.mean(avg_mae_current_vals)) if avg_mae_current_vals else None
        avg_mae_retrain = float(np.mean(avg_mae_retrain_vals)) if avg_mae_retrain_vals else None
        result["slices"][slice_name] = {
            "summary": {
                "start": start_s,
                "end": end_s,
                "rows_raw": rows_raw,
                "rows_scored": int(len(slice_df)),
                "avg_mae_current": avg_mae_current,
                "avg_mae_retrain": avg_mae_retrain,
                "avg_mae_delta_retrain_minus_current": (
                    None
                    if avg_mae_current is None or avg_mae_retrain is None
                    else float(avg_mae_retrain - avg_mae_current)
                ),
            },
            "per_target": per_target,
        }

    return result


def assess_eval_guardrails(
    eval_summary: dict[str, Any],
    *,
    max_avg_mae_delta: float = 0.0,
    max_head_mae_regression: float = 0.001,
) -> dict[str, Any]:
    failing_slices: list[dict[str, Any]] = []
    worst_head: dict[str, Any] | None = None

    slices = eval_summary.get("slices") or {}
    for slice_name, slice_payload in slices.items():
        summary = slice_payload.get("summary") or {}
        avg_delta = summary.get("avg_mae_delta_retrain_minus_current")
        if avg_delta is not None and float(avg_delta) > max_avg_mae_delta:
            failing_slices.append(
                {
                    "slice": slice_name,
                    "avg_mae_delta_retrain_minus_current": float(avg_delta),
                    "threshold": max_avg_mae_delta,
                }
            )

        for target, payload in (slice_payload.get("per_target") or {}).items():
            delta = payload.get("delta_mae_retrain_minus_current")
            if delta is None:
                continue
            delta_val = float(delta)
            if worst_head is None or delta_val > float(worst_head["delta_mae_retrain_minus_current"]):
                worst_head = {
                    "slice": slice_name,
                    "target": target,
                    "delta_mae_retrain_minus_current": delta_val,
                }

    pass_worst_head = worst_head is None or float(worst_head["delta_mae_retrain_minus_current"]) <= max_head_mae_regression
    passed = (len(failing_slices) == 0) and pass_worst_head
    return {
        "passed": passed,
        "max_avg_mae_delta": max_avg_mae_delta,
        "max_head_mae_regression": max_head_mae_regression,
        "failing_slices": failing_slices,
        "worst_head_regression": worst_head,
    }


@task(name="refresh-rates-training-base-window", retries=1, retry_delay_seconds=120)
def refresh_rates_training_base_window_task(
    *,
    data_root: Path,
    start_date: str,
    end_date: str,
) -> dict[str, str]:
    logger = get_run_logger()
    logger.info("[rates-retrain] refreshing rates_training_base start=%s end=%s", start_date, end_date)
    _run_python_module(
        "scripts.rates.build_training_base",
        [
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--data-root",
            str(data_root),
        ],
        data_root=data_root,
        timeout_s=60 * 60,
    )
    return {"status": "ok", "start_date": start_date, "end_date": end_date}


@task(name="rates-retrain", retries=0)
def retrain_rates_task(
    *,
    run_id: str | None,
    data_root: Path,
    start_date: str,
    end_date: str,
    train_end_date: str,
    cal_end_date: str,
    feature_set: str,
    run_tag: str,
    recency_half_life_days: float,
) -> dict[str, str]:
    logger = get_run_logger()
    effective_run_id = run_id.strip() if run_id else _default_retrain_run_id()
    logger.info(
        "[rates-retrain] run_id=%s window=[%s..%s] train_end=%s cal_end=%s feature_set=%s half_life=%.2f",
        effective_run_id,
        start_date,
        end_date,
        train_end_date,
        cal_end_date,
        feature_set,
        recency_half_life_days,
    )
    _run_python_module(
        "scripts.rates.train_rates_v1",
        [
            "--data-root",
            str(data_root),
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--train-end-date",
            train_end_date,
            "--cal-end-date",
            cal_end_date,
            "--feature-set",
            feature_set,
            "--run-tag",
            run_tag,
            "--run-id",
            effective_run_id,
            "--recency-half-life-days",
            str(recency_half_life_days),
        ],
        data_root=data_root,
        timeout_s=60 * 60 * 2,
    )
    run_dir = data_root / "artifacts" / "rates_v1" / "runs" / effective_run_id
    if not (run_dir / "meta.json").exists():
        raise RuntimeError(f"retrain completed but meta missing: {run_dir / 'meta.json'}")
    if not (run_dir / "metrics.json").exists():
        raise RuntimeError(f"retrain completed but metrics missing: {run_dir / 'metrics.json'}")
    return {"run_id": effective_run_id, "run_dir": str(run_dir)}


@task(name="rates-calibration-diagnostics", retries=0)
def run_rates_calibration_diagnostics_task(
    *,
    rates_run_id: str,
    data_root: Path,
    start_date: str,
    end_date: str,
    cal_end_date: str,
    output_root: Path,
) -> dict[str, str]:
    logger = get_run_logger()
    output_root.mkdir(parents=True, exist_ok=True)
    # train_end_date is metadata-only in eval_efficiency_heads; use same day as cal_end_date.
    logger.info(
        "[rates-calibration] run_id=%s val_window=(%s, %s] output_root=%s",
        rates_run_id,
        cal_end_date,
        end_date,
        output_root,
    )
    _run_python_module(
        "scripts.rates.eval_efficiency_heads",
        [
            "--data-root",
            str(data_root),
            "--rates-run-id",
            rates_run_id,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--train-end-date",
            cal_end_date,
            "--cal-end-date",
            cal_end_date,
            "--output-root",
            str(output_root),
        ],
        data_root=data_root,
        timeout_s=60 * 30,
    )
    summary_path = output_root / f"efficiency_eval_{rates_run_id}.json"
    if not summary_path.exists():
        raise RuntimeError(f"calibration diagnostics missing expected output: {summary_path}")
    return {"status": "ok", "summary_path": str(summary_path)}


@task(name="rates-head-to-head-eval", retries=0)
def evaluate_candidate_task(
    *,
    data_root: Path,
    current_run_id: str,
    retrain_run_id: str,
    train_end_date: str,
    cal_end_date: str,
    end_date: str,
    normal_slice_start: str | None = None,
    normal_slice_end: str | None = None,
    chaos_slice_start: str | None = None,
    chaos_slice_end: str | None = None,
) -> dict[str, str]:
    logger = get_run_logger()
    default_slices = _default_eval_slices(
        train_end_date=train_end_date,
        cal_end_date=cal_end_date,
        end_date=end_date,
    )
    slices = {
        "normal_pre_deadline": (
            normal_slice_start or default_slices["normal_pre_deadline"][0],
            normal_slice_end or default_slices["normal_pre_deadline"][1],
        ),
        "chaos_deadline": (
            chaos_slice_start or default_slices["chaos_deadline"][0],
            chaos_slice_end or default_slices["chaos_deadline"][1],
        ),
    }
    logger.info(
        "[rates-eval] current=%s retrain=%s normal=%s chaos=%s",
        current_run_id,
        retrain_run_id,
        slices["normal_pre_deadline"],
        slices["chaos_deadline"],
    )
    summary = evaluate_head_to_head_from_rates_base(
        data_root=data_root,
        current_run_id=current_run_id,
        retrain_run_id=retrain_run_id,
        slices=slices,
    )
    out_path = (
        data_root
        / "artifacts"
        / "rates_v1"
        / "runs"
        / retrain_run_id
        / "head_to_head_eval_normal_vs_chaos.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary_path": str(out_path)}


@task(name="rates-promote", retries=0)
def promote_rates_task(
    *,
    data_root: Path,
    source_config_path: Path,
    runtime_config_path: Path,
    repo_config_path: Path,
    new_run_id: str,
    eval_summary_path: str | None,
    guardrail_result: dict[str, Any],
    sync_repo_selector: bool,
) -> dict[str, str]:
    payload = _read_selector_payload(source_config_path)
    if not payload and source_config_path != repo_config_path:
        payload = _read_selector_payload(repo_config_path)
    previous_run_id = str(payload.get("run_id", ""))
    payload["run_id"] = new_run_id
    runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if sync_repo_selector:
        repo_config_path.parent.mkdir(parents=True, exist_ok=True)
        repo_config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    promotions_root = data_root / "artifacts" / "rates_v1" / "promotions"
    promotions_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "timestamp": timestamp,
        "previous_run_id": previous_run_id,
        "new_run_id": new_run_id,
        "eval_summary_path": eval_summary_path,
        "guardrails": guardrail_result,
        "source_config_path": str(source_config_path),
        "runtime_config_path": str(runtime_config_path),
        "repo_config_path": str(repo_config_path),
        "sync_repo_selector": bool(sync_repo_selector),
    }
    history_path = promotions_root / f"promotion_{timestamp}_{new_run_id}.json"
    latest_path = promotions_root / "latest_promotion.json"
    rollback_path = promotions_root / "rollback_pointer.json"
    history_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    rollback_path.write_text(
        json.dumps(
            {
                "rollback_to_run_id": previous_run_id,
                "current_run_id": new_run_id,
                "updated_at": timestamp,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "previous_run_id": previous_run_id,
        "new_run_id": new_run_id,
        "source_selector_path": str(source_config_path),
        "runtime_selector_path": str(runtime_config_path),
        "repo_selector_path": str(repo_config_path),
        "history_path": str(history_path),
        "rollback_pointer_path": str(rollback_path),
    }


@flow(name="rates-retrain-pipeline", log_prints=True)
def rates_retrain_flow(
    *,
    run_id: str | None = None,
    train_window_days: int = 365,
    cal_window_days: int = 14,
    val_window_days: int = 14,
    recency_half_life_days: float = 75.0,
    feature_set: str = "stage5_fta_tracking",
    run_tag: str = "rates_v1_stage5_recency_h75",
    refresh_training_base: bool = True,
    run_calibration_diagnostics: bool = True,
    calibration_window_days: int = 28,
    evaluate_head_to_head: bool = True,
    auto_promote: bool = True,
    max_avg_mae_delta: float = 0.0,
    max_head_mae_regression: float = 0.001,
    allow_guardrail_failure: bool = False,
    biweekly_gate_enabled: bool = True,
    biweekly_anchor_date: str = "2026-02-03",
    normal_slice_start: str | None = None,
    normal_slice_end: str | None = None,
    chaos_slice_start: str | None = None,
    chaos_slice_end: str | None = None,
    sync_repo_selector: bool = False,
) -> dict[str, Any]:
    logger = get_run_logger()
    data_root = paths.get_data_root()
    logger.info("[rates-retrain-flow] data_root=%s", data_root)

    run_day = datetime.now(tz=UTC).date()
    if biweekly_gate_enabled:
        anchor_day = pd.Timestamp(biweekly_anchor_date).date()
        if not _should_run_biweekly(run_day=run_day, anchor_day=anchor_day):
            logger.info(
                "[rates-retrain-flow] biweekly gate skip: run_day=%s anchor_day=%s",
                run_day,
                anchor_day,
            )
            return {
                "status": "skipped_biweekly_gate",
                "run_day": str(run_day),
                "anchor_day": str(anchor_day),
            }

    labels_min, labels_max = _labels_date_bounds(data_root)
    if labels_max is None:
        raise RuntimeError("Cannot retrain rates: gold/labels_minutes_v1 is missing or empty.")
    windows = _resolve_training_window(
        labels_max_date=labels_max,
        labels_min_date=labels_min,
        train_window_days=train_window_days,
        cal_window_days=cal_window_days,
        val_window_days=val_window_days,
    )
    logger.info("[rates-retrain-flow] windows=%s", windows)

    if refresh_training_base:
        refresh_rates_training_base_window_task(
            data_root=data_root,
            start_date=windows["start_date"],
            end_date=windows["end_date"],
        )

    retrain_result = retrain_rates_task(
        run_id=run_id,
        data_root=data_root,
        start_date=windows["start_date"],
        end_date=windows["end_date"],
        train_end_date=windows["train_end_date"],
        cal_end_date=windows["cal_end_date"],
        feature_set=feature_set,
        run_tag=run_tag,
        recency_half_life_days=recency_half_life_days,
    )

    result: dict[str, Any] = {"status": "ok", "windows": windows, "retrain": retrain_result}
    calibration_result: dict[str, str] | None = None
    if run_calibration_diagnostics:
        end_ts = pd.Timestamp(windows["end_date"]).normalize()
        cal_diag_end = end_ts - pd.Timedelta(days=max(calibration_window_days, 1))
        cal_diag_start = end_ts - pd.Timedelta(days=max(calibration_window_days + 14, 21))
        calibration_result = run_rates_calibration_diagnostics_task(
            rates_run_id=retrain_result["run_id"],
            data_root=data_root,
            start_date=str(cal_diag_start.date()),
            end_date=str(end_ts.date()),
            cal_end_date=str(cal_diag_end.date()),
            output_root=Path(retrain_result["run_dir"]) / "calibration",
        )
        result["calibration"] = calibration_result

    eval_result: dict[str, str] | None = None
    guardrail_result: dict[str, Any] | None = None
    active_selector_path = model_selectors.active_rates_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    runtime_selector_path = model_selectors.runtime_rates_selector_path(data_root=data_root)
    repo_selector_path = model_selectors.repo_rates_selector_path(project_root=PROJECT_ROOT)
    logger.info(
        "[rates-retrain-flow] selectors active=%s runtime=%s repo=%s",
        active_selector_path,
        runtime_selector_path,
        repo_selector_path,
    )
    current_run_id = _read_current_rates_run_id(active_selector_path)
    result["current_run_id"] = current_run_id
    result["selector_path"] = str(active_selector_path)
    if evaluate_head_to_head:
        eval_result = evaluate_candidate_task(
            data_root=data_root,
            current_run_id=current_run_id,
            retrain_run_id=retrain_result["run_id"],
            train_end_date=windows["train_end_date"],
            cal_end_date=windows["cal_end_date"],
            end_date=windows["end_date"],
            normal_slice_start=normal_slice_start,
            normal_slice_end=normal_slice_end,
            chaos_slice_start=chaos_slice_start,
            chaos_slice_end=chaos_slice_end,
        )
        result["eval"] = eval_result
        eval_summary = json.loads(Path(eval_result["summary_path"]).read_text(encoding="utf-8"))
        guardrail_result = assess_eval_guardrails(
            eval_summary,
            max_avg_mae_delta=max_avg_mae_delta,
            max_head_mae_regression=max_head_mae_regression,
        )
        result["guardrails"] = guardrail_result
        if not guardrail_result["passed"] and not allow_guardrail_failure:
            raise RuntimeError(
                f"Rates retrain guardrails failed: {json.dumps(guardrail_result, sort_keys=True)}"
            )

    if auto_promote:
        if evaluate_head_to_head and guardrail_result is not None and not guardrail_result["passed"] and not allow_guardrail_failure:
            logger.info("[rates-retrain-flow] skipping promotion due to guardrail failure")
            result["promotion"] = {"status": "skipped_guardrail_failure"}
            return result

        promotion = promote_rates_task(
            data_root=data_root,
            source_config_path=active_selector_path,
            runtime_config_path=runtime_selector_path,
            repo_config_path=repo_selector_path,
            new_run_id=retrain_result["run_id"],
            eval_summary_path=eval_result["summary_path"] if eval_result else None,
            guardrail_result=guardrail_result or {"passed": True, "note": "head-to-head evaluation disabled"},
            sync_repo_selector=sync_repo_selector,
        )
        result["promotion"] = promotion

    return result


@flow(name="rates-calibration-monitor", log_prints=True)
def rates_calibration_monitor_flow(
    *,
    run_id: str | None = None,
    calibration_window_days: int = 28,
    lookback_days: int = 120,
) -> dict[str, Any]:
    data_root = paths.get_data_root()
    labels_min, labels_max = _labels_date_bounds(data_root)
    if labels_max is None:
        raise RuntimeError("Cannot run rates calibration monitor: gold/labels_minutes_v1 is missing or empty.")

    active_selector_path = model_selectors.active_rates_selector_path(
        data_root=data_root,
        project_root=PROJECT_ROOT,
    )
    effective_run_id = run_id.strip() if run_id else _read_current_rates_run_id(active_selector_path)
    end_ts = labels_max.normalize()
    start_ts = max(labels_min or end_ts, end_ts - pd.Timedelta(days=max(lookback_days, calibration_window_days + 14)))
    cal_end_ts = end_ts - pd.Timedelta(days=max(calibration_window_days, 1))
    output_root = data_root / "artifacts" / "rates_v1" / "runs" / effective_run_id / "calibration_monitor"
    result = run_rates_calibration_diagnostics_task(
        rates_run_id=effective_run_id,
        data_root=data_root,
        start_date=str(start_ts.date()),
        end_date=str(end_ts.date()),
        cal_end_date=str(cal_end_ts.date()),
        output_root=output_root,
    )
    return {
        "run_id": effective_run_id,
        "labels_max_date": str(end_ts.date()),
        "calibration_window_days": calibration_window_days,
        "result": result,
    }


if __name__ == "__main__":  # pragma: no cover
    rates_retrain_flow()
