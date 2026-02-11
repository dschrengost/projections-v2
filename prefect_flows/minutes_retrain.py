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
import warnings

import pandas as pd
from prefect import flow, get_run_logger, task

from projections import model_selectors, paths
from projections.validation.data_quality import MINUTES_FEATURE_BOUNDS, validate_feature_ranges


PROJECT_ROOT = paths.get_project_root()
_DEFAULT_UV_PATH = Path("/home/daniel/.local/bin/uv")

DEFAULT_MIN_EVAL_ROWS_PER_SLICE = 1000
DEFAULT_MAX_WEIGHTED_BRIER_DELTA = 0.005
DEFAULT_MAX_WEIGHTED_FALSE_ACTIVE_DELTA = 0.010
DEFAULT_MAX_WEIGHTED_MAE_P50_COND_DELTA = 0.100
DEFAULT_MAX_SLICE_BRIER_DELTA = 0.010
DEFAULT_MAX_SLICE_FALSE_ACTIVE_DELTA = 0.015
DEFAULT_MAX_SLICE_MAE_P50_COND_DELTA = 0.200
DEFAULT_MAX_WEIGHTED_MAE_P50_COND_DELTA_SOFT = 0.400
DEFAULT_MAX_SLICE_MAE_P50_COND_DELTA_SOFT = 0.600
DEFAULT_MAX_ABS_P10_COVERAGE_DELTA = 0.040
DEFAULT_MAX_ABS_P90_COVERAGE_DELTA = 0.020
DEFAULT_EVAL_CANDIDATE_VARIANT = "retrain_occupancy_v0"
DEFAULT_ADAPTIVE_LOOKBACK_RUNS = 20
DEFAULT_ADAPTIVE_MIN_HISTORY_RUNS = 6
DEFAULT_ADAPTIVE_IQR_MULTIPLIER = 1.5
DEFAULT_ADAPTIVE_THRESHOLD_CAP_MULTIPLIER = 3.0
DEFAULT_DRIFT_OVERRIDE_RELAX_MULTIPLIER = 1.25
DEFAULT_DRIFT_OVERRIDE_MIN_TRIGGER_METRICS = 2
DEFAULT_DRIFT_OVERRIDE_IQR_MULTIPLIER = 1.5
DEFAULT_DRIFT_MIN_ABS_BRIER = 0.01
DEFAULT_DRIFT_MIN_ABS_FALSE_ACTIVE = 0.01
DEFAULT_DRIFT_MIN_ABS_MAE = 0.5


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


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _history_diagnosis_paths(*, data_root: Path) -> list[Path]:
    root = data_root / "artifacts" / "minutes_eval_runs"
    if not root.exists():
        return []
    paths = list(root.glob("*/guardrail_diagnosis.json"))
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def _collect_recent_minutes_gate_history(
    *,
    data_root: Path,
    exclude_eval_run_id: str | None,
    lookback_runs: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in _history_diagnosis_paths(data_root=data_root):
        if len(out) >= max(lookback_runs, 0):
            break
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        eval_run_id = str(Path(payload.get("eval_summary_path", "")).parent.name or "")
        if exclude_eval_run_id and eval_run_id == exclude_eval_run_id:
            continue
        out.append(payload)
    return out


def _extract_history_metric_values(
    *,
    history_runs: list[dict[str, Any]],
    metric: str,
    source: str,
    absolute: bool,
    positive_only: bool,
) -> list[float]:
    values: list[float] = []
    for run in history_runs:
        if not bool((run.get("data_quality") or {}).get("passed", True)):
            continue
        guardrails = run.get("guardrails") or {}
        if source == "weighted":
            triplet = (guardrails.get("weighted_metrics") or {}).get(metric) or {}
            value = _safe_float(triplet.get("delta"))
            if value is None:
                continue
            parsed = abs(value) if absolute else value
            if positive_only and parsed <= 0:
                continue
            values.append(parsed)
            continue

        if source == "slice":
            for payload in (guardrails.get("slice_diagnostics") or {}).values():
                delta_map = payload.get("delta_retrain_minus_current") or {}
                value = _safe_float(delta_map.get(metric))
                if value is None:
                    continue
                parsed = abs(value) if absolute else value
                if positive_only and parsed <= 0:
                    continue
                values.append(parsed)
            continue
    return values


def _adaptive_threshold(
    *,
    base_threshold: float,
    history_values: list[float],
    min_history_runs: int,
    iqr_multiplier: float,
    cap_multiplier: float,
) -> tuple[float, dict[str, Any]]:
    if len(history_values) < max(min_history_runs, 1):
        return base_threshold, {"used_history": False, "history_count": len(history_values)}
    series = pd.Series(history_values, dtype=float)
    med = float(series.median())
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = max(0.0, q3 - q1)
    candidate = med + max(0.0, float(iqr_multiplier)) * iqr
    threshold = max(float(base_threshold), candidate)
    if base_threshold > 0 and cap_multiplier > 0:
        threshold = min(threshold, float(base_threshold) * float(cap_multiplier))
    return threshold, {
        "used_history": True,
        "history_count": len(history_values),
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "candidate": candidate,
        "base_threshold": float(base_threshold),
        "threshold": threshold,
    }


def _assess_current_bundle_drift(
    *,
    eval_summary: dict[str, Any],
    history_runs: list[dict[str, Any]],
    min_history_runs: int,
    iqr_multiplier: float,
    min_trigger_metrics: int,
) -> dict[str, Any]:
    metric_min_abs = {
        "brier_play_prob": DEFAULT_DRIFT_MIN_ABS_BRIER,
        "false_active_rate_p_ge_0_5": DEFAULT_DRIFT_MIN_ABS_FALSE_ACTIVE,
        "mae_p50_conditional": DEFAULT_DRIFT_MIN_ABS_MAE,
    }
    triggers: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []
    for metric, min_abs in metric_min_abs.items():
        current_triplet = _weighted_metric_triplet(eval_summary, metric, candidate_variant="retrain")
        current_value = _safe_float(current_triplet.get("current"))
        history_values: list[float] = []
        for run in history_runs:
            if not bool((run.get("data_quality") or {}).get("passed", True)):
                continue
            weighted = ((run.get("guardrails") or {}).get("weighted_metrics") or {}).get(metric) or {}
            value = _safe_float(weighted.get("current"))
            if value is not None:
                history_values.append(value)
        info: dict[str, Any] = {
            "metric": metric,
            "current": current_value,
            "history_count": len(history_values),
        }
        if current_value is None or len(history_values) < max(min_history_runs, 1):
            inspected.append(info)
            continue
        series = pd.Series(history_values, dtype=float)
        med = float(series.median())
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = max(0.0, q3 - q1)
        line = med + max(float(min_abs), max(0.0, float(iqr_multiplier)) * iqr)
        trigger = current_value > line
        info.update(
            {
                "median": med,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "trigger_line": line,
                "triggered": trigger,
            }
        )
        if trigger:
            triggers.append(info)
        inspected.append(info)
    degraded = len(triggers) >= max(int(min_trigger_metrics), 1)
    return {
        "degraded": degraded,
        "trigger_count": len(triggers),
        "min_trigger_metrics": int(min_trigger_metrics),
        "triggers": triggers,
        "inspected": inspected,
    }


def _candidate_metric_keys(candidate_variant: str) -> tuple[str, str]:
    variant = str(candidate_variant or "retrain").strip().lower()
    if variant in {"retrain", "base", "retrain_base"}:
        return "metrics_retrain", "delta_retrain_minus_current"
    if variant in {
        "retrain_occupancy_v0",
        "occupancy_sparse_v0",
        "occupancy_v0",
    }:
        return "metrics_retrain_occupancy_v0", "delta_retrain_occupancy_v0_minus_current"
    raise ValueError(f"Unsupported eval candidate variant: {candidate_variant}")


def _weighted_metric_triplet(
    summary: dict[str, Any],
    metric: str,
    *,
    candidate_variant: str,
) -> dict[str, float | None]:
    slices = summary.get("slices") or {}
    candidate_metrics_key, _ = _candidate_metric_keys(candidate_variant)
    total_rows = 0.0
    weighted_current = 0.0
    weighted_retrain = 0.0
    for payload in slices.values():
        metrics_current = payload.get("metrics_current") or {}
        metrics_retrain = payload.get(candidate_metrics_key) or payload.get("metrics_retrain") or {}
        rows = _safe_float(metrics_current.get("rows"))
        cur = _safe_float(metrics_current.get(metric))
        ret = _safe_float(metrics_retrain.get(metric))
        if rows is None or rows <= 0 or cur is None or ret is None:
            continue
        total_rows += rows
        weighted_current += rows * cur
        weighted_retrain += rows * ret
    if total_rows <= 0:
        return {"rows": 0.0, "current": None, "retrain": None, "delta": None}
    current = weighted_current / total_rows
    retrain = weighted_retrain / total_rows
    return {"rows": total_rows, "current": current, "retrain": retrain, "delta": retrain - current}


def assess_minutes_eval_guardrails(
    eval_summary: dict[str, Any],
    *,
    eval_candidate_variant: str = DEFAULT_EVAL_CANDIDATE_VARIANT,
    history_runs: list[dict[str, Any]] | None = None,
    adaptive_thresholds_enabled: bool = True,
    adaptive_min_history_runs: int = DEFAULT_ADAPTIVE_MIN_HISTORY_RUNS,
    adaptive_iqr_multiplier: float = DEFAULT_ADAPTIVE_IQR_MULTIPLIER,
    adaptive_threshold_cap_multiplier: float = DEFAULT_ADAPTIVE_THRESHOLD_CAP_MULTIPLIER,
    drift_override_enabled: bool = True,
    drift_override_relax_multiplier: float = DEFAULT_DRIFT_OVERRIDE_RELAX_MULTIPLIER,
    drift_override_min_trigger_metrics: int = DEFAULT_DRIFT_OVERRIDE_MIN_TRIGGER_METRICS,
    drift_override_iqr_multiplier: float = DEFAULT_DRIFT_OVERRIDE_IQR_MULTIPLIER,
    min_eval_rows_per_slice: int = DEFAULT_MIN_EVAL_ROWS_PER_SLICE,
    max_weighted_brier_delta: float = DEFAULT_MAX_WEIGHTED_BRIER_DELTA,
    max_weighted_false_active_delta: float = DEFAULT_MAX_WEIGHTED_FALSE_ACTIVE_DELTA,
    max_weighted_mae_p50_cond_delta: float = DEFAULT_MAX_WEIGHTED_MAE_P50_COND_DELTA,
    max_weighted_mae_p50_cond_delta_soft: float | None = DEFAULT_MAX_WEIGHTED_MAE_P50_COND_DELTA_SOFT,
    max_slice_brier_delta: float = DEFAULT_MAX_SLICE_BRIER_DELTA,
    max_slice_false_active_delta: float = DEFAULT_MAX_SLICE_FALSE_ACTIVE_DELTA,
    max_slice_mae_p50_cond_delta: float = DEFAULT_MAX_SLICE_MAE_P50_COND_DELTA,
    max_slice_mae_p50_cond_delta_soft: float | None = DEFAULT_MAX_SLICE_MAE_P50_COND_DELTA_SOFT,
    max_abs_p10_coverage_delta: float = DEFAULT_MAX_ABS_P10_COVERAGE_DELTA,
    max_abs_p90_coverage_delta: float = DEFAULT_MAX_ABS_P90_COVERAGE_DELTA,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    soft_failures: list[dict[str, Any]] = []
    slices = eval_summary.get("slices") or {}
    slice_diagnostics: dict[str, dict[str, Any]] = {}
    history_payloads = history_runs or []
    candidate_metrics_key, candidate_delta_key = _candidate_metric_keys(eval_candidate_variant)

    final_max_weighted_brier_delta = float(max_weighted_brier_delta)
    final_max_weighted_false_active_delta = float(max_weighted_false_active_delta)
    final_max_weighted_mae_p50_cond_delta = float(max_weighted_mae_p50_cond_delta)
    requested_weighted_mae_soft = _safe_float(max_weighted_mae_p50_cond_delta_soft)
    final_max_weighted_mae_p50_cond_delta_soft = (
        max(float(max_weighted_mae_p50_cond_delta), requested_weighted_mae_soft)
        if requested_weighted_mae_soft is not None
        else float(max_weighted_mae_p50_cond_delta)
    )
    final_max_slice_brier_delta = float(max_slice_brier_delta)
    final_max_slice_false_active_delta = float(max_slice_false_active_delta)
    final_max_slice_mae_p50_cond_delta = float(max_slice_mae_p50_cond_delta)
    requested_slice_mae_soft = _safe_float(max_slice_mae_p50_cond_delta_soft)
    final_max_slice_mae_p50_cond_delta_soft = (
        max(float(max_slice_mae_p50_cond_delta), requested_slice_mae_soft)
        if requested_slice_mae_soft is not None
        else float(max_slice_mae_p50_cond_delta)
    )
    final_max_abs_p10_coverage_delta = float(max_abs_p10_coverage_delta)
    final_max_abs_p90_coverage_delta = float(max_abs_p90_coverage_delta)

    adaptive_details: dict[str, Any] = {}
    if adaptive_thresholds_enabled:
        history_weighted_brier = _extract_history_metric_values(
            history_runs=history_payloads,
            metric="brier_play_prob",
            source="weighted",
            absolute=False,
            positive_only=True,
        )
        history_weighted_false_active = _extract_history_metric_values(
            history_runs=history_payloads,
            metric="false_active_rate_p_ge_0_5",
            source="weighted",
            absolute=False,
            positive_only=True,
        )
        history_weighted_mae = _extract_history_metric_values(
            history_runs=history_payloads,
            metric="mae_p50_conditional",
            source="weighted",
            absolute=False,
            positive_only=True,
        )
        history_slice_brier = _extract_history_metric_values(
            history_runs=history_payloads,
            metric="brier_play_prob",
            source="slice",
            absolute=False,
            positive_only=True,
        )
        history_slice_false_active = _extract_history_metric_values(
            history_runs=history_payloads,
            metric="false_active_rate_p_ge_0_5",
            source="slice",
            absolute=False,
            positive_only=True,
        )
        history_slice_mae = _extract_history_metric_values(
            history_runs=history_payloads,
            metric="mae_p50_conditional",
            source="slice",
            absolute=False,
            positive_only=True,
        )
        history_slice_p10_abs = _extract_history_metric_values(
            history_runs=history_payloads,
            metric="p10_coverage_leq",
            source="slice",
            absolute=True,
            positive_only=False,
        )
        history_slice_p90_abs = _extract_history_metric_values(
            history_runs=history_payloads,
            metric="p90_coverage_leq",
            source="slice",
            absolute=True,
            positive_only=False,
        )

        final_max_weighted_brier_delta, adaptive_details["max_weighted_brier_delta"] = _adaptive_threshold(
            base_threshold=final_max_weighted_brier_delta,
            history_values=history_weighted_brier,
            min_history_runs=adaptive_min_history_runs,
            iqr_multiplier=adaptive_iqr_multiplier,
            cap_multiplier=adaptive_threshold_cap_multiplier,
        )
        final_max_weighted_false_active_delta, adaptive_details["max_weighted_false_active_delta"] = _adaptive_threshold(
            base_threshold=final_max_weighted_false_active_delta,
            history_values=history_weighted_false_active,
            min_history_runs=adaptive_min_history_runs,
            iqr_multiplier=adaptive_iqr_multiplier,
            cap_multiplier=adaptive_threshold_cap_multiplier,
        )
        final_max_weighted_mae_p50_cond_delta, adaptive_details["max_weighted_mae_p50_cond_delta"] = _adaptive_threshold(
            base_threshold=final_max_weighted_mae_p50_cond_delta,
            history_values=history_weighted_mae,
            min_history_runs=adaptive_min_history_runs,
            iqr_multiplier=adaptive_iqr_multiplier,
            cap_multiplier=adaptive_threshold_cap_multiplier,
        )
        final_max_slice_brier_delta, adaptive_details["max_slice_brier_delta"] = _adaptive_threshold(
            base_threshold=final_max_slice_brier_delta,
            history_values=history_slice_brier,
            min_history_runs=adaptive_min_history_runs,
            iqr_multiplier=adaptive_iqr_multiplier,
            cap_multiplier=adaptive_threshold_cap_multiplier,
        )
        final_max_slice_false_active_delta, adaptive_details["max_slice_false_active_delta"] = _adaptive_threshold(
            base_threshold=final_max_slice_false_active_delta,
            history_values=history_slice_false_active,
            min_history_runs=adaptive_min_history_runs,
            iqr_multiplier=adaptive_iqr_multiplier,
            cap_multiplier=adaptive_threshold_cap_multiplier,
        )
        final_max_slice_mae_p50_cond_delta, adaptive_details["max_slice_mae_p50_cond_delta"] = _adaptive_threshold(
            base_threshold=final_max_slice_mae_p50_cond_delta,
            history_values=history_slice_mae,
            min_history_runs=adaptive_min_history_runs,
            iqr_multiplier=adaptive_iqr_multiplier,
            cap_multiplier=adaptive_threshold_cap_multiplier,
        )
        final_max_abs_p10_coverage_delta, adaptive_details["max_abs_p10_coverage_delta"] = _adaptive_threshold(
            base_threshold=final_max_abs_p10_coverage_delta,
            history_values=history_slice_p10_abs,
            min_history_runs=adaptive_min_history_runs,
            iqr_multiplier=adaptive_iqr_multiplier,
            cap_multiplier=adaptive_threshold_cap_multiplier,
        )
        final_max_abs_p90_coverage_delta, adaptive_details["max_abs_p90_coverage_delta"] = _adaptive_threshold(
            base_threshold=final_max_abs_p90_coverage_delta,
            history_values=history_slice_p90_abs,
            min_history_runs=adaptive_min_history_runs,
            iqr_multiplier=adaptive_iqr_multiplier,
            cap_multiplier=adaptive_threshold_cap_multiplier,
        )

    drift_assessment = {
        "degraded": False,
        "trigger_count": 0,
        "min_trigger_metrics": int(drift_override_min_trigger_metrics),
        "triggers": [],
        "inspected": [],
    }
    drift_override_applied = False
    if drift_override_enabled:
        drift_assessment = _assess_current_bundle_drift(
            eval_summary=eval_summary,
            history_runs=history_payloads,
            min_history_runs=adaptive_min_history_runs,
            iqr_multiplier=drift_override_iqr_multiplier,
            min_trigger_metrics=drift_override_min_trigger_metrics,
        )
        if drift_assessment.get("degraded"):
            relax = max(1.0, float(drift_override_relax_multiplier))
            final_max_weighted_brier_delta *= relax
            final_max_weighted_false_active_delta *= relax
            final_max_weighted_mae_p50_cond_delta *= relax
            final_max_slice_brier_delta *= relax
            final_max_slice_false_active_delta *= relax
            final_max_slice_mae_p50_cond_delta *= relax
            drift_override_applied = True

    final_max_weighted_mae_p50_cond_delta_soft = max(
        float(final_max_weighted_mae_p50_cond_delta_soft),
        float(final_max_weighted_mae_p50_cond_delta),
    )
    final_max_slice_mae_p50_cond_delta_soft = max(
        float(final_max_slice_mae_p50_cond_delta_soft),
        float(final_max_slice_mae_p50_cond_delta),
    )

    if not slices:
        failures.append(
            {
                "scope": "eval",
                "metric": "slices",
                "reason": "missing_slices",
            }
        )

    for slice_name, payload in slices.items():
        metrics_current = payload.get("metrics_current") or {}
        metrics_candidate_raw = payload.get(candidate_metrics_key)
        delta_candidate_raw = payload.get(candidate_delta_key)
        fallback_used = False
        if not isinstance(metrics_candidate_raw, dict) or not isinstance(delta_candidate_raw, dict):
            metrics_candidate_raw = payload.get("metrics_retrain") or {}
            delta_candidate_raw = payload.get("delta_retrain_minus_current") or {}
            fallback_used = True
        metrics_retrain = metrics_candidate_raw or {}
        delta = delta_candidate_raw or {}
        rows = int(_safe_float(metrics_current.get("rows")) or 0)
        slice_diagnostics[slice_name] = {
            "rows": rows,
            "metrics_current": metrics_current,
            "metrics_retrain": metrics_retrain,
            "delta_retrain_minus_current": delta,
            "candidate_variant_requested": eval_candidate_variant,
            "candidate_variant_used": "retrain" if fallback_used else eval_candidate_variant,
        }

        if rows < min_eval_rows_per_slice:
            failures.append(
                {
                    "scope": "slice",
                    "slice": slice_name,
                    "metric": "rows",
                    "rows": rows,
                    "threshold": min_eval_rows_per_slice,
                    "reason": "insufficient_eval_rows",
                }
            )

        for metric, threshold in (
            ("brier_play_prob", final_max_slice_brier_delta),
            ("false_active_rate_p_ge_0_5", final_max_slice_false_active_delta),
            ("mae_p50_conditional", final_max_slice_mae_p50_cond_delta),
        ):
            delta_value = _safe_float(delta.get(metric))
            if delta_value is None:
                continue
            threshold_to_use = threshold
            is_soft_mae = metric == "mae_p50_conditional"
            if is_soft_mae:
                threshold_to_use = float(final_max_slice_mae_p50_cond_delta_soft)
            if delta_value > threshold_to_use:
                failures.append(
                    {
                        "scope": "slice",
                        "slice": slice_name,
                        "metric": metric,
                        "delta": delta_value,
                        "threshold": threshold_to_use,
                        "current": _safe_float(metrics_current.get(metric)),
                        "retrain": _safe_float(metrics_retrain.get(metric)),
                        "hard_threshold": threshold if is_soft_mae else None,
                    }
                )
            elif is_soft_mae and delta_value > threshold:
                soft_failures.append(
                    {
                        "scope": "slice",
                        "slice": slice_name,
                        "metric": metric,
                        "delta": delta_value,
                        "threshold": threshold,
                        "soft_threshold": threshold_to_use,
                        "current": _safe_float(metrics_current.get(metric)),
                        "retrain": _safe_float(metrics_retrain.get(metric)),
                    }
                )

        for metric, threshold in (
            ("p10_coverage_leq", final_max_abs_p10_coverage_delta),
            ("p90_coverage_leq", final_max_abs_p90_coverage_delta),
        ):
            delta_value = _safe_float(delta.get(metric))
            if delta_value is None:
                continue
            if abs(delta_value) > threshold:
                failures.append(
                    {
                        "scope": "slice",
                        "slice": slice_name,
                        "metric": metric,
                        "delta": delta_value,
                        "threshold": threshold,
                        "current": _safe_float(metrics_current.get(metric)),
                        "retrain": _safe_float(metrics_retrain.get(metric)),
                        "reason": "coverage_drift",
                    }
                )

    weighted_checks = {
        "brier_play_prob": final_max_weighted_brier_delta,
        "false_active_rate_p_ge_0_5": final_max_weighted_false_active_delta,
        "mae_p50_conditional": final_max_weighted_mae_p50_cond_delta,
    }
    weighted_metrics: dict[str, dict[str, float | None]] = {}
    for metric, threshold in weighted_checks.items():
        triplet = _weighted_metric_triplet(
            eval_summary,
            metric,
            candidate_variant=eval_candidate_variant,
        )
        weighted_metrics[metric] = triplet
        delta_value = triplet.get("delta")
        if delta_value is None:
            failures.append(
                {
                    "scope": "weighted",
                    "metric": metric,
                    "reason": "metric_missing",
                }
            )
            continue
        threshold_to_use = threshold
        is_soft_mae = metric == "mae_p50_conditional"
        if is_soft_mae:
            threshold_to_use = float(final_max_weighted_mae_p50_cond_delta_soft)
        if delta_value > threshold_to_use:
            failures.append(
                {
                    "scope": "weighted",
                    "metric": metric,
                    "delta": delta_value,
                    "threshold": threshold_to_use,
                    "current": triplet.get("current"),
                    "retrain": triplet.get("retrain"),
                    "hard_threshold": threshold if is_soft_mae else None,
                }
            )
        elif is_soft_mae and delta_value > threshold:
            soft_failures.append(
                {
                    "scope": "weighted",
                    "metric": metric,
                    "delta": delta_value,
                    "threshold": threshold,
                    "soft_threshold": threshold_to_use,
                    "current": triplet.get("current"),
                    "retrain": triplet.get("retrain"),
                }
            )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "soft_failures": soft_failures,
        "candidate_variant_requested": eval_candidate_variant,
        "candidate_metric_key": candidate_metrics_key,
        "slice_diagnostics": slice_diagnostics,
        "weighted_metrics": weighted_metrics,
        "adaptive": {
            "enabled": bool(adaptive_thresholds_enabled),
            "history_runs_considered": len(history_payloads),
            "history_runs_with_data_quality": int(
                sum(1 for run in history_payloads if bool((run.get("data_quality") or {}).get("passed", True)))
            ),
            "details": adaptive_details,
        },
        "drift_override": {
            "enabled": bool(drift_override_enabled),
            "applied": drift_override_applied,
            "relax_multiplier": float(drift_override_relax_multiplier),
            "assessment": drift_assessment,
        },
        "thresholds": {
            "min_eval_rows_per_slice": min_eval_rows_per_slice,
            "max_weighted_brier_delta": final_max_weighted_brier_delta,
            "max_weighted_false_active_delta": final_max_weighted_false_active_delta,
            "max_weighted_mae_p50_cond_delta": final_max_weighted_mae_p50_cond_delta,
            "max_weighted_mae_p50_cond_delta_soft": final_max_weighted_mae_p50_cond_delta_soft,
            "max_slice_brier_delta": final_max_slice_brier_delta,
            "max_slice_false_active_delta": final_max_slice_false_active_delta,
            "max_slice_mae_p50_cond_delta": final_max_slice_mae_p50_cond_delta,
            "max_slice_mae_p50_cond_delta_soft": final_max_slice_mae_p50_cond_delta_soft,
            "max_abs_p10_coverage_delta": final_max_abs_p10_coverage_delta,
            "max_abs_p90_coverage_delta": final_max_abs_p90_coverage_delta,
            "base_eval_candidate_variant": str(eval_candidate_variant),
            "base_max_weighted_brier_delta": float(max_weighted_brier_delta),
            "base_max_weighted_false_active_delta": float(max_weighted_false_active_delta),
            "base_max_weighted_mae_p50_cond_delta": float(max_weighted_mae_p50_cond_delta),
            "base_max_weighted_mae_p50_cond_delta_soft": _safe_float(max_weighted_mae_p50_cond_delta_soft),
            "base_max_slice_brier_delta": float(max_slice_brier_delta),
            "base_max_slice_false_active_delta": float(max_slice_false_active_delta),
            "base_max_slice_mae_p50_cond_delta": float(max_slice_mae_p50_cond_delta),
            "base_max_slice_mae_p50_cond_delta_soft": _safe_float(max_slice_mae_p50_cond_delta_soft),
            "base_max_abs_p10_coverage_delta": float(max_abs_p10_coverage_delta),
            "base_max_abs_p90_coverage_delta": float(max_abs_p90_coverage_delta),
        },
    }


def _summarize_retrain_data_quality(*, data_root: Path, retrain_run_id: str) -> dict[str, Any]:
    run_dir = data_root / "artifacts" / "minutes_retrain_runs" / retrain_run_id
    meta_path = run_dir / "meta.json"
    dataset_path = run_dir / "dataset.parquet"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw_path = meta.get("dataset_path")
        if raw_path:
            dataset_path = Path(str(raw_path))

    payload: dict[str, Any] = {
        "retrain_run_id": retrain_run_id,
        "run_dir": str(run_dir),
        "meta_path": str(meta_path),
        "dataset_path": str(dataset_path),
    }

    if not dataset_path.exists():
        payload["passed"] = False
        payload["violations"] = [f"dataset missing: {dataset_path}"]
        payload["violation_count"] = 1
        return payload

    df = pd.read_parquet(dataset_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        violations = validate_feature_ranges(df, bounds=MINUTES_FEATURE_BOUNDS, strict=False)

    payload["rows"] = int(len(df))
    payload["violations"] = violations
    payload["violation_count"] = len(violations)
    payload["passed"] = len(violations) == 0
    if "minutes" in df.columns:
        payload["minutes_max"] = _safe_float(pd.to_numeric(df["minutes"], errors="coerce").max())
    if "days_since_last" in df.columns:
        days = pd.to_numeric(df["days_since_last"], errors="coerce")
        payload["days_since_last_min"] = _safe_float(days.min())
        payload["days_since_last_negative_rows"] = int((days < 0).sum())
    return payload


def classify_minutes_quality_outcome(
    *,
    guardrails: dict[str, Any],
    data_quality: dict[str, Any],
) -> dict[str, Any]:
    dq_violations = data_quality.get("violations") or []
    if dq_violations:
        return {
            "classification": "data_issue",
            "reason": "data_quality_violations",
            "details": dq_violations[:10],
        }

    if guardrails.get("passed"):
        return {"classification": "pass", "reason": "all_gates_passed", "details": []}

    failures = guardrails.get("failures") or []
    thresholds = guardrails.get("thresholds") or {}
    slice_diags = guardrails.get("slice_diagnostics") or {}

    material_regression = False
    for failure in failures:
        metric = str(failure.get("metric"))
        delta_value = _safe_float(failure.get("delta"))
        threshold = _safe_float(failure.get("threshold"))
        if delta_value is None or threshold is None:
            continue
        if metric in {"brier_play_prob", "false_active_rate_p_ge_0_5", "mae_p50_conditional"}:
            if delta_value > (threshold * 1.5):
                material_regression = True
                break

    max_slice_brier_delta = float(thresholds.get("max_slice_brier_delta", DEFAULT_MAX_SLICE_BRIER_DELTA))
    max_slice_false_active_delta = float(
        thresholds.get("max_slice_false_active_delta", DEFAULT_MAX_SLICE_FALSE_ACTIVE_DELTA)
    )
    max_slice_mae_delta = float(thresholds.get("max_slice_mae_p50_cond_delta", DEFAULT_MAX_SLICE_MAE_P50_COND_DELTA))

    hard_slice_signals: list[dict[str, Any]] = []
    for slice_name, payload in slice_diags.items():
        metrics_current = payload.get("metrics_current") or {}
        delta = payload.get("delta_retrain_minus_current") or {}

        current_brier = _safe_float(metrics_current.get("brier_play_prob"))
        delta_brier = _safe_float(delta.get("brier_play_prob"))
        if current_brier is not None and delta_brier is not None:
            if current_brier >= 0.17 and delta_brier <= max_slice_brier_delta:
                hard_slice_signals.append(
                    {
                        "slice": slice_name,
                        "metric": "brier_play_prob",
                        "current": current_brier,
                        "delta": delta_brier,
                    }
                )

        current_false_active = _safe_float(metrics_current.get("false_active_rate_p_ge_0_5"))
        delta_false_active = _safe_float(delta.get("false_active_rate_p_ge_0_5"))
        if current_false_active is not None and delta_false_active is not None:
            if current_false_active >= 0.14 and delta_false_active <= max_slice_false_active_delta:
                hard_slice_signals.append(
                    {
                        "slice": slice_name,
                        "metric": "false_active_rate_p_ge_0_5",
                        "current": current_false_active,
                        "delta": delta_false_active,
                    }
                )

        current_mae = _safe_float(metrics_current.get("mae_p50_conditional"))
        delta_mae = _safe_float(delta.get("mae_p50_conditional"))
        if current_mae is not None and delta_mae is not None:
            if current_mae >= 6.8 and delta_mae <= max_slice_mae_delta:
                hard_slice_signals.append(
                    {
                        "slice": slice_name,
                        "metric": "mae_p50_conditional",
                        "current": current_mae,
                        "delta": delta_mae,
                    }
                )

    if hard_slice_signals and not material_regression:
        return {
            "classification": "hard_slice",
            "reason": "current_bundle_underperforms_on_slice_with_small_candidate_delta",
            "details": hard_slice_signals[:10],
        }

    return {
        "classification": "model_regression",
        "reason": "candidate_regresses_beyond_thresholds",
        "details": failures[:10],
    }


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


@task(name="minutes-quality-gates", retries=0)
def assess_quality_task(
    *,
    retrain_run_id: str,
    eval_summary_path: str,
    data_root: Path,
    eval_candidate_variant: str,
    adaptive_thresholds_enabled: bool,
    adaptive_lookback_runs: int,
    adaptive_min_history_runs: int,
    adaptive_iqr_multiplier: float,
    adaptive_threshold_cap_multiplier: float,
    drift_override_enabled: bool,
    drift_override_relax_multiplier: float,
    drift_override_min_trigger_metrics: int,
    drift_override_iqr_multiplier: float,
    min_eval_rows_per_slice: int,
    max_weighted_brier_delta: float,
    max_weighted_false_active_delta: float,
    max_weighted_mae_p50_cond_delta: float,
    max_weighted_mae_p50_cond_delta_soft: float | None,
    max_slice_brier_delta: float,
    max_slice_false_active_delta: float,
    max_slice_mae_p50_cond_delta: float,
    max_slice_mae_p50_cond_delta_soft: float | None,
    max_abs_p10_coverage_delta: float,
    max_abs_p90_coverage_delta: float,
) -> dict[str, Any]:
    summary_path = Path(eval_summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history_runs = _collect_recent_minutes_gate_history(
        data_root=data_root,
        exclude_eval_run_id=str(summary.get("eval_run_id") or ""),
        lookback_runs=adaptive_lookback_runs,
    )
    guardrails = assess_minutes_eval_guardrails(
        summary,
        eval_candidate_variant=eval_candidate_variant,
        history_runs=history_runs,
        adaptive_thresholds_enabled=adaptive_thresholds_enabled,
        adaptive_min_history_runs=adaptive_min_history_runs,
        adaptive_iqr_multiplier=adaptive_iqr_multiplier,
        adaptive_threshold_cap_multiplier=adaptive_threshold_cap_multiplier,
        drift_override_enabled=drift_override_enabled,
        drift_override_relax_multiplier=drift_override_relax_multiplier,
        drift_override_min_trigger_metrics=drift_override_min_trigger_metrics,
        drift_override_iqr_multiplier=drift_override_iqr_multiplier,
        min_eval_rows_per_slice=min_eval_rows_per_slice,
        max_weighted_brier_delta=max_weighted_brier_delta,
        max_weighted_false_active_delta=max_weighted_false_active_delta,
        max_weighted_mae_p50_cond_delta=max_weighted_mae_p50_cond_delta,
        max_weighted_mae_p50_cond_delta_soft=max_weighted_mae_p50_cond_delta_soft,
        max_slice_brier_delta=max_slice_brier_delta,
        max_slice_false_active_delta=max_slice_false_active_delta,
        max_slice_mae_p50_cond_delta=max_slice_mae_p50_cond_delta,
        max_slice_mae_p50_cond_delta_soft=max_slice_mae_p50_cond_delta_soft,
        max_abs_p10_coverage_delta=max_abs_p10_coverage_delta,
        max_abs_p90_coverage_delta=max_abs_p90_coverage_delta,
    )
    data_quality = _summarize_retrain_data_quality(data_root=data_root, retrain_run_id=retrain_run_id)
    diagnosis = classify_minutes_quality_outcome(guardrails=guardrails, data_quality=data_quality)
    passed = bool(guardrails.get("passed")) and bool(data_quality.get("passed"))

    payload = {
        "passed": passed,
        "classification": diagnosis["classification"],
        "reason": diagnosis["reason"],
        "details": diagnosis["details"],
        "guardrails": guardrails,
        "data_quality": data_quality,
        "eval_candidate_variant": eval_candidate_variant,
        "eval_summary_path": str(summary_path),
        "retrain_run_id": retrain_run_id,
        "history_runs_used": len(history_runs),
    }
    diagnosis_path = summary_path.with_name("guardrail_diagnosis.json")
    diagnosis_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["diagnosis_path"] = str(diagnosis_path)
    return payload


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
    assess_quality_gates: bool = True,
    block_on_quality_gate_failure: bool = False,
    eval_candidate_variant: str = DEFAULT_EVAL_CANDIDATE_VARIANT,
    adaptive_thresholds_enabled: bool = True,
    adaptive_lookback_runs: int = DEFAULT_ADAPTIVE_LOOKBACK_RUNS,
    adaptive_min_history_runs: int = DEFAULT_ADAPTIVE_MIN_HISTORY_RUNS,
    adaptive_iqr_multiplier: float = DEFAULT_ADAPTIVE_IQR_MULTIPLIER,
    adaptive_threshold_cap_multiplier: float = DEFAULT_ADAPTIVE_THRESHOLD_CAP_MULTIPLIER,
    drift_override_enabled: bool = True,
    drift_override_relax_multiplier: float = DEFAULT_DRIFT_OVERRIDE_RELAX_MULTIPLIER,
    drift_override_min_trigger_metrics: int = DEFAULT_DRIFT_OVERRIDE_MIN_TRIGGER_METRICS,
    drift_override_iqr_multiplier: float = DEFAULT_DRIFT_OVERRIDE_IQR_MULTIPLIER,
    min_eval_rows_per_slice: int = DEFAULT_MIN_EVAL_ROWS_PER_SLICE,
    max_weighted_brier_delta: float = DEFAULT_MAX_WEIGHTED_BRIER_DELTA,
    max_weighted_false_active_delta: float = DEFAULT_MAX_WEIGHTED_FALSE_ACTIVE_DELTA,
    max_weighted_mae_p50_cond_delta: float = DEFAULT_MAX_WEIGHTED_MAE_P50_COND_DELTA,
    max_weighted_mae_p50_cond_delta_soft: float | None = DEFAULT_MAX_WEIGHTED_MAE_P50_COND_DELTA_SOFT,
    max_slice_brier_delta: float = DEFAULT_MAX_SLICE_BRIER_DELTA,
    max_slice_false_active_delta: float = DEFAULT_MAX_SLICE_FALSE_ACTIVE_DELTA,
    max_slice_mae_p50_cond_delta: float = DEFAULT_MAX_SLICE_MAE_P50_COND_DELTA,
    max_slice_mae_p50_cond_delta_soft: float | None = DEFAULT_MAX_SLICE_MAE_P50_COND_DELTA_SOFT,
    max_abs_p10_coverage_delta: float = DEFAULT_MAX_ABS_P10_COVERAGE_DELTA,
    max_abs_p90_coverage_delta: float = DEFAULT_MAX_ABS_P90_COVERAGE_DELTA,
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
        if assess_quality_gates:
            quality_result = assess_quality_task(
                retrain_run_id=retrain_result["run_id"],
                eval_summary_path=eval_result["summary_path"],
                data_root=data_root,
                eval_candidate_variant=eval_candidate_variant,
                adaptive_thresholds_enabled=adaptive_thresholds_enabled,
                adaptive_lookback_runs=adaptive_lookback_runs,
                adaptive_min_history_runs=adaptive_min_history_runs,
                adaptive_iqr_multiplier=adaptive_iqr_multiplier,
                adaptive_threshold_cap_multiplier=adaptive_threshold_cap_multiplier,
                drift_override_enabled=drift_override_enabled,
                drift_override_relax_multiplier=drift_override_relax_multiplier,
                drift_override_min_trigger_metrics=drift_override_min_trigger_metrics,
                drift_override_iqr_multiplier=drift_override_iqr_multiplier,
                min_eval_rows_per_slice=min_eval_rows_per_slice,
                max_weighted_brier_delta=max_weighted_brier_delta,
                max_weighted_false_active_delta=max_weighted_false_active_delta,
                max_weighted_mae_p50_cond_delta=max_weighted_mae_p50_cond_delta,
                max_weighted_mae_p50_cond_delta_soft=max_weighted_mae_p50_cond_delta_soft,
                max_slice_brier_delta=max_slice_brier_delta,
                max_slice_false_active_delta=max_slice_false_active_delta,
                max_slice_mae_p50_cond_delta=max_slice_mae_p50_cond_delta,
                max_slice_mae_p50_cond_delta_soft=max_slice_mae_p50_cond_delta_soft,
                max_abs_p10_coverage_delta=max_abs_p10_coverage_delta,
                max_abs_p90_coverage_delta=max_abs_p90_coverage_delta,
            )
            result["quality"] = quality_result
            logger.info(
                "[minutes-quality] passed=%s classification=%s reason=%s diagnosis=%s",
                quality_result.get("passed"),
                quality_result.get("classification"),
                quality_result.get("reason"),
                quality_result.get("diagnosis_path"),
            )
            if not quality_result.get("passed") and block_on_quality_gate_failure:
                raise RuntimeError(
                    "Minutes retrain quality gates failed: "
                    f"classification={quality_result.get('classification')} "
                    f"reason={quality_result.get('reason')} "
                    f"diagnosis={quality_result.get('diagnosis_path')}"
                )
        else:
            logger.info("[minutes-retrain-flow] assess_quality_gates=False; skipping gate diagnostics")
    else:
        logger.info("[minutes-retrain-flow] run_head_to_head_eval=False; skipping evaluation stage")
        if assess_quality_gates:
            logger.info("[minutes-retrain-flow] quality gates skipped because head-to-head eval is disabled")
    return result


if __name__ == "__main__":  # pragma: no cover
    minutes_retrain_flow()
