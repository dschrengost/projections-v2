"""Head-to-head evaluation of two minutes bundles on identical eval datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import typer

from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest
from projections.minutes_alloc.occupancy_sparse import (
    DEFAULT_OCCUPANCY_CAP_MAX,
    DEFAULT_OCCUPANCY_FRINGE_CAP_MAX,
    DEFAULT_OCCUPANCY_K_MAX,
    DEFAULT_OCCUPANCY_K_MIN,
    DEFAULT_OCCUPANCY_P_CUTOFF,
    DEFAULT_OCCUPANCY_SCALE,
    DEFAULT_OCCUPANCY_STARTER_FLOOR,
    OccupancySparseConfig,
    apply_occupancy_sparse_allocation,
)
from projections.models import minutes_lgbm as ml


app = typer.Typer(help=__doc__)


DEFAULT_CURRENT_BUNDLE = Path(
    "artifacts/minutes_lgbm/minutes_v1_safe_starter_20260127_dnp_playprob_dedicated"
)
DEFAULT_RETRAIN_BUNDLE = Path(
    "/home/daniel/projections-data/artifacts/minutes_lgbm/minutes_v1_recency_h35_20260207T110500Z"
)
DEFAULT_DATA_ROOT = Path("/home/daniel/projections-data")
DEFAULT_LABELS = Path("/home/daniel/projections-data/labels/season=2025/boxscore_labels.parquet")
DEFAULT_EVAL_ROOT = Path("/home/daniel/projections-data/artifacts/minutes_eval_runs")
DEFAULT_REPORT_PATH = Path("reports/minutes_head_to_head_eval_20260207.md")
@dataclass(frozen=True)
class EvalSlice:
    name: str
    requested_start: date
    requested_end: date


@dataclass(frozen=True)
class BuiltSlice:
    name: str
    requested_start: date
    requested_end: date
    clamped_start: date
    clamped_end: date
    effective_start: date
    effective_end: date
    slice_dir: Path
    eval_dataset_path: Path
    meta_path: Path


def _iso_day(value: date) -> str:
    return value.isoformat()


def _to_date(value: str | datetime | date) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.date()


def _iter_days(start_day: date, end_day: date) -> list[date]:
    if end_day < start_day:
        return []
    total = (end_day - start_day).days + 1
    return [start_day + timedelta(days=offset) for offset in range(total)]


def _git_sha_or_unknown() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _safe_status_upper(series: pd.Series | None, *, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series("", index=index, dtype="string")
    return series.astype("string", copy=False).fillna("").str.upper()


def _coerce_bool_series(series: pd.Series | None, *, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index, dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0.0).astype(float).ne(0.0)

    text = series.astype("string", copy=False).str.strip().str.lower()
    return text.fillna("").isin({"1", "true", "t", "yes", "y"})


def _read_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels parquet: {labels_path}")
    labels = pd.read_parquet(labels_path)
    required = {"game_date", "minutes", *KEY_COLUMNS}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label parquet missing required columns: {sorted(missing)}")
    labels = labels.copy()
    labels["game_date"] = pd.to_datetime(labels["game_date"], errors="coerce").dt.date
    labels = labels.dropna(subset=["game_date"])
    labels["minutes"] = pd.to_numeric(labels["minutes"], errors="coerce")
    labels = labels.dropna(subset=["minutes"])
    labels = labels.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
    return labels


def _largest_contiguous_labeled_subrange(
    start_day: date,
    end_day: date,
    *,
    label_dates: set[date],
) -> tuple[date, date]:
    days = _iter_days(start_day, end_day)
    if not days:
        raise ValueError("Requested range is empty after clamping.")

    best_start: date | None = None
    best_end: date | None = None
    best_len = 0

    cur_start: date | None = None
    cur_len = 0
    for day in days:
        if day in label_dates:
            if cur_start is None:
                cur_start = day
                cur_len = 1
            else:
                cur_len += 1
            cur_end = day
            if cur_len > best_len:
                best_start = cur_start
                best_end = cur_end
                best_len = cur_len
        else:
            cur_start = None
            cur_len = 0

    if best_start is None or best_end is None:
        raise ValueError(
            f"No labeled dates found inside requested range [{_iso_day(start_day)} .. {_iso_day(end_day)}]"
        )
    return best_start, best_end


def _contiguous_labeled_segments(
    start_day: date,
    end_day: date,
    *,
    label_dates: set[date],
) -> list[tuple[date, date]]:
    days = _iter_days(start_day, end_day)
    segments: list[tuple[date, date]] = []
    cur_start: date | None = None
    cur_end: date | None = None
    for day in days:
        if day in label_dates:
            if cur_start is None:
                cur_start = day
            cur_end = day
        elif cur_start is not None and cur_end is not None:
            segments.append((cur_start, cur_end))
            cur_start = None
            cur_end = None
    if cur_start is not None and cur_end is not None:
        segments.append((cur_start, cur_end))
    return segments


def _candidate_subranges(segments: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Generate contiguous labeled candidates ordered from largest to smallest."""

    candidates: list[tuple[date, date]] = []
    for seg_start, seg_end in segments:
        seg_days = _iter_days(seg_start, seg_end)
        n = len(seg_days)
        for length in range(n, 0, -1):
            for i in range(0, n - length + 1):
                candidates.append((seg_days[i], seg_days[i + length - 1]))
    # Order globally by length desc then latest end date.
    candidates.sort(key=lambda p: ((p[1] - p[0]).days + 1, p[1]), reverse=True)
    # Deduplicate while preserving order.
    deduped: list[tuple[date, date]] = []
    seen: set[tuple[date, date]] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _run_canonical_builder(
    *,
    data_root: Path,
    season: int,
    start_day: date,
    end_day: date,
    out_path: Path,
) -> list[str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "projections.pipelines.build_features_minutes_v1",
        "--start-date",
        _iso_day(start_day),
        "--end-date",
        _iso_day(end_day),
        "--data-root",
        str(data_root),
        "--season",
        str(season),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return cmd


def _build_eval_dataset_slice(
    *,
    data_root: Path,
    labels_path: Path,
    season: int,
    eval_run_dir: Path,
    slice_cfg: EvalSlice,
) -> BuiltSlice:
    labels = _read_labels(labels_path)
    label_dates = set(labels["game_date"].tolist())
    label_min = min(label_dates)
    label_max = max(label_dates)

    clamped_start = max(slice_cfg.requested_start, label_min)
    clamped_end = min(slice_cfg.requested_end, label_max)
    if clamped_end < clamped_start:
        raise ValueError(
            f"Slice {slice_cfg.name} has no labeled overlap after clamping "
            f"[{_iso_day(slice_cfg.requested_start)}..{_iso_day(slice_cfg.requested_end)}]"
        )

    slice_dir = eval_run_dir / slice_cfg.name
    features_path = slice_dir / "features.parquet"
    eval_dataset_path = slice_dir / "eval_dataset.parquet"
    meta_path = slice_dir / "meta.json"
    segments = _contiguous_labeled_segments(clamped_start, clamped_end, label_dates=label_dates)
    if not segments:
        raise ValueError(
            f"Slice {slice_cfg.name} has no labeled dates inside clamped range "
            f"[{_iso_day(clamped_start)}..{_iso_day(clamped_end)}]"
        )
    candidates = _candidate_subranges(segments)
    attempts: list[dict[str, Any]] = []
    canonical_cmd: list[str] | None = None
    effective_start: date | None = None
    effective_end: date | None = None
    last_error: str | None = None
    for cand_start, cand_end in candidates:
        try:
            canonical_cmd = _run_canonical_builder(
                data_root=data_root,
                season=season,
                start_day=cand_start,
                end_day=cand_end,
                out_path=features_path,
            )
            effective_start = cand_start
            effective_end = cand_end
            attempts.append(
                {
                    "start": _iso_day(cand_start),
                    "end": _iso_day(cand_end),
                    "status": "success",
                }
            )
            break
        except subprocess.CalledProcessError as exc:
            last_error = str(exc)
            attempts.append(
                {
                    "start": _iso_day(cand_start),
                    "end": _iso_day(cand_end),
                    "status": "failed",
                    "error": last_error,
                }
            )
            continue
    if effective_start is None or effective_end is None or canonical_cmd is None:
        raise RuntimeError(
            f"Slice {slice_cfg.name} could not build features for any labeled contiguous subrange "
            f"within [{_iso_day(clamped_start)}..{_iso_day(clamped_end)}]. Last error: {last_error}"
        )

    if not features_path.exists():
        raise FileNotFoundError(f"Canonical builder did not write features at {features_path}")
    features = pd.read_parquet(features_path)
    required_features = {"game_date", "feature_as_of_ts", "tip_ts", *KEY_COLUMNS}
    missing_features = required_features - set(features.columns)
    if missing_features:
        raise ValueError(
            f"Slice {slice_cfg.name} features missing required columns: {sorted(missing_features)}"
        )

    features = features.copy()
    features["game_date"] = pd.to_datetime(features["game_date"], errors="coerce").dt.date
    features = features.dropna(subset=["game_date"])
    features = features[
        (features["game_date"] >= effective_start) & (features["game_date"] <= effective_end)
    ].copy()

    feature_as_of = pd.to_datetime(features["feature_as_of_ts"], utc=True, errors="coerce")
    tip_ts = pd.to_datetime(features["tip_ts"], utc=True, errors="coerce")
    leakage_mask = feature_as_of > tip_ts
    leakage_rows = int(leakage_mask.fillna(False).sum())
    if leakage_rows > 0:
        raise RuntimeError(
            f"Leakage violation in slice {slice_cfg.name}: {leakage_rows} rows have feature_as_of_ts > tip_ts"
        )

    pre_dedup_rows = int(len(features))
    features = deduplicate_latest(features, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])

    labels_slice = labels[
        (labels["game_date"] >= effective_start) & (labels["game_date"] <= effective_end)
    ][list(KEY_COLUMNS) + ["game_date", "minutes"]].copy()

    joined = features.merge(
        labels_slice.rename(columns={"minutes": "actual_minutes", "game_date": "label_game_date"}),
        on=list(KEY_COLUMNS),
        how="left",
    )

    missing_label_mask = joined["actual_minutes"].isna()
    dropped_missing_labels = int(missing_label_mask.sum())
    dropped_by_date: dict[str, int] = {}
    if dropped_missing_labels > 0:
        dropped_counts = (
            pd.to_datetime(joined.loc[missing_label_mask, "game_date"], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .value_counts()
            .sort_index()
        )
        dropped_by_date = {str(idx): int(val) for idx, val in dropped_counts.items()}

    eval_df = joined.loc[~missing_label_mask].copy()
    if eval_df.empty:
        raise ValueError(f"Slice {slice_cfg.name} has zero labeled rows after label join.")

    eval_df["actual_minutes"] = pd.to_numeric(eval_df["actual_minutes"], errors="coerce").fillna(0.0)
    eval_df["plays_target"] = (eval_df["actual_minutes"] > 0.0).astype(int)

    slice_dir.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(eval_dataset_path, index=False)

    payload = {
        "slice_name": slice_cfg.name,
        "requested_window": {
            "start": _iso_day(slice_cfg.requested_start),
            "end": _iso_day(slice_cfg.requested_end),
        },
        "clamped_window": {
            "start": _iso_day(clamped_start),
            "end": _iso_day(clamped_end),
        },
        "effective_window": {
            "start": _iso_day(effective_start),
            "end": _iso_day(effective_end),
        },
        "label_path": str(labels_path),
        "label_date_bounds": {
            "min": _iso_day(label_min),
            "max": _iso_day(label_max),
        },
        "feature_builder": {
            "module": "projections.pipelines.build_features_minutes_v1",
            "command": canonical_cmd,
            "features_path": str(features_path),
            "candidate_segments": [
                {"start": _iso_day(seg_start), "end": _iso_day(seg_end)}
                for seg_start, seg_end in segments
            ],
            "attempts": attempts,
        },
        "row_counts": {
            "features_pre_dedup": pre_dedup_rows,
            "features_post_dedup": int(len(features)),
            "eval_rows": int(len(eval_df)),
            "dropped_missing_labels": dropped_missing_labels,
        },
        "dropped_missing_labels_by_date": dropped_by_date,
        "leakage_rows": leakage_rows,
        "git_sha": _git_sha_or_unknown(),
        "eval_dataset_path": str(eval_dataset_path),
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return BuiltSlice(
        name=slice_cfg.name,
        requested_start=slice_cfg.requested_start,
        requested_end=slice_cfg.requested_end,
        clamped_start=clamped_start,
        clamped_end=clamped_end,
        effective_start=effective_start,
        effective_end=effective_end,
        slice_dir=slice_dir,
        eval_dataset_path=eval_dataset_path,
        meta_path=meta_path,
    )


def _load_minutes_bundle(bundle_path: Path) -> dict[str, Any]:
    resolved = bundle_path.expanduser().resolve()
    quantile_path = resolved / "lgbm_quantiles.joblib"
    if not quantile_path.exists():
        raise FileNotFoundError(f"Missing lgbm_quantiles.joblib at {quantile_path}")
    bundle = joblib.load(quantile_path)
    bundle.setdefault("bucket_mode", "none")
    bundle.setdefault("bucket_offsets", {"__global__": {"d10": 0.0, "d90": 0.0, "n": 0}})
    bundle.setdefault("conformal_mode", "tail-deltas")
    return bundle


def score_bundle_on_eval_dataset(
    eval_df: pd.DataFrame,
    *,
    bundle: dict[str, Any],
    bundle_label: str,
) -> pd.DataFrame:
    if eval_df.empty:
        raise ValueError("Eval dataframe is empty.")

    feature_cols = bundle.get("feature_columns")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("Bundle missing feature_columns metadata.")
    working_eval = eval_df.copy()
    missing_features = sorted(set(feature_cols) - set(working_eval.columns))
    if missing_features:
        for col in missing_features:
            # Training-only weighting columns should default to neutral weight at eval time.
            if col == "weight_recency":
                working_eval[col] = 1.0
            else:
                working_eval[col] = 0.0

    feature_matrix = working_eval[feature_cols]
    preds = ml.modeling.predict_quantiles(bundle["quantiles"], feature_matrix)
    p10_raw = np.minimum(preds[0.1], preds[0.5])
    p90_raw = np.maximum(preds[0.9], preds[0.5])

    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        p10_cal, p90_cal = calibrator.calibrate(p10_raw, p90_raw)
    else:
        p10_cal, p90_cal = p10_raw, p90_raw

    working = working_eval.copy()
    working["p10_pred"] = p10_cal
    working["p50_pred"] = preds[0.5]
    working["p90_pred"] = p90_cal
    working = ml.apply_conformal(
        working,
        bundle["bucket_offsets"],
        mode=bundle["conformal_mode"],
        bucket_mode=bundle["bucket_mode"],
    )

    play_prob_artifacts = bundle.get("play_probability")
    if play_prob_artifacts is not None:
        # Use the full eval frame so dedicated play-prob features (for example
        # prior_play_prob) are available even when excluded from quantile features.
        play_prob = ml.predict_play_probability(play_prob_artifacts, working)
    else:
        play_prob = np.ones(len(working), dtype=float)

    out = pd.DataFrame(
        {
            "game_id": working["game_id"],
            "player_id": working["player_id"],
            "team_id": working["team_id"],
            "game_date": working["game_date"],
            "actual_minutes": pd.to_numeric(working["actual_minutes"], errors="coerce").fillna(0.0),
            "plays_target": pd.to_numeric(working["plays_target"], errors="coerce").fillna(0).astype(int),
            "play_prob": np.clip(np.asarray(play_prob, dtype=float), 0.0, 1.0),
            "pred_p10_minutes": pd.to_numeric(working["p10_adj"], errors="coerce").fillna(0.0),
            "pred_p50_minutes": pd.to_numeric(working["p50_adj"], errors="coerce").fillna(0.0),
            "pred_p90_minutes": pd.to_numeric(working["p90_adj"], errors="coerce").fillna(0.0),
            "bundle_label": bundle_label,
        }
    )
    return out


def apply_occupancy_sparse_layer(
    pred_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    p_cutoff: float = DEFAULT_OCCUPANCY_P_CUTOFF,
    k_min: int = DEFAULT_OCCUPANCY_K_MIN,
    k_max: int = DEFAULT_OCCUPANCY_K_MAX,
    cap_max: float = DEFAULT_OCCUPANCY_CAP_MAX,
    fringe_cap_max: float = DEFAULT_OCCUPANCY_FRINGE_CAP_MAX,
    occupancy_scale: float = DEFAULT_OCCUPANCY_SCALE,
    starter_floor: float = DEFAULT_OCCUPANCY_STARTER_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the shared occupancy+sparse allocator to retrain eval predictions."""
    if pred_df.empty:
        raise ValueError("pred_df is empty.")
    if eval_df.empty:
        raise ValueError("eval_df is empty.")

    required_pred_cols = {
        "game_id",
        "team_id",
        "player_id",
        "play_prob",
        "pred_p10_minutes",
        "pred_p50_minutes",
        "pred_p90_minutes",
    }
    missing_pred = sorted(required_pred_cols - set(pred_df.columns))
    if missing_pred:
        raise ValueError(f"pred_df missing required columns: {missing_pred}")

    join_cols = ["game_id", "team_id", "player_id"]
    context_cols = [
        "status",
        "is_out",
        "lineup_role",
        "starter_flag",
        "is_projected_starter",
        "is_confirmed_starter",
        "is_starter",
        "spread_home",
        "total",
        "home_flag",
    ]
    available_context = [c for c in context_cols if c in eval_df.columns]
    context = eval_df[join_cols + available_context].drop_duplicates(subset=join_cols, keep="last")
    merged = pred_df.merge(context, on=join_cols, how="left", validate="one_to_one").copy()

    canonical = merged.rename(
        columns={
            "pred_p10_minutes": "minutes_p10",
            "pred_p50_minutes": "minutes_p50",
            "pred_p90_minutes": "minutes_p90",
        }
    )
    occ_cfg = OccupancySparseConfig(
        p_cutoff=float(p_cutoff),
        k_min=int(k_min),
        k_max=int(k_max),
        cap_max=float(cap_max),
        fringe_cap_max=float(fringe_cap_max),
        occupancy_scale=float(occupancy_scale),
        starter_floor=float(starter_floor),
    )
    occ_df, diagnostics = apply_occupancy_sparse_allocation(canonical, config=occ_cfg)

    out = pred_df.copy()
    out["play_prob"] = pd.to_numeric(occ_df["play_prob_occ"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["pred_p10_minutes"] = pd.to_numeric(occ_df["minutes_p10_occ"], errors="coerce").fillna(0.0)
    out["pred_p50_minutes"] = pd.to_numeric(occ_df["minutes_occ"], errors="coerce").fillna(0.0)
    out["pred_p90_minutes"] = pd.to_numeric(occ_df["minutes_p90_occ"], errors="coerce").fillna(0.0)
    out["eligible_flag_occ_v0"] = pd.to_numeric(occ_df["eligible_flag_occ"], errors="coerce").fillna(0).astype(int)
    out["bench_share_pred_occ_v0"] = pd.to_numeric(occ_df["bench_share_pred_occ"], errors="coerce").fillna(0.0)
    return out, diagnostics


def compute_head_to_head_metrics(pred_df: pd.DataFrame) -> dict[str, Any]:
    if pred_df.empty:
        raise ValueError("Prediction dataframe is empty.")

    y = pd.to_numeric(pred_df["actual_minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    plays = pd.to_numeric(pred_df["plays_target"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    p_play = pd.to_numeric(pred_df["play_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    pred_p50 = pd.to_numeric(pred_df["pred_p50_minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    metrics: dict[str, Any] = {
        "rows": int(len(pred_df)),
        "positive_rows": int(np.sum(plays == 1)),
        "brier_play_prob": float(np.mean((plays - p_play) ** 2)),
        "false_active_rate_p_ge_0_5": float(np.mean((p_play >= 0.5) & (plays == 0))),
        "false_inactive_rate_p_le_0_2": float(np.mean((p_play <= 0.2) & (plays == 1))),
        "bench_smear_proxy": float(np.mean((pred_p50 > 10.0) & (y < 1.0))),
    }

    cond_mask = plays == 1
    if int(np.sum(cond_mask)) > 0:
        metrics["mae_p50_conditional"] = float(np.mean(np.abs(pred_p50[cond_mask] - y[cond_mask])))
    else:
        metrics["mae_p50_conditional"] = None

    has_p10 = "pred_p10_minutes" in pred_df.columns
    has_p90 = "pred_p90_minutes" in pred_df.columns
    if has_p10 and has_p90:
        pred_p10 = pd.to_numeric(pred_df["pred_p10_minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pred_p90 = pd.to_numeric(pred_df["pred_p90_minutes"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        metrics["p10_coverage_leq"] = float(np.mean(y <= pred_p10))
        metrics["p90_coverage_leq"] = float(np.mean(y <= pred_p90))
    else:
        metrics["p10_coverage_leq"] = None
        metrics["p90_coverage_leq"] = None

    return metrics


def _metric_delta(current: dict[str, Any], retrain: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = sorted(set(current.keys()) | set(retrain.keys()))
    for key in keys:
        c_val = current.get(key)
        r_val = retrain.get(key)
        if isinstance(c_val, (int, float)) and isinstance(r_val, (int, float)):
            out[key] = float(r_val) - float(c_val)
        else:
            out[key] = None
    return out


def _write_slice_metrics(
    *,
    slice_dir: Path,
    metrics_current: dict[str, Any],
    metrics_retrain: dict[str, Any],
    metrics_retrain_occupancy_v0: dict[str, Any] | None = None,
) -> Path:
    path = slice_dir / "metrics_head_to_head.json"
    payload = {
        "current": metrics_current,
        "retrain": metrics_retrain,
        "delta_retrain_minus_current": _metric_delta(metrics_current, metrics_retrain),
    }
    if metrics_retrain_occupancy_v0 is not None:
        payload["retrain_occupancy_v0"] = metrics_retrain_occupancy_v0
        payload["delta_retrain_occupancy_v0_minus_current"] = _metric_delta(
            metrics_current, metrics_retrain_occupancy_v0
        )
        payload["delta_retrain_occupancy_v0_minus_retrain"] = _metric_delta(
            metrics_retrain, metrics_retrain_occupancy_v0
        )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _render_report(summary: dict[str, Any]) -> str:
    def _fmt_metric(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return f"{float(value):.6f}"

    lines: list[str] = [
        "# Minutes Bundle Head-to-Head Eval (2026-02-07)",
        "",
        f"- Eval run id: `{summary['eval_run_id']}`",
        f"- Current bundle: `{summary['current_bundle']}`",
        f"- Retrain bundle: `{summary['retrain_bundle']}`",
        "",
    ]

    metric_order = [
        "rows",
        "positive_rows",
        "brier_play_prob",
        "false_active_rate_p_ge_0_5",
        "false_inactive_rate_p_le_0_2",
        "mae_p50_conditional",
        "bench_smear_proxy",
        "p10_coverage_leq",
        "p90_coverage_leq",
    ]

    for slice_name, payload in summary["slices"].items():
        meta = payload["meta"]
        metrics_current = payload["metrics_current"]
        metrics_retrain = payload["metrics_retrain"]
        delta = payload["delta_retrain_minus_current"]
        metrics_occ = payload.get("metrics_retrain_occupancy_v0")
        delta_occ_minus_retrain = payload.get("delta_retrain_occupancy_v0_minus_retrain") or {}
        lines.extend(
            [
                f"## {slice_name}",
                "",
                f"- Requested window: `{meta['requested_window']['start']}` .. `{meta['requested_window']['end']}`",
                f"- Effective window: `{meta['effective_window']['start']}` .. `{meta['effective_window']['end']}`",
                f"- Eval rows: `{meta['row_counts']['eval_rows']}`",
                "",
                "| metric | current | retrain | delta (retrain-current) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for key in metric_order:
            c_val = metrics_current.get(key)
            r_val = metrics_retrain.get(key)
            d_val = delta.get(key)
            lines.append(
                f"| {key} | {_fmt_metric(c_val)} | {_fmt_metric(r_val)} | {_fmt_metric(d_val)} |"
            )
        lines.append("")

        if metrics_occ is not None:
            lines.extend(
                [
                    "| metric | retrain | retrain_occupancy_v0 | delta (occupancy-retrain) |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for key in metric_order:
                r_val = metrics_retrain.get(key)
                o_val = metrics_occ.get(key)
                d_val = delta_occ_minus_retrain.get(key)
                lines.append(
                    f"| {key} | {_fmt_metric(r_val)} | {_fmt_metric(o_val)} | {_fmt_metric(d_val)} |"
                )
            lines.append("")

    return "\n".join(lines)


@app.command()
def main(
    eval_run_id: str = typer.Option(
        "",
        "--eval-run-id",
        help="Output run id under /home/daniel/projections-data/artifacts/minutes_eval_runs/.",
    ),
    current_bundle: Path = typer.Option(DEFAULT_CURRENT_BUNDLE, "--current-bundle"),
    retrain_bundle: Path = typer.Option(DEFAULT_RETRAIN_BUNDLE, "--retrain-bundle"),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root"),
    labels_path: Path = typer.Option(DEFAULT_LABELS, "--labels-path"),
    season: int = typer.Option(2025, "--season"),
    eval_root: Path = typer.Option(DEFAULT_EVAL_ROOT, "--eval-root"),
    report_path: Path = typer.Option(DEFAULT_REPORT_PATH, "--report-path"),
    run_occupancy_sparse_diagnostic: bool = typer.Option(
        True,
        "--run-occupancy-sparse-diagnostic/--skip-occupancy-sparse-diagnostic",
        help="Run occupancy+sparse allocation diagnostic on retrain predictions.",
    ),
    occupancy_p_cutoff: float = typer.Option(DEFAULT_OCCUPANCY_P_CUTOFF, "--occupancy-p-cutoff"),
    occupancy_k_min: int = typer.Option(DEFAULT_OCCUPANCY_K_MIN, "--occupancy-k-min"),
    occupancy_k_max: int = typer.Option(DEFAULT_OCCUPANCY_K_MAX, "--occupancy-k-max"),
    occupancy_cap_max: float = typer.Option(DEFAULT_OCCUPANCY_CAP_MAX, "--occupancy-cap-max"),
    occupancy_fringe_cap_max: float = typer.Option(
        DEFAULT_OCCUPANCY_FRINGE_CAP_MAX, "--occupancy-fringe-cap-max"
    ),
    occupancy_scale: float = typer.Option(DEFAULT_OCCUPANCY_SCALE, "--occupancy-scale"),
    occupancy_starter_floor: float = typer.Option(
        DEFAULT_OCCUPANCY_STARTER_FLOOR, "--occupancy-starter-floor"
    ),
) -> None:
    run_id = eval_run_id.strip() or datetime.now(tz=UTC).strftime("minutes_head_to_head_%Y%m%dT%H%M%SZ")
    eval_run_dir = eval_root / run_id
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    slices = [
        EvalSlice("deadline_chaos", requested_start=date(2026, 2, 1), requested_end=date(2026, 2, 5)),
        EvalSlice("pre_deadline_stability", requested_start=date(2026, 1, 15), requested_end=date(2026, 1, 31)),
    ]

    current_bundle_obj = _load_minutes_bundle(current_bundle)
    retrain_bundle_obj = _load_minutes_bundle(retrain_bundle)

    summary: dict[str, Any] = {
        "eval_run_id": run_id,
        "git_sha": _git_sha_or_unknown(),
        "current_bundle": str(current_bundle.expanduser().resolve()),
        "retrain_bundle": str(retrain_bundle.expanduser().resolve()),
        "occupancy_sparse_v0": {
            "enabled": bool(run_occupancy_sparse_diagnostic),
            "p_cutoff": float(occupancy_p_cutoff),
            "k_min": int(occupancy_k_min),
            "k_max": int(occupancy_k_max),
            "cap_max": float(occupancy_cap_max),
            "fringe_cap_max": float(occupancy_fringe_cap_max),
            "scale": float(occupancy_scale),
            "starter_floor": float(occupancy_starter_floor),
        },
        "slices": {},
    }

    for slice_cfg in slices:
        built = _build_eval_dataset_slice(
            data_root=data_root,
            labels_path=labels_path,
            season=season,
            eval_run_dir=eval_run_dir,
            slice_cfg=slice_cfg,
        )
        eval_df = pd.read_parquet(built.eval_dataset_path)

        preds_current = score_bundle_on_eval_dataset(eval_df, bundle=current_bundle_obj, bundle_label="current")
        preds_retrain = score_bundle_on_eval_dataset(eval_df, bundle=retrain_bundle_obj, bundle_label="retrain")

        preds_current_path = built.slice_dir / "preds_current.parquet"
        preds_retrain_path = built.slice_dir / "preds_retrain.parquet"
        preds_current.to_parquet(preds_current_path, index=False)
        preds_retrain.to_parquet(preds_retrain_path, index=False)

        metrics_current = compute_head_to_head_metrics(preds_current)
        metrics_retrain = compute_head_to_head_metrics(preds_retrain)
        delta = _metric_delta(metrics_current, metrics_retrain)

        preds_retrain_occ: pd.DataFrame | None = None
        occupancy_diag_df: pd.DataFrame | None = None
        preds_retrain_occ_path: Path | None = None
        occupancy_diag_path: Path | None = None
        metrics_retrain_occ: dict[str, Any] | None = None
        delta_occ_minus_current: dict[str, Any] | None = None
        delta_occ_minus_retrain: dict[str, Any] | None = None
        if run_occupancy_sparse_diagnostic:
            preds_retrain_occ, occupancy_diag_df = apply_occupancy_sparse_layer(
                preds_retrain,
                eval_df,
                p_cutoff=float(occupancy_p_cutoff),
                k_min=int(occupancy_k_min),
                k_max=int(occupancy_k_max),
                cap_max=float(occupancy_cap_max),
                fringe_cap_max=float(occupancy_fringe_cap_max),
                occupancy_scale=float(occupancy_scale),
                starter_floor=float(occupancy_starter_floor),
            )
            preds_retrain_occ_path = built.slice_dir / "preds_retrain_occupancy_v0.parquet"
            preds_retrain_occ.to_parquet(preds_retrain_occ_path, index=False)
            occupancy_diag_path = built.slice_dir / "occupancy_sparse_v0_team_diagnostics.parquet"
            assert occupancy_diag_df is not None
            occupancy_diag_df.to_parquet(occupancy_diag_path, index=False)
            metrics_retrain_occ = compute_head_to_head_metrics(preds_retrain_occ)
            delta_occ_minus_current = _metric_delta(metrics_current, metrics_retrain_occ)
            delta_occ_minus_retrain = _metric_delta(metrics_retrain, metrics_retrain_occ)
        metrics_path = _write_slice_metrics(
            slice_dir=built.slice_dir,
            metrics_current=metrics_current,
            metrics_retrain=metrics_retrain,
            metrics_retrain_occupancy_v0=metrics_retrain_occ,
        )

        meta_payload = json.loads(built.meta_path.read_text(encoding="utf-8"))
        paths_payload: dict[str, str] = {
            "eval_dataset": str(built.eval_dataset_path),
            "preds_current": str(preds_current_path),
            "preds_retrain": str(preds_retrain_path),
            "metrics": str(metrics_path),
        }
        if preds_retrain_occ_path is not None:
            paths_payload["preds_retrain_occupancy_v0"] = str(preds_retrain_occ_path)
        if occupancy_diag_path is not None:
            paths_payload["occupancy_sparse_v0_team_diagnostics"] = str(occupancy_diag_path)

        slice_payload: dict[str, Any] = {
            "meta": meta_payload,
            "paths": paths_payload,
            "metrics_current": metrics_current,
            "metrics_retrain": metrics_retrain,
            "delta_retrain_minus_current": delta,
        }
        if metrics_retrain_occ is not None:
            slice_payload["metrics_retrain_occupancy_v0"] = metrics_retrain_occ
            slice_payload["delta_retrain_occupancy_v0_minus_current"] = delta_occ_minus_current
            slice_payload["delta_retrain_occupancy_v0_minus_retrain"] = delta_occ_minus_retrain
        summary["slices"][slice_cfg.name] = slice_payload

        line = (
            f"[slice:{slice_cfg.name}] rows={metrics_current['rows']} "
            f"brier(current={metrics_current['brier_play_prob']:.6f}, retrain={metrics_retrain['brier_play_prob']:.6f})"
        )
        if metrics_retrain_occ is not None:
            line += f", occupancy_v0={metrics_retrain_occ['brier_play_prob']:.6f}"
        typer.echo(line)

    summary_path = eval_run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    report = _render_report(summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    typer.echo(f"[done] summary={summary_path}")
    typer.echo(f"[done] report={report_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
