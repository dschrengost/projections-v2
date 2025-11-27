"""Status-stratified evaluation for the production minutes bundle."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.features.availability import normalize_status
from projections.labels import derive_starter_flag_labels
from projections.metrics.minutes import compute_mae_by_actual_minutes_bucket
from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest, load_feature_frame
from projections.minutes_v1.production import load_production_minutes_bundle
from projections.models import minutes_lgbm as ml
from projections.models.minutes_features import MINUTES_TARGET_COL


UTC = timezone.utc
app = typer.Typer(help=__doc__)

STATUS_CANDIDATES: tuple[str, ...] = ("status", "injury_status", "availability_status")


def _build_windows() -> dict[str, ml.DateWindow]:
    return {
        "train": ml.DateWindow.from_bounds("train", datetime(2022, 10, 1, tzinfo=UTC), datetime(2025, 2, 28, tzinfo=UTC)),
        "cal": ml.DateWindow.from_bounds("cal", datetime(2025, 3, 1, tzinfo=UTC), datetime(2025, 4, 30, tzinfo=UTC)),
        "val": ml.DateWindow.from_bounds("val", datetime(2025, 10, 1, tzinfo=UTC), datetime(2025, 11, 14, tzinfo=UTC)),
    }


def _resolve_status_column(df: pd.DataFrame, override: str | None = None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"Requested status column '{override}' missing from features.")
        return override
    for candidate in STATUS_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Unable to locate status column; checked {', '.join(STATUS_CANDIDATES)}")


def _prepare_slice(df: pd.DataFrame, window: ml.DateWindow) -> pd.DataFrame:
    sliced = window.slice(df)
    filtered = ml._filter_out_players(sliced)
    deduped = deduplicate_latest(filtered, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])
    return deduped


def score_with_bundle(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    feature_columns: list[str] = bundle["feature_columns"]
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"Feature frame missing required columns: {', '.join(sorted(missing))}")
    quantiles = bundle["quantiles"]
    calibrator = bundle.get("calibrator")

    feature_matrix = df[feature_columns]
    preds = ml.modeling.predict_quantiles(quantiles, feature_matrix)
    p10_raw = np.minimum(preds[0.1], preds[0.5])
    p90_raw = np.maximum(preds[0.9], preds[0.5])
    if calibrator is not None:
        p10_cal, p90_cal = calibrator.calibrate(p10_raw, p90_raw)
    else:
        p10_cal, p90_cal = p10_raw, p90_raw

    working = df.copy()
    working["p10_pred"] = p10_cal
    working["p50_pred"] = preds[0.5]
    working["p90_pred"] = p90_cal
    working = ml.apply_conformal(
        working,
        bundle["bucket_offsets"],
        mode=bundle.get("conformal_mode", "tail-deltas"),
        bucket_mode=bundle.get("bucket_mode", "starter,p50bins"),
    )

    working["minutes_p10"] = working["p10_adj"]
    working["minutes_p50"] = working["p50_adj"]
    working["minutes_p90"] = working["p90_adj"]
    for base in ("minutes_p10", "minutes_p50", "minutes_p90"):
        working[f"{base}_cond"] = working[base]

    play_prob_artifacts = bundle.get("play_probability")
    play_prob = None
    if play_prob_artifacts is not None:
        play_prob = ml.predict_play_probability(play_prob_artifacts, feature_matrix)
        working["play_prob"] = play_prob
        working = ml.apply_play_probability_mixture(working, play_prob)
        working["minutes_expected_p50"] = working["minutes_p50_cond"] * working["play_prob"]
    else:
        working["minutes_expected_p50"] = np.nan
    return working


def compute_status_metrics(
    df: pd.DataFrame,
    *,
    status_col: str,
    target_col: str,
) -> dict[str, Dict[str, Any]]:
    if df.empty:
        return {}
    ensure_cols = {status_col, target_col, "minutes_p10", "minutes_p50", "minutes_p90", "starter_flag"}
    missing = ensure_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for metrics: {', '.join(sorted(missing))}")
    working = df.copy()
    working["starter_flag"] = pd.to_numeric(working["starter_flag"], errors="coerce").fillna(0).astype(int)
    working["status_canonical"] = working[status_col].apply(lambda value: normalize_status(value).value)
    results: dict[str, Dict[str, Any]] = {}
    for (status_value, starter_value), group in working.groupby(["status_canonical", "starter_flag"], dropna=False):
        key = f"status={status_value},starter={starter_value}"
        actual = pd.to_numeric(group[target_col], errors="coerce")
        p10 = pd.to_numeric(group["minutes_p10"], errors="coerce")
        p50 = pd.to_numeric(group["minutes_p50"], errors="coerce")
        p90 = pd.to_numeric(group["minutes_p90"], errors="coerce")
        mae_mask = actual.notna() & p50.notna()
        coverage_mask = actual.notna() & p10.notna() & p90.notna()
        cond_mask = coverage_mask & (actual > 0)
        mae_value = float(np.abs(actual[mae_mask] - p50[mae_mask]).mean()) if mae_mask.any() else None
        coverage_value = (
            float(((actual[coverage_mask] >= p10[coverage_mask]) & (actual[coverage_mask] <= p90[coverage_mask])).mean())
            if coverage_mask.any()
            else None
        )
        cond_coverage_value = (
            float(((actual[cond_mask] >= p10[cond_mask]) & (actual[cond_mask] <= p90[cond_mask])).mean())
            if cond_mask.any()
            else None
        )
        group_metrics: Dict[str, Any] = {
            "rows": int(len(group)),
            "mae_q50": mae_value,
            "coverage_q10_q90": coverage_value,
            "cond_coverage_q10_q90": cond_coverage_value,
        }
        if mae_mask.any():
            bucket_metrics = compute_mae_by_actual_minutes_bucket(
                actual[mae_mask].to_numpy(dtype=float),
                p50[mae_mask].to_numpy(dtype=float),
            )
            group_metrics.update(bucket_metrics)
        results[key] = group_metrics
    return results


@app.command()
def main(
    features_root: Path = typer.Option(
        paths.data_path("gold", "features_minutes_v1"),
        "--features-root",
        help="Root directory containing minutes_v1 features (season=YYYY/month=MM partitions).",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Base data root used to resolve default feature paths.",
    ),
    out_path: Path = typer.Option(
        Path("artifacts/minutes_lgbm/status_eval_minutes.json"),
        "--out-path",
        help="Where to write the aggregated JSON metrics.",
    ),
    target_col: str = typer.Option(
        MINUTES_TARGET_COL,
        "--target-col",
        help="Name of the actual minutes column in the feature frame.",
    ),
    status_column: str | None = typer.Option(
        None,
        "--status-column",
        help="Optional override for the status column name.",
    ),
) -> None:
    features_root = features_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    out_path = out_path.expanduser().resolve()

    typer.echo(f"[minutes-status-eval] Loading features from {features_root}")
    feature_df = load_feature_frame(
        features_path=features_root,
        data_root=data_root,
        season=None,
        month=None,
    )
    if target_col not in feature_df.columns:
        raise ValueError(f"Target column '{target_col}' missing from feature frame.")
    feature_df = derive_starter_flag_labels(feature_df, output_col="starter_flag")

    status_col = _resolve_status_column(feature_df, override=status_column)
    bundle = load_production_minutes_bundle()
    windows = _build_windows()

    payload: dict[str, dict[str, Dict[str, Any]]] = {}
    for split_name, window in windows.items():
        typer.echo(f"[minutes-status-eval] Scoring {split_name} window {window.start.date()} → {window.end.date()}")
        slice_df = _prepare_slice(feature_df, window)
        scored = score_with_bundle(slice_df, bundle)
        payload[split_name] = compute_status_metrics(scored, status_col=status_col, target_col=target_col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(json.dumps(payload, indent=2))
    typer.echo(f"[minutes-status-eval] Wrote metrics to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
