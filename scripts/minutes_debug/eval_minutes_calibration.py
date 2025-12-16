"""Evaluate raw vs calibrated tail coverage on VAL for minutes_v1 (no p_play bundle)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.labels import derive_starter_flag_labels
from projections.minutes_v1.calibration import MinutesCalibrationParams, apply_k_params
from projections.minutes_v1.datasets import load_feature_frame
from scripts.minutes_debug.status_eval_minutes import _build_windows, _prepare_slice, score_with_bundle

app = typer.Typer(help=__doc__)

DEFAULT_BUNDLE_DIR = Path("artifacts/minutes_lgbm/lgbm_full_v1_no_p_play_20251202")
DEFAULT_FEATURES_ROOT = paths.data_path("gold", "features_minutes_v1")
DEFAULT_DATA_ROOT = paths.get_data_root()
DEFAULT_OUTPUT_DIR = Path("artifacts/minutes_v1/calibration/lgbm_full_v1_no_p_play_20251202")


def _load_bundle(bundle_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_path = bundle_dir / "lgbm_quantiles.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle missing at {bundle_path}")
    bundle = joblib.load(bundle_path)
    meta_path = bundle_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return bundle, meta


def _load_scored_slice(
    bundle_dir: Path,
    window_name: str,
    *,
    features_root: Path,
    data_root: Path,
) -> pd.DataFrame:
    bundle, _ = _load_bundle(bundle_dir)
    windows = _build_windows()
    if window_name not in windows:
        raise KeyError(f"Unknown window '{window_name}'")
    feature_df = load_feature_frame(
        features_path=features_root,
        data_root=data_root,
        season=None,
        month=None,
    )
    feature_df = derive_starter_flag_labels(feature_df, output_col="starter_flag")
    slice_df = _prepare_slice(feature_df, windows[window_name])
    scored = score_with_bundle(slice_df, bundle)
    return scored


def _playable_mask(df: pd.DataFrame, threshold: float) -> pd.Series:
    mask = pd.to_numeric(df["minutes"], errors="coerce") >= threshold
    if "status" in df.columns:
        mask &= df["status"].astype(str).str.upper() != "OUT"
    return mask


def _metrics(
    df: pd.DataFrame,
    *,
    p10_col: str,
    p90_col: str,
    mask: pd.Series,
) -> dict[str, float]:
    y = pd.to_numeric(df.loc[mask, "minutes"], errors="coerce")
    p10 = pd.to_numeric(df.loc[mask, p10_col], errors="coerce")
    p90 = pd.to_numeric(df.loc[mask, p90_col], errors="coerce")
    if y.empty:
        return {"p10": 0.0, "p90": 0.0, "inside": 0.0, "mpiwn": 0.0, "mae_p50": 0.0}
    return {
        "p10": float((y < p10).mean()),
        "p90": float((y <= p90).mean()),
        "inside": float(((y >= p10) & (y <= p90)).mean()),
        "mpiwn": float(np.mean(p90 - p10)),
        "mae_p50": float(np.abs(y - pd.to_numeric(df.loc[mask, "minutes_p50"], errors="coerce")).mean()),
    }


@app.command()
def main(
    bundle_dir: Path = typer.Option(DEFAULT_BUNDLE_DIR, "--bundle-dir"),
    calib_path: Path = typer.Option(
        DEFAULT_OUTPUT_DIR / "k_params.json",
        "--calib-path",
        help="Path to k_params.json produced by fit_minutes_calibration.py",
    ),
    features_root: Path = typer.Option(DEFAULT_FEATURES_ROOT, "--features-root"),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir"),
) -> None:
    """Compare raw vs calibrated tail metrics on VAL."""

    bundle_dir = bundle_dir.expanduser().resolve()
    features_root = features_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    calib_path = calib_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = bundle_dir / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    playable_threshold = float(metrics_payload.get("playable_minutes_threshold", 10.0))
    run_id = bundle_dir.name

    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration params missing at {calib_path}")
    params = MinutesCalibrationParams.from_dict(json.loads(calib_path.read_text(encoding="utf-8")))

    typer.echo(f"[calibration-eval] loading VAL slice from {features_root}")
    val_df = _load_scored_slice(
        bundle_dir,
        "val",
        features_root=features_root,
        data_root=data_root,
    )
    val_df["is_playable"] = _playable_mask(val_df, playable_threshold)
    mask = val_df["is_playable"].astype(bool)

    raw_metrics = _metrics(val_df, p10_col="minutes_p10", p90_col="minutes_p90", mask=mask)
    calibrated = apply_k_params(val_df, params)
    cal_metrics = _metrics(calibrated, p10_col="minutes_p10_cal", p90_col="minutes_p90_cal", mask=mask)

    summary = {
        "run_id": run_id,
        "group_keys": params.group_keys,
        "raw": {
            "p10_cond_playable": raw_metrics["p10"],
            "p90_cond_playable": raw_metrics["p90"],
            "inside_cond_playable": raw_metrics["inside"],
            "val_mae_p50": raw_metrics["mae_p50"],
            "mpiwn": raw_metrics["mpiwn"],
        },
        "calibrated": {
            "p10_cond_playable": cal_metrics["p10"],
            "p90_cond_playable": cal_metrics["p90"],
            "inside_cond_playable": cal_metrics["inside"],
            "val_mae_p50": cal_metrics["mae_p50"],
            "mpiwn": cal_metrics["mpiwn"],
        },
    }

    out_path = output_dir / "calibration_eval_val.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"[calibration-eval] wrote summary to {out_path}")
    typer.echo(
        "[calibration-eval] VAL playable (all): "
        f"raw p10={raw_metrics['p10']:.3f}, p90={raw_metrics['p90']:.3f}, inside={raw_metrics['inside']:.3f} | "
        f"cal p10={cal_metrics['p10']:.3f}, p90={cal_metrics['p90']:.3f}, inside={cal_metrics['inside']:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
