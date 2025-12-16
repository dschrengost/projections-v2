"""Fit a star-only upper-tail calibration (k_high) for minutes_v1 (lab bundle)."""

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
from projections.minutes_v1.calibration import apply_star_k_high, fit_star_k_high
from projections.minutes_v1.datasets import load_feature_frame
from scripts.minutes_debug.status_eval_minutes import _build_windows, _prepare_slice, score_with_bundle

app = typer.Typer(help=__doc__)

DEFAULT_BUNDLE_DIR = Path("artifacts/minutes_lgbm/lgbm_full_v1_no_p_play_20251202")
DEFAULT_OUTPUT_DIR = Path("artifacts/minutes_v1/calibration/lgbm_full_v1_no_p_play_20251202")
DEFAULT_FEATURES_ROOT = paths.data_path("gold", "features_minutes_v1")
DEFAULT_DATA_ROOT = paths.get_data_root()


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


@app.command()
def main(
    bundle_dir: Path = typer.Option(DEFAULT_BUNDLE_DIR, "--bundle-dir"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir"),
    features_root: Path = typer.Option(DEFAULT_FEATURES_ROOT, "--features-root"),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root"),
    p50_threshold: float = typer.Option(32.0, "--p50-threshold", help="Star definition: minutes_p50 >= threshold."),
) -> None:
    """Fit a single k_high for stars (starter & high p50) on CAL; lab-only."""

    bundle_dir = bundle_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    features_root = features_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = bundle_dir / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    playable_threshold = float(metrics_payload.get("playable_minutes_threshold", 10.0))

    typer.echo(f"[star-k-high] loading CAL slice from {features_root}")
    cal_df = _load_scored_slice(
        bundle_dir,
        "cal",
        features_root=features_root,
        data_root=data_root,
    )
    cal_df["is_playable"] = _playable_mask(cal_df, playable_threshold)
    params = fit_star_k_high(
        cal_df,
        p50_threshold=p50_threshold,
        minutes_actual_col="minutes",
        minutes_p50_col="minutes_p50",
        minutes_p90_col="minutes_p90",
        is_playable_col="is_playable",
        status_col="status",
    )

    params_path = output_dir / "star_k_high_params.json"
    payload = params.to_dict()
    payload["run_id"] = bundle_dir.name
    params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"[star-k-high] wrote params to {params_path}")

    # Coverage summary on CAL, stars only.
    star_mask = (cal_df["starter_flag"] == 1) & (cal_df["minutes_p50"] >= p50_threshold) & cal_df["is_playable"]
    star_slice = cal_df.loc[star_mask].copy()
    if star_slice.empty:
        typer.echo("[star-k-high] no star rows found on CAL; nothing to report.")
        return
    y = pd.to_numeric(star_slice["minutes"], errors="coerce")
    p10 = pd.to_numeric(star_slice["minutes_p10"], errors="coerce")
    p50 = pd.to_numeric(star_slice["minutes_p50"], errors="coerce")
    p90 = pd.to_numeric(star_slice["minutes_p90"], errors="coerce")
    raw_p10_under = float((y < p10).mean())
    raw_p90_cov = float((y <= p90).mean())
    raw_inside = float(((y >= p10) & (y <= p90)).mean())
    raw_width = float(np.mean(p90 - p10))

    calibrated = apply_star_k_high(
        star_slice,
        params,
        starter_col="starter_flag",
        minutes_p50_col="minutes_p50",
        minutes_p90_col="minutes_p90",
        out_p90_col="minutes_p90_star_cal",
    )
    p90_cal = pd.to_numeric(calibrated["minutes_p90_star_cal"], errors="coerce")
    cal_p90_cov = float((y <= p90_cal).mean())
    cal_inside = float(((y >= p10) & (y <= p90_cal)).mean())
    cal_width = float(np.mean(p90_cal - p10))

    typer.echo(
        f"[star-k-high] CAL stars (n={len(star_slice)}): "
        f"raw p10_under={raw_p10_under:.3f}, p90_cov={raw_p90_cov:.3f}, inside={raw_inside:.3f}, width={raw_width:.3f} | "
        f"cal p10_under={raw_p10_under:.3f}, p90_cov={cal_p90_cov:.3f}, inside={cal_inside:.3f}, width={cal_width:.3f}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
