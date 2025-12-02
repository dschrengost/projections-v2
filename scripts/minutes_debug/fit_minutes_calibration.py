"""Fit simple post-hoc tail calibration for minutes_v1 quantiles (no p_play bundle)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import typer

from projections import paths
from projections.labels import derive_starter_flag_labels
from projections.minutes_v1.calibration import MinutesCalibrationParams, apply_k_params, fit_k_params
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


def _coverage_metrics(
    df: pd.DataFrame,
    *,
    p10_col: str,
    p90_col: str,
    mask: pd.Series,
) -> tuple[float, float, float]:
    y = pd.to_numeric(df.loc[mask, "minutes"], errors="coerce")
    p10 = pd.to_numeric(df.loc[mask, p10_col], errors="coerce")
    p90 = pd.to_numeric(df.loc[mask, p90_col], errors="coerce")
    if y.empty:
        return 0.0, 0.0, 0.0
    p10_cov = float((y < p10).mean())
    p90_cov = float((y <= p90).mean())
    inside = float(((y >= p10) & (y <= p90)).mean())
    return p10_cov, p90_cov, inside


def _print_group_coverage(df: pd.DataFrame, params: MinutesCalibrationParams, threshold: float) -> None:
    mask = _playable_mask(df, threshold)
    raw_p10, raw_p90, raw_inside = _coverage_metrics(df, p10_col="minutes_p10", p90_col="minutes_p90", mask=mask)
    calibrated = apply_k_params(df, params)
    cal_p10, cal_p90, cal_inside = _coverage_metrics(
        calibrated, p10_col="minutes_p10_cal", p90_col="minutes_p90_cal", mask=mask
    )
    typer.echo(
        f"[calibration] playable coverage (all): raw p10={raw_p10:.3f}, p90={raw_p90:.3f}, inside={raw_inside:.3f} | "
        f"cal p10={cal_p10:.3f}, p90={cal_p90:.3f}, inside={cal_inside:.3f}"
    )
    for starter_flag, group in calibrated.groupby("starter_flag"):
        gmask = mask.loc[group.index]
        gp10, gp90, ginside = _coverage_metrics(
            group, p10_col="minutes_p10", p90_col="minutes_p90", mask=gmask
        )
        cp10, cp90, cinside = _coverage_metrics(
            group, p10_col="minutes_p10_cal", p90_col="minutes_p90_cal", mask=gmask
        )
        typer.echo(
            f"[calibration] starter_flag={int(starter_flag)} playable coverage: "
            f"raw p10={gp10:.3f}, p90={gp90:.3f}, inside={ginside:.3f} | "
            f"cal p10={cp10:.3f}, p90={cp90:.3f}, inside={cinside:.3f}"
        )


@app.command()
def main(
    bundle_dir: Path = typer.Option(DEFAULT_BUNDLE_DIR, "--bundle-dir"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, "--output-dir"),
    features_root: Path = typer.Option(DEFAULT_FEATURES_ROOT, "--features-root"),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, "--data-root"),
) -> None:
    """Fit k_low/k_high on CAL for the no_p_play bundle."""

    bundle_dir = bundle_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    features_root = features_root.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = bundle_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    playable_threshold = float(metrics.get("playable_minutes_threshold", 10.0))

    typer.echo(f"[calibration] loading CAL slice from {features_root}")
    cal_df = _load_scored_slice(
        bundle_dir,
        "cal",
        features_root=features_root,
        data_root=data_root,
    )
    cal_df["is_playable"] = _playable_mask(cal_df, playable_threshold)
    params = fit_k_params(
        cal_df,
        group_keys=["starter_flag"],
        target_col="minutes",
        minutes_p10_col="minutes_p10",
        minutes_p50_col="minutes_p50",
        minutes_p90_col="minutes_p90",
        status_col="status",
        is_playable_col="is_playable",
        playable_minutes_threshold=playable_threshold,
    )

    payload = params.to_dict()
    payload["run_id"] = bundle_dir.name
    params_path = output_dir / "k_params.json"
    params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"[calibration] wrote params to {params_path}")

    _print_group_coverage(cal_df, params, playable_threshold)


if __name__ == "__main__":  # pragma: no cover
    app()
