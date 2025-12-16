"""Debug utilities for the rates_v1 efficiency heads (fg2/fg3/ft pct).

Scores a parquet slice with a loaded rates bundle and reports MAE/RMSE plus
simple decile calibration tables for the three pct heads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from projections.rates_v1.production import load_production_rates_bundle
from projections.rates_v1.loader import load_rates_bundle
from projections.rates_v1.score import predict_rates
from projections.rates_v1.schemas import EFFICIENCY_TARGETS

app = typer.Typer(add_completion=False)


def _load_bundle(run_id: Optional[str], artifacts_root: Optional[Path]):
    if run_id:
        return load_rates_bundle(run_id, base_artifacts_root=artifacts_root)
    return load_production_rates_bundle()


def _clamp_preds(df: pd.DataFrame) -> pd.DataFrame:
    pct_clamps = {
        "fg2_pct": (0.3, 0.75),
        "fg3_pct": (0.2, 0.55),
        "ft_pct": (0.5, 0.95),
    }
    for col, (lo, hi) in pct_clamps.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def _evaluate(df: pd.DataFrame, pred_col: str, label_col: str) -> dict:
    mask = df[label_col].notna() & df[pred_col].notna()
    if not mask.any():
        return {"n": 0}
    sub = df.loc[mask].copy()
    err = sub[pred_col] - sub[label_col]
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    # Calibration by decile
    try:
        sub["bucket"] = pd.qcut(sub[pred_col], 10, duplicates="drop")
        calib = (
            sub.groupby("bucket")[[pred_col, label_col]]
            .mean()
            .reset_index()
            .rename(columns={pred_col: "pred_mean", label_col: "label_mean"})
        )
    except ValueError:
        calib = pd.DataFrame()
    return {"n": int(mask.sum()), "mae": mae, "rmse": rmse, "calibration": calib}


@app.command()
def main(
    parquet_path: Path = typer.Argument(..., help="Path to rates_training_base parquet slice."),
    run_id: Optional[str] = typer.Option(None, help="Explicit rates run_id (defaults to production)."),
    artifacts_root: Optional[Path] = typer.Option(None, help="Optional artifacts root for run_id."),
):
    df = pd.read_parquet(parquet_path)
    bundle = _load_bundle(run_id, artifacts_root)
    missing = [c for c in bundle.feature_cols if c not in df.columns]
    if missing:
        raise typer.BadParameter(f"Missing feature columns: {missing}")

    preds = predict_rates(df, bundle)
    preds = _clamp_preds(preds)
    label_map = {
        "fg2_pct": "fg2_pct_label",
        "fg3_pct": "fg3_pct_label",
        "ft_pct": "ft_pct_label",
    }
    report = {}
    for target in EFFICIENCY_TARGETS:
        label_col = label_map[target]
        if label_col not in df.columns or target not in preds.columns:
            typer.echo(f"[debug] skipping {target}: missing label or prediction")
            continue
        metrics = _evaluate(pd.concat([df[[label_col]].reset_index(drop=True), preds[[target]].reset_index(drop=True)], axis=1), target, label_col)
        report[target] = metrics

    for target, metrics in report.items():
        typer.echo(f"\n=== {target} ===")
        typer.echo(f"n={metrics['n']} mae={metrics.get('mae')} rmse={metrics.get('rmse')}")
        calib = metrics.get("calibration")
        if calib is not None and not calib.empty:
            typer.echo("calibration (deciles):")
            typer.echo(calib.to_string(index=False))


if __name__ == "__main__":
    app()
