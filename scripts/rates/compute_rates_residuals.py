"""
Compute residual summaries for the current rates_v1 bundle.

For each target, residuals are computed as:
    actual_total = target_per_min * minutes_actual
    pred_total   = minutes_pred_p50 * rate_pred_per_min
    residual     = actual_total - pred_total

Results are written to artifacts/rates_v1/residuals/<run_id>_<split>_residuals.json.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import typer

from projections.paths import data_path
from projections.rates_v1.current import load_current_rates_bundle
from projections.rates_v1.score import predict_rates
from scripts.rates.train_rates_v1 import _load_training_base, _split_by_date

Split = Literal["train", "cal", "val", "all"]

app = typer.Typer(add_completion=False)


def _load_split_df(split: Split, bundle_meta: dict, data_root: Path) -> pd.DataFrame:
    window = bundle_meta.get("date_window") or {}
    start = window.get("start")
    end = window.get("end")
    train_end = window.get("train_end")
    cal_end = window.get("cal_end")
    if not (start and end and train_end and cal_end):
        raise RuntimeError("bundle meta missing date_window with start/end/train_end/cal_end")

    df = _load_training_base(
        data_root,
        start=pd.Timestamp(start).normalize(),
        end=pd.Timestamp(end).normalize(),
    )
    train_df, cal_df, val_df = _split_by_date(
        df,
        train_end=pd.Timestamp(train_end).normalize(),
        cal_end=pd.Timestamp(cal_end).normalize(),
    )
    if split == "train":
        return train_df
    if split == "cal":
        return cal_df
    if split == "val":
        return val_df
    return df


def _coerce_feature_types(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Cast feature columns to numeric types accepted by LightGBM."""

    working = df.copy()
    for col in feature_cols:
        if col not in working.columns:
            continue
        working[col] = pd.to_numeric(working[col], errors="coerce")
    return working


def _same_team_corr(residuals: pd.Series, game_ids: pd.Series, team_ids: pd.Series) -> float | None:
    valid_mask = residuals.notna() & game_ids.notna() & team_ids.notna()
    if not valid_mask.any():
        return None
    r = residuals[valid_mask]
    g = game_ids[valid_mask]
    t = team_ids[valid_mask]
    team_mean = r.groupby([g, t]).transform("mean")
    if team_mean.isna().all():
        return None
    # Correlation between player residual and team mean residual (including self).
    corr = r.corr(team_mean)
    return float(corr) if pd.notna(corr) else None


@app.command()
def main(
    split: Split = typer.Option("val", "--split", case_sensitive=False, help="Split to evaluate: train|cal|val|all"),
    output_root: Optional[Path] = typer.Option(
        None,
        "--output-root",
        help="Root for residual outputs (default: $DATA_ROOT/artifacts/rates_v1/residuals)",
    ),
) -> None:
    split_norm: Split = split.lower()  # type: ignore[assignment]
    bundle = load_current_rates_bundle()
    run_id = bundle.meta.get("run_id") or "unknown_run"
    data_root = data_path()
    out_root = output_root or (data_root / "artifacts" / "rates_v1" / "residuals")
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_split_df(split_norm, bundle.meta, data_root)
    n_raw = len(df)
    df = df[df["minutes_pred_p50"].notna()].copy()
    n_filtered = len(df)
    if df.empty:
        raise RuntimeError(f"No rows after filtering minutes_pred_p50 for split={split_norm}")

    feature_cols = bundle.feature_cols
    target_cols = bundle.meta.get("targets")
    if not target_cols:
        raise RuntimeError("Bundle meta missing targets.")

    # Derive minutes_pred_spread if absent but quantiles available.
    if "minutes_pred_spread" in feature_cols and "minutes_pred_spread" not in df.columns:
        if {"minutes_pred_p90", "minutes_pred_p10"} <= set(df.columns):
            df["minutes_pred_spread"] = df["minutes_pred_p90"] - df["minutes_pred_p10"]
    missing_features = [c for c in feature_cols if c not in df.columns]
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_features:
        raise RuntimeError(f"Missing feature columns in rates_training_base: {missing_features}")
    if missing_targets:
        raise RuntimeError(f"Missing target columns in rates_training_base: {missing_targets}")

    # Drop rows with NA in any feature to avoid scoring errors.
    df_features = _coerce_feature_types(df, feature_cols).dropna(subset=feature_cols).copy()
    preds = predict_rates(df_features[feature_cols], bundle)

    summary: dict[str, dict[str, float | int | None]] = {}
    for target in target_cols:
        # Align predictions back to the filtered rows.
        pred_series = preds.loc[df_features.index, target]
        # Ensure availability of necessary columns.
        required_cols = ["minutes_actual", "minutes_pred_p50", target]
        available_mask = (
            df_features[target].notna()
            & df_features["minutes_actual"].notna()
            & df_features["minutes_pred_p50"].notna()
            & pred_series.notna()
        )
        if not available_mask.any():
            summary[target] = {"count": 0, "mean_resid": None, "var_resid": None, "std_resid": None, "same_team_corr": None}
            continue
        minutes_pred = df_features.loc[available_mask, "minutes_pred_p50"]
        minutes_actual = df_features.loc[available_mask, "minutes_actual"]
        y_true_per_min = df_features.loc[available_mask, target]
        y_true_total = y_true_per_min * minutes_actual
        y_pred_total = minutes_pred * pred_series.loc[available_mask]
        residual = y_true_total - y_pred_total
        count = int(residual.shape[0])
        mean_resid = float(residual.mean())
        var_resid = float(residual.var(ddof=0))
        std_resid = float(sqrt(var_resid))
        same_team = _same_team_corr(
            residual,
            df_features.loc[available_mask, "game_id"],
            df_features.loc[available_mask, "team_id"],
        )
        summary[target] = {
            "count": count,
            "mean_resid": mean_resid,
            "var_resid": var_resid,
            "std_resid": std_resid,
            "same_team_corr": same_team,
        }

    payload = {
        "run_id": run_id,
        "split": split_norm,
        "n_rows_raw": n_raw,
        "n_rows_minutes_pred": n_filtered,
        "targets": summary,
    }
    out_path = out_root / f"{run_id}_{split_norm}_residuals.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(
        f"[residuals] run_id={run_id} split={split_norm} rows_raw={n_raw} rows_minutes_pred={n_filtered} "
        f"written={out_path}"
    )


if __name__ == "__main__":
    app()
