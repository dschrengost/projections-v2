"""Evaluate rates_v1 efficiency heads against season pct baselines.

Usage example:

uv run python -m scripts.rates.eval_efficiency_heads \
  --data-root /home/daniel/projections-data \
  --rates-run-id rates_v1_stage3_efficiency_20251205_212357 \
  --start-date 2023-10-01 \
  --end-date   2025-12-05 \
  --train-end-date 2024-06-30 \
  --cal-end-date    2025-03-01 \
  --output-root /home/daniel/projections-data/artifacts/rates_v1/analysis
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import typer

from projections.rates_v1.features import FEATURES_STAGE3_CONTEXT
from projections.rates_v1.schemas import EFFICIENCY_TARGETS

app = typer.Typer(add_completion=False)


def _iter_partitions(base_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    root = base_root / "gold" / "rates_training_base"
    if not root.exists():
        raise FileNotFoundError(f"Missing rates_training_base root at {root}")
    parts: list[Path] = []
    for season_dir in root.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            candidate = day_dir / "rates_training_base.parquet"
            if candidate.exists():
                parts.append(candidate)
    if not parts:
        raise FileNotFoundError("No rates_training_base partitions matched the requested window.")
    return parts


def _load_val_slice(
    base_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cal_end: pd.Timestamp,
) -> pd.DataFrame:
    parts = _iter_partitions(base_root, start, end)
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df = df[(df["game_date"] > cal_end) & (df["game_date"] <= end)].copy()
    label_cols = ["fg2_pct_label", "fg3_pct_label", "ft_pct_label"]
    df = df[df[label_cols].notna().any(axis=1)].copy()
    return df


def _load_models(run_dir: Path) -> dict[str, lgb.Booster]:
    models: dict[str, lgb.Booster] = {}
    for target in EFFICIENCY_TARGETS:
        model_path = run_dir / f"model_{target}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model for target={target} at {model_path}")
        models[target] = lgb.Booster(model_file=str(model_path))
    return models


def _clip_preds(df: pd.DataFrame) -> pd.DataFrame:
    clamps = {
        "fg2_pct_pred": (0.3, 0.75),
        "fg3_pct_pred": (0.2, 0.55),
        "ft_pct_pred": (0.5, 0.95),
    }
    for col, (lo, hi) in clamps.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _calibration(y_true: np.ndarray, y_pred: np.ndarray, q: int = 10) -> list[dict]:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    try:
        df["bin"] = pd.qcut(df["y_pred"], q=q, duplicates="drop")
    except ValueError:
        return []
    calib = (
        df.groupby("bin")
        .agg(pred_mean=("y_pred", "mean"), actual_mean=("y_true", "mean"), count=("y_true", "size"))
        .reset_index()
    )
    calib["bin"] = calib["bin"].astype(str)
    return calib.to_dict(orient="records")


def _pos_bucket(row: pd.Series) -> str:
    flags = {pos: row.get(f"position_flags_{pos}", 0) for pos in ["PG", "SG", "SF", "PF", "C"]}
    on_flags = [pos for pos, val in flags.items() if val == 1]
    if len(on_flags) == 1:
        return on_flags[0]
    return "multi"


def _archetype_splits(df: pd.DataFrame, stat: str, min_rows: int = 100) -> list[dict]:
    splits: list[dict] = []
    label_col = f"{stat}_pct_label"
    base_col = f"season_{stat}_pct"
    pred_col = f"{stat}_pct_pred"
    if not {label_col, base_col, pred_col}.issubset(df.columns):
        return splits
    df = df[df[[label_col, base_col, pred_col]].notna().all(axis=1)].copy()
    if df.empty:
        return splits
    df["pos_bucket"] = df.apply(_pos_bucket, axis=1)
    df["is_starter_flag"] = df.get("is_starter", 0).fillna(0).astype(int)
    grouped = df.groupby(["pos_bucket", "is_starter_flag"])
    for (pos_bucket, starter_flag), frame in grouped:
        if len(frame) < min_rows:
            continue
        y_true = frame[label_col].to_numpy()
        y_base = frame[base_col].to_numpy()
        y_pred = frame[pred_col].to_numpy()
        splits.append(
            {
                "pos_bucket": pos_bucket,
                "is_starter": bool(starter_flag),
                "n": int(len(frame)),
                "baseline_mae": _mae(y_true, y_base),
                "model_mae": _mae(y_true, y_pred),
            }
        )
    return splits


@app.command()
def main(
    data_root: Path = typer.Option(..., help="Base data root (contains gold/rates_training_base)."),
    rates_run_id: str = typer.Option(..., help="rates_v1 run id under artifacts/rates_v1/runs."),
    start_date: str = typer.Option(..., help="Start date YYYY-MM-DD."),
    end_date: str = typer.Option(..., help="End date YYYY-MM-DD."),
    train_end_date: str = typer.Option(..., help="Train end date YYYY-MM-DD."),
    cal_end_date: str = typer.Option(..., help="Cal end date YYYY-MM-DD."),
    output_root: Path = typer.Option(None, help="Where to write analysis artifacts."),
):
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    train_end_ts = pd.Timestamp(train_end_date).normalize()
    cal_end_ts = pd.Timestamp(cal_end_date).normalize()

    base_root = data_root.expanduser().resolve()
    run_dir = base_root / "artifacts" / "rates_v1" / "runs" / rates_run_id
    if output_root is None:
        output_root = base_root / "artifacts" / "rates_v1" / "analysis"
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[eval] loading val slice {cal_end_ts.date()} -> {end_ts.date()}")
    df_val = _load_val_slice(base_root, start_ts, end_ts, cal_end_ts)

    # Backfill minutes_pred_spread if absent
    if "minutes_pred_spread" not in df_val.columns:
        if {"minutes_pred_p90", "minutes_pred_p10"}.issubset(df_val.columns):
            df_val["minutes_pred_spread"] = df_val["minutes_pred_p90"] - df_val["minutes_pred_p10"]
        else:
            df_val["minutes_pred_spread"] = 0.0

    models = _load_models(run_dir)
    missing_features = [c for c in FEATURES_STAGE3_CONTEXT if c not in df_val.columns]
    if missing_features:
        typer.echo(f"[eval] WARNING: filling missing features with 0: {missing_features}", err=True)
        for col in missing_features:
            df_val[col] = 0.0

    X_val = df_val[FEATURES_STAGE3_CONTEXT]
    for target, model in models.items():
        df_val[f"{target}_pred"] = model.predict(X_val, num_iteration=model.best_iteration)
    df_val = _clip_preds(df_val)

    stats = ["fg2", "fg3", "ft"]
    metrics: dict[str, dict] = {}
    calibration: dict[str, list[dict]] = {}
    archetypes: dict[str, list[dict]] = {}

    for stat in stats:
        label_col = f"{stat}_pct_label"
        base_col = f"season_{stat}_pct"
        pred_col = f"{stat}_pct_pred"
        if not {label_col, base_col, pred_col}.issubset(df_val.columns):
            typer.echo(f"[eval] skipping {stat}: missing columns")
            continue
        mask = df_val[[label_col, base_col, pred_col]].notna().all(axis=1)
        if not mask.any():
            typer.echo(f"[eval] skipping {stat}: no rows with complete data")
            continue
        y_true = df_val.loc[mask, label_col].to_numpy()
        y_base = df_val.loc[mask, base_col].to_numpy()
        y_pred = df_val.loc[mask, pred_col].to_numpy()
        metrics[stat] = {
            "n": int(mask.sum()),
            "baseline_mae": _mae(y_true, y_base),
            "baseline_rmse": _rmse(y_true, y_base),
            "model_mae": _mae(y_true, y_pred),
            "model_rmse": _rmse(y_true, y_pred),
        }
        calibration[stat] = _calibration(y_true, y_pred)
        archetypes[stat] = _archetype_splits(df_val.loc[mask].copy(), stat)

    # Print summary
    typer.echo(f"\n== Efficiency head eval (run {rates_run_id}) ==")
    for stat in stats:
        if stat not in metrics:
            continue
        m = metrics[stat]
        typer.echo(
            f"  {stat}_pct: n={m['n']}  baseline_mae={m['baseline_mae']:.3f}  model_mae={m['model_mae']:.3f}  "
            f"baseline_rmse={m['baseline_rmse']:.3f}  model_rmse={m['model_rmse']:.3f}"
        )

    payload = {
        "rates_run_id": rates_run_id,
        "date_generated": datetime.now(tz=UTC).isoformat(),
        "windows": {
            "start_date": start_ts.date().isoformat(),
            "end_date": end_ts.date().isoformat(),
            "train_end_date": train_end_ts.date().isoformat(),
            "cal_end_date": cal_end_ts.date().isoformat(),
        },
        "metrics": metrics,
        "calibration": calibration,
        "archetype_splits": archetypes,
    }

    json_path = output_root / f"efficiency_eval_{rates_run_id}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Also write calibration CSV for convenience
    for stat, rows in calibration.items():
        if not rows:
            continue
        calib_df = pd.DataFrame(rows)
        calib_df.to_csv(output_root / f"calibration_{stat}_{rates_run_id}.csv", index=False)

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv(output_root / f"metrics_{rates_run_id}.csv")

    typer.echo(f"[eval] wrote artifacts to {output_root}")


if __name__ == "__main__":
    app()
