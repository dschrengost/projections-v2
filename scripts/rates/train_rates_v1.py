"""
Train rates_v1 stage-0 LightGBM models (one regressor per per-minute target).

Inputs:
- gold/rates_training_base/season=YYYY/game_date=YYYY-MM-DD/rates_training_base.parquet

Outputs (per run_id):
- artifacts/rates_v1/runs/<run_id>/model_<target>.txt (LightGBM boosters)
- artifacts/rates_v1/runs/<run_id>/feature_cols.json
- artifacts/rates_v1/runs/<run_id>/meta.json
- artifacts/rates_v1/runs/<run_id>/metrics.json

Usage example (multi-season window):
    uv run python -m scripts.rates.train_rates_v1 \
        --start-date     2023-10-01 \
        --end-date       2025-11-26 \
        --train-end-date 2024-06-30 \
        --cal-end-date   2025-03-01 \
        --data-root      /home/daniel/projections-data

Notes:
- Uses minutes_actual as a feature (leaky for live). TODO: replace with historical minutes
  predictions (minutes_expected_p50, minutes_spread, play_prob) in future runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)

TARGETS = [
    "fga2_per_min",
    "fga3_per_min",
    "fta_per_min",
    "ast_per_min",
    "tov_per_min",
    "oreb_per_min",
    "dreb_per_min",
    "stl_per_min",
    "blk_per_min",
]

FEATURE_COLS = [
    "minutes_actual",
    "is_starter",
    "position_flags_PG",
    "position_flags_SG",
    "position_flags_SF",
    "position_flags_PF",
    "position_flags_C",
    "season_fga_per_min",
    "season_3pa_per_min",
    "season_fta_per_min",
    "season_ast_per_min",
    "season_tov_per_min",
    "season_reb_per_min",
    "season_stl_per_min",
    "season_blk_per_min",
    "home_flag",
    "days_rest",
    "spread_close",
    "total_close",
    "team_itt",
    "opp_itt",
    "has_odds",
]

BASE_PARAMS: dict[str, object] = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "max_depth": -1,
    "lambda_l2": 1.0,
}


def _iter_partitions(root: Path, start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[Path]:
    base = root / "gold" / "rates_training_base"
    if not base.exists():
        raise FileNotFoundError(f"Missing rates_training_base root at {base}")
    partitions: list[Path] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if start is not None and day < start:
                continue
            if end is not None and day > end:
                continue
            candidate = day_dir / "rates_training_base.parquet"
            if candidate.exists():
                partitions.append(candidate)
    if not partitions:
        raise FileNotFoundError("No rates_training_base partitions matched the requested window.")
    return partitions


def _load_training_base(root: Path, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    paths = _iter_partitions(root, start, end)
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("is_starter", "home_flag"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    odds_cols = ["spread_close", "total_close", "team_itt", "opp_itt"]
    if "has_odds" not in df.columns:
        raise KeyError("has_odds missing from rates_training_base; rebuild base to include it.")
    # Fill season aggregates and rest with zeros when absent (common for early dates)
    fill_zero_cols = [
        "season_fga_per_min",
        "season_3pa_per_min",
        "season_fta_per_min",
        "season_ast_per_min",
        "season_tov_per_min",
        "season_reb_per_min",
        "season_stl_per_min",
        "season_blk_per_min",
        "days_rest",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def _split_by_date(
    df: pd.DataFrame, train_end: pd.Timestamp, cal_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["game_date"] < train_end].copy()
    cal_df = df[(df["game_date"] >= train_end) & (df["game_date"] < cal_end)].copy()
    val_df = df[df["game_date"] >= cal_end].copy()
    return train_df, cal_df, val_df


def _impute_odds(train_df: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    odds_cols = ["spread_close", "total_close", "team_itt", "opp_itt"]
    medians = {}
    for col in odds_cols:
        med = train_df[col].median(skipna=True)
        medians[col] = 0.0 if pd.isna(med) else med
    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        for col in odds_cols:
            frame[col] = frame[col].fillna(medians[col])
        frame["has_odds"] = frame["has_odds"].fillna(0).astype(int)
        return frame
    out_frames = [_apply(train_df)] + [_apply(df) for df in others]
    return tuple(out_frames)


def _clean_frame(df: pd.DataFrame, targets: list[str], features: list[str]) -> pd.DataFrame:
    df = df.copy()
    cols_needed = set(targets + features)
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=targets + features)
    return df


def _train_one(
    target: str, train_df: pd.DataFrame, cal_df: pd.DataFrame, features: list[str]
) -> tuple[lgb.Booster, dict]:
    X_train = train_df[features]
    y_train = train_df[target]
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

    callbacks = []
    valid_sets = []
    if not cal_df.empty:
        X_cal = cal_df[features]
        y_cal = cal_df[target]
        cal_set = lgb.Dataset(X_cal, label=y_cal, reference=train_set, free_raw_data=False)
        valid_sets = [cal_set]
        callbacks.append(lgb.early_stopping(stopping_rounds=200, verbose=False))

    booster = lgb.train(
        params=BASE_PARAMS,
        train_set=train_set,
        valid_sets=valid_sets,
        num_boost_round=5000,
        callbacks=callbacks,
    )
    metrics = {
        "best_iteration": booster.best_iteration,
        "cal_l2": booster.best_score.get("valid_0", {}).get("l2") if valid_sets else None,
    }
    return booster, metrics


def _eval_split(booster: lgb.Booster, df: pd.DataFrame, features: list[str], target: str) -> dict:
    if df.empty:
        return {"mae": None, "rmse": None, "n": 0}
    preds = booster.predict(df[features], num_iteration=booster.best_iteration)
    y_true = df[target].values
    err = preds - y_true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"mae": mae, "rmse": rmse, "n": int(len(df))}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.command()
def main(
    start_date: Optional[str] = typer.Option(None, help="Optional start date YYYY-MM-DD."),
    end_date: Optional[str] = typer.Option(None, help="Optional end date YYYY-MM-DD."),
    train_end_date: str = typer.Option("2024-06-30", help="Cutoff: game_date < train_end_date goes to train (default covers full 23-24)."),
    cal_end_date: str = typer.Option("2025-03-01", help="Cutoff: train_end_date <= date < cal_end_date goes to cal; rest to val."),
    data_root: Optional[Path] = typer.Option(None, help="Root containing gold/rates_training_base."),
    output_root: Optional[Path] = typer.Option(None, help="Base artifacts root (defaults to data_root/artifacts/rates_v1/runs)."),
    run_id: Optional[str] = typer.Option(None, help="Override run id; defaults to rates_v1_stage0_<timestamp>."),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize() if start_date else None
    end = pd.Timestamp(end_date).normalize() if end_date else None
    train_cutoff = pd.Timestamp(train_end_date).normalize()
    cal_cutoff = pd.Timestamp(cal_end_date).normalize()
    default_run_id = f"rates_v1_stage0_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    resolved_run_id = run_id or default_run_id
    base_output = output_root or (root / "artifacts" / "rates_v1" / "runs")
    run_dir = base_output / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"[train] run_id={resolved_run_id} data_root={root} "
        f"date_window=({start_date} to {end_date}) train_end={train_cutoff.date()} cal_end={cal_cutoff.date()}"
    )

    df = _load_training_base(root, start, end)
    df = _prepare_features(df)

    train_df, cal_df, val_df = _split_by_date(df, train_cutoff, cal_cutoff)
    if train_df.empty or cal_df.empty or val_df.empty:
        typer.echo(
            f"[train] warning: split sizes train={len(train_df)}, cal={len(cal_df)}, val={len(val_df)}"
        )
    train_df, cal_df, val_df = _impute_odds(train_df, cal_df, val_df)
    train_df = _clean_frame(train_df, TARGETS, FEATURE_COLS)
    cal_df = _clean_frame(cal_df, TARGETS, FEATURE_COLS)
    val_df = _clean_frame(val_df, TARGETS, FEATURE_COLS)

    metrics: dict[str, dict] = {}
    model_paths: dict[str, str] = {}
    for target in TARGETS:
        typer.echo(f"[train] training target={target}")
        booster, train_metrics = _train_one(target, train_df, cal_df, FEATURE_COLS)
        cal_metrics = _eval_split(booster, cal_df, FEATURE_COLS, target)
        val_metrics = _eval_split(booster, val_df, FEATURE_COLS, target)
        metrics[target] = {
            **train_metrics,
            **{f"cal_{k}": v for k, v in cal_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        model_path = run_dir / f"model_{target}.txt"
        booster.save_model(str(model_path))
        model_paths[target] = str(model_path)

    _write_json(run_dir / "feature_cols.json", {"feature_cols": FEATURE_COLS})
    meta = {
        "run_id": resolved_run_id,
        "targets": TARGETS,
        "feature_cols": FEATURE_COLS,
        "params": BASE_PARAMS,
        "train_rows": len(train_df),
        "cal_rows": len(cal_df),
        "val_rows": len(val_df),
        "date_window": {
            "start": start_date,
            "end": end_date,
            "train_end": train_end_date,
            "cal_end": cal_end_date,
        },
        "notes": [
            "Stage 0 uses minutes_actual (leaky); replace with minutes model predictions in future runs."
        ],
        "models": model_paths,
    }
    _write_json(run_dir / "meta.json", meta)
    _write_json(run_dir / "metrics.json", metrics)

    typer.echo(f"[train] completed. artifacts at {run_dir}")
    typer.echo(f"[train] rows train={len(train_df):,} cal={len(cal_df):,} val={len(val_df):,}")
    typer.echo("[train] val metrics (per target):")
    for target, vals in metrics.items():
        val_n = vals.get("val_n")
        if val_n and val_n > 0:
            typer.echo(
                f"  {target}: val_mae={vals.get('val_mae')} val_rmse={vals.get('val_rmse')} n={val_n}"
            )
        else:
            typer.echo(f"  {target}: val set empty; no holdout metrics")


if __name__ == "__main__":
    app()
