"""
Train rates_v1 LightGBM models (one regressor per per-minute target).

Inputs:
- gold/rates_training_base/season=YYYY/game_date=YYYY-MM-DD/rates_training_base.parquet

Outputs (per run_id):
- artifacts/rates_v1/runs/<run_id>/model_<target>.txt (LightGBM boosters)
- artifacts/rates_v1/runs/<run_id>/feature_cols.json
- artifacts/rates_v1/runs/<run_id>/meta.json
- artifacts/rates_v1/runs/<run_id>/metrics.json
- config/rates_current_run.json tracks the provisional production run_id; load via projections.rates_v1.current.

Usage example (multi-season window):
    uv run python -m scripts.rates.train_rates_v1 \
        --start-date     2023-10-01 \
        --end-date       2025-11-26 \
        --train-end-date 2024-06-30 \
        --cal-end-date   2025-03-01 \
        --data-root      /home/daniel/projections-data \
        --run-tag        rates_v1_stage1

Stage definitions:
- Stage 0 (leaky upper bound): uses minutes_actual as a feature.
- Stage 1 (non-leaky): uses minutes_pred_p50/minutes_pred_spread/minutes_pred_play_prob from minutes_for_rates.
- Stage 2 (tracking): Stage 1 plus tracking rates and role cluster features from tracking_roles.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import typer

from projections.paths import data_path
from projections.rates_v1.features import get_rates_feature_sets
from projections.rates_v1.schemas import EFFICIENCY_TARGETS
from projections.registry.manifest import (
    load_manifest,
    save_manifest,
    register_model,
)

app = typer.Typer(add_completion=False)

RATE_TARGETS = [
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

TARGETS = RATE_TARGETS + EFFICIENCY_TARGETS

TARGET_LABEL_MAP: dict[str, str] = {target: target for target in RATE_TARGETS}
TARGET_LABEL_MAP.update(
    {
        "fg2_pct": "fg2_pct_label",
        "fg3_pct": "fg3_pct_label",
        "ft_pct": "ft_pct_label",
    }
)

_FEATURE_SETS = get_rates_feature_sets()
STAGE0_FEATURES = _FEATURE_SETS["stage0"]
STAGE1_FEATURES = _FEATURE_SETS["stage1"]
TRACKING_FEATURES = [c for c in _FEATURE_SETS["stage2_tracking"] if c not in STAGE1_FEATURES]
FEATURES_STAGE0 = STAGE0_FEATURES
FEATURES_STAGE1 = STAGE1_FEATURES
FEATURES_STAGE2_TRACKING = _FEATURE_SETS["stage2_tracking"]
CONTEXT_FEATURES = [c for c in _FEATURE_SETS["stage3_context"] if c not in FEATURES_STAGE2_TRACKING]
FEATURES_STAGE3_CONTEXT = _FEATURE_SETS["stage3_context"]

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


def _prepare_features(
    df: pd.DataFrame,
    *,
    use_predicted_minutes: bool,
    fallback_minutes_with_actual: bool,
    use_tracking_features: bool,
) -> pd.DataFrame:
    for col in ("is_starter", "home_flag"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
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
        "season_fg2_pct",
        "season_fg3_pct",
        "season_ft_pct",
        "days_rest",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

        if use_predicted_minutes:
            if "minutes_pred_spread" not in df.columns:
                df["minutes_pred_spread"] = np.nan
            if "minutes_pred_p90" in df.columns and "minutes_pred_p10" in df.columns:
                df["minutes_pred_spread"] = df["minutes_pred_p90"] - df["minutes_pred_p10"]
        if fallback_minutes_with_actual:
            pred_cols = ["minutes_pred_p50", "minutes_pred_spread", "minutes_pred_play_prob"]
            missing_mask = df[pred_cols].isna().any(axis=1)
            if missing_mask.any():
                fallback_count = int(missing_mask.sum())
                typer.echo(
                    f"[train] minutes_pred_* missing for {fallback_count:,} rows; "
                    "falling back to minutes_actual (spread=0, play_prob=1)."
                )
                df.loc[missing_mask, "minutes_pred_p50"] = df.loc[missing_mask, "minutes_actual"]
                df.loc[missing_mask, "minutes_pred_spread"] = 0.0
                df.loc[missing_mask, "minutes_pred_play_prob"] = 1.0
                if "minutes_pred_p10" in df.columns:
                    df.loc[missing_mask, "minutes_pred_p10"] = df.loc[missing_mask, "minutes_actual"]
                if "minutes_pred_p90" in df.columns:
                    df.loc[missing_mask, "minutes_pred_p90"] = df.loc[missing_mask, "minutes_actual"]
    if use_tracking_features:
        tracking_cols = [
            "track_touches_per_min_szn",
            "track_sec_per_touch_szn",
            "track_pot_ast_per_min_szn",
            "track_drives_per_min_szn",
        ]
        for col in tracking_cols:
            if col not in df.columns:
                df[col] = np.nan
            mean_val = df[col].mean(skipna=True)
            fill_val = 0.0 if pd.isna(mean_val) else mean_val
            df[col] = df[col].fillna(fill_val)
        if "track_role_cluster" in df.columns:
            df["track_role_cluster"] = df["track_role_cluster"].fillna(-1).astype(int)
        else:
            df["track_role_cluster"] = -1
        if "track_role_is_low_minutes" in df.columns:
            df["track_role_is_low_minutes"] = df["track_role_is_low_minutes"].fillna(True).astype(int)
        else:
            df["track_role_is_low_minutes"] = 1
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


def _clean_frame(df: pd.DataFrame, label_map: dict[str, str], features: list[str]) -> pd.DataFrame:
    df = df.copy()
    cols_needed = set(features + list(label_map.values()))
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with missing features; labels may be NaN for low-attempt games
    df = df.dropna(subset=features)
    return df


def _train_one(
    target: str,
    label_col: str,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    features: list[str],
) -> tuple[lgb.Booster | None, dict]:
    train_mask = train_df[label_col].notna()
    cal_mask = cal_df[label_col].notna()
    if not train_mask.any():
        return None, {"best_iteration": None, "cal_l2": None, "train_rows": 0, "cal_rows": int(cal_mask.sum())}

    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, label_col]
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

    callbacks = []
    valid_sets = []
    if cal_mask.any():
        X_cal = cal_df.loc[cal_mask, features]
        y_cal = cal_df.loc[cal_mask, label_col]
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
        "train_rows": int(train_mask.sum()),
        "cal_rows": int(cal_mask.sum()),
    }
    return booster, metrics


def _eval_split(booster: lgb.Booster, df: pd.DataFrame, features: list[str], target: str) -> dict:
    if df.empty:
        return {"mae": None, "rmse": None, "n": 0}
    mask = df[target].notna()
    if not mask.any():
        return {"mae": None, "rmse": None, "n": 0}
    preds = booster.predict(df.loc[mask, features], num_iteration=booster.best_iteration)
    y_true = df.loc[mask, target].values
    err = preds - y_true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    return {"mae": mae, "rmse": rmse, "n": int(mask.sum())}


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
    run_id: Optional[str] = typer.Option(None, help="Override run id; defaults to <run_tag>_<timestamp>."),
    run_tag: str = typer.Option(
        "rates_v1_stage1",
        help="Run tag determines feature set and default run_id prefix (use stage0 for legacy minutes_actual).",
    ),
    feature_set: str = typer.Option(
        "stage1",
        help="Feature set to use: stage0, stage1 (minutes_pred), stage2_tracking (minutes_pred + tracking roles), or stage3_context (tracking + pace/injury context).",
        case_sensitive=False,
    ),
    allow_minutes_actual_fallback: bool = typer.Option(
        True,
        "--allow-minutes-actual-fallback/--no-minutes-actual-fallback",
        help="When using predicted minutes, fall back to minutes_actual if minutes_pred_* are missing.",
    ),
) -> None:
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize() if start_date else None
    end = pd.Timestamp(end_date).normalize() if end_date else None
    train_cutoff = pd.Timestamp(train_end_date).normalize()
    cal_cutoff = pd.Timestamp(cal_end_date).normalize()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = run_tag if run_tag.startswith("rates_v1_") else f"rates_v1_{run_tag}"
    default_run_id = f"{prefix}_{timestamp}"
    resolved_run_id = run_id or default_run_id
    base_output = output_root or (root / "artifacts" / "rates_v1" / "runs")
    run_dir = base_output / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_set_key = feature_set.lower()
    feature_map = {
        "stage0": FEATURES_STAGE0,
        "stage1": FEATURES_STAGE1,
        "stage2_tracking": FEATURES_STAGE2_TRACKING,
        "stage3_context": FEATURES_STAGE3_CONTEXT,
    }
    if feature_set_key not in feature_map:
        raise typer.BadParameter(f"feature_set must be one of {list(feature_map.keys())}")
    feature_cols = feature_map[feature_set_key]
    use_predicted_minutes = feature_set_key in {"stage1", "stage2_tracking", "stage3_context"}
    fallback_minutes = use_predicted_minutes and allow_minutes_actual_fallback
    use_tracking_features = feature_set_key in {"stage2_tracking", "stage3_context"}

    typer.echo(
        f"[train] run_id={resolved_run_id} data_root={root} "
        f"date_window=({start_date} to {end_date}) train_end={train_cutoff.date()} cal_end={cal_cutoff.date()} "
        f"features={feature_set_key}"
    )

    # Configure MLFlow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("rates_v1")
    mlflow.start_run(run_name=resolved_run_id)
    mlflow.log_params({
        "run_tag": run_tag,
        "feature_set": feature_set_key,
        "train_end": train_end_date,
        "cal_end": cal_end_date,
        "start_date": start_date or "None",
        "end_date": end_date or "None",
        **{f"lgb_{k}": v for k, v in BASE_PARAMS.items()},
    })

    df = _load_training_base(root, start, end)
    df = _prepare_features(
        df,
        use_predicted_minutes=use_predicted_minutes,
        fallback_minutes_with_actual=fallback_minutes,
        use_tracking_features=use_tracking_features,
    )

    train_df, cal_df, val_df = _split_by_date(df, train_cutoff, cal_cutoff)
    if train_df.empty or cal_df.empty or val_df.empty:
        typer.echo(
            f"[train] warning: split sizes train={len(train_df)}, cal={len(cal_df)}, val={len(val_df)}"
        )
    train_df, cal_df, val_df = _impute_odds(train_df, cal_df, val_df)
    train_df = _clean_frame(train_df, TARGET_LABEL_MAP, feature_cols)
    cal_df = _clean_frame(cal_df, TARGET_LABEL_MAP, feature_cols)
    val_df = _clean_frame(val_df, TARGET_LABEL_MAP, feature_cols)

    metrics: dict[str, dict] = {}
    model_paths: dict[str, str] = {}
    for target in TARGETS:
        label_col = TARGET_LABEL_MAP.get(target, target)
        typer.echo(f"[train] training target={target} (label={label_col})")
        booster, train_metrics = _train_one(target, label_col, train_df, cal_df, feature_cols)
        if booster is None:
            typer.echo(f"[train] skipping target={target}: no rows with non-null labels", err=True)
            metrics[target] = {**train_metrics, "cal_mae": None, "cal_rmse": None, "val_mae": None, "val_rmse": None}
            continue
        cal_metrics = _eval_split(booster, cal_df, feature_cols, label_col)
        val_metrics = _eval_split(booster, val_df, feature_cols, label_col)
        metrics[target] = {
            **train_metrics,
            **{f"cal_{k}": v for k, v in cal_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        model_path = run_dir / f"model_{target}.txt"
        booster.save_model(str(model_path))
        model_paths[target] = str(model_path)
        
        # Log model to MLFlow for registry support
        mlflow.lightgbm.log_model(booster, artifact_path=f"model_{target}")

    _write_json(run_dir / "feature_cols.json", {"feature_cols": feature_cols})
    meta = {
        "run_id": resolved_run_id,
        "run_tag": run_tag,
        "feature_set": feature_set_key,
        "targets": TARGETS,
        "label_map": TARGET_LABEL_MAP,
        "feature_cols": feature_cols,
        "params": BASE_PARAMS,
        "train_rows": len(train_df),
        "cal_rows": len(cal_df),
        "val_rows": len(val_df),
    }
    # Log split sizes to MLFlow
    mlflow.log_metrics({
        "train_rows": len(train_df),
        "cal_rows": len(cal_df),
        "val_rows": len(val_df),
    })
    meta.update({
        "date_window": {
            "start": start_date,
            "end": end_date,
            "train_end": train_end_date,
            "cal_end": cal_end_date,
        },
        "notes": [
            "Stage 0 uses minutes_actual (leaky upper bound).",
            "Stage 1 uses minutes_pred_* from minutes_for_rates (non-leaky); missing preds fall back to minutes_actual.",
            "Stage 2 adds tracking-based rates and role cluster features on top of Stage 1.",
        ],
        "models": model_paths,
    })
    _write_json(run_dir / "meta.json", meta)
    _write_json(run_dir / "metrics.json", metrics)

    # Log per-target metrics to MLFlow
    for target, vals in metrics.items():
        if vals.get("val_mae") is not None:
            mlflow.log_metric(f"{target}_val_mae", vals["val_mae"])
        if vals.get("val_rmse") is not None:
            mlflow.log_metric(f"{target}_val_rmse", vals["val_rmse"])
        if vals.get("cal_mae") is not None:
            mlflow.log_metric(f"{target}_cal_mae", vals["cal_mae"])

    # Log aggregate metrics
    val_maes = [m.get("val_mae") for m in metrics.values() if m.get("val_mae") is not None]
    if val_maes:
        mlflow.log_metric("avg_val_mae", float(np.mean(val_maes)))

    # Log artifacts
    mlflow.log_artifact(str(run_dir / "meta.json"))
    mlflow.log_artifact(str(run_dir / "metrics.json"))
    mlflow.log_artifact(str(run_dir / "feature_cols.json"))

    mlflow.end_run()

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

    # Auto-register model in registry
    try:
        manifest = load_manifest()
        # Compute aggregate metrics across all targets
        val_maes = [m.get("val_mae") for m in metrics.values() if m.get("val_mae") is not None]
        avg_val_mae = float(np.mean(val_maes)) if val_maes else None
        register_model(
            manifest,
            model_name="rates_v1_lgbm",
            version=resolved_run_id,
            run_id=resolved_run_id,
            artifact_path=str(run_dir),
            training_start=start_date or "2023-10-01",
            training_end=train_end_date,
            feature_schema_version=feature_set_key,
            metrics={
                "val_mae_avg": avg_val_mae,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "num_targets": len(TARGETS),
            },
            description=f"Train to {train_end_date} | {feature_set_key} | {len(TARGETS)} targets",
        )
        save_manifest(manifest)
        typer.echo(f"[registry] Registered rates_v1_lgbm v{resolved_run_id} (stage=dev)")
    except Exception as e:
        typer.echo(f"[registry] Warning: Failed to register model: {e}", err=True)


if __name__ == "__main__":
    app()
