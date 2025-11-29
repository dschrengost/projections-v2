from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import typer

from projections.rates_v1.features import get_rates_feature_sets
from projections.fpts_v2.features import build_fpts_design_matrix, CATEGORICAL_FEATURES_DEFAULT

app = typer.Typer(add_completion=False)

TARGET = "dk_fpts_actual"
DEFAULT_START = "2023-10-01"
DEFAULT_END = "2025-11-26"
DEFAULT_DATA_ROOT = Path(os.environ.get("PROJECTIONS_DATA_ROOT") or "/home/daniel/projections-data")

MINUTES_FEATURES = ["minutes_p10", "minutes_p50", "minutes_p90", "play_prob", "is_starter"]
RATE_PRED_FEATURES = [
    "pred_fga2_per_min",
    "pred_fga3_per_min",
    "pred_fta_per_min",
    "pred_ast_per_min",
    "pred_tov_per_min",
    "pred_oreb_per_min",
    "pred_dreb_per_min",
    "pred_stl_per_min",
    "pred_blk_per_min",
]
CONTEXT_FEATURES = [
    "home_flag",
    "team_itt",
    "opp_itt",
    "team_pace_szn",
    "team_off_rtg_szn",
    "team_def_rtg_szn",
    "opp_pace_szn",
    "opp_def_rtg_szn",
    "vac_min_szn",
    "vac_fga_szn",
    "vac_ast_szn",
    "position_flags_PG",
    "position_flags_SG",
    "position_flags_SF",
    "position_flags_PF",
    "position_flags_C",
]
CATEGORICAL_FEATURES = ["track_role_cluster", "track_role_is_low_minutes"]
FEATURE_COLS = MINUTES_FEATURES + RATE_PRED_FEATURES + CONTEXT_FEATURES + CATEGORICAL_FEATURES

MODEL_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "n_estimators": 4000,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _iter_partitions(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    base = root / "gold" / "fpts_training_base"
    partitions: list[Path] = []
    for season_dir in base.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day = pd.Timestamp(day_dir.name.split("=", 1)[1]).normalize()
            except ValueError:
                continue
            if day < start or day > end:
                continue
            candidate = day_dir / "fpts_training_base.parquet"
            if candidate.exists():
                partitions.append(candidate)
    if not partitions:
        raise FileNotFoundError("No fpts_training_base partitions matched the requested window.")
    return partitions


def _load_training_base(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    paths = _iter_partitions(root, start, end)
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def _split_by_date(
    df: pd.DataFrame, train_end: pd.Timestamp, cal_end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["game_date"] <= train_end].copy()
    cal_df = df[(df["game_date"] > train_end) & (df["game_date"] <= cal_end)].copy()
    val_df = df[df["game_date"] > cal_end].copy()
    return train_df, cal_df, val_df


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    design_matrix = build_fpts_design_matrix(
        df,
        feature_cols,
        categorical_cols=CATEGORICAL_FEATURES_DEFAULT,
        fill_missing_with_zero=True,
        warn_missing=True,
    )
    design = df.copy()
    for col in design_matrix.columns:
        design[col] = design_matrix[col]

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in design_matrix.columns]
    num_cols = [c for c in design_matrix.columns if c not in cat_cols]

    for col in cat_cols:
        design[col] = design[col].astype("category")

    design = design.dropna(subset=[TARGET])
    return design, num_cols, cat_cols


def _compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    err = preds - y_true
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else None
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(y_true))}


def _train_model(train_df: pd.DataFrame, cal_df: pd.DataFrame, feature_cols: list[str]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    fit_kwargs = {
        "X": train_df[feature_cols],
        "y": train_df[TARGET],
        "eval_metric": "l2",
    }
    callbacks = []
    if not cal_df.empty:
        fit_kwargs["eval_set"] = [(cal_df[feature_cols], cal_df[TARGET])]
        callbacks.append(lgb.early_stopping(100, verbose=False))
    if callbacks:
        fit_kwargs["callbacks"] = callbacks
    model.fit(**fit_kwargs)
    return model


def _compute_baselines(df: pd.DataFrame) -> pd.DataFrame:
    working = df.sort_values(["season", "player_id", "game_date"]).copy()
    minutes_actual = pd.to_numeric(working.get("minutes_actual"), errors="coerce")
    minutes_actual = minutes_actual.replace(0.0, np.nan)

    fpts_per_min_actual = working[TARGET] / minutes_actual
    working["fpts_per_min_actual"] = fpts_per_min_actual
    working["season_fpts_per_min_prior"] = (
        working.groupby(["season", "player_id"])["fpts_per_min_actual"]
        .transform(lambda s: s.shift().expanding().mean())
    )
    global_fpm = fpts_per_min_actual.mean()
    working["season_fpts_per_min_prior"] = working["season_fpts_per_min_prior"].fillna(global_fpm)
    working["baseline_season"] = working["season_fpts_per_min_prior"] * working["minutes_p50"]

    coeffs = {"pts": 1.0, "reb": 1.25, "ast": 1.5, "stl": 2.0, "blk": 2.0, "tov": -0.5}
    for col in coeffs:
        per_min = pd.to_numeric(working.get(col), errors="coerce") / minutes_actual
        working[f"{col}_per_min_actual"] = per_min
        working[f"season_{col}_per_min_prior"] = (
            working.groupby(["season", "player_id"])[f"{col}_per_min_actual"]
            .transform(lambda s: s.shift().expanding().mean())
        )
        working[f"season_{col}_per_min_prior"] = working[f"season_{col}_per_min_prior"].fillna(
            per_min.mean()
        )

    working["season_naive_fpts_per_min"] = 0.0
    for col, weight in coeffs.items():
        working["season_naive_fpts_per_min"] += weight * working[f"season_{col}_per_min_prior"]
    working["baseline_naive_rates"] = working["season_naive_fpts_per_min"] * working["minutes_p50"]
    return working


def _score_split(model: lgb.LGBMRegressor, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    if df.empty:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    num_iter = model.best_iteration_
    preds = model.predict(df[feature_cols], num_iteration=num_iter)
    return _compute_metrics(df[TARGET].values, preds)


def _score_baseline(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    mask = df[col].notna() & df[TARGET].notna()
    if not mask.any():
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    return _compute_metrics(df.loc[mask, TARGET].values, df.loc[mask, col].values)


@app.command()
def main(
    data_root: Optional[Path] = typer.Option(None, help="Root containing gold/fpts_training_base."),
    start_date: str = typer.Option(DEFAULT_START, help="Start date (YYYY-MM-DD)."),
    end_date: str = typer.Option(DEFAULT_END, help="End date (YYYY-MM-DD)."),
    train_end_date: str = typer.Option("2024-06-30", help="Train cutoff (game_date <= train_end_date)."),
    cal_end_date: str = typer.Option("2025-03-01", help="Calibration cutoff (game_date <= cal_end_date)."),
    output_root: Optional[Path] = typer.Option(None, help="Artifacts root (defaults to <data_root>/artifacts/fpts_v2/runs)."),
    run_id: Optional[str] = typer.Option(None, help="Override run id (defaults to <run_tag>_<timestamp>)."),
    run_tag: str = typer.Option("fpts_v2_stage0", help="Run tag prefix for artifacts."),
    feature_set: str = typer.Option(
        "stage0",
        help="Feature set: stage0 (core minutes+rates+basic context), stage1 (alias of stage0), stage3_context (adds rates Stage3 context).",
        case_sensitive=False,
    ),
) -> None:
    root = data_root or DEFAULT_DATA_ROOT
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    train_cutoff = pd.Timestamp(train_end_date).normalize()
    cal_cutoff = pd.Timestamp(cal_end_date).normalize()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    resolved_run_id = run_id or f"{run_tag}_{timestamp}"
    artifacts_root = output_root or (root / "artifacts" / "fpts_v2" / "runs")
    run_dir = artifacts_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"[fpts_train] run_id={resolved_run_id} data_root={root} window=({start.date()} to {end.date()}) "
        f"train_end={train_cutoff.date()} cal_end={cal_cutoff.date()}"
    )

    df = _load_training_base(root, start, end)
    df = _compute_baselines(df)

    feature_set_key = feature_set.lower()
    if feature_set_key not in {"stage0", "stage1", "stage3_context"}:
        raise typer.BadParameter("feature_set must be one of: stage0, stage1, stage3_context")
    core_features = MINUTES_FEATURES + RATE_PRED_FEATURES + CONTEXT_FEATURES + CATEGORICAL_FEATURES
    feature_cols = list(core_features)
    if feature_set_key == "stage3_context":
        rates_feature_sets = get_rates_feature_sets()
        stage3_cols = rates_feature_sets.get("stage3_context", [])
        skip_cols = {"minutes_pred_p50", "minutes_pred_spread", "minutes_pred_play_prob"}
        extra_context = [c for c in stage3_cols if c not in skip_cols and c not in feature_cols and c in df.columns]
        feature_cols.extend(extra_context)

    df, numeric_cols, categorical_cols = _prepare_features(df, feature_cols)
    feature_cols = numeric_cols + categorical_cols

    train_df, cal_df, val_df = _split_by_date(df, train_cutoff, cal_cutoff)
    if train_df.empty or val_df.empty:
        typer.echo(
            f"[fpts_train] warning: split sizes train={len(train_df)}, cal={len(cal_df)}, val={len(val_df)}"
        )

    model = _train_model(train_df, cal_df, feature_cols)

    metrics = {
        "train": _score_split(model, train_df, feature_cols),
        "cal": _score_split(model, cal_df, feature_cols),
        "val": _score_split(model, val_df, feature_cols),
    }

    baseline_metrics = {
        "baseline_season": {
            "train": _score_baseline(train_df, "baseline_season"),
            "cal": _score_baseline(cal_df, "baseline_season"),
            "val": _score_baseline(val_df, "baseline_season"),
        },
        "baseline_naive_rates": {
            "train": _score_baseline(train_df, "baseline_naive_rates"),
            "cal": _score_baseline(cal_df, "baseline_naive_rates"),
            "val": _score_baseline(val_df, "baseline_naive_rates"),
        },
    }

    model_path = run_dir / "model.txt"
    model.booster_.save_model(str(model_path))

    (run_dir / "feature_cols.json").write_text(json.dumps({"feature_cols": feature_cols}, indent=2), encoding="utf-8")
    meta = {
        "run_id": resolved_run_id,
        "run_tag": run_tag,
        "data_root": str(root),
        "feature_set": feature_set_key,
        "date_window": {
            "start": start_date,
            "end": end_date,
            "train_end": train_end_date,
            "cal_end": cal_end_date,
        },
        "feature_cols": feature_cols,
        "params": MODEL_PARAMS,
        "rows": {"train": len(train_df), "cal": len(cal_df), "val": len(val_df)},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"model": metrics, "baselines": baseline_metrics}, indent=2), encoding="utf-8"
    )

    typer.echo(f"[fpts_train] saved model -> {model_path}")
    typer.echo(f"[fpts_train] val metrics: {metrics['val']}")


if __name__ == "__main__":
    app()
