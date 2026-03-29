#!/usr/bin/env python3
"""Train targeted tree-based rate heads and evaluate a simple GTv2 hybrid.

This experiment keeps GTv2 minutes fixed and replaces AST / REB means with:

    ast_mean_hybrid = pred_minutes * pred_ast_per_min
    reb_mean_hybrid = pred_minutes * (pred_oreb_per_min + pred_dreb_per_min)

It is intentionally a mean-layer experiment. World generation is unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


KEY_COLS = ["game_id", "team_id", "player_id", "game_date"]
AST_REB_TARGETS = ["ast_per_min", "oreb_per_min", "dreb_per_min"]
FULL_RATE_TARGETS = [
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
EFF_TARGETS = ["fg2_pct_label", "fg3_pct_label", "ft_pct_label"]
FULL_TARGETS = FULL_RATE_TARGETS + EFF_TARGETS
EXCLUDE_COLS = {
    *KEY_COLS,
    "minutes_actual",
    "rates_non_null_count",
    "efficiency_non_null_count",
    "rates_label_available_any",
    "rates_label_available_all_rate_targets",
    "rates_loss_eligible",
    "fg2_pct_label",
    "fg3_pct_label",
    "ft_pct_label",
    "fga2_per_min",
    "fga3_per_min",
    "fta_per_min",
    "tov_per_min",
    "stl_per_min",
    "blk_per_min",
    *FULL_TARGETS,
}

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
    "verbosity": -1,
}


@dataclass
class TargetModelResult:
    target: str
    model: Any
    best_iteration: int | None
    val_mae: float
    val_rmse: float
    train_rows: int
    cal_rows: int


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _coerce_keys(df: pd.DataFrame, *, require_game_date: bool = True) -> pd.DataFrame:
    out = df.copy()
    for col in ("game_id", "team_id", "player_id"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    elif require_game_date:
        raise KeyError("game_date")
    return out


def _numeric_feature_cols(frame: pd.DataFrame, *, allowed_live_cols: set[str] | None = None) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in EXCLUDE_COLS:
            continue
        if allowed_live_cols is not None and col not in allowed_live_cols:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns found for training.")
    return cols


def _split_train_cal(
    train_df: pd.DataFrame,
    *,
    cal_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_days = sorted(train_df["game_date"].dropna().unique().tolist())
    if not unique_days:
        raise ValueError("No training dates available.")
    cal_days = max(1, int(cal_days))
    if len(unique_days) <= cal_days:
        return train_df.iloc[0:0].copy(), train_df.copy()
    cal_dates = set(unique_days[-cal_days:])
    cal_df = train_df.loc[train_df["game_date"].isin(cal_dates)].copy()
    fit_df = train_df.loc[~train_df["game_date"].isin(cal_dates)].copy()
    return fit_df, cal_df


def _clean_features(df: pd.DataFrame, *, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _sample_weights(
    df: pd.DataFrame,
    *,
    target: str,
    ast_line_threshold: float,
    ast_weight_mult: float,
    reb_line_threshold: float,
    reb_weight_mult: float,
) -> np.ndarray | None:
    weights = np.ones(len(df), dtype=float)
    if target == "ast_per_min" and ast_weight_mult > 1.0 and "an_ast_line" in df.columns:
        line = pd.to_numeric(df["an_ast_line"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weights[line >= float(ast_line_threshold)] *= float(ast_weight_mult)
    elif target in {"oreb_per_min", "dreb_per_min"} and reb_weight_mult > 1.0 and "an_reb_line" in df.columns:
        line = pd.to_numeric(df["an_reb_line"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        weights[line >= float(reb_line_threshold)] *= float(reb_weight_mult)
    if np.allclose(weights, 1.0):
        return None
    return weights


def _predict_model(model: Any, X: pd.DataFrame, *, model_type: str, best_iteration: int | None) -> np.ndarray:
    if model_type == "lgbm":
        return model.predict(X, num_iteration=best_iteration)
    if model_type == "xgb":
        iteration_range = (0, int(best_iteration) + 1) if best_iteration is not None else None
        return model.predict(X, iteration_range=iteration_range)
    raise ValueError(f"Unsupported model_type={model_type}")


def _train_target(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    *,
    model_type: str,
    target: str,
    feature_cols: list[str],
    num_boost_round: int,
    ast_line_threshold: float,
    ast_weight_mult: float,
    reb_line_threshold: float,
    reb_weight_mult: float,
) -> TargetModelResult:
    fit_mask = train_df[target].notna()
    cal_mask = cal_df[target].notna()
    if not fit_mask.any():
        raise ValueError(f"No non-null training labels for target={target}")

    X_train = train_df.loc[fit_mask, feature_cols]
    y_train = train_df.loc[fit_mask, target]
    X_cal = cal_df.loc[cal_mask, feature_cols] if cal_mask.any() else None
    y_cal = cal_df.loc[cal_mask, target] if cal_mask.any() else None
    train_weights = _sample_weights(
        train_df.loc[fit_mask].copy(),
        target=target,
        ast_line_threshold=ast_line_threshold,
        ast_weight_mult=ast_weight_mult,
        reb_line_threshold=reb_line_threshold,
        reb_weight_mult=reb_weight_mult,
    )
    cal_weights = _sample_weights(
        cal_df.loc[cal_mask].copy(),
        target=target,
        ast_line_threshold=ast_line_threshold,
        ast_weight_mult=ast_weight_mult,
        reb_line_threshold=reb_line_threshold,
        reb_weight_mult=reb_weight_mult,
    )

    if model_type == "lgbm":
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            weight=train_weights,
            free_raw_data=False,
        )
        valid_sets = []
        callbacks = []
        if cal_mask.any():
            cal_set = lgb.Dataset(
                X_cal,
                label=y_cal,
                weight=cal_weights,
                reference=train_set,
                free_raw_data=False,
            )
            valid_sets = [cal_set]
            callbacks.append(lgb.early_stopping(stopping_rounds=200, verbose=False))

        model = lgb.train(
            params=BASE_PARAMS,
            train_set=train_set,
            valid_sets=valid_sets,
            num_boost_round=int(num_boost_round),
            callbacks=callbacks,
        )
        best_iteration = model.best_iteration
    elif model_type == "xgb":
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(num_boost_round),
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
            n_jobs=8,
        )
        fit_kwargs: dict[str, Any] = {"sample_weight": train_weights}
        if cal_mask.any():
            fit_kwargs["eval_set"] = [(X_cal, y_cal)]
            fit_kwargs["sample_weight_eval_set"] = [cal_weights if cal_weights is not None else np.ones(len(y_cal), dtype=float)]
            fit_kwargs["verbose"] = False
        try:
            model.fit(
                X_train,
                y_train,
                early_stopping_rounds=200 if cal_mask.any() else None,
                **fit_kwargs,
            )
        except TypeError:
            model.fit(X_train, y_train, **fit_kwargs)
        best_iteration = getattr(model, "best_iteration", None)
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    preds = _predict_model(model, X_cal, model_type=model_type, best_iteration=best_iteration) if cal_mask.any() else np.array([])
    truth = cal_df.loc[cal_mask, target].to_numpy(dtype=float) if cal_mask.any() else np.array([])
    mae = float(np.mean(np.abs(preds - truth))) if len(truth) else float("nan")
    rmse = float(np.sqrt(np.mean((preds - truth) ** 2))) if len(truth) else float("nan")
    return TargetModelResult(
        target=target,
        model=model,
        best_iteration=int(best_iteration) if best_iteration is not None else None,
        val_mae=mae,
        val_rmse=rmse,
        train_rows=int(fit_mask.sum()),
        cal_rows=int(cal_mask.sum()),
    )


def _metric_block(df: pd.DataFrame, *, pred_col: str, actual_col: str) -> dict[str, float]:
    work = df[[pred_col, actual_col]].dropna()
    if work.empty:
        return {"n": 0, "mae": float("nan"), "bias": float("nan")}
    err = work[pred_col] - work[actual_col]
    return {
        "n": int(len(work)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def _slice_summary(df: pd.DataFrame, *, pred_col: str, actual_col: str, line_col: str, thresholds: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for thresh in thresholds:
        subset = df.loc[pd.to_numeric(df[line_col], errors="coerce") >= float(thresh)].copy()
        if subset.empty:
            rows.append({"line_ge": thresh, "n": 0})
            continue
        err = subset[pred_col] - subset[actual_col]
        rows.append(
            {
                "line_ge": thresh,
                "n": int(len(subset)),
                "pred_mean": float(subset[pred_col].mean()),
                "actual_mean": float(subset[actual_col].mean()),
                "line_mean": float(subset[line_col].mean()),
                "bias": float(err.mean()),
                "mae": float(np.abs(err).mean()),
            }
        )
    return rows


def _add_boxscore_tree_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    minutes_col = "pred_minutes" if "pred_minutes" in out.columns else "minutes_mean"
    minutes = pd.to_numeric(out[minutes_col], errors="coerce").fillna(0.0)

    def _num(col: str) -> pd.Series:
        return pd.to_numeric(out.get(col, 0.0), errors="coerce").fillna(0.0)

    if "pred_fga2_per_min" in out.columns:
        out["pred_fga2_tree"] = minutes * _num("pred_fga2_per_min")
        out["pred_fga3_tree"] = minutes * _num("pred_fga3_per_min")
        out["pred_fta_tree"] = minutes * _num("pred_fta_per_min")
        out["pred_fg2m_tree"] = out["pred_fga2_tree"] * _num("pred_fg2_pct_label").clip(lower=0.0, upper=1.0)
        out["pred_fg3m_tree"] = out["pred_fga3_tree"] * _num("pred_fg3_pct_label").clip(lower=0.0, upper=1.0)
        out["pred_ftm_tree"] = out["pred_fta_tree"] * _num("pred_ft_pct_label").clip(lower=0.0, upper=1.0)
        out["pred_pts_tree"] = 2.0 * out["pred_fg2m_tree"] + 3.0 * out["pred_fg3m_tree"] + out["pred_ftm_tree"]
        out["pred_stl_tree"] = minutes * _num("pred_stl_per_min")
        out["pred_blk_tree"] = minutes * _num("pred_blk_per_min")
        out["pred_tov_tree"] = minutes * _num("pred_tov_per_min")
    if "pred_ast_per_min" in out.columns:
        out["pred_ast_tree"] = minutes * _num("pred_ast_per_min")
    if "pred_oreb_per_min" in out.columns and "pred_dreb_per_min" in out.columns:
        out["pred_reb_tree"] = minutes * (_num("pred_oreb_per_min") + _num("pred_dreb_per_min"))

    if {"pred_pts_tree", "pred_reb_tree", "pred_ast_tree", "pred_stl_tree", "pred_blk_tree", "pred_tov_tree", "pred_fg3m_tree"}.issubset(out.columns):
        out["pred_dk_fpts_tree_naive"] = (
            out["pred_pts_tree"]
            + 1.25 * out["pred_reb_tree"]
            + 1.5 * out["pred_ast_tree"]
            + 2.0 * out["pred_stl_tree"]
            + 2.0 * out["pred_blk_tree"]
            - 0.5 * out["pred_tov_tree"]
            + 0.5 * out["pred_fg3m_tree"]
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--baseline-player-summary-csv", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model-type", choices=["lgbm", "xgb"], default="lgbm")
    parser.add_argument("--target-set", choices=["astreb", "full"], default="astreb")
    parser.add_argument("--cal-days", type=int, default=30)
    parser.add_argument("--num-boost-round", type=int, default=5000)
    parser.add_argument("--ast-line-threshold", type=float, default=7.0)
    parser.add_argument("--ast-weight-mult", type=float, default=1.0)
    parser.add_argument("--reb-line-threshold", type=float, default=10.0)
    parser.add_argument("--reb-weight-mult", type=float, default=1.0)
    parser.add_argument("--live-features-path", default=None)
    parser.add_argument("--live-projections-path", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    baseline_path = Path(args.baseline_player_summary_csv).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else Path("/home/daniel/projections-data/training/runs") / f"lgbm_rate_hybrid_eval_{_utc_now_compact()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    features = _coerce_keys(pd.read_parquet(dataset_dir / "features.parquet"))
    labels = _coerce_keys(pd.read_parquet(dataset_dir / "labels_rates.parquet"))
    counts = _coerce_keys(pd.read_parquet(dataset_dir / "labels_boxscore_counts.parquet"))
    base = features.merge(labels, on=KEY_COLS, how="inner")

    baseline = _coerce_keys(pd.read_csv(baseline_path))
    eval_keys = baseline[KEY_COLS].drop_duplicates().copy()
    eval_join = eval_keys.assign(_is_eval=1)
    base = base.merge(eval_join, on=KEY_COLS, how="left")
    eval_df = base.loc[base["_is_eval"] == 1].copy()
    train_pool = base.loc[base["_is_eval"] != 1].copy()

    live_allowed_cols: set[str] | None = None
    live_features: pd.DataFrame | None = None
    live_proj: pd.DataFrame | None = None
    if args.live_features_path and args.live_projections_path:
        live_features = _coerce_keys(
            pd.read_parquet(Path(args.live_features_path).expanduser().resolve()),
            require_game_date=False,
        )
        live_proj = _coerce_keys(pd.read_parquet(Path(args.live_projections_path).expanduser().resolve()))
        if "game_date" not in live_features.columns and "game_date" in live_proj.columns:
            live_features = live_features.merge(
                live_proj[KEY_COLS].drop_duplicates(["game_id", "team_id", "player_id"]),
                on=["game_id", "team_id", "player_id"],
                how="left",
            )
        live_allowed_cols = set(live_features.columns)

    target_list = FULL_TARGETS if str(args.target_set) == "full" else AST_REB_TARGETS

    feature_cols = _numeric_feature_cols(train_pool, allowed_live_cols=live_allowed_cols)
    train_pool = _clean_features(train_pool, feature_cols=feature_cols)
    eval_df = _clean_features(eval_df, feature_cols=feature_cols)

    fit_df, cal_df = _split_train_cal(train_pool, cal_days=int(args.cal_days))
    if fit_df.empty:
        raise ValueError("Training split is empty after calibration split.")

    model_results: dict[str, TargetModelResult] = {}
    eval_preds = eval_df.loc[:, KEY_COLS].copy()
    for target in target_list:
        res = _train_target(
            fit_df,
            cal_df,
            model_type=str(args.model_type),
            target=target,
            feature_cols=feature_cols,
            num_boost_round=int(args.num_boost_round),
            ast_line_threshold=float(args.ast_line_threshold),
            ast_weight_mult=float(args.ast_weight_mult),
            reb_line_threshold=float(args.reb_line_threshold),
            reb_weight_mult=float(args.reb_weight_mult),
        )
        model_results[target] = res
        eval_preds[f"pred_{target}"] = _predict_model(
            res.model,
            eval_df[feature_cols],
            model_type=str(args.model_type),
            best_iteration=res.best_iteration,
        )

    merged = baseline.merge(eval_preds, on=KEY_COLS, how="inner")
    merged = _add_boxscore_tree_predictions(merged)
    if "pred_ast_tree" in merged.columns:
        merged["pred_ast_lgbm"] = merged["pred_ast_tree"]
    if "pred_reb_tree" in merged.columns:
        merged["pred_reb_lgbm"] = merged["pred_reb_tree"]
    if "pred_dk_fpts_tree_naive" in merged.columns:
        merged["pred_dk_fpts_lgbm_naive"] = merged["pred_dk_fpts_tree_naive"]
    elif {"pred_ast_lgbm", "pred_reb_lgbm"}.issubset(merged.columns):
        merged["pred_dk_fpts_lgbm_naive"] = (
            merged["pred_dk_fpts"]
            + 1.5 * (merged["pred_ast_lgbm"] - merged["pred_ast"])
            + 1.25 * (merged["pred_reb_lgbm"] - merged["pred_reb"])
        )

    count_cols = ["fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov", "minutes"]
    counts_eval = counts[KEY_COLS + [c for c in count_cols if c in counts.columns]].drop_duplicates(KEY_COLS)
    merged = merged.merge(counts_eval, on=KEY_COLS, how="left")
    if {"fg2m", "fg3m", "ftm"}.issubset(merged.columns):
        merged["actual_pts_counts"] = 2.0 * pd.to_numeric(merged["fg2m"], errors="coerce").fillna(0.0) + 3.0 * pd.to_numeric(merged["fg3m"], errors="coerce").fillna(0.0) + pd.to_numeric(merged["ftm"], errors="coerce").fillna(0.0)
    if {"oreb", "dreb"}.issubset(merged.columns):
        merged["actual_reb_counts"] = pd.to_numeric(merged["oreb"], errors="coerce").fillna(0.0) + pd.to_numeric(merged["dreb"], errors="coerce").fillna(0.0)

    prop_cols = ["an_has_ast", "an_ast_line", "an_has_reb", "an_reb_line", "player_name"]
    prop_cols = [c for c in prop_cols if c in eval_df.columns]
    merged = merged.merge(eval_df[KEY_COLS + prop_cols].drop_duplicates(KEY_COLS), on=KEY_COLS, how="left")

    summary = {
        "dataset_dir": str(dataset_dir),
        "baseline_player_summary_csv": str(baseline_path),
        "model_type": str(args.model_type),
        "target_set": str(args.target_set),
        "ast_line_threshold": float(args.ast_line_threshold),
        "ast_weight_mult": float(args.ast_weight_mult),
        "reb_line_threshold": float(args.reb_line_threshold),
        "reb_weight_mult": float(args.reb_weight_mult),
        "n_eval_rows": int(len(merged)),
        "feature_count": int(len(feature_cols)),
        "targets": {
            target: {
                "best_iteration": int(res.best_iteration) if res.best_iteration is not None else None,
                "val_mae": res.val_mae,
                "val_rmse": res.val_rmse,
                "train_rows": res.train_rows,
                "cal_rows": res.cal_rows,
            }
            for target, res in model_results.items()
        },
        "eval_metrics": {
            "ast_transformer": _metric_block(merged, pred_col="pred_ast", actual_col="actual_ast"),
            "ast_lgbm_hybrid": _metric_block(merged, pred_col="pred_ast_lgbm", actual_col="actual_ast"),
            "reb_transformer": _metric_block(merged, pred_col="pred_reb", actual_col="actual_reb"),
            "reb_lgbm_hybrid": _metric_block(merged, pred_col="pred_reb_lgbm", actual_col="actual_reb"),
            "dk_transformer": _metric_block(merged, pred_col="pred_dk_fpts", actual_col="actual_dk_fpts"),
            "dk_lgbm_hybrid_naive": _metric_block(merged, pred_col="pred_dk_fpts_lgbm_naive", actual_col="actual_dk_fpts"),
        },
        "ast_top_line_slices": _slice_summary(
            merged.loc[pd.to_numeric(merged.get("an_has_ast"), errors="coerce").fillna(0) > 0].copy(),
            pred_col="pred_ast_lgbm",
            actual_col="actual_ast",
            line_col="an_ast_line",
            thresholds=[5, 7, 9, 10],
        ),
        "ast_top_line_slices_transformer": _slice_summary(
            merged.loc[pd.to_numeric(merged.get("an_has_ast"), errors="coerce").fillna(0) > 0].copy(),
            pred_col="pred_ast",
            actual_col="actual_ast",
            line_col="an_ast_line",
            thresholds=[5, 7, 9, 10],
        ),
        "reb_top_line_slices": _slice_summary(
            merged.loc[pd.to_numeric(merged.get("an_has_reb"), errors="coerce").fillna(0) > 0].copy(),
            pred_col="pred_reb_lgbm",
            actual_col="actual_reb",
            line_col="an_reb_line",
            thresholds=[8, 10, 12],
        ),
        "reb_top_line_slices_transformer": _slice_summary(
            merged.loc[pd.to_numeric(merged.get("an_has_reb"), errors="coerce").fillna(0) > 0].copy(),
            pred_col="pred_reb",
            actual_col="actual_reb",
            line_col="an_reb_line",
            thresholds=[8, 10, 12],
        ),
    }

    if "pred_pts_tree" in merged.columns:
        summary["eval_metrics"]["pts_transformer"] = _metric_block(merged, pred_col="pred_pts", actual_col="actual_pts")
        summary["eval_metrics"]["pts_tree_hybrid"] = _metric_block(merged, pred_col="pred_pts_tree", actual_col="actual_pts")
    if "pred_stl_tree" in merged.columns:
        summary["eval_metrics"]["stl_transformer"] = _metric_block(merged, pred_col="pred_stl", actual_col="actual_stl")
        summary["eval_metrics"]["stl_tree_hybrid"] = _metric_block(merged, pred_col="pred_stl_tree", actual_col="actual_stl")
    if "pred_blk_tree" in merged.columns:
        summary["eval_metrics"]["blk_transformer"] = _metric_block(merged, pred_col="pred_blk", actual_col="actual_blk")
        summary["eval_metrics"]["blk_tree_hybrid"] = _metric_block(merged, pred_col="pred_blk_tree", actual_col="actual_blk")
    if "pred_tov_tree" in merged.columns:
        summary["eval_metrics"]["tov_transformer"] = _metric_block(merged, pred_col="pred_tov", actual_col="actual_tov")
        summary["eval_metrics"]["tov_tree_hybrid"] = _metric_block(merged, pred_col="pred_tov_tree", actual_col="actual_tov")
    if "pred_fg3m_tree" in merged.columns and "fg3m" in merged.columns:
        summary["eval_metrics"]["fg3m_tree_hybrid"] = _metric_block(merged, pred_col="pred_fg3m_tree", actual_col="fg3m")
    if "pred_ftm_tree" in merged.columns and "ftm" in merged.columns:
        summary["eval_metrics"]["ftm_tree_hybrid"] = _metric_block(merged, pred_col="pred_ftm_tree", actual_col="ftm")

    merged.to_csv(out_dir / "eval_player_rows.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if live_features is not None and live_proj is not None:
        missing_live = [c for c in feature_cols if c not in live_features.columns]
        if missing_live:
            raise KeyError(f"Live features missing required columns: {missing_live[:20]}")
        live_work = _clean_features(live_features, feature_cols=feature_cols)
        live_preds = live_work.loc[:, KEY_COLS].copy()
        for target, res in model_results.items():
            live_preds[f"pred_{target}"] = _predict_model(
                res.model,
                live_work[feature_cols],
                model_type=str(args.model_type),
                best_iteration=res.best_iteration,
            )
        live_merged = live_proj.merge(live_preds, on=KEY_COLS, how="inner")
        live_merged = _add_boxscore_tree_predictions(live_merged)
        if "pred_ast_tree" in live_merged.columns:
            live_merged["pred_ast_lgbm"] = live_merged["pred_ast_tree"]
        if "pred_reb_tree" in live_merged.columns:
            live_merged["pred_reb_lgbm"] = live_merged["pred_reb_tree"]

        live_join_cols = [c for c in ["an_ast_line", "an_reb_line", "an_has_ast", "an_has_reb"] if c in live_work.columns]
        if live_join_cols:
            live_merged = live_merged.merge(
                live_work[KEY_COLS + live_join_cols].drop_duplicates(KEY_COLS),
                on=KEY_COLS,
                how="left",
            )
        top_ast = (
            live_merged.loc[pd.to_numeric(live_merged.get("an_has_ast"), errors="coerce").fillna(0) > 0]
            .sort_values("an_ast_line", ascending=False)
            .loc[:, [c for c in ["player_name", "minutes_mean", "ast_mean", "pred_ast_lgbm", "an_ast_line"] if c in live_merged.columns]]
            .head(20)
        )
        top_reb = (
            live_merged.loc[pd.to_numeric(live_merged.get("an_has_reb"), errors="coerce").fillna(0) > 0]
            .sort_values("an_reb_line", ascending=False)
            .loc[:, [c for c in ["player_name", "minutes_mean", "reb_mean", "pred_reb_lgbm", "an_reb_line"] if c in live_merged.columns]]
            .head(20)
        )
        live_merged.to_csv(out_dir / "live_player_rows.csv", index=False)
        top_ast.to_csv(out_dir / "live_top_ast.csv", index=False)
        top_reb.to_csv(out_dir / "live_top_reb.csv", index=False)


if __name__ == "__main__":
    main()
