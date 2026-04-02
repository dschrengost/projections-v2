"""
LightGBM scoring rate experiment.

Train opportunity + efficiency models for 2P/3P/FT shooting, then evaluate
how the resulting points predictions compare to GTv2 flow head output.

Models:
  - Opportunity rates: fga2_per_min, fga3_per_min, fta_per_min
  - Efficiency rates: fg2_pct, fg3_pct, ft_pct
  - Derived: fg2m = min * fga2_per_min * fg2_pct, etc.
  - Points = 2*fg2m + 3*fg3m + ftm

Also trains the existing ast/reb rates for a complete comparison.

Usage:
    uv run python scripts/experiments/lgbm_scoring_rates_v1.py \
        --dataset-dir /home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_trackingctx_prodparity_lineupavailfix_priorplayprobstarterneutral_20260326T201310Z
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]

# Targets
OPP_RATE_TARGETS = ["fga2_per_min", "fga3_per_min", "fta_per_min"]
EFF_TARGETS = ["fg2_pct_label", "fg3_pct_label", "ft_pct_label"]
COUNTING_RATE_TARGETS = ["ast_per_min", "oreb_per_min", "dreb_per_min",
                          "stl_per_min", "blk_per_min", "tov_per_min"]
ALL_TARGETS = OPP_RATE_TARGETS + EFF_TARGETS + COUNTING_RATE_TARGETS

# Feature exclusions (same as tree_rate_bundle.py)
EXCLUDE_COLS = {
    *JOIN_KEYS, "minutes_actual",
    "rates_non_null_count", "efficiency_non_null_count",
    "rates_label_available_any", "rates_label_available_all_rate_targets",
    "rates_loss_eligible",
    *OPP_RATE_TARGETS, *EFF_TARGETS, *COUNTING_RATE_TARGETS,
    # Box score label columns joined for evaluation — must NOT be features
    "fg2m", "fg3m", "ftm", "fga2", "fga3", "fta",
    "minutes", "played",
}

# Also exclude non-numeric / identifier columns from features
ID_COLS = {
    "player_name", "team_name", "team_tricode",
    "opponent_team_name", "opponent_team_tricode",
    "status", "restriction_flag", "ramp_flag",
    "archetype", "pos_bucket", "lineup_role", "lineup_status",
    "lineup_roster_status", "lineup_timestamp",
    "injury_as_of_ts", "roster_as_of_ts", "feature_as_of_ts",
    "snapshot_ts", "frozen_at", "snapshot_type", "tip_ts", "odds_as_of_ts",
    "source",
}


def _select_features(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS or col in ID_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    return sorted(cols)


def _time_split(df: pd.DataFrame, *, val_frac: float = 0.15, test_frac: float = 0.10):
    dates = sorted(df["game_date"].unique())
    n = len(dates)
    test_start = dates[int(n * (1 - test_frac))]
    val_start = dates[int(n * (1 - test_frac - val_frac))]
    return (
        df[df["game_date"] < val_start].copy(),
        df[(df["game_date"] >= val_start) & (df["game_date"] < test_start)].copy(),
        df[df["game_date"] >= test_start].copy(),
    )


def _train_lgbm(
    X_train, y_train, X_val, y_val, *,
    objective="regression", metric="mae",
    num_leaves=64, lr=0.05, n_rounds=3000,
    min_child_samples=50, weight_col=None,
    sample_weights_train=None, sample_weights_val=None,
):
    params = {
        "objective": objective,
        "metric": metric,
        "num_leaves": num_leaves,
        "learning_rate": lr,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": min_child_samples,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    ds_train = lgb.Dataset(X_train, label=y_train, weight=sample_weights_train)
    ds_val = lgb.Dataset(X_val, label=y_val, weight=sample_weights_val, reference=ds_train)
    model = lgb.train(
        params, ds_train,
        num_boost_round=n_rounds,
        valid_sets=[ds_val], valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    )
    return model


def _upweight_high_usage(df: pd.DataFrame, target: str, threshold_col: str,
                          threshold: float, mult: float) -> np.ndarray | None:
    """Upweight high-usage players so model doesn't under-predict stars."""
    if mult <= 1.0 or threshold_col not in df.columns:
        return None
    weights = np.ones(len(df), dtype=float)
    line = pd.to_numeric(df[threshold_col], errors="coerce").fillna(0.0).values
    weights[line >= threshold] *= mult
    return weights if not np.allclose(weights, 1.0) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--n-rounds", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--upweight-stars", type=float, default=3.0,
                        help="Weight multiplier for high-usage players")
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir).expanduser().resolve()
    print(f"Loading from {ds_dir}")

    feat = pd.read_parquet(ds_dir / "features.parquet")
    labels_rates = pd.read_parquet(ds_dir / "labels_rates.parquet")
    labels_box = pd.read_parquet(ds_dir / "labels_boxscore_counts.parquet")

    # Join features + rate labels + box labels (for ground truth counting stats)
    df = feat.merge(labels_rates, on=JOIN_KEYS, how="inner")
    df = df.merge(
        labels_box[JOIN_KEYS + ["fg2m", "fg3m", "ftm", "fga2", "fga3", "fta", "minutes", "played"]],
        on=JOIN_KEYS, how="left", suffixes=("", "_box"),
    )

    # Filter to players with rate labels (played with >0 minutes)
    active_mask = df["minutes_actual"].notna() & (df["minutes_actual"] >= 4.0)
    df_active = df[active_mask].copy()
    print(f"Active rows (≥4 min): {len(df_active)}")
    print(f"Total rows: {len(df)}")

    feature_cols = _select_features(df_active)
    print(f"Features: {len(feature_cols)}")

    train, val, test = _time_split(df_active)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # =========================================================================
    # TRAIN MODELS
    # =========================================================================

    models: dict[str, lgb.Booster] = {}

    # Upweight config per target
    upweight_config = {
        "fga2_per_min": ("an_pts_line", 20.0),
        "fga3_per_min": ("an_3pm_line", 2.0),
        "fta_per_min": ("an_pts_line", 20.0),
        "fg2_pct_label": ("an_pts_line", 20.0),
        "fg3_pct_label": ("an_3pm_line", 2.0),
        "ft_pct_label": ("an_pts_line", 20.0),
        "ast_per_min": ("an_ast_line", 5.0),
        "oreb_per_min": ("an_reb_line", 7.0),
        "dreb_per_min": ("an_reb_line", 7.0),
        "stl_per_min": ("an_stl_line", 1.0),
        "blk_per_min": ("an_blk_line", 1.0),
        "tov_per_min": ("an_pts_line", 20.0),
    }

    for target in ALL_TARGETS:
        print(f"\n{'='*60}")
        print(f"Training: {target}")
        print(f"{'='*60}")

        train_mask = train[target].notna()
        val_mask = val[target].notna()
        if train_mask.sum() < 100:
            print(f"  Skipping {target}: only {train_mask.sum()} non-null rows")
            continue

        thr_col, thr_val = upweight_config.get(target, ("", 0.0))
        w_train = _upweight_high_usage(
            train[train_mask], target, thr_col, thr_val, args.upweight_stars
        )
        w_val = _upweight_high_usage(
            val[val_mask], target, thr_col, thr_val, args.upweight_stars
        )

        model = _train_lgbm(
            train.loc[train_mask, feature_cols], train.loc[train_mask, target],
            val.loc[val_mask, feature_cols], val.loc[val_mask, target],
            n_rounds=args.n_rounds, lr=args.lr,
            sample_weights_train=w_train, sample_weights_val=w_val,
        )
        models[target] = model
        print(f"  best_iteration={model.best_iteration}")

    # =========================================================================
    # EVALUATE — Scoring Rates
    # =========================================================================
    print(f"\n{'='*70}")
    print("EVALUATION — Per-Minute Rate Accuracy")
    print(f"{'='*70}")

    test_pred = test[JOIN_KEYS + ["minutes_actual"]].copy()
    for target in ALL_TARGETS:
        if target not in models:
            continue
        mask = test[target].notna()
        test_pred[f"pred_{target}"] = np.nan
        if mask.any():
            test_pred.loc[mask, f"pred_{target}"] = models[target].predict(
                test.loc[mask, feature_cols],
                num_iteration=models[target].best_iteration,
            )
        actual = test.loc[mask, target].values
        pred = test_pred.loc[mask, f"pred_{target}"].values
        mae = float(np.mean(np.abs(actual - pred)))
        bias = float(np.mean(pred - actual))
        corr = float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 10 else 0.0
        print(f"  {target:20s}: MAE={mae:.4f}  bias={bias:+.4f}  r={corr:.3f}  n={mask.sum()}")

    # =========================================================================
    # EVALUATE — Counting Stats & Points
    # =========================================================================
    print(f"\n{'='*70}")
    print("EVALUATION — Counting Stats (predicted rate × actual minutes)")
    print(f"{'='*70}")

    # Use actual minutes to isolate rate accuracy from minutes error
    minutes = test_pred["minutes_actual"].values

    # LightGBM predicted counting stats
    pred_fga2 = test_pred["pred_fga2_per_min"].fillna(0).values * minutes
    pred_fga3 = test_pred["pred_fga3_per_min"].fillna(0).values * minutes
    pred_fta = test_pred["pred_fta_per_min"].fillna(0).values * minutes
    pred_fg2_pct = test_pred["pred_fg2_pct_label"].fillna(0.5).values
    pred_fg3_pct = test_pred["pred_fg3_pct_label"].fillna(0.35).values
    pred_ft_pct = test_pred["pred_ft_pct_label"].fillna(0.77).values

    pred_fg2m = pred_fga2 * pred_fg2_pct
    pred_fg3m = pred_fga3 * pred_fg3_pct
    pred_ftm = pred_fta * pred_ft_pct
    pred_pts = 2 * pred_fg2m + 3 * pred_fg3m + pred_ftm

    # Actual counting stats
    actual_fg2m = test["fg2m"].fillna(0).values
    actual_fg3m = test["fg3m"].fillna(0).values
    actual_ftm = test["ftm"].fillna(0).values
    actual_fga2 = test["fga2"].fillna(0).values
    actual_fga3 = test["fga3"].fillna(0).values
    actual_fta = test["fta"].fillna(0).values
    actual_pts = 2 * actual_fg2m + 3 * actual_fg3m + actual_ftm

    # GTv2 flow implied stats (from rate × minutes, using actual box score)
    # We can't directly compare to GTv2 here, but we can check our accuracy

    for stat, pred, actual in [
        ("fg2m", pred_fg2m, actual_fg2m),
        ("fg3m", pred_fg3m, actual_fg3m),
        ("ftm", pred_ftm, actual_ftm),
        ("fga2", pred_fga2, actual_fga2),
        ("fga3", pred_fga3, actual_fga3),
        ("fta", pred_fta, actual_fta),
        ("pts", pred_pts, actual_pts),
    ]:
        mae = float(np.mean(np.abs(actual - pred)))
        bias = float(np.mean(pred - actual))
        corr = float(np.corrcoef(actual, pred)[0, 1]) if len(actual) > 10 else 0.0
        print(f"  {stat:6s}: MAE={mae:.2f}  bias={bias:+.2f}  r={corr:.3f}")

    # Also compute ast/reb/stl/blk for completeness
    pred_ast = test_pred["pred_ast_per_min"].fillna(0).values * minutes
    pred_oreb = test_pred["pred_oreb_per_min"].fillna(0).values * minutes
    pred_dreb = test_pred["pred_dreb_per_min"].fillna(0).values * minutes
    actual_ast = test["ast"].fillna(0).values if "ast" in test.columns else np.zeros(len(test))
    actual_oreb = test["oreb"].fillna(0).values if "oreb" in test.columns else np.zeros(len(test))
    actual_dreb = test["dreb"].fillna(0).values if "dreb" in test.columns else np.zeros(len(test))

    # Join box labels for ast/reb
    test_box = test.merge(
        pd.read_parquet(ds_dir / "labels_boxscore_counts.parquet")[JOIN_KEYS + ["ast", "oreb", "dreb", "stl", "blk", "tov"]],
        on=JOIN_KEYS, how="left", suffixes=("", "_label"),
    )
    for stat, rate_col in [("ast", "ast_per_min"), ("oreb", "oreb_per_min"), ("dreb", "dreb_per_min")]:
        label_col = f"{stat}_label" if f"{stat}_label" in test_box.columns else stat
        if label_col in test_box.columns:
            actual_vals = test_box[label_col].fillna(0).values
            pred_vals = test_pred[f"pred_{rate_col}"].fillna(0).values * minutes
            mae = float(np.mean(np.abs(actual_vals - pred_vals)))
            bias = float(np.mean(pred_vals - actual_vals))
            corr = float(np.corrcoef(actual_vals, pred_vals)[0, 1]) if len(actual_vals) > 10 else 0.0
            print(f"  {stat:6s}: MAE={mae:.2f}  bias={bias:+.2f}  r={corr:.3f}")

    # =========================================================================
    # Conditional analysis — by minutes bucket
    # =========================================================================
    print(f"\n{'='*70}")
    print("POINTS ERROR BY MINUTES BUCKET (rate × actual minutes)")
    print(f"{'='*70}")

    test_eval = pd.DataFrame({
        "minutes": minutes,
        "actual_pts": actual_pts,
        "pred_pts": pred_pts,
    })
    bins = [4, 10, 15, 20, 25, 30, 35, 48]
    test_eval["bucket"] = pd.cut(test_eval["minutes"], bins=bins, right=False)
    for bucket, g in test_eval.groupby("bucket", observed=False):
        if len(g) < 20:
            continue
        mae = float(np.mean(np.abs(g["actual_pts"] - g["pred_pts"])))
        bias = float(np.mean(g["pred_pts"] - g["actual_pts"]))
        print(f"  {str(bucket):15s}  n={len(g):4d}  pts_MAE={mae:.2f}  pts_bias={bias:+.2f}")

    # =========================================================================
    # DK FPTS using all LightGBM rates
    # =========================================================================
    print(f"\n{'='*70}")
    print("FULL DK FPTS — All LightGBM rates × actual minutes")
    print(f"{'='*70}")

    pred_stl = test_pred["pred_stl_per_min"].fillna(0).values * minutes
    pred_blk = test_pred["pred_blk_per_min"].fillna(0).values * minutes
    pred_tov = test_pred["pred_tov_per_min"].fillna(0).values * minutes
    pred_reb = pred_oreb + pred_dreb

    pred_dk = (pred_pts + 1.25 * pred_reb + 1.5 * pred_ast
               + 2.0 * pred_stl + 2.0 * pred_blk - 0.5 * pred_tov
               + 0.5 * pred_fg3m)

    # Actual DK (from box labels)
    if all(c in test_box.columns for c in ["ast_label", "oreb_label", "dreb_label"]):
        a_ast = test_box["ast_label"].fillna(0).values
        a_oreb = test_box["oreb_label"].fillna(0).values
        a_dreb = test_box["dreb_label"].fillna(0).values
    else:
        # Try without suffix
        a_ast = test_box["ast"].fillna(0).values
        a_oreb = test_box["oreb"].fillna(0).values
        a_dreb = test_box["dreb"].fillna(0).values

    a_stl = test_box.get("stl_label", test_box.get("stl", pd.Series(0))).fillna(0).values
    a_blk = test_box.get("blk_label", test_box.get("blk", pd.Series(0))).fillna(0).values
    a_tov = test_box.get("tov_label", test_box.get("tov", pd.Series(0))).fillna(0).values
    a_reb = a_oreb + a_dreb

    actual_dk = (actual_pts + 1.25 * a_reb + 1.5 * a_ast
                 + 2.0 * a_stl + 2.0 * a_blk - 0.5 * a_tov
                 + 0.5 * actual_fg3m)

    dk_mae = float(np.mean(np.abs(actual_dk - pred_dk)))
    dk_bias = float(np.mean(pred_dk - actual_dk))
    dk_corr = float(np.corrcoef(actual_dk, pred_dk)[0, 1])
    print(f"  DK FPTS MAE:  {dk_mae:.2f}")
    print(f"  DK FPTS bias: {dk_bias:+.2f}")
    print(f"  DK FPTS corr: {dk_corr:.3f}")

    # By minutes bucket
    test_dk = pd.DataFrame({"minutes": minutes, "actual_dk": actual_dk, "pred_dk": pred_dk})
    test_dk["bucket"] = pd.cut(test_dk["minutes"], bins=bins, right=False)
    for bucket, g in test_dk.groupby("bucket", observed=False):
        if len(g) < 20:
            continue
        mae = float(np.mean(np.abs(g["actual_dk"] - g["pred_dk"])))
        bias = float(np.mean(g["pred_dk"] - g["actual_dk"]))
        print(f"  {str(bucket):15s}  n={len(g):4d}  dk_MAE={mae:.2f}  dk_bias={bias:+.2f}")

    # =========================================================================
    # Feature importance — scoring models
    # =========================================================================
    print(f"\n{'='*70}")
    print("TOP 15 FEATURES — Opportunity Rates")
    print(f"{'='*70}")

    for target in OPP_RATE_TARGETS:
        if target not in models:
            continue
        imp = models[target].feature_importance(importance_type="gain")
        top = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:15]
        print(f"\n  {target}:")
        for name, v in top:
            print(f"    {name:50s} {v:.0f}")

    print(f"\n{'='*70}")
    print("TOP 15 FEATURES — Efficiency Rates")
    print(f"{'='*70}")

    for target in EFF_TARGETS:
        if target not in models:
            continue
        imp = models[target].feature_importance(importance_type="gain")
        top = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:10]
        print(f"\n  {target}:")
        for name, v in top:
            print(f"    {name:50s} {v:.0f}")

    # =========================================================================
    # Save
    # =========================================================================
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = ds_dir.parent / f"lgbm_scoring_rates_v1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for target, model in models.items():
        model.save_model(str(models_dir / f"{target}.lgb"), num_iteration=model.best_iteration)

    (out_dir / "feature_columns.json").write_text(
        json.dumps({"columns": feature_cols}, indent=2), encoding="utf-8"
    )

    test_pred.to_parquet(out_dir / "test_predictions.parquet", index=False)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(ds_dir),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_features": len(feature_cols),
        "best_iterations": {t: m.best_iteration for t, m in models.items()},
        "dk_fpts_mae": dk_mae,
        "dk_fpts_bias": dk_bias,
        "dk_fpts_corr": dk_corr,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
