"""
LightGBM scoring rate experiment v2 — prior-efficiency edition.

Key insight from v1: single-game shooting percentages are mostly noise.
LightGBM efficiency models (fg2_pct, fg3_pct, ft_pct) achieved r<0.23
without leakage. Instead, use rolling season-average efficiency priors
computed from box score history (strictly anti-leak: excludes current game).

Strategy:
  - LightGBM for OPPORTUNITY rates: fga2_per_min, fga3_per_min, fta_per_min
  - Rolling prior for EFFICIENCY: cumulative FGM/FGA up to (not including) game
  - Derived: fg2m = min × fga2_per_min × prior_fg2_pct, etc.
  - Points = 2×fg2m + 3×fg3m + ftm

Also trains counting rates (ast/reb/stl/blk/tov) for a complete DK FPTS eval.

Usage:
    uv run python scripts/experiments/lgbm_scoring_rates_v2.py \
        --dataset-dir /home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_trackingctx_prodparity_lineupavailfix_priorplayprobstarterneutral_20260326T201310Z
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]

# LightGBM targets (opportunity + counting rates only — no efficiency)
OPP_RATE_TARGETS = ["fga2_per_min", "fga3_per_min", "fta_per_min"]
COUNTING_RATE_TARGETS = ["ast_per_min", "oreb_per_min", "dreb_per_min",
                          "stl_per_min", "blk_per_min", "tov_per_min"]
LGBM_TARGETS = OPP_RATE_TARGETS + COUNTING_RATE_TARGETS

# Efficiency labels (used only for evaluation, not trained)
EFF_TARGETS = ["fg2_pct_label", "fg3_pct_label", "ft_pct_label"]

# Feature exclusions
EXCLUDE_COLS = {
    *JOIN_KEYS, "minutes_actual",
    "rates_non_null_count", "efficiency_non_null_count",
    "rates_label_available_any", "rates_label_available_all_rate_targets",
    "rates_loss_eligible",
    *OPP_RATE_TARGETS, *EFF_TARGETS, *COUNTING_RATE_TARGETS,
    # Box score label columns joined for evaluation — must NOT be features
    "fg2m", "fg3m", "ftm", "fga2", "fga3", "fta",
    "minutes", "played",
    # Prior efficiency columns we compute (not features for opportunity models)
    "prior_fg2_pct", "prior_fg3_pct", "prior_ft_pct",
    "prior_fg2_n", "prior_fg3_n", "prior_ft_n",
}

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

# League-average fallbacks (2024-25 NBA season approx)
LEAGUE_FG2_PCT = 0.545
LEAGUE_FG3_PCT = 0.365
LEAGUE_FT_PCT = 0.780


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


def _compute_rolling_efficiency(box: pd.DataFrame, min_attempts: int = 10) -> pd.DataFrame:
    """Compute cumulative season-average shooting percentages per player.

    For each game, the prior is computed from ALL previous games in the dataset
    (strictly before the current game_date). This is anti-leak by construction.

    For players with fewer than `min_attempts`, we shrink toward league average
    using a Bayesian-style blend: prior = (player_makes + k*league) / (player_att + k)
    where k = min_attempts.
    """
    box = box.sort_values(["player_id", "game_date", "game_id"]).copy()

    # Cumulative sums per player (shifted by 1 to exclude current game)
    priors = []
    for pid, grp in box.groupby("player_id"):
        g = grp.sort_values("game_date").copy()

        # Cumulative attempts and makes (shifted = prior to this game)
        for stat, made, att in [("fg2", "fg2m", "fga2"), ("fg3", "fg3m", "fga3"), ("ft", "ftm", "fta")]:
            league_avg = {"fg2": LEAGUE_FG2_PCT, "fg3": LEAGUE_FG3_PCT, "ft": LEAGUE_FT_PCT}[stat]
            cum_made = g[made].fillna(0).cumsum().shift(1, fill_value=0)
            cum_att = g[att].fillna(0).cumsum().shift(1, fill_value=0)

            # Bayesian shrinkage toward league average
            k = min_attempts
            g[f"prior_{stat}_pct"] = (cum_made + k * league_avg) / (cum_att + k)
            g[f"prior_{stat}_n"] = cum_att

        priors.append(g[JOIN_KEYS + [
            "prior_fg2_pct", "prior_fg3_pct", "prior_ft_pct",
            "prior_fg2_n", "prior_fg3_n", "prior_ft_n",
        ]])

    return pd.concat(priors, ignore_index=True)


def _train_lgbm(
    X_train, y_train, X_val, y_val, *,
    objective="regression", metric="mae",
    num_leaves=64, lr=0.05, n_rounds=3000,
    min_child_samples=50,
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


def _upweight_high_usage(df: pd.DataFrame, threshold_col: str,
                          threshold: float, mult: float) -> np.ndarray | None:
    if mult <= 1.0 or threshold_col not in df.columns:
        return None
    weights = np.ones(len(df), dtype=float)
    line = pd.to_numeric(df[threshold_col], errors="coerce").fillna(0.0).values
    weights[line >= threshold] *= mult
    return weights if not np.allclose(weights, 1.0) else None


def _safe_corr(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) <= 1 or len(pred) <= 1:
        return 0.0
    if np.allclose(actual, actual[0]) or np.allclose(pred, pred[0]):
        return 0.0
    corr = np.corrcoef(actual, pred)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _metric_row(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if len(actual) == 0:
        return {"n": 0, "mae": float("nan"), "bias": float("nan"), "corr": float("nan")}
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(actual - pred))),
        "bias": float(np.mean(pred - actual)),
        "corr": _safe_corr(actual, pred),
    }


def _clip_non_negative(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, 0.0, None)


def _clip_pct(values: pd.Series | np.ndarray, fallback: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, fallback)
    return np.clip(arr, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--n-rounds", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--upweight-stars", type=float, default=3.0)
    parser.add_argument("--eff-min-attempts", type=int, default=10,
                        help="Min FGA/FTA before prior stops shrinking to league avg")
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir).expanduser().resolve()
    print(f"Loading from {ds_dir}")

    feat = pd.read_parquet(ds_dir / "features.parquet")
    labels_rates = pd.read_parquet(ds_dir / "labels_rates.parquet")
    labels_box = pd.read_parquet(ds_dir / "labels_boxscore_counts.parquet")

    # =========================================================================
    # COMPUTE ROLLING EFFICIENCY PRIORS
    # =========================================================================
    print("Computing rolling efficiency priors from box score history...")
    eff_priors = _compute_rolling_efficiency(labels_box, min_attempts=args.eff_min_attempts)
    print(f"  Prior rows: {len(eff_priors)}")

    # Sanity check priors
    active_priors = eff_priors.merge(
        labels_rates[labels_rates["minutes_actual"].notna() & (labels_rates["minutes_actual"] >= 4)][JOIN_KEYS],
        on=JOIN_KEYS, how="inner",
    )
    for stat in ["fg2", "fg3", "ft"]:
        vals = active_priors[f"prior_{stat}_pct"].dropna()
        n_vals = active_priors[f"prior_{stat}_n"].dropna()
        print(f"  prior_{stat}_pct: mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"median_n={n_vals.median():.0f}")

    # =========================================================================
    # JOIN DATA
    # =========================================================================
    df = feat.merge(labels_rates, on=JOIN_KEYS, how="inner")
    df = df.merge(
        labels_box[JOIN_KEYS + ["fg2m", "fg3m", "ftm", "fga2", "fga3", "fta", "minutes", "played"]],
        on=JOIN_KEYS, how="left", suffixes=("", "_box"),
    )
    df = df.merge(eff_priors, on=JOIN_KEYS, how="left")

    active_mask = df["minutes_actual"].notna() & (df["minutes_actual"] >= 4.0)
    df_active = df[active_mask].copy()
    print(f"\nActive rows (≥4 min): {len(df_active)}")
    print(f"Total rows: {len(df)}")

    feature_cols = _select_features(df_active)
    print(f"Features: {len(feature_cols)}")

    train, val, test = _time_split(df_active)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # =========================================================================
    # TRAIN LightGBM — opportunity + counting rates only
    # =========================================================================
    models: dict[str, lgb.Booster] = {}

    upweight_config = {
        "fga2_per_min": ("an_pts_line", 20.0),
        "fga3_per_min": ("an_3pm_line", 2.0),
        "fta_per_min": ("an_pts_line", 20.0),
        "ast_per_min": ("an_ast_line", 5.0),
        "oreb_per_min": ("an_reb_line", 7.0),
        "dreb_per_min": ("an_reb_line", 7.0),
        "stl_per_min": ("an_stl_line", 1.0),
        "blk_per_min": ("an_blk_line", 1.0),
        "tov_per_min": ("an_pts_line", 20.0),
    }

    for target in LGBM_TARGETS:
        print(f"\n{'='*60}")
        print(f"Training: {target}")
        print(f"{'='*60}")

        train_mask = train[target].notna()
        val_mask = val[target].notna()
        if train_mask.sum() < 100:
            print(f"  Skipping {target}: only {train_mask.sum()} non-null rows")
            continue

        thr_col, thr_val = upweight_config.get(target, ("", 0.0))
        w_train = _upweight_high_usage(train[train_mask], thr_col, thr_val, args.upweight_stars)
        w_val = _upweight_high_usage(val[val_mask], thr_col, thr_val, args.upweight_stars)

        model = _train_lgbm(
            train.loc[train_mask, feature_cols], train.loc[train_mask, target],
            val.loc[val_mask, feature_cols], val.loc[val_mask, target],
            n_rounds=args.n_rounds, lr=args.lr,
            sample_weights_train=w_train, sample_weights_val=w_val,
        )
        models[target] = model
        print(f"  best_iteration={model.best_iteration}")

    # =========================================================================
    # EVALUATE — Per-Minute Opportunity Rates
    # =========================================================================
    print(f"\n{'='*70}")
    print("EVALUATION — Per-Minute Rate Accuracy (LightGBM)")
    print(f"{'='*70}")

    test_pred = test[
        JOIN_KEYS
        + [
            "player_name",
            "team_tricode",
            "opponent_team_tricode",
            "minutes_actual",
            "prior_fg2_pct",
            "prior_fg3_pct",
            "prior_ft_pct",
            "prior_fg2_n",
            "prior_fg3_n",
            "prior_ft_n",
        ]
    ].copy()
    rate_metrics: dict[str, dict[str, float]] = {}

    for target in LGBM_TARGETS:
        if target not in models:
            continue
        mask = test[target].notna()
        test_pred[f"pred_{target}"] = np.nan
        if mask.any():
            preds = models[target].predict(
                test.loc[mask, feature_cols],
                num_iteration=models[target].best_iteration,
            )
            test_pred.loc[mask, f"pred_{target}"] = _clip_non_negative(preds)
        actual = test.loc[mask, target].values
        pred = test_pred.loc[mask, f"pred_{target}"].values
        metrics = _metric_row(actual, pred)
        rate_metrics[target] = metrics
        print(
            f"  {target:20s}: MAE={metrics['mae']:.4f}  "
            f"bias={metrics['bias']:+.4f}  r={metrics['corr']:.3f}  n={metrics['n']}"
        )

    # =========================================================================
    # EVALUATE — Rolling Efficiency Priors vs Actual
    # =========================================================================
    print(f"\n{'='*70}")
    print("EVALUATION — Rolling Efficiency Priors vs Actual Game %")
    print(f"{'='*70}")

    efficiency_prior_metrics: dict[str, dict[str, float]] = {}
    for stat, pct_label, prior_col in [
        ("FG2%", "fg2_pct_label", "prior_fg2_pct"),
        ("FG3%", "fg3_pct_label", "prior_fg3_pct"),
        ("FT%",  "ft_pct_label",  "prior_ft_pct"),
    ]:
        mask = test[pct_label].notna() & test[prior_col].notna()
        if mask.sum() < 10:
            continue
        actual = test.loc[mask, pct_label].values
        prior = _clip_pct(test.loc[mask, prior_col].values, 0.0)
        metrics = _metric_row(actual, prior)
        efficiency_prior_metrics[stat] = metrics
        print(
            f"  {stat:6s}: MAE={metrics['mae']:.4f}  "
            f"bias={metrics['bias']:+.4f}  r={metrics['corr']:.3f}  n={metrics['n']}"
        )

    # =========================================================================
    # EVALUATE — Counting Stats & Points (LightGBM opp × prior eff)
    # =========================================================================
    print(f"\n{'='*70}")
    print("EVALUATION — Counting Stats (LightGBM opp rate × prior eff × actual min)")
    print(f"{'='*70}")

    minutes = test_pred["minutes_actual"].values

    # LightGBM opportunity × prior efficiency
    pred_fga2 = _clip_non_negative(test_pred["pred_fga2_per_min"].fillna(0).values) * minutes
    pred_fga3 = _clip_non_negative(test_pred["pred_fga3_per_min"].fillna(0).values) * minutes
    pred_fta = _clip_non_negative(test_pred["pred_fta_per_min"].fillna(0).values) * minutes

    prior_fg2_pct = _clip_pct(test_pred["prior_fg2_pct"].fillna(LEAGUE_FG2_PCT).values, LEAGUE_FG2_PCT)
    prior_fg3_pct = _clip_pct(test_pred["prior_fg3_pct"].fillna(LEAGUE_FG3_PCT).values, LEAGUE_FG3_PCT)
    prior_ft_pct = _clip_pct(test_pred["prior_ft_pct"].fillna(LEAGUE_FT_PCT).values, LEAGUE_FT_PCT)

    pred_fg2m = pred_fga2 * prior_fg2_pct
    pred_fg3m = pred_fga3 * prior_fg3_pct
    pred_ftm = pred_fta * prior_ft_pct
    pred_pts = 2 * pred_fg2m + 3 * pred_fg3m + pred_ftm

    # Actual counting stats
    actual_fg2m = test["fg2m"].fillna(0).values
    actual_fg3m = test["fg3m"].fillna(0).values
    actual_ftm = test["ftm"].fillna(0).values
    actual_fga2 = test["fga2"].fillna(0).values
    actual_fga3 = test["fga3"].fillna(0).values
    actual_fta = test["fta"].fillna(0).values
    actual_pts = 2 * actual_fg2m + 3 * actual_fg3m + actual_ftm

    counting_metrics: dict[str, dict[str, float]] = {}
    for stat, pred, actual in [
        ("fg2m", pred_fg2m, actual_fg2m),
        ("fg3m", pred_fg3m, actual_fg3m),
        ("ftm", pred_ftm, actual_ftm),
        ("fga2", pred_fga2, actual_fga2),
        ("fga3", pred_fga3, actual_fga3),
        ("fta", pred_fta, actual_fta),
        ("pts", pred_pts, actual_pts),
    ]:
        metrics = _metric_row(actual, pred)
        counting_metrics[stat] = metrics
        print(f"  {stat:6s}: MAE={metrics['mae']:.2f}  bias={metrics['bias']:+.2f}  r={metrics['corr']:.3f}")

    # ast/reb from box labels
    test_box = test.merge(
        labels_box[JOIN_KEYS + ["ast", "oreb", "dreb", "stl", "blk", "tov"]],
        on=JOIN_KEYS, how="left", suffixes=("", "_label"),
    )
    for stat, rate_col in [
        ("ast", "ast_per_min"),
        ("oreb", "oreb_per_min"),
        ("dreb", "dreb_per_min"),
        ("stl", "stl_per_min"),
        ("blk", "blk_per_min"),
        ("tov", "tov_per_min"),
    ]:
        label_col = f"{stat}_label" if f"{stat}_label" in test_box.columns else stat
        if label_col in test_box.columns:
            actual_vals = test_box[label_col].fillna(0).values
            pred_vals = _clip_non_negative(test_pred[f"pred_{rate_col}"].fillna(0).values) * minutes
            metrics = _metric_row(actual_vals, pred_vals)
            counting_metrics[stat] = metrics
            print(f"  {stat:6s}: MAE={metrics['mae']:.2f}  bias={metrics['bias']:+.2f}  r={metrics['corr']:.3f}")

    # =========================================================================
    # Points by minutes bucket
    # =========================================================================
    print(f"\n{'='*70}")
    print("POINTS ERROR BY MINUTES BUCKET (LightGBM opp × prior eff × actual min)")
    print(f"{'='*70}")

    bins = [4, 10, 15, 20, 25, 30, 35, 48]
    test_eval = pd.DataFrame({"minutes": minutes, "actual_pts": actual_pts, "pred_pts": pred_pts})
    test_eval["bucket"] = pd.cut(test_eval["minutes"], bins=bins, right=False)
    points_bucket_metrics: list[dict[str, Any]] = []
    for bucket, g in test_eval.groupby("bucket", observed=False):
        if len(g) < 20:
            continue
        metrics = _metric_row(g["actual_pts"].to_numpy(dtype=float), g["pred_pts"].to_numpy(dtype=float))
        points_bucket_metrics.append({"bucket": str(bucket), **metrics})
        print(
            f"  {str(bucket):15s}  n={metrics['n']:4d}  "
            f"pts_MAE={metrics['mae']:.2f}  pts_bias={metrics['bias']:+.2f}"
        )

    # =========================================================================
    # DK FPTS
    # =========================================================================
    print(f"\n{'='*70}")
    print("FULL DK FPTS — LightGBM opp × prior eff + LightGBM counting × actual min")
    print(f"{'='*70}")

    pred_ast = _clip_non_negative(test_pred["pred_ast_per_min"].fillna(0).values) * minutes
    pred_oreb = _clip_non_negative(test_pred["pred_oreb_per_min"].fillna(0).values) * minutes
    pred_dreb = _clip_non_negative(test_pred["pred_dreb_per_min"].fillna(0).values) * minutes
    pred_stl = _clip_non_negative(test_pred["pred_stl_per_min"].fillna(0).values) * minutes
    pred_blk = _clip_non_negative(test_pred["pred_blk_per_min"].fillna(0).values) * minutes
    pred_tov = _clip_non_negative(test_pred["pred_tov_per_min"].fillna(0).values) * minutes
    pred_reb = pred_oreb + pred_dreb

    pred_dk = (pred_pts + 1.25 * pred_reb + 1.5 * pred_ast
               + 2.0 * pred_stl + 2.0 * pred_blk - 0.5 * pred_tov
               + 0.5 * pred_fg3m)

    # Actual DK
    if all(c in test_box.columns for c in ["ast_label", "oreb_label", "dreb_label"]):
        a_ast = test_box["ast_label"].fillna(0).values
        a_oreb = test_box["oreb_label"].fillna(0).values
        a_dreb = test_box["dreb_label"].fillna(0).values
    else:
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

    dk_metrics = _metric_row(actual_dk, pred_dk)
    dk_mae = dk_metrics["mae"]
    dk_bias = dk_metrics["bias"]
    dk_corr = dk_metrics["corr"]
    print(f"  DK FPTS MAE:  {dk_mae:.2f}")
    print(f"  DK FPTS bias: {dk_bias:+.2f}")
    print(f"  DK FPTS corr: {dk_corr:.3f}")

    test_dk = pd.DataFrame({"minutes": minutes, "actual_dk": actual_dk, "pred_dk": pred_dk})
    test_dk["bucket"] = pd.cut(test_dk["minutes"], bins=bins, right=False)
    dk_bucket_metrics: list[dict[str, Any]] = []
    for bucket, g in test_dk.groupby("bucket", observed=False):
        if len(g) < 20:
            continue
        metrics = _metric_row(g["actual_dk"].to_numpy(dtype=float), g["pred_dk"].to_numpy(dtype=float))
        dk_bucket_metrics.append({"bucket": str(bucket), **metrics})
        print(
            f"  {str(bucket):15s}  n={metrics['n']:4d}  "
            f"dk_MAE={metrics['mae']:.2f}  dk_bias={metrics['bias']:+.2f}"
        )

    # =========================================================================
    # Efficiency prior quality (attempt-weighted on scoring allocation)
    # =========================================================================
    print(f"\n{'='*70}")
    print("EFFICIENCY PRIOR QUALITY — attempt-weighted error on scoring allocation")
    print(f"{'='*70}")

    # v1 style: LightGBM predicts efficiency too (same opp rates though)
    # We don't have the v1 models here, so compare prior eff to actual game eff
    # to show the ceiling of what LightGBM eff could add

    attempt_weighted_efficiency_metrics: dict[str, dict[str, float]] = {}
    for stat, prior_col, actual_col, att_col in [
        ("FG2%", "prior_fg2_pct", "fg2_pct_label", "fga2"),
        ("FG3%", "prior_fg3_pct", "fg3_pct_label", "fga3"),
        ("FT%",  "prior_ft_pct",  "ft_pct_label",  "fta"),
    ]:
        mask = test[actual_col].notna() & test[prior_col].notna()
        if mask.sum() < 10:
            continue
        actual = test.loc[mask, actual_col].values
        prior = _clip_pct(test.loc[mask, prior_col].values, 0.0)

        # Weighted by attempts (more attempts = more impact on points)
        att = test.loc[mask, att_col].fillna(0).values
        wmae = float(np.average(np.abs(actual - prior), weights=np.maximum(att, 1)))
        wbias = float(np.average(prior - actual, weights=np.maximum(att, 1)))
        attempt_weighted_efficiency_metrics[stat] = {
            "n": int(mask.sum()),
            "attempt_weighted_mae": wmae,
            "attempt_weighted_bias": wbias,
        }
        print(f"  {stat:6s}: attempt-weighted MAE={wmae:.4f}  bias={wbias:+.4f}  n={mask.sum()}")

    # =========================================================================
    # Feature importance
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

    # =========================================================================
    # Save
    # =========================================================================
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = ds_dir.parent / f"lgbm_scoring_rates_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for target, model in models.items():
        model.save_model(str(models_dir / f"{target}.lgb"), num_iteration=model.best_iteration)

    (out_dir / "feature_columns.json").write_text(
        json.dumps({"columns": feature_cols}, indent=2), encoding="utf-8"
    )

    test_pred["pred_fga2"] = pred_fga2
    test_pred["pred_fga3"] = pred_fga3
    test_pred["pred_fta"] = pred_fta
    test_pred["pred_fg2m"] = pred_fg2m
    test_pred["pred_fg3m"] = pred_fg3m
    test_pred["pred_ftm"] = pred_ftm
    test_pred["pred_pts"] = pred_pts
    test_pred["pred_ast"] = pred_ast
    test_pred["pred_oreb"] = pred_oreb
    test_pred["pred_dreb"] = pred_dreb
    test_pred["pred_stl"] = pred_stl
    test_pred["pred_blk"] = pred_blk
    test_pred["pred_tov"] = pred_tov
    test_pred["pred_reb"] = pred_reb
    test_pred["pred_dk"] = pred_dk
    test_pred["actual_fga2"] = actual_fga2
    test_pred["actual_fga3"] = actual_fga3
    test_pred["actual_fta"] = actual_fta
    test_pred["actual_fg2m"] = actual_fg2m
    test_pred["actual_fg3m"] = actual_fg3m
    test_pred["actual_ftm"] = actual_ftm
    test_pred["actual_pts"] = actual_pts
    test_pred["actual_ast"] = a_ast
    test_pred["actual_oreb"] = a_oreb
    test_pred["actual_dreb"] = a_dreb
    test_pred["actual_stl"] = a_stl
    test_pred["actual_blk"] = a_blk
    test_pred["actual_tov"] = a_tov
    test_pred["actual_reb"] = a_reb
    test_pred["actual_dk"] = actual_dk

    test_pred.to_parquet(out_dir / "test_predictions.parquet", index=False)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(ds_dir),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_features": len(feature_cols),
        "best_iterations": {t: m.best_iteration for t, m in models.items()},
        "efficiency_source": "rolling_prior",
        "eff_min_attempts": args.eff_min_attempts,
        "rate_metrics": rate_metrics,
        "efficiency_prior_metrics": efficiency_prior_metrics,
        "counting_metrics": counting_metrics,
        "points_bucket_metrics": points_bucket_metrics,
        "dk_fpts_mae": dk_mae,
        "dk_fpts_bias": dk_bias,
        "dk_fpts_corr": dk_corr,
        "dk_bucket_metrics": dk_bucket_metrics,
        "attempt_weighted_efficiency_metrics": attempt_weighted_efficiency_metrics,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
