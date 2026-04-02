"""
LightGBM rotation + minutes experiment.

Three-stage model:
  1. P(active)   — binary classification (will they get minutes?)
  2. P(starter | active) — binary classification
  3. Minutes | active — quantile regression with simplex projection

Trains on the same dataset as GTv2 and reports rotation accuracy, minutes
calibration, and team-budget consistency.

Usage:
    uv run python scripts/experiments/lgbm_rotation_minutes_v1.py \
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]

ID_COLS = {
    "game_id", "player_id", "team_id", "home_team_id", "away_team_id",
    "opponent_team_id", "season", "player_name", "team_name", "team_tricode",
    "opponent_team_name", "opponent_team_tricode", "game_date", "tip_ts",
    "injury_as_of_ts", "roster_as_of_ts", "feature_as_of_ts", "snapshot_ts",
    "frozen_at", "snapshot_type", "status", "restriction_flag", "ramp_flag",
    "archetype", "pos_bucket", "lineup_role", "lineup_status",
    "lineup_roster_status", "lineup_timestamp", "odds_as_of_ts",
}

EXCLUDE_LEAKY = {
    "minutes", "minutes_label", "starter_flag_label", "starter_flag",
    "source", "first_in_time_real", "last_out_time_real", "time_unit_detected",
}

EXCLUDE_DNP_BLIND = {
    "min_last1", "min_last3", "min_last5", "roll_mean_3", "roll_mean_5",
    "roll_mean_10", "roll_iqr_5", "z_vs_10",
}

EXCLUDE_SAME_GAME = {
    "depth_6", "depth_10", "depth_14", "effective_n", "bench_conc_top1",
    "bench_conc_top2", "starter_pool_minutes", "bench_pool_minutes",
    "team_total_minutes_from_stints", "num_stints",
    "max_stint_len_real", "minutes_from_stints",
    "started_proxy", "rotation_team_missing", "rotation_missing",
    "rotation_player_row_missing_raw", "rotation_player_filled_zero",
}

EXCLUDE_UNSTABLE = {
    "vac_min_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn",
}

ALL_EXCLUDE = ID_COLS | EXCLUDE_LEAKY | EXCLUDE_DNP_BLIND | EXCLUDE_SAME_GAME | EXCLUDE_UNSTABLE

MINUTES_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def _select_feature_cols(feat: pd.DataFrame) -> list[str]:
    numeric = feat.select_dtypes(include=["number"]).columns.tolist()
    return sorted(c for c in numeric if c not in ALL_EXCLUDE)


def _time_split(
    df: pd.DataFrame,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["game_date"].unique())
    n = len(dates)
    test_start = dates[int(n * (1 - test_frac))]
    val_start = dates[int(n * (1 - test_frac - val_frac))]
    train = df[df["game_date"] < val_start].copy()
    val = df[(df["game_date"] >= val_start) & (df["game_date"] < test_start)].copy()
    test = df[df["game_date"] >= test_start].copy()
    return train, val, test


def _build_team_context_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Teammate and opponent aggregate features."""
    agg_candidates = [
        "prior_play_prob", "an_implied_minutes", "an_pts_line", "an_reb_line",
        "an_ast_line", "an_pra_line", "sum_min_7d", "recent_start_pct_10",
        "prior_minutes_share_20",
    ]
    agg_cols = [c for c in agg_candidates if c in feature_cols]
    if not agg_cols:
        return df

    team_totals = df.groupby(["game_id", "team_id"])[agg_cols].transform("sum")
    team_counts = df.groupby(["game_id", "team_id"])[agg_cols[0]].transform("count")
    for col in agg_cols:
        self_val = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        team_sum = pd.to_numeric(team_totals[col], errors="coerce").fillna(0.0)
        df[f"tm_mean_{col}"] = (team_sum - self_val) / (team_counts - 1).clip(lower=1)
        df[f"tm_sum_{col}"] = team_sum - self_val

    # Opponent aggregates
    opp_cols = [c for c in ["an_implied_minutes", "an_pts_line", "an_pra_line", "prior_minutes_share_20"]
                if c in feature_cols]
    if opp_cols:
        game_totals = df.groupby("game_id")[opp_cols].transform("sum")
        opp_totals = df.groupby(["game_id", "team_id"])[opp_cols].transform("sum")
        for col in opp_cols:
            df[f"opp_sum_{col}"] = (
                pd.to_numeric(game_totals[col], errors="coerce").fillna(0.0)
                - pd.to_numeric(opp_totals[col], errors="coerce").fillna(0.0)
            )

    # Team-level counts
    df["tm_n_roster"] = df.groupby(["game_id", "team_id"])["player_id"].transform("count")
    df["tm_sum_play_prob"] = df.groupby(["game_id", "team_id"])["prior_play_prob"].transform("sum")
    df["tm_n_implied_active"] = df.groupby(["game_id", "team_id"])["an_implied_minutes"].transform(
        lambda x: (x > 0).sum()
    )

    return df


def _simplex_project(
    minutes: np.ndarray,
    team_ids: np.ndarray,
    game_ids: np.ndarray,
    target_total: float = 240.0,
    max_per_player: float = 48.0,
) -> np.ndarray:
    """Scale minutes within each team-game to sum to target_total."""
    result = minutes.copy()
    keys = list(zip(game_ids, team_ids))
    unique_keys = sorted(set(keys))
    key_to_idx: dict[tuple, list[int]] = {k: [] for k in unique_keys}
    for i, k in enumerate(keys):
        key_to_idx[k].append(i)

    for key, idxs in key_to_idx.items():
        team_mins = result[idxs]
        total = team_mins.sum()
        if total > 1e-6:
            scale = target_total / total
            team_mins = np.clip(team_mins * scale, 0, max_per_player)
            # Re-scale after clipping
            clipped_total = team_mins.sum()
            if clipped_total > 1e-6 and abs(clipped_total - target_total) > 0.1:
                unclipped = team_mins < max_per_player
                if unclipped.any():
                    deficit = target_total - team_mins[~unclipped].sum()
                    team_mins[unclipped] *= deficit / team_mins[unclipped].sum()
            result[idxs] = team_mins

    return result


def _train_binary_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    *, n_estimators: int = 2000, lr: float = 0.05,
    num_leaves: int = 63, min_child_samples: int = 50,
    scale_pos_weight: float = 1.0,
) -> lgb.Booster:
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    ds_train = lgb.Dataset(X_train, label=y_train)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
    model = lgb.train(
        params, ds_train,
        num_boost_round=n_estimators,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    return model


def _train_quantile_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    alpha: float, *, n_estimators: int = 2000, lr: float = 0.05,
    num_leaves: int = 127, min_child_samples: int = 50,
) -> lgb.Booster:
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    ds_train = lgb.Dataset(X_train, label=y_train)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
    model = lgb.train(
        params, ds_train,
        num_boost_round=n_estimators,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    return model


def _train_mean_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    *, n_estimators: int = 2000, lr: float = 0.05,
    num_leaves: int = 127, min_child_samples: int = 50,
) -> lgb.Booster:
    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    ds_train = lgb.Dataset(X_train, label=y_train)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
    model = lgb.train(
        params, ds_train,
        num_boost_round=n_estimators,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    ds_dir = Path(str(args.dataset_dir)).expanduser().resolve()
    print(f"Loading from {ds_dir}")

    feat = pd.read_parquet(ds_dir / "features.parquet")
    labels_min = pd.read_parquet(ds_dir / "labels_minutes.parquet")
    labels_box = pd.read_parquet(ds_dir / "labels_boxscore_counts.parquet")

    # Join
    min_cols = [c for c in ["game_id", "team_id", "player_id", "game_date",
                            "minutes_label", "starter_flag_label"] if c in labels_min.columns]
    df = feat.merge(labels_min[min_cols], on=JOIN_KEYS, how="inner")
    # Also pull starter_flag from box labels if not in minutes labels
    if "starter_flag_label" not in df.columns and "starter_flag" in labels_box.columns:
        box_starter = labels_box[["game_id", "team_id", "player_id", "game_date", "starter_flag"]].copy()
        box_starter = box_starter.rename(columns={"starter_flag": "starter_flag_label"})
        df = df.merge(box_starter, on=JOIN_KEYS, how="left")
    # minutes_label may not exist — fall back to labels_box
    if "minutes_label" not in df.columns:
        df = df.merge(
            labels_box[["game_id", "team_id", "player_id", "game_date", "minutes"]].rename(
                columns={"minutes": "minutes_label"}
            ),
            on=JOIN_KEYS, how="left",
        )
    df["minutes_actual"] = pd.to_numeric(df["minutes_label"], errors="coerce").fillna(0.0)
    df["played_actual"] = (df["minutes_actual"] > 0).astype(int)
    # starter_flag_label: in some datasets this is 1 for ALL rows (useless); detect and use box labels
    if "starter_flag_label" in df.columns:
        sf = pd.to_numeric(df["starter_flag_label"], errors="coerce")
        if sf.nunique(dropna=True) <= 1:
            # Useless — try to derive from minutes ranking (top 5 per team = starters)
            df["starter_actual"] = 0
            for (gid, tid), grp in df[df["played_actual"] == 1].groupby(["game_id", "team_id"]):
                top5 = grp.nlargest(5, "minutes_actual").index
                df.loc[top5, "starter_actual"] = 1
            print("  (starter_flag_label was constant — deriving starters from top-5 minutes)")
        else:
            df["starter_actual"] = sf.fillna(0).astype(int)
    else:
        df["starter_actual"] = 0

    print(f"Rows: {len(df)}, Games: {df['game_id'].nunique()}")
    print(f"Active rate: {df['played_actual'].mean():.1%}")
    print(f"Starter rate (of active): {df.loc[df['played_actual']==1, 'starter_actual'].mean():.1%}")

    # Features
    feature_cols = _select_feature_cols(feat)
    df = _build_team_context_features(df, feature_cols)
    extra = [c for c in df.columns if c.startswith("tm_") or c.startswith("opp_sum_")]
    feature_cols = feature_cols + sorted(extra)
    print(f"Features: {len(feature_cols)}")

    # Split
    train, val, test = _time_split(df)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"  train: {train['game_date'].min()} → {train['game_date'].max()}")
    print(f"  val:   {val['game_date'].min()} → {val['game_date'].max()}")
    print(f"  test:  {test['game_date'].min()} → {test['game_date'].max()}")

    X_train_all = train[feature_cols]
    X_val_all = val[feature_cols]
    X_test_all = test[feature_cols]

    # =========================================================================
    # STAGE 1: P(active)
    # =========================================================================
    print(f"\n{'='*70}")
    print("STAGE 1: P(active) — binary classification")
    print(f"{'='*70}")

    active_model = _train_binary_model(
        X_train_all, train["played_actual"],
        X_val_all, val["played_actual"],
        n_estimators=args.n_estimators, lr=args.lr,
    )
    test_p_active = active_model.predict(X_test_all)

    # Evaluate active model
    from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, f1_score
    auc = roc_auc_score(test["played_actual"], test_p_active)
    ll = log_loss(test["played_actual"], test_p_active)

    # Find optimal threshold on val
    val_p_active = active_model.predict(X_val_all)
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.1, 0.9, 0.02):
        f1 = f1_score(val["played_actual"], (val_p_active >= thr).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    test_active_pred = (test_p_active >= best_thr).astype(int)
    test_f1 = f1_score(test["played_actual"], test_active_pred)

    print(f"\n  AUC:        {auc:.4f}")
    print(f"  Log loss:   {ll:.4f}")
    print(f"  Best thr:   {best_thr:.2f} (val F1={best_f1:.4f})")
    print(f"  Test F1:    {test_f1:.4f}")
    print(f"  Pred active rate: {test_active_pred.mean():.1%} (actual: {test['played_actual'].mean():.1%})")

    # Confusion at threshold
    tp = int(((test_active_pred == 1) & (test["played_actual"] == 1)).sum())
    fp = int(((test_active_pred == 1) & (test["played_actual"] == 0)).sum())
    fn = int(((test_active_pred == 0) & (test["played_actual"] == 1)).sum())
    tn = int(((test_active_pred == 0) & (test["played_actual"] == 0)).sum())
    print(f"  Confusion:  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision:  {tp/(tp+fp):.4f}" if tp+fp > 0 else "  Precision: N/A")
    print(f"  Recall:     {tp/(tp+fn):.4f}" if tp+fn > 0 else "  Recall: N/A")

    # Per-team accuracy (correct count of active players)
    test_team = test[["game_id", "team_id", "played_actual"]].copy()
    test_team["pred_active"] = test_active_pred
    team_agg = test_team.groupby(["game_id", "team_id"]).agg(
        actual_n=("played_actual", "sum"),
        pred_n=("pred_active", "sum"),
    ).reset_index()
    team_agg["count_error"] = team_agg["pred_n"] - team_agg["actual_n"]
    print(f"\n  Team active-count error: mean={team_agg['count_error'].mean():.2f}, "
          f"std={team_agg['count_error'].std():.2f}, "
          f"mae={team_agg['count_error'].abs().mean():.2f}")
    print(f"  Exact match: {(team_agg['count_error'] == 0).mean():.1%}")

    # Top features
    imp = active_model.feature_importance(importance_type="gain")
    top = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:15]
    print(f"\n  Top 15 features (active):")
    for name, v in top:
        print(f"    {name:50s} {v:.0f}")

    # =========================================================================
    # STAGE 2: P(starter | active)
    # =========================================================================
    print(f"\n{'='*70}")
    print("STAGE 2: P(starter | active) — binary classification")
    print(f"{'='*70}")

    train_active = train[train["played_actual"] == 1]
    val_active = val[val["played_actual"] == 1]
    test_active = test[test["played_actual"] == 1]

    starter_model = _train_binary_model(
        train_active[feature_cols], train_active["starter_actual"],
        val_active[feature_cols], val_active["starter_actual"],
        n_estimators=args.n_estimators, lr=args.lr,
    )
    test_p_starter = starter_model.predict(test_active[feature_cols])
    starter_auc = roc_auc_score(test_active["starter_actual"], test_p_starter)

    # Top-5 per team approach: for each team-game, pick top 5 by P(starter)
    test_starter_eval = test_active[["game_id", "team_id", "player_id", "starter_actual"]].copy()
    test_starter_eval["p_starter"] = test_p_starter
    test_starter_eval["pred_starter"] = 0

    for (gid, tid), grp in test_starter_eval.groupby(["game_id", "team_id"]):
        top5_idx = grp.nlargest(5, "p_starter").index
        test_starter_eval.loc[top5_idx, "pred_starter"] = 1

    starter_acc = (test_starter_eval["pred_starter"] == test_starter_eval["starter_actual"]).mean()
    starter_precision = (
        test_starter_eval.loc[test_starter_eval["pred_starter"] == 1, "starter_actual"].mean()
    )
    starter_recall = (
        test_starter_eval.loc[test_starter_eval["starter_actual"] == 1, "pred_starter"].mean()
    )
    # How many starters did we get exactly right per team?
    starter_team = test_starter_eval.groupby(["game_id", "team_id"]).apply(
        lambda g: (g["pred_starter"] & g["starter_actual"]).sum(), include_groups=False
    )
    print(f"\n  Starter AUC:       {starter_auc:.4f}")
    print(f"  Starter accuracy:  {starter_acc:.4f}")
    print(f"  Starter precision: {starter_precision:.4f}")
    print(f"  Starter recall:    {starter_recall:.4f}")
    print(f"  Correct starters per team: mean={starter_team.mean():.2f}/5, "
          f"all-5={int((starter_team == 5).sum())}/{len(starter_team)}")

    imp = starter_model.feature_importance(importance_type="gain")
    top = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:10]
    print(f"\n  Top 10 features (starter):")
    for name, v in top:
        print(f"    {name:50s} {v:.0f}")

    # =========================================================================
    # STAGE 3: Minutes | active — quantile regression + mean
    # =========================================================================
    print(f"\n{'='*70}")
    print("STAGE 3: Minutes | active — quantile + mean regression")
    print(f"{'='*70}")

    # Train on active players only
    y_train_min = train_active["minutes_actual"]
    y_val_min = val_active["minutes_actual"]
    y_test_min = test_active["minutes_actual"]

    # Mean model
    print("\nTraining mean regression...")
    minutes_mean_model = _train_mean_model(
        train_active[feature_cols], y_train_min,
        val_active[feature_cols], y_val_min,
        n_estimators=args.n_estimators, lr=args.lr,
    )
    test_minutes_mean = minutes_mean_model.predict(test_active[feature_cols])

    # Quantile models
    minutes_q_models: dict[float, lgb.Booster] = {}
    test_minutes_q: dict[float, np.ndarray] = {}
    for q in MINUTES_QUANTILES:
        print(f"\nTraining quantile={q:.2f}...")
        model = _train_quantile_model(
            train_active[feature_cols], y_train_min,
            val_active[feature_cols], y_val_min,
            alpha=q, n_estimators=args.n_estimators, lr=args.lr,
        )
        minutes_q_models[q] = model
        test_minutes_q[q] = model.predict(test_active[feature_cols])

    # Raw metrics (before simplex)
    mae_raw = float(np.mean(np.abs(y_test_min.values - test_minutes_mean)))
    median_mae_raw = float(np.mean(np.abs(y_test_min.values - test_minutes_q[0.50])))
    print(f"\n--- Raw Minutes Metrics (active only, before simplex) ---")
    print(f"  Mean MAE:   {mae_raw:.2f}")
    print(f"  Median MAE: {median_mae_raw:.2f}")
    for q in MINUTES_QUANTILES:
        cov = float(np.mean(y_test_min.values <= test_minutes_q[q]))
        print(f"  q{int(q*100):02d} coverage: {cov:.4f} (target {q:.2f})")

    # Simplex-projected metrics
    test_minutes_simplex = _simplex_project(
        test_minutes_mean,
        test_active["team_id"].values,
        test_active["game_id"].values,
    )
    mae_simplex = float(np.mean(np.abs(y_test_min.values - test_minutes_simplex)))
    print(f"\n--- After Simplex Projection ---")
    print(f"  Mean MAE:   {mae_simplex:.2f}")

    # Team budget check
    test_budget = test_active[["game_id", "team_id"]].copy()
    test_budget["pred_minutes"] = test_minutes_simplex
    test_budget["actual_minutes"] = y_test_min.values
    team_totals = test_budget.groupby(["game_id", "team_id"]).agg(
        pred_total=("pred_minutes", "sum"),
        actual_total=("actual_minutes", "sum"),
    ).reset_index()
    print(f"  Team total (pred):   mean={team_totals['pred_total'].mean():.1f}, std={team_totals['pred_total'].std():.1f}")
    print(f"  Team total (actual): mean={team_totals['actual_total'].mean():.1f}, std={team_totals['actual_total'].std():.1f}")

    # Conditional calibration by minutes bucket
    print(f"\n--- Minutes Calibration by Bucket (active only) ---")
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 48]
    test_eval = pd.DataFrame({
        "actual": y_test_min.values,
        "pred_mean": test_minutes_mean,
        "pred_simplex": test_minutes_simplex,
    })
    for q in MINUTES_QUANTILES:
        test_eval[f"q{int(q*100):02d}"] = test_minutes_q[q]
    test_eval["bucket"] = pd.cut(test_eval["pred_mean"], bins=bins, right=False)

    for bucket, g in test_eval.groupby("bucket", observed=False):
        if len(g) < 20:
            continue
        mae = float(np.mean(np.abs(g["actual"].values - g["pred_mean"].values)))
        q10_cov = float(np.mean(g["actual"].values <= g["q10"].values))
        q50_cov = float(np.mean(g["actual"].values <= g["q50"].values))
        q90_cov = float(np.mean(g["actual"].values <= g["q90"].values))
        print(f"  {str(bucket):15s}  n={len(g):4d}  MAE={mae:.2f}  "
              f"q10={q10_cov:.3f}  q50={q50_cov:.3f}  q90={q90_cov:.3f}")

    # Feature importance for minutes model
    imp = minutes_mean_model.feature_importance(importance_type="gain")
    top = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:20]
    print(f"\n  Top 20 features (minutes|active):")
    for name, v in top:
        print(f"    {name:50s} {v:.0f}")

    # =========================================================================
    # COMBINED PIPELINE EVAL — unconditional minutes
    # =========================================================================
    print(f"\n{'='*70}")
    print("COMBINED: Full rotation pipeline (active prediction → minutes)")
    print(f"{'='*70}")

    # For all test players: predict P(active), then minutes for predicted-active
    test_all_p_active = active_model.predict(X_test_all)
    test_all_pred_active = (test_all_p_active >= best_thr).astype(int)

    # Predict minutes for everyone (but zero out predicted-inactive)
    test_all_minutes_raw = minutes_mean_model.predict(X_test_all)
    test_all_minutes = np.where(test_all_pred_active == 1, test_all_minutes_raw, 0.0)

    # Simplex project (only active players contribute)
    test_all_minutes_proj = _simplex_project(
        test_all_minutes,
        test["team_id"].values,
        test["game_id"].values,
    )

    # Compare to actual
    actual_minutes = test["minutes_actual"].values
    uncond_mae = float(np.mean(np.abs(actual_minutes - test_all_minutes_proj)))
    active_mask = test["played_actual"].values == 1
    cond_mae = float(np.mean(np.abs(actual_minutes[active_mask] - test_all_minutes_proj[active_mask])))

    # Naive baseline: use prior_minutes_share_20 * 240
    prior_share = test["prior_minutes_share_20"].fillna(0).values
    naive_minutes = prior_share * 240.0
    naive_mae = float(np.mean(np.abs(actual_minutes - naive_minutes)))
    naive_cond_mae = float(np.mean(np.abs(actual_minutes[active_mask] - naive_minutes[active_mask])))

    # Props baseline: an_implied_minutes
    props_minutes = test["an_implied_minutes"].fillna(0).values
    props_mae = float(np.mean(np.abs(actual_minutes - props_minutes)))
    props_cond_mae = float(np.mean(np.abs(actual_minutes[active_mask] - props_minutes[active_mask])))

    print(f"\n  Unconditional MAE (all players):")
    print(f"    LightGBM pipeline: {uncond_mae:.2f}")
    print(f"    Prior share×240:   {naive_mae:.2f}")
    print(f"    Props implied:     {props_mae:.2f}")
    print(f"\n  Conditional MAE (active only):")
    print(f"    LightGBM pipeline: {cond_mae:.2f}")
    print(f"    Prior share×240:   {naive_cond_mae:.2f}")
    print(f"    Props implied:     {props_cond_mae:.2f}")

    # Per-player error by role
    test_with_pred = test[["game_id", "team_id", "player_id", "played_actual",
                           "minutes_actual", "starter_actual", "prior_play_prob",
                           "an_implied_minutes", "prior_minutes_share_20"]].copy()
    test_with_pred["pred_active"] = test_all_pred_active
    test_with_pred["pred_minutes"] = test_all_minutes_proj
    test_with_pred["minutes_error"] = test_with_pred["pred_minutes"] - test_with_pred["minutes_actual"]

    print(f"\n  Error by role:")
    for label, mask in [
        ("Starters", test_with_pred["starter_actual"] == 1),
        ("Bench (active)", (test_with_pred["played_actual"] == 1) & (test_with_pred["starter_actual"] == 0)),
        ("DNP", test_with_pred["played_actual"] == 0),
    ]:
        if mask.sum() == 0:
            continue
        sub = test_with_pred[mask]
        mae = sub["minutes_error"].abs().mean()
        bias = sub["minutes_error"].mean()
        print(f"    {label:20s}: n={len(sub):5d}  MAE={mae:.2f}  bias={bias:+.2f}")

    # =========================================================================
    # Save
    # =========================================================================
    if args.out_dir:
        out_dir = Path(str(args.out_dir)).expanduser().resolve()
    else:
        out_dir = ds_dir.parent / f"lgbm_rotation_minutes_v1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    active_model.save_model(str(out_dir / "model_active.lgb"))
    starter_model.save_model(str(out_dir / "model_starter.lgb"))
    minutes_mean_model.save_model(str(out_dir / "model_minutes_mean.lgb"))
    for q in MINUTES_QUANTILES:
        minutes_q_models[q].save_model(str(out_dir / f"model_minutes_q{int(q*100):02d}.lgb"))

    test_with_pred.to_parquet(out_dir / "test_predictions.parquet", index=False)
    (out_dir / "feature_columns.json").write_text(
        json.dumps({"columns": feature_cols}, indent=2), encoding="utf-8"
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(ds_dir),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_features": len(feature_cols),
        "active_auc": auc, "active_f1": test_f1, "active_threshold": best_thr,
        "starter_auc": starter_auc, "starter_accuracy": starter_acc,
        "minutes_mae_raw": mae_raw, "minutes_mae_simplex": mae_simplex,
        "combined_mae_uncond": uncond_mae, "combined_mae_cond": cond_mae,
        "baseline_naive_mae": naive_mae, "baseline_props_mae": props_mae,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
