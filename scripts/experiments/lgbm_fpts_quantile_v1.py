"""
LightGBM pinball-loss FPTS quantile experiment.

Train per-quantile LightGBM models on the same feature set as GTv2 and compare
marginal calibration against the current GTv2 worlds output.

Usage:
    uv run python scripts/experiments/lgbm_fpts_quantile_v1.py \
        --dataset-dir /home/daniel/projections-data/training/datasets/joint_rotation_rates_v1_trackingctx_prodparity_lineupavailfix_priorplayprobstarterneutral_20260326T201310Z \
        --out-dir /home/daniel/projections-data/training/runs/lgbm_fpts_quantile_v1
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

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# DK FPTS weights applied to box-score labels
# pts = 2*fg2m + 3*fg3m + ftm, then DK scoring adds 0.5*fg3m bonus
DK_WEIGHTS = {
    "fg2m": 2.0,
    "fg3m": 3.5,   # 3 for points + 0.5 three-pointer bonus
    "ftm": 1.0,
    "oreb": 1.25,
    "dreb": 1.25,
    "ast": 1.5,
    "stl": 2.0,
    "blk": 2.0,
    "tov": -0.5,
}


def _compute_dk_fpts(box: pd.DataFrame) -> pd.Series:
    """Compute DK FPTS from box-score labels (ignoring DD/TD bonus)."""
    fpts = pd.Series(0.0, index=box.index)
    for col, weight in DK_WEIGHTS.items():
        fpts += pd.to_numeric(box[col], errors="coerce").fillna(0.0) * weight
    return fpts


def _select_feature_cols(feat: pd.DataFrame) -> list[str]:
    numeric = feat.select_dtypes(include=["number"]).columns.tolist()
    return sorted(c for c in numeric if c not in ALL_EXCLUDE)


def _time_split(
    df: pd.DataFrame,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split: train | val | test by game_date."""
    dates = sorted(df["game_date"].unique())
    n = len(dates)
    test_start = dates[int(n * (1 - test_frac))]
    val_start = dates[int(n * (1 - test_frac - val_frac))]
    train = df[df["game_date"] < val_start].copy()
    val = df[(df["game_date"] >= val_start) & (df["game_date"] < test_start)].copy()
    test = df[df["game_date"] >= test_start].copy()
    return train, val, test


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball (quantile) loss."""
    residual = y_true - y_pred
    return float(np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1) * residual)))


def _calibration_report(
    y_true: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> dict[str, float]:
    """Fraction of y_true below each predicted quantile (should match the quantile)."""
    report = {}
    for q, pred in sorted(quantile_preds.items()):
        actual_coverage = float(np.mean(y_true <= pred))
        report[f"q{int(q*100):02d}_target"] = q
        report[f"q{int(q*100):02d}_actual"] = round(actual_coverage, 4)
        report[f"q{int(q*100):02d}_gap"] = round(actual_coverage - q, 4)
    return report


def _train_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    alpha: float,
    *,
    n_estimators: int = 2000,
    lr: float = 0.05,
    num_leaves: int = 127,
    min_child_samples: int = 50,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 100,
) -> lgb.Booster:
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": lr,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    ds_train = lgb.Dataset(X_train, label=y_train)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
    callbacks = [
        lgb.early_stopping(early_stopping_rounds),
        lgb.log_evaluation(200),
    ]
    model = lgb.train(
        params,
        ds_train,
        num_boost_round=n_estimators,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=callbacks,
    )
    return model


def _build_team_context_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Add teammate-aggregate features to capture intrateam context.

    For each player-game, compute team-level aggregates (excluding self) from
    the same features available at prediction time.  This gives the model some
    awareness of who else is playing — a lightweight substitute for the
    transformer's attention over teammates.
    """
    # Subset of features worth aggregating at team level
    team_agg_candidates = [
        "prior_play_prob", "an_implied_minutes", "an_pts_line", "an_reb_line",
        "an_ast_line", "an_pra_line", "sum_min_7d", "recent_start_pct_10",
    ]
    agg_cols = [c for c in team_agg_candidates if c in feature_cols]
    if not agg_cols:
        return df

    # Team totals (including self)
    team_totals = (
        df.groupby(["game_id", "team_id"])[agg_cols]
        .transform("sum")
    )
    team_counts = (
        df.groupby(["game_id", "team_id"])[agg_cols[0]]
        .transform("count")
    )
    for col in agg_cols:
        self_val = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        team_sum = pd.to_numeric(team_totals[col], errors="coerce").fillna(0.0)
        teammates_sum = team_sum - self_val
        teammates_count = team_counts - 1
        df[f"tm_mean_{col}"] = teammates_sum / teammates_count.clip(lower=1)
        df[f"tm_sum_{col}"] = teammates_sum

    # Opponent aggregates
    opp_agg_cols = ["an_implied_minutes", "an_pts_line", "an_pra_line"]
    opp_cols = [c for c in opp_agg_cols if c in feature_cols]
    if opp_cols:
        opp_totals = (
            df.groupby(["game_id", "team_id"])[opp_cols]
            .transform("sum")
        )
        # Each game has 2 teams; game total minus own team = opponent
        game_totals = (
            df.groupby("game_id")[opp_cols]
            .transform("sum")
        )
        for col in opp_cols:
            df[f"opp_sum_{col}"] = (
                pd.to_numeric(game_totals[col], errors="coerce").fillna(0.0)
                - pd.to_numeric(opp_totals[col], errors="coerce").fillna(0.0)
            )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM FPTS quantile experiment")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--active-only", action="store_true", default=False,
                        help="Train only on rows where player actually played (minutes > 0)")
    parser.add_argument("--include-minutes-target", action="store_true", default=False,
                        help="Also train a minutes regression model for comparison")
    parser.add_argument("--add-team-context", action="store_true", default=True,
                        help="Add teammate-aggregate features")
    args = parser.parse_args()

    ds_dir = Path(str(args.dataset_dir)).expanduser().resolve()
    print(f"Loading dataset from {ds_dir}")

    feat = pd.read_parquet(ds_dir / "features.parquet")
    labels_box = pd.read_parquet(ds_dir / "labels_boxscore_counts.parquet")

    # Join
    df = feat.merge(labels_box, on=["game_id", "team_id", "player_id", "game_date"], how="inner")
    print(f"Joined: {len(df)} rows, {df['game_id'].nunique()} games")

    # Compute target
    df["dk_fpts"] = _compute_dk_fpts(df)
    df["played"] = pd.to_numeric(df["played"], errors="coerce").fillna(0).astype(int)

    # Add double-double / triple-double bonuses
    df["pts"] = 2.0 * df["fg2m"].fillna(0) + 3.0 * df["fg3m"].fillna(0) + df["ftm"].fillna(0)
    df["reb"] = df["oreb"].fillna(0) + df["dreb"].fillna(0)
    counting = pd.DataFrame({
        "pts": df["pts"], "reb": df["reb"], "ast": df["ast"].fillna(0),
        "stl": df["stl"].fillna(0), "blk": df["blk"].fillna(0),
    })
    qualifying = (counting >= 10).sum(axis=1)
    df["dk_fpts"] += np.where(qualifying >= 3, 3.0, np.where(qualifying >= 2, 1.5, 0.0))

    if args.active_only:
        df = df[df["played"] == 1].copy()
        print(f"Active-only filter: {len(df)} rows")

    # Select features
    feature_cols = _select_feature_cols(feat)
    print(f"Feature columns: {len(feature_cols)}")

    # Add team context features
    if args.add_team_context:
        df = _build_team_context_features(df, feature_cols)
        extra = [c for c in df.columns if c.startswith("tm_") or c.startswith("opp_sum_")]
        feature_cols = feature_cols + sorted(extra)
        print(f"With team context: {len(feature_cols)} features")

    # Time split
    train, val, test = _time_split(df)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"  train dates: {train['game_date'].min()} to {train['game_date'].max()}")
    print(f"  val dates:   {val['game_date'].min()} to {val['game_date'].max()}")
    print(f"  test dates:  {test['game_date'].min()} to {test['game_date'].max()}")

    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y_train = train["dk_fpts"]
    y_val = val["dk_fpts"]
    y_test = test["dk_fpts"]

    # Train quantile models
    models: dict[float, lgb.Booster] = {}
    train_times: dict[float, float] = {}
    for q in QUANTILES:
        print(f"\n{'='*60}")
        print(f"Training quantile={q:.2f}")
        print(f"{'='*60}")
        t0 = time.time()
        model = _train_quantile_model(
            X_train, y_train, X_val, y_val,
            alpha=q,
            n_estimators=int(args.n_estimators),
            lr=float(args.lr),
            num_leaves=int(args.num_leaves),
        )
        elapsed = time.time() - t0
        models[q] = model
        train_times[q] = elapsed
        print(f"  Trained in {elapsed:.1f}s, best_iteration={model.best_iteration}")

    # Also train a mean regression model for comparison
    print(f"\n{'='*60}")
    print(f"Training mean regression (L2)")
    print(f"{'='*60}")
    t0 = time.time()
    mean_params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": float(args.lr),
        "num_leaves": int(args.num_leaves),
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": -1,
    }
    ds_train = lgb.Dataset(X_train, label=y_train)
    ds_val_lgb = lgb.Dataset(X_val, label=y_val, reference=ds_train)
    mean_model = lgb.train(
        mean_params, ds_train,
        num_boost_round=int(args.n_estimators),
        valid_sets=[ds_val_lgb],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    mean_elapsed = time.time() - t0
    print(f"  Trained in {mean_elapsed:.1f}s, best_iteration={mean_model.best_iteration}")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION ({len(test)} rows)")
    print(f"{'='*60}")

    quantile_preds: dict[float, np.ndarray] = {}
    for q in QUANTILES:
        quantile_preds[q] = models[q].predict(X_test)

    mean_pred = mean_model.predict(X_test)

    # 1. Calibration
    cal = _calibration_report(y_test.values, quantile_preds)
    print("\n--- Quantile Calibration ---")
    for q in QUANTILES:
        k = f"q{int(q*100):02d}"
        print(f"  {k}: target={cal[f'{k}_target']:.2f}, actual={cal[f'{k}_actual']:.4f}, gap={cal[f'{k}_gap']:+.4f}")

    # 2. MAE / pinball loss
    mean_mae = float(np.mean(np.abs(y_test.values - mean_pred)))
    median_mae = float(np.mean(np.abs(y_test.values - quantile_preds[0.50])))
    print(f"\n--- Error Metrics ---")
    print(f"  Mean model MAE:   {mean_mae:.3f}")
    print(f"  Median model MAE: {median_mae:.3f}")
    for q in QUANTILES:
        pl = _pinball_loss(y_test.values, quantile_preds[q], q)
        print(f"  Pinball loss q{int(q*100):02d}: {pl:.3f}")

    # 3. Conditional calibration by minutes bucket
    print(f"\n--- Conditional Calibration (by predicted minutes bucket) ---")
    test_with_preds = test[["game_id", "team_id", "player_id", "dk_fpts", "an_implied_minutes", "prior_play_prob"]].copy()
    test_with_preds["pred_mean"] = mean_pred
    test_with_preds["pred_p50"] = quantile_preds[0.50]
    test_with_preds["pred_p10"] = quantile_preds[0.10]
    test_with_preds["pred_p90"] = quantile_preds[0.90]
    test_with_preds["pred_p25"] = quantile_preds[0.25]
    test_with_preds["pred_p75"] = quantile_preds[0.75]

    # Bucket by predicted median FPTS
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 100]
    test_with_preds["fpts_bucket"] = pd.cut(test_with_preds["pred_p50"], bins=bins, right=False)
    for bucket, g in test_with_preds.groupby("fpts_bucket", observed=False):
        if len(g) < 20:
            continue
        y = g["dk_fpts"].values
        p10_cov = float(np.mean(y <= g["pred_p10"].values))
        p50_cov = float(np.mean(y <= g["pred_p50"].values))
        p90_cov = float(np.mean(y <= g["pred_p90"].values))
        mae = float(np.mean(np.abs(y - g["pred_mean"].values)))
        print(f"  {str(bucket):20s}  n={len(g):5d}  MAE={mae:.2f}  "
              f"p10_cov={p10_cov:.3f}(0.10)  p50_cov={p50_cov:.3f}(0.50)  p90_cov={p90_cov:.3f}(0.90)")

    # 4. Active vs DNP breakdown
    print(f"\n--- Active vs DNP ---")
    for played_val, label in [(1, "Active"), (0, "DNP")]:
        mask = test["played"] == played_val
        if mask.sum() == 0:
            continue
        y_sub = y_test.values[mask]
        mae = float(np.mean(np.abs(y_sub - mean_pred[mask])))
        p50_cov = float(np.mean(y_sub <= quantile_preds[0.50][mask]))
        print(f"  {label:8s}: n={mask.sum():5d}  MAE={mae:.2f}  p50_cov={p50_cov:.3f}")

    # 5. Feature importance (top 30)
    print(f"\n--- Top 30 Features (median model) ---")
    importance = models[0.50].feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    for name, imp in feat_imp[:30]:
        print(f"  {name:50s}  {imp:.0f}")

    # 6. Interval width analysis
    print(f"\n--- Interval Widths ---")
    p90_width = quantile_preds[0.90] - quantile_preds[0.10]
    p50_width = quantile_preds[0.75] - quantile_preds[0.25]
    print(f"  80% interval (p10-p90): mean={p90_width.mean():.2f}, std={p90_width.std():.2f}")
    print(f"  50% interval (p25-p75): mean={p50_width.mean():.2f}, std={p50_width.std():.2f}")

    # 7. Quantile crossing check
    crossings = 0
    for i in range(len(QUANTILES) - 1):
        q_lo = QUANTILES[i]
        q_hi = QUANTILES[i + 1]
        crossings += int(np.sum(quantile_preds[q_hi] < quantile_preds[q_lo]))
    print(f"\n  Quantile crossings: {crossings} / {len(y_test) * (len(QUANTILES)-1)}")

    # Save results
    if args.out_dir:
        out_dir = Path(str(args.out_dir)).expanduser().resolve()
    else:
        out_dir = ds_dir.parent / f"lgbm_fpts_quantile_v1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    for q in QUANTILES:
        models[q].save_model(str(out_dir / f"model_q{int(q*100):02d}.lgb"))
    mean_model.save_model(str(out_dir / "model_mean.lgb"))

    # Save test predictions
    test_out = test[JOIN_KEYS + ["dk_fpts", "played"]].copy()
    test_out["pred_mean"] = mean_pred
    for q in QUANTILES:
        test_out[f"pred_q{int(q*100):02d}"] = quantile_preds[q]
    test_out.to_parquet(out_dir / "test_predictions.parquet", index=False)

    # Save feature columns
    (out_dir / "feature_columns.json").write_text(
        json.dumps({"columns": feature_cols}, indent=2), encoding="utf-8"
    )

    # Save report
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(ds_dir),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_features": len(feature_cols),
        "active_only": bool(args.active_only),
        "quantiles": QUANTILES,
        "mean_mae": mean_mae,
        "median_mae": median_mae,
        "calibration": cal,
        "train_times": {f"q{int(q*100):02d}": t for q, t in train_times.items()},
        "best_iterations": {f"q{int(q*100):02d}": models[q].best_iteration for q in QUANTILES},
        "crossings": crossings,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
