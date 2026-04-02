"""
LightGBM rotation + minutes v2 — anti-smear edition.

Key differences from v1:
  - Starters are assumed known (from RotoWire), not predicted
  - Active prediction uses team-constrained top-K selection instead of
    independent thresholding
  - Rotation size is predicted at the team level first
  - Minutes allocation uses rank-aware share prediction to reduce bench smear

Usage:
    uv run python scripts/experiments/lgbm_rotation_minutes_v2.py \
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
    df: pd.DataFrame, *, val_frac: float = 0.15, test_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["game_date"].unique())
    n = len(dates)
    test_start = dates[int(n * (1 - test_frac))]
    val_start = dates[int(n * (1 - test_frac - val_frac))]
    return (
        df[df["game_date"] < val_start].copy(),
        df[(df["game_date"] >= val_start) & (df["game_date"] < test_start)].copy(),
        df[df["game_date"] >= test_start].copy(),
    )


def _build_team_context_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
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

    # Player's rank within team by prior minutes share
    df["tm_rank_prior_share"] = df.groupby(["game_id", "team_id"])["prior_minutes_share_20"].rank(
        ascending=False, method="first"
    )
    # How many teammates have higher play prob
    df["tm_rank_play_prob"] = df.groupby(["game_id", "team_id"])["prior_play_prob"].rank(
        ascending=False, method="first"
    )

    df["tm_n_roster"] = df.groupby(["game_id", "team_id"])["player_id"].transform("count")
    df["tm_sum_play_prob"] = df.groupby(["game_id", "team_id"])["prior_play_prob"].transform("sum")
    df["tm_n_with_props"] = df.groupby(["game_id", "team_id"])["an_implied_minutes"].transform(
        lambda x: (x > 0).sum()
    )

    # Bench minutes budget: 240 - sum of starter implied minutes
    # (In production we know starters; here approximate with top-5 by prior share)
    def _starter_minutes_budget(grp: pd.DataFrame) -> float:
        top5 = grp.nlargest(5, "prior_minutes_share_20")["an_implied_minutes"]
        return float(top5.sum())

    starter_budget = df.groupby(["game_id", "team_id"]).apply(
        _starter_minutes_budget, include_groups=False
    ).rename("tm_starter_implied_minutes")
    df = df.merge(starter_budget, on=["game_id", "team_id"], how="left")
    df["tm_bench_budget"] = 240.0 - df["tm_starter_implied_minutes"].fillna(0.0)

    return df


def _simplex_project(
    minutes: np.ndarray, team_ids: np.ndarray, game_ids: np.ndarray,
    target_total: float = 240.0, max_per_player: float = 48.0,
) -> np.ndarray:
    result = minutes.copy()
    index_map: dict[tuple, list[int]] = {}
    for i, k in enumerate(zip(game_ids, team_ids)):
        index_map.setdefault(k, []).append(i)
    for idxs in index_map.values():
        team_mins = result[idxs]
        total = team_mins.sum()
        if total > 1e-6:
            team_mins = np.clip(team_mins * (target_total / total), 0, max_per_player)
            clipped_total = team_mins.sum()
            if abs(clipped_total - target_total) > 0.1:
                unclipped = team_mins < max_per_player
                if unclipped.any():
                    deficit = target_total - team_mins[~unclipped].sum()
                    team_mins[unclipped] *= deficit / team_mins[unclipped].sum()
            result[idxs] = team_mins
    return result


def _train_model(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    *, objective: str, metric: str, alpha: float = 0.5,
    n_estimators: int = 2000, lr: float = 0.05,
    num_leaves: int = 127, min_child_samples: int = 50,
    scale_pos_weight: float = 1.0,
) -> lgb.Booster:
    params = {
        "objective": objective,
        "metric": metric,
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
    if objective == "binary":
        params["scale_pos_weight"] = scale_pos_weight
    if objective == "quantile":
        params["alpha"] = alpha
    ds_train = lgb.Dataset(X_train, label=y_train)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
    return lgb.train(
        params, ds_train,
        num_boost_round=n_estimators,
        valid_sets=[ds_val], valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )


def _select_active_top_k(
    p_active: np.ndarray,
    game_ids: np.ndarray,
    team_ids: np.ndarray,
    is_starter: np.ndarray,
    rotation_size: dict[tuple, int],
) -> np.ndarray:
    """Select top-K non-starter players by P(active) per team, where K is
    predicted rotation size minus number of starters.

    Starters are always active.
    """
    result = is_starter.copy().astype(int)
    index_map: dict[tuple, list[int]] = {}
    for i, k in enumerate(zip(game_ids, team_ids)):
        index_map.setdefault(k, []).append(i)

    for key, idxs in index_map.items():
        n_rotation = rotation_size.get(key, 10)
        n_starters = int(is_starter[idxs].sum())
        bench_slots = max(0, n_rotation - n_starters)
        # Among non-starters, take top bench_slots by P(active)
        bench_idxs = [i for i in idxs if is_starter[i] == 0]
        if bench_idxs and bench_slots > 0:
            bench_probs = [(i, p_active[i]) for i in bench_idxs]
            bench_probs.sort(key=lambda x: -x[1])
            for i, _ in bench_probs[:bench_slots]:
                result[i] = 1

    return result


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
    min_cols = [c for c in ["game_id", "team_id", "player_id", "game_date", "minutes_label"]
                if c in labels_min.columns]
    df = feat.merge(labels_min[min_cols], on=JOIN_KEYS, how="inner")
    if "minutes_label" not in df.columns:
        df = df.merge(
            labels_box[["game_id", "team_id", "player_id", "game_date", "minutes"]].rename(
                columns={"minutes": "minutes_label"}),
            on=JOIN_KEYS, how="left",
        )
    df["minutes_actual"] = pd.to_numeric(df["minutes_label"], errors="coerce").fillna(0.0)
    df["played_actual"] = (df["minutes_actual"] > 0).astype(int)

    # Derive starters from top-5 minutes (simulates having RotoWire starters)
    df["starter_actual"] = 0
    for (gid, tid), grp in df[df["played_actual"] == 1].groupby(["game_id", "team_id"]):
        top5 = grp.nlargest(5, "minutes_actual").index
        df.loc[top5, "starter_actual"] = 1

    # Derive "known starter" feature: in production this comes from RotoWire.
    # For training, we use lineup_starter_announced + started_proxy_rate as proxy.
    # The key insight: starters are given, so we can use is_starter as a feature.
    df["is_known_starter"] = 0
    # Simulate RotoWire: use started_proxy_rate_prior_5 >= 0.8 as "announced starter"
    if "started_proxy_rate_prior_5" in df.columns:
        df["is_known_starter"] = (
            pd.to_numeric(df["started_proxy_rate_prior_5"], errors="coerce").fillna(0.0) >= 0.8
        ).astype(int)
    if "lineup_starter_announced" in df.columns:
        announced = pd.to_numeric(df["lineup_starter_announced"], errors="coerce").fillna(0.0)
        df["is_known_starter"] = df["is_known_starter"] | (announced >= 1).astype(int)

    # Actual rotation size per team-game
    rotation_actual = df[df["played_actual"] == 1].groupby(
        ["game_id", "team_id"]
    )["player_id"].count().reset_index(name="rotation_size_actual")
    df = df.merge(rotation_actual, on=["game_id", "team_id"], how="left")

    print(f"Rows: {len(df)}, Games: {df['game_id'].nunique()}")
    print(f"Active rate: {df['played_actual'].mean():.1%}")
    print(f"Known-starter rate: {df['is_known_starter'].mean():.1%}")
    print(f"Rotation size: mean={rotation_actual['rotation_size_actual'].mean():.1f}, "
          f"std={rotation_actual['rotation_size_actual'].std():.1f}")

    # Features
    feature_cols = _select_feature_cols(feat)
    df = _build_team_context_features(df, feature_cols)
    extra = [c for c in df.columns if c.startswith("tm_") or c.startswith("opp_sum_")]
    feature_cols = feature_cols + sorted(extra)
    # Add is_known_starter as a feature (this is a production input, not a label)
    feature_cols = feature_cols + ["is_known_starter"]
    print(f"Features: {len(feature_cols)}")

    # Split
    train, val, test = _time_split(df)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # =========================================================================
    # STAGE 1: Rotation size per team-game
    # =========================================================================
    print(f"\n{'='*70}")
    print("STAGE 1: Rotation size prediction (team-level)")
    print(f"{'='*70}")

    # Build team-level features
    team_agg_features = [
        "tm_n_roster", "tm_sum_play_prob", "tm_n_with_props",
        "tm_bench_budget", "tm_starter_implied_minutes",
    ]
    # Also add game-level features
    game_features = ["total", "spread_home", "blowout_index", "is_b2b"]
    team_level_cols = []
    for col in team_agg_features + game_features:
        if col in df.columns:
            team_level_cols.append(col)

    # Aggregate to team-game level
    team_df = df.groupby(["game_id", "team_id", "game_date"]).agg(
        rotation_size_actual=("rotation_size_actual", "first"),
        n_roster=("player_id", "count"),
        sum_play_prob=("prior_play_prob", "sum"),
        mean_play_prob=("prior_play_prob", "mean"),
        sum_implied_minutes=("an_implied_minutes", "sum"),
        n_with_props=("an_implied_minutes", lambda x: (x > 0).sum()),
        n_starters=("is_known_starter", "sum"),
        sum_starter_minutes=("an_implied_minutes", lambda x: x[df.loc[x.index, "is_known_starter"] == 1].sum()),
        total=("total", "first"),
        spread_home=("spread_home", "first"),
        is_b2b=("is_b2b", "first"),
        team_pace=("team_pace_szn", "first"),
    ).reset_index()
    team_df["bench_budget"] = 240.0 - team_df["sum_starter_minutes"]
    team_df["bench_n"] = team_df["n_roster"] - team_df["n_starters"]

    team_feat_cols = ["n_roster", "sum_play_prob", "mean_play_prob",
                      "sum_implied_minutes", "n_with_props", "n_starters",
                      "sum_starter_minutes", "bench_budget", "bench_n",
                      "total", "spread_home", "is_b2b", "team_pace"]
    team_feat_cols = [c for c in team_feat_cols if c in team_df.columns]

    team_train, team_val, team_test = _time_split(team_df)

    rot_model = _train_model(
        team_train[team_feat_cols], team_train["rotation_size_actual"],
        team_val[team_feat_cols], team_val["rotation_size_actual"],
        objective="regression", metric="mae",
        n_estimators=args.n_estimators, lr=args.lr, num_leaves=31,
    )
    team_test["pred_rotation_size"] = rot_model.predict(team_test[team_feat_cols])
    # Round to nearest integer, clip to [8, 15]
    team_test["pred_rotation_int"] = np.clip(np.round(team_test["pred_rotation_size"]), 8, 15).astype(int)

    rot_mae = float(np.mean(np.abs(
        team_test["rotation_size_actual"].values - team_test["pred_rotation_size"].values
    )))
    rot_exact = float((team_test["pred_rotation_int"] == team_test["rotation_size_actual"]).mean())
    rot_within1 = float((
        (team_test["pred_rotation_int"] - team_test["rotation_size_actual"]).abs() <= 1
    ).mean())
    print(f"\n  Rotation size MAE:    {rot_mae:.2f}")
    print(f"  Exact match:          {rot_exact:.1%}")
    print(f"  Within ±1:            {rot_within1:.1%}")
    print(f"  Actual dist: {team_test['rotation_size_actual'].value_counts().sort_index().to_dict()}")
    print(f"  Pred dist:   {team_test['pred_rotation_int'].value_counts().sort_index().to_dict()}")

    # =========================================================================
    # STAGE 2: P(active | bench) — rank-aware bench selection
    # =========================================================================
    print(f"\n{'='*70}")
    print("STAGE 2: P(active) for bench players — rank-aware selection")
    print(f"{'='*70}")

    # For bench players only (starters are always active)
    bench_mask_train = train["is_known_starter"] == 0
    bench_mask_val = val["is_known_starter"] == 0
    bench_mask_test = test["is_known_starter"] == 0

    bench_active_model = _train_model(
        train.loc[bench_mask_train, feature_cols], train.loc[bench_mask_train, "played_actual"],
        val.loc[bench_mask_val, feature_cols], val.loc[bench_mask_val, "played_actual"],
        objective="binary", metric="auc",
        n_estimators=args.n_estimators, lr=args.lr, num_leaves=63,
    )

    # Predict P(active) for all test bench players
    test_bench_p_active = np.zeros(len(test))
    test_bench_p_active[bench_mask_test.values] = bench_active_model.predict(
        test.loc[bench_mask_test, feature_cols]
    )
    # Starters get P(active) = 1
    test_bench_p_active[~bench_mask_test.values] = 1.0

    from sklearn.metrics import roc_auc_score
    bench_auc = roc_auc_score(
        test.loc[bench_mask_test, "played_actual"],
        test_bench_p_active[bench_mask_test.values],
    )
    print(f"\n  Bench P(active) AUC: {bench_auc:.4f}")

    # Method A: Independent threshold (v1 baseline)
    from sklearn.metrics import f1_score
    val_bench_p = bench_active_model.predict(val.loc[bench_mask_val, feature_cols])
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.1, 0.9, 0.02):
        f1 = f1_score(val.loc[bench_mask_val, "played_actual"], (val_bench_p >= thr).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    test_active_indep = np.where(
        test["is_known_starter"].values == 1, 1,
        (test_bench_p_active >= best_thr).astype(int)
    )

    # Method B: Top-K selection using predicted rotation size
    rotation_lookup = dict(zip(
        zip(team_test["game_id"], team_test["team_id"]),
        team_test["pred_rotation_int"],
    ))
    test_active_topk = _select_active_top_k(
        test_bench_p_active,
        test["game_id"].values,
        test["team_id"].values,
        test["is_known_starter"].values,
        rotation_lookup,
    )

    # Compare methods
    for method_name, pred_active in [("Independent threshold", test_active_indep),
                                      ("Top-K constrained", test_active_topk)]:
        tp = int(((pred_active == 1) & (test["played_actual"].values == 1)).sum())
        fp = int(((pred_active == 1) & (test["played_actual"].values == 0)).sum())
        fn = int(((pred_active == 0) & (test["played_actual"].values == 1)).sum())
        tn = int(((pred_active == 0) & (test["played_actual"].values == 0)).sum())
        f1 = f1_score(test["played_actual"], pred_active)
        pred_rate = pred_active.mean()

        team_eval = pd.DataFrame({
            "game_id": test["game_id"].values,
            "team_id": test["team_id"].values,
            "actual": test["played_actual"].values,
            "pred": pred_active,
        })
        team_counts = team_eval.groupby(["game_id", "team_id"]).agg(
            actual_n=("actual", "sum"), pred_n=("pred", "sum"),
        )
        count_mae = float((team_counts["pred_n"] - team_counts["actual_n"]).abs().mean())
        count_bias = float((team_counts["pred_n"] - team_counts["actual_n"]).mean())

        print(f"\n  {method_name}:")
        print(f"    F1={f1:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"    Pred active rate: {pred_rate:.1%} (actual: {test['played_actual'].mean():.1%})")
        print(f"    Team count: MAE={count_mae:.2f}, bias={count_bias:+.2f}")

    # =========================================================================
    # STAGE 3: Minutes | active — with anti-smear
    # =========================================================================
    print(f"\n{'='*70}")
    print("STAGE 3: Minutes prediction (active players only)")
    print(f"{'='*70}")

    train_active = train[train["played_actual"] == 1]
    val_active = val[val["played_actual"] == 1]
    test_active = test[test["played_actual"] == 1]

    # Mean model
    print("\nTraining minutes mean model...")
    minutes_model = _train_model(
        train_active[feature_cols], train_active["minutes_actual"],
        val_active[feature_cols], val_active["minutes_actual"],
        objective="regression", metric="mae",
        n_estimators=args.n_estimators, lr=args.lr,
    )

    # Quantile models
    minutes_q_models: dict[float, lgb.Booster] = {}
    for q in MINUTES_QUANTILES:
        print(f"Training minutes q{int(q*100):02d}...")
        minutes_q_models[q] = _train_model(
            train_active[feature_cols], train_active["minutes_actual"],
            val_active[feature_cols], val_active["minutes_actual"],
            objective="quantile", metric="quantile", alpha=q,
            n_estimators=args.n_estimators, lr=args.lr,
        )

    # =========================================================================
    # FULL PIPELINE COMPARISON
    # =========================================================================
    print(f"\n{'='*70}")
    print("FULL PIPELINE COMPARISON")
    print(f"{'='*70}")

    # Predict minutes for all test players
    all_minutes_raw = minutes_model.predict(test[feature_cols])
    all_minutes_q: dict[float, np.ndarray] = {}
    for q in MINUTES_QUANTILES:
        all_minutes_q[q] = minutes_q_models[q].predict(test[feature_cols])

    results: dict[str, dict[str, float]] = {}

    for method_name, pred_active in [("Independent thr", test_active_indep),
                                      ("Top-K constrained", test_active_topk)]:
        # Zero out inactive players
        minutes_masked = np.where(pred_active == 1, all_minutes_raw, 0.0)
        minutes_proj = _simplex_project(
            minutes_masked, test["team_id"].values, test["game_id"].values,
        )

        actual = test["minutes_actual"].values
        is_starter = test["starter_actual"].values == 1
        is_active = test["played_actual"].values == 1
        is_bench_active = is_active & ~is_starter
        is_dnp = ~is_active

        uncond_mae = float(np.mean(np.abs(actual - minutes_proj)))
        starter_mae = float(np.mean(np.abs(actual[is_starter] - minutes_proj[is_starter])))
        starter_bias = float(np.mean(minutes_proj[is_starter] - actual[is_starter]))
        bench_mae = float(np.mean(np.abs(actual[is_bench_active] - minutes_proj[is_bench_active])))
        bench_bias = float(np.mean(minutes_proj[is_bench_active] - actual[is_bench_active]))
        dnp_mae = float(np.mean(np.abs(actual[is_dnp] - minutes_proj[is_dnp])))
        dnp_bias = float(np.mean(minutes_proj[is_dnp] - actual[is_dnp]))

        print(f"\n  {method_name}:")
        print(f"    Unconditional MAE:  {uncond_mae:.2f}")
        print(f"    Starters:           MAE={starter_mae:.2f}  bias={starter_bias:+.2f}")
        print(f"    Bench (active):     MAE={bench_mae:.2f}  bias={bench_bias:+.2f}")
        print(f"    DNP:                MAE={dnp_mae:.2f}  bias={dnp_bias:+.2f}")

        results[method_name] = {
            "uncond_mae": uncond_mae,
            "starter_mae": starter_mae, "starter_bias": starter_bias,
            "bench_mae": bench_mae, "bench_bias": bench_bias,
            "dnp_mae": dnp_mae, "dnp_bias": dnp_bias,
        }

    # Baselines
    prior_share = test["prior_minutes_share_20"].fillna(0).values * 240.0
    props = test["an_implied_minutes"].fillna(0).values
    actual = test["minutes_actual"].values
    is_starter = test["starter_actual"].values == 1
    print(f"\n  Baselines:")
    print(f"    Prior share×240: uncond MAE={np.mean(np.abs(actual - prior_share)):.2f}, "
          f"starter bias={np.mean(prior_share[is_starter] - actual[is_starter]):+.2f}")
    print(f"    Props implied:   uncond MAE={np.mean(np.abs(actual - props)):.2f}, "
          f"starter bias={np.mean(props[is_starter] - actual[is_starter]):+.2f}")

    # Minutes quantile calibration for active players (top-K method)
    print(f"\n--- Minutes Quantile Calibration (Top-K, active only) ---")
    active_mask = test_active_topk == 1
    for q in MINUTES_QUANTILES:
        q_pred = all_minutes_q[q]
        cov = float(np.mean(actual[active_mask] <= q_pred[active_mask]))
        print(f"  q{int(q*100):02d}: coverage={cov:.4f} (target={q:.2f})")

    # Feature importance
    imp = minutes_model.feature_importance(importance_type="gain")
    top = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:15]
    print(f"\n--- Top 15 features (minutes) ---")
    for name, v in top:
        print(f"  {name:50s} {v:.0f}")

    imp = bench_active_model.feature_importance(importance_type="gain")
    top = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:15]
    print(f"\n--- Top 15 features (bench active) ---")
    for name, v in top:
        print(f"  {name:50s} {v:.0f}")

    # Save
    if args.out_dir:
        out_dir = Path(str(args.out_dir)).expanduser().resolve()
    else:
        out_dir = ds_dir.parent / f"lgbm_rotation_minutes_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_active_model.save_model(str(out_dir / "model_bench_active.lgb"))
    rot_model.save_model(str(out_dir / "model_rotation_size.lgb"))
    minutes_model.save_model(str(out_dir / "model_minutes_mean.lgb"))
    for q in MINUTES_QUANTILES:
        minutes_q_models[q].save_model(str(out_dir / f"model_minutes_q{int(q*100):02d}.lgb"))

    test_out = test[JOIN_KEYS + ["minutes_actual", "played_actual", "starter_actual",
                                  "is_known_starter", "prior_minutes_share_20",
                                  "an_implied_minutes"]].copy()
    test_out["p_active"] = test_bench_p_active
    test_out["pred_active_indep"] = test_active_indep
    test_out["pred_active_topk"] = test_active_topk
    test_out["pred_minutes_raw"] = all_minutes_raw
    for q in MINUTES_QUANTILES:
        test_out[f"pred_minutes_q{int(q*100):02d}"] = all_minutes_q[q]
    test_out.to_parquet(out_dir / "test_predictions.parquet", index=False)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(ds_dir),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_features": len(feature_cols),
        "rotation_size_mae": rot_mae,
        "bench_active_auc": bench_auc,
        "results": results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
