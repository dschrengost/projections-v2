"""
Train ownership_v1 LightGBM model.

Inputs:
    gold/ownership_training_base/ownership_training_base.parquet

Outputs (per run_id):
    artifacts/ownership_v1/runs/{run_id}/
        model.txt
        feature_cols.json
        meta.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from projections.paths import data_path
from projections.ownership_v1.calibration import SoftmaxCalibrator
from projections.ownership_v1.features import (
    OWNERSHIP_FEATURES,
    OWNERSHIP_FEATURES_V2,
    OWNERSHIP_FEATURES_V3,
    OWNERSHIP_FEATURES_V4,
    OWNERSHIP_FEATURES_V5,
    OWNERSHIP_FEATURES_V6,
)


BASE_PARAMS: dict[str, object] = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 32,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 100,
    "max_depth": -1,
    "lambda_l2": 1.0,
    "verbose": -1,
}

# Stronger regularization for limited data (~5k samples)
REGULARIZED_PARAMS: dict[str, object] = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 16,            # Down from 32
    "learning_rate": 0.01,       # Down from 0.05
    "feature_fraction": 0.6,     # Down from 0.8
    "bagging_fraction": 0.6,     # Down from 0.8
    "bagging_freq": 1,
    "min_data_in_leaf": 200,     # Up from 100
    "max_depth": 5,              # Explicit limit
    "lambda_l1": 0.5,            # Add L1
    "lambda_l2": 5.0,            # Up from 1.0
    "verbose": -1,
}


def load_training_base(path: Path) -> pd.DataFrame:
    """Load ownership training base."""
    if not path.exists():
        raise FileNotFoundError(f"Training base not found: {path}")
    return pd.read_parquet(path)


def prepare_features(
    df: pd.DataFrame,
    compute_historical: bool = True,
    compute_slate_features: bool = False,
    compute_value_leverage: bool = False,
    compute_player_popularity: bool = False,
    compute_interactions: bool = False,
) -> pd.DataFrame:
    """Compute ownership features from training data."""
    result = df.copy()
    
    # Sort by date for historical features
    result = result.sort_values(["game_date", "slate_id"]).reset_index(drop=True)
    
    # Value per $1k (already in data, but recompute for safety)
    if "value_per_k" not in result.columns or result["value_per_k"].isna().any():
        result["value_per_k"] = result["proj_fpts"] / (result["salary"] / 1000.0)
    
    # Salary tiers
    result["is_value_tier"] = (result["salary"] < 5000).astype(int)
    result["is_mid_tier"] = ((result["salary"] >= 5000) & (result["salary"] < 7500)).astype(int)
    result["is_high_tier"] = (result["salary"] >= 7500).astype(int)
    
    # Position one-hot encoding
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        result[f"pos_{pos}"] = result["pos"].str.contains(pos, na=False).astype(int)
    
    # Within-slate ranks (1 = highest)
    result["salary_rank"] = result.groupby("slate_id")["salary"].rank(
        ascending=False, method="min"
    )
    result["proj_fpts_rank"] = result.groupby("slate_id")["proj_fpts"].rank(
        ascending=False, method="min"
    )
    
    # Z-score within slate
    slate_means = result.groupby("slate_id")["proj_fpts"].transform("mean")
    slate_stds = result.groupby("slate_id")["proj_fpts"].transform("std")
    result["proj_fpts_zscore"] = (result["proj_fpts"] - slate_means) / slate_stds.replace(0, 1)
    
    # Injury features - fill missing with 0
    if "player_is_out" not in result.columns:
        result["player_is_out"] = 0
    if "player_is_questionable" not in result.columns:
        result["player_is_questionable"] = 0
    if "team_outs_count" not in result.columns:
        result["team_outs_count"] = 0
    
    result["player_is_out"] = result["player_is_out"].fillna(0).astype(int)
    result["player_is_questionable"] = result["player_is_questionable"].fillna(0).astype(int)
    result["team_outs_count"] = result["team_outs_count"].fillna(0).astype(int)
    
    # Historical ownership: player's average ownership over prior slates
    if compute_historical:
        print("  Computing historical ownership features...")
        result["player_own_avg_10"] = compute_historical_ownership(result, window=10)
    else:
        result["player_own_avg_10"] = np.nan
    
    # Slate structure features
    if compute_slate_features:
        print("  Computing slate-structure features...")
        result = compute_slate_structure_features(result)

    # Value leverage features
    if compute_value_leverage:
        print("  Computing value leverage features...")
        result = compute_value_leverage_features(result)

    # Player popularity features
    if compute_player_popularity:
        print("  Computing player popularity features...")
        result = compute_player_popularity_features(result)

    # Interaction features
    if compute_interactions:
        print("  Computing interaction features...")
        result = compute_interaction_features(result)

    return result


def compute_value_leverage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute value leverage features (relative to slate).

    Features:
    - value_vs_slate_avg: value_per_k minus slate average
    - salary_vs_median: (salary - slate median) / slate median
    - is_min_priced_by_pos: cheapest at position on this slate
    - game_count_on_slate: number of games on slate
    """
    result = df.copy()

    # Value vs slate average
    slate_value_avg = result.groupby("slate_id")["value_per_k"].transform("mean")
    result["value_vs_slate_avg"] = result["value_per_k"] - slate_value_avg

    # Salary vs median
    slate_salary_median = result.groupby("slate_id")["salary"].transform("median")
    result["salary_vs_median"] = (result["salary"] - slate_salary_median) / slate_salary_median

    # Is min-priced by position (cheapest at this position on slate)
    def is_min_by_pos(group):
        # For each position the player has, check if they're the cheapest
        min_by_pos = {}
        for _, row in group.iterrows():
            positions = [p for p in ["PG", "SG", "SF", "PF", "C"] if row.get(f"pos_{p}", 0) == 1]
            if not positions:
                positions = [row.get("pos", "")]
            for pos in positions:
                if pos not in min_by_pos or row["salary"] < min_by_pos[pos]:
                    min_by_pos[pos] = row["salary"]

        is_min = []
        for _, row in group.iterrows():
            positions = [p for p in ["PG", "SG", "SF", "PF", "C"] if row.get(f"pos_{p}", 0) == 1]
            if not positions:
                positions = [row.get("pos", "")]
            is_min.append(any(row["salary"] == min_by_pos.get(pos, float("inf")) for pos in positions))
        return pd.Series(is_min, index=group.index, dtype=int)

    result["is_min_priced_by_pos"] = result.groupby("slate_id", group_keys=False).apply(
        is_min_by_pos, include_groups=False
    )

    # Game count on slate (approximate from unique matchups if available)
    if "matchup" in result.columns:
        result["game_count_on_slate"] = result.groupby("slate_id")["matchup"].transform("nunique")
    else:
        # Fallback: estimate from team count / 2
        result["game_count_on_slate"] = result.groupby("slate_id")["team"].transform("nunique") // 2

    return result


def compute_player_popularity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute player popularity features from historical ownership.

    Features:
    - player_own_median: median historical ownership
    - player_own_variance: ownership volatility (std)
    - player_chalk_rate: % of slates with >30% ownership
    """
    result = df.copy()

    # Sort by date for proper temporal ordering
    result = result.sort_values(["game_date", "slate_id"]).reset_index(drop=True)

    def player_stats(group):
        group = group.sort_values("game_date")
        # Expanding statistics with shift to avoid leakage
        median = group["actual_own_pct"].expanding().median().shift(1)
        variance = group["actual_own_pct"].expanding().std().shift(1)
        # Chalk rate: % of prior slates with >30% ownership
        is_chalk = (group["actual_own_pct"] > 30).astype(float)
        chalk_rate = is_chalk.expanding().mean().shift(1)
        return pd.DataFrame({
            "player_own_median": median,
            "player_own_variance": variance,
            "player_chalk_rate": chalk_rate,
        }, index=group.index)

    pop_features = result.groupby("player_id", group_keys=False).apply(
        player_stats, include_groups=False
    )

    # Fill NaN with overall averages for players with no history
    overall_median = result["actual_own_pct"].median()
    overall_var = result["actual_own_pct"].std()
    overall_chalk_rate = (result["actual_own_pct"] > 30).mean()

    result["player_own_median"] = pop_features["player_own_median"].fillna(overall_median)
    result["player_own_variance"] = pop_features["player_own_variance"].fillna(overall_var)
    result["player_chalk_rate"] = pop_features["player_chalk_rate"].fillna(overall_chalk_rate)

    return result


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction features.

    Features:
    - value_x_value_tier: value_per_k * is_value_tier (punt play detection)
    - outs_x_salary_rank: team_outs_count * salary_rank (injury beneficiary)
    """
    result = df.copy()

    result["value_x_value_tier"] = result["value_per_k"] * result["is_value_tier"]
    result["outs_x_salary_rank"] = result["team_outs_count"] * result["salary_rank"]

    return result


def compute_slate_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute slate-level structure features."""
    result = df.copy()
    
    # Slate size
    result["slate_size"] = result.groupby("slate_id")["player_id"].transform("count")
    
    # Salary structure
    result["slate_min_salary"] = result.groupby("slate_id")["salary"].transform("min")
    result["slate_max_salary"] = result.groupby("slate_id")["salary"].transform("max")
    result["salary_pct_of_max"] = result["salary"] / result["slate_max_salary"]
    result["is_min_salary"] = (result["salary"] == result["slate_min_salary"]).astype(int)
    
    # Count of near-min salary players (punt options)
    min_thresh = result["slate_min_salary"] + 200
    result["is_near_min"] = (result["salary"] <= min_thresh).astype(int)
    result["slate_near_min_count"] = result.groupby("slate_id")["is_near_min"].transform("sum")
    
    # Clean up temp columns
    result.drop(columns=["slate_min_salary", "slate_max_salary", "is_near_min"], inplace=True, errors="ignore")
    
    return result


def compute_historical_ownership(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Compute each player's average ownership over previous slates.
    
    Uses expanding window with shift to avoid data leakage.
    """
    # Get unique slates in chronological order
    slate_order = df.groupby("slate_id")["game_date"].first().sort_values()
    slate_to_idx = {slate: idx for idx, slate in enumerate(slate_order.index)}
    df["_slate_idx"] = df["slate_id"].map(slate_to_idx)
    
    # For each player, compute rolling average of prior slates
    # Group by player, sort by slate, compute expanding mean with shift
    def player_rolling_avg(group):
        group = group.sort_values("_slate_idx")
        # shift(1) to exclude current row, expanding mean of prior values
        return group["actual_own_pct"].expanding().mean().shift(1)
    
    result = df.groupby("player_id", group_keys=False).apply(
        player_rolling_avg, include_groups=False
    )
    
    # For players with no history, use overall mean
    overall_mean = df["actual_own_pct"].mean()
    result = result.fillna(overall_mean)
    
    df.drop(columns=["_slate_idx"], inplace=True, errors="ignore")
    
    return result


def split_by_season(
    df: pd.DataFrame,
    val_seasons: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by season for train/val."""
    val_mask = df["season"].isin(val_seasons)
    return df[~val_mask].copy(), df[val_mask].copy()


def split_by_date(
    df: pd.DataFrame,
    val_start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date for train/val.

    Args:
        df: DataFrame with game_date column
        val_start_date: Start date for validation (YYYY-MM-DD)

    Returns:
        (train_df, val_df) tuple
    """
    val_mask = df["game_date"] >= val_start_date
    return df[~val_mask].copy(), df[val_mask].copy()


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target: str = "actual_own_pct",
    params: dict | None = None,
) -> tuple[lgb.Booster, dict]:
    """Train LightGBM model with early stopping."""
    params = params or BASE_PARAMS.copy()
    
    # Clean data
    train_clean = train_df.dropna(subset=features + [target])
    val_clean = val_df.dropna(subset=features + [target])
    
    print(f"Training rows: {len(train_clean):,} (dropped {len(train_df) - len(train_clean):,} with NaN)")
    print(f"Validation rows: {len(val_clean):,} (dropped {len(val_df) - len(val_clean):,} with NaN)")
    
    X_train = train_clean[features]
    y_train = train_clean[target]
    X_val = val_clean[features]
    y_val = val_clean[target]
    
    # Transform target if requested
    y_train_raw = y_train.copy()
    y_val_raw = y_val.copy()
    
    if params.get("target_transform") == "logit":
        print("Applying logit transformation to target...")
        # Clip to avoid inf
        y_train = np.clip(y_train / 100.0, 0.005, 0.995)
        y_val = np.clip(y_val / 100.0, 0.005, 0.995)
        
        # Logit transform: log(p / (1-p))
        y_train = np.log(y_train / (1 - y_train))
        y_val = np.log(y_val / (1 - y_val))
    
    # Sample weighting
    train_weight = None
    if params.get("sample_weighting"):
        chalk_weight = params.get("chalk_weight", 3.0)  # Default 3x
        source_weight = params.get("source_weight", 2.0)  # DK data weight multiplier
        print(f"Applying sample weighting (chalk_weight={chalk_weight}x, source_weight={source_weight}x for DK)...")

        # Use raw target for weighting logic
        train_weight = np.ones(len(y_train))

        # Source-based weighting (DK data is higher quality)
        if "data_source" in train_clean.columns:
            dk_mask = train_clean["data_source"].values == "dk"
            train_weight[dk_mask] *= source_weight
            print(f"  DK source rows: {dk_mask.sum():,} (weighted {source_weight}x)")

        # Progressive weighting for high ownership (chalk)
        # More aggressive for higher ownership
        if chalk_weight > 1:
            # >30% ownership: 2x
            train_weight[y_train_raw > 30] *= 2.0
            # >40% ownership: chalk_weight * 0.5
            train_weight[y_train_raw > 40] *= chalk_weight * 0.5
            # >50% ownership: chalk_weight
            train_weight[y_train_raw > 50] *= chalk_weight
            # >60% ownership: chalk_weight * 1.5
            train_weight[y_train_raw > 60] *= chalk_weight * 1.5
            # >70% ownership: chalk_weight * 2
            train_weight[y_train_raw > 70] *= chalk_weight * 2.0

        # Also weight low ownership but less aggressively
        train_weight[y_train_raw < 5] *= 1.5

        print(f"  Weighted rows: {(train_weight > 1).sum():,} / {len(train_weight):,}")
        print(f"  >50% ownership weighted: {(y_train_raw > 50).sum():,} samples")

    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weight)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train regressor
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )
    
    # Train classifier if requested
    clf_model = None
    if params.get("chalk_classifier"):
        print("\nTraining Chalk Classifier (>50% ownership)...")
        # Binary target: 1 if > 50%, 0 otherwise
        y_train_clf = (y_train_raw > 50).astype(int)
        y_val_clf = (y_val_raw > 50).astype(int)
        
        # Weight positive class heavily (since it's rare ~1%)
        # Calculate ratio
        pos_ratio = y_train_clf.mean()
        scale_pos_weight = (1 - pos_ratio) / pos_ratio
        print(f"  Positive class ratio: {pos_ratio:.1%} -> scale_pos_weight: {scale_pos_weight:.1f}")
        
        clf_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 32,
            "learning_rate": 0.05,
            "scale_pos_weight": scale_pos_weight,  # Handle imbalance
            "verbose": -1,
        }
        seed = int(params.get("seed", 1337))
        num_threads = int(params.get("num_threads", 1))
        clf_params.update(
            {
                "seed": seed,
                "feature_fraction_seed": seed,
                "bagging_seed": seed,
                "data_random_seed": seed,
                "deterministic": True,
                "force_row_wise": True,
                "num_threads": num_threads,
            }
        )
        
        train_data_clf = lgb.Dataset(X_train, label=y_train_clf)
        val_data_clf = lgb.Dataset(X_val, label=y_val_clf, reference=train_data_clf)
        
        clf_model = lgb.train(
            clf_params,
            train_data_clf,
            num_boost_round=500,
            valid_sets=[train_data_clf, val_data_clf],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50),
            ],
        )
    
    # Compute metrics
    train_preds = model.predict(X_train, num_iteration=model.best_iteration)
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Apply classifier boost if available
    if clf_model is not None:
        print("\nApplying chalk boost...")
        # Get probabilities
        val_probs = clf_model.predict(X_val, num_iteration=clf_model.best_iteration)
        
        # Boost logic: if P(chalk) > 0.7, ensure prediction is at least 40%
        # And blend it upwards
        high_conf_mask = val_probs > 0.7
        print(f"  Boosting {high_conf_mask.sum()} predictions (P > 0.7)")
        
        # Simple boost: max(pred, 40) for high confidence
        # Or maybe: pred = pred * (1 + prob)
        # Let's try: max(pred, 40) to force it into chalk territory
        val_preds[high_conf_mask] = np.maximum(val_preds[high_conf_mask], 40.0)
    
    # Inverse transform if needed
    if params.get("target_transform") == "logit":
        # Sigmoid: 1 / (1 + exp(-x))
        train_preds = 1 / (1 + np.exp(-train_preds))
        val_preds = 1 / (1 + np.exp(-val_preds))
        
        # Scale back to 0-100
        train_preds *= 100.0
        val_preds *= 100.0
        
        # Restore raw targets for metric computation
        y_train = y_train_raw
        y_val = y_val_raw
    
    # Compute correlation safely (handle index mismatch and zero variance)
    if len(val_preds) > 1 and y_val.std() > 0:
        val_corr = float(np.corrcoef(val_preds, y_val.values)[0, 1])
    else:
        val_corr = float('nan')

    metrics = {
        "train_mae": float(np.abs(train_preds - y_train.values).mean()),
        "train_rmse": float(np.sqrt(((train_preds - y_train.values) ** 2).mean())),
        "val_mae": float(np.abs(val_preds - y_val.values).mean()),
        "val_rmse": float(np.sqrt(((val_preds - y_val.values) ** 2).mean())),
        "val_corr": val_corr,
        "best_iteration": model.best_iteration,
        "num_features": len(features),
    }
    
    print("\n--- Metrics ---")
    print(f"Train MAE: {metrics['train_mae']:.3f}%")
    print(f"Val MAE: {metrics['val_mae']:.3f}%")
    corr_str = f"{metrics['val_corr']:.3f}" if not np.isnan(metrics['val_corr']) else "NaN (check val set)"
    print(f"Val Correlation: {corr_str}")
    print(f"Best iteration: {metrics['best_iteration']}")
    
    return model, metrics


def write_artifacts(
    run_dir: Path,
    model: lgb.Booster,
    features: list[str],
    metrics: dict,
    params: dict,
    train_seasons: list[int],
    val_seasons: list[int],
) -> None:
    """Write model artifacts to run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    model.save_model(str(run_dir / "model.txt"))
    
    # Feature columns
    (run_dir / "feature_cols.json").write_text(
        json.dumps({"feature_cols": features}, indent=2)
    )
    
    # Metadata
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": features,
        "params": params,
        "metrics": metrics,
        "train_seasons": train_seasons,
        "val_seasons": val_seasons,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    
    print(f"\nArtifacts written to: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train ownership_v1 model")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for artifacts (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--training-base",
        type=Path,
        default=None,
        help="Path to training base parquet",
    )
    parser.add_argument(
        "--val-seasons",
        type=int,
        nargs="+",
        default=[2025],
        help="Seasons to use for validation (default: 2025)",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["v1", "v2", "v3", "v4", "v5", "v6"],
        default="v2",
        help="Feature set: v1=base, v2=+injuries, v3=+historical, v4=+slate, v5=+value_leverage, v6=+popularity+interactions (default: v2)",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=32,
        help="LightGBM num_leaves",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="LightGBM learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for deterministic training (default: 1337)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="LightGBM num_threads (default: 1 for determinism)",
    )
    parser.add_argument(
        "--target-transform",
        type=str,
        choices=["none", "logit"],
        default="none",
        help="Target transformation (default: none)",
    )
    parser.add_argument(
        "--fit-calibrator",
        action="store_true",
        help="Fit and save a softmax calibrator (per-slate sum constraint) from training predictions",
    )
    parser.add_argument(
        "--target-sum-pct",
        type=float,
        default=800.0,
        help="Target slate sum in percent (DK classic=800, FD classic=900; default: 800)",
    )
    parser.add_argument(
        "--sample-weighting",
        action="store_true",
        help="Apply sample weighting (progressive for chalk plays)",
    )
    parser.add_argument(
        "--chalk-weight",
        type=float,
        default=3.0,
        help="Weight multiplier for chalk plays (used with --sample-weighting, default: 3.0)",
    )
    parser.add_argument(
        "--source-weight",
        type=float,
        default=2.0,
        help="Weight multiplier for DK source data (used with --sample-weighting, default: 2.0)",
    )
    parser.add_argument(
        "--chalk-classifier",
        action="store_true",
        help="Train a secondary classifier to boost chalk predictions",
    )
    parser.add_argument(
        "--use-merged-base",
        action="store_true",
        help="Use merged Linestar+DK training base instead of Linestar-only",
    )
    parser.add_argument(
        "--val-start-date",
        type=str,
        default=None,
        help="Use date-based split instead of season. Format: YYYY-MM-DD",
    )
    parser.add_argument(
        "--regularized",
        action="store_true",
        help="Use stronger regularization params (recommended for <5k samples)",
    )

    args = parser.parse_args()
    
    # Defaults
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.training_base is None:
        if args.use_merged_base:
            args.training_base = data_path() / "gold" / "ownership_merged_base" / "ownership_merged_base.parquet"
        else:
            args.training_base = data_path() / "gold" / "ownership_training_base" / "ownership_training_base.parquet"
    
    # Select feature set
    feature_sets = {
        "v1": OWNERSHIP_FEATURES,
        "v2": OWNERSHIP_FEATURES_V2,
        "v3": OWNERSHIP_FEATURES_V3,
        "v4": OWNERSHIP_FEATURES_V4,
        "v5": OWNERSHIP_FEATURES_V5,
        "v6": OWNERSHIP_FEATURES_V6,
    }
    features = list(feature_sets[args.feature_set])

    print(f"Run ID: {args.run_id}")
    print(f"Training base: {args.training_base}")
    print(f"Validation seasons: {args.val_seasons}")
    print(f"Feature set: {args.feature_set} ({len(features)} features)")

    # Load and prepare data
    print("\nLoading training data...")
    df = load_training_base(args.training_base)
    print(f"Loaded {len(df):,} rows")

    print("\nPreparing features...")
    compute_historical = args.feature_set in ["v3", "v4", "v5", "v6"]
    compute_slate = args.feature_set in ["v4", "v5", "v6"]
    compute_value_leverage = args.feature_set in ["v5", "v6"]
    compute_player_popularity = args.feature_set == "v6"
    compute_interactions = args.feature_set == "v6"
    df = prepare_features(
        df,
        compute_historical=compute_historical,
        compute_slate_features=compute_slate,
        compute_value_leverage=compute_value_leverage,
        compute_player_popularity=compute_player_popularity,
        compute_interactions=compute_interactions,
    )
    
    # Split train/val
    if args.val_start_date:
        # Date-based split (recommended for merged data)
        print(f"\nUsing date-based split: val >= {args.val_start_date}")
        train_df, val_df = split_by_date(df, args.val_start_date)
        val_seasons = sorted(val_df["season"].unique().tolist())
        train_seasons = sorted(train_df["season"].unique().tolist())
        print(f"Train: {train_df['game_date'].min()} to {train_df['game_date'].max()} ({len(train_df):,} rows)")
        print(f"Val: {val_df['game_date'].min()} to {val_df['game_date'].max()} ({len(val_df):,} rows)")

        # Show data source distribution
        if "data_source" in train_df.columns:
            print(f"Train sources: {train_df['data_source'].value_counts().to_dict()}")
            print(f"Val sources: {val_df['data_source'].value_counts().to_dict()}")
    else:
        # Season-based split (legacy)
        # If 2025 is empty (e.g. filtered out because no contest files), use 2024 as val
        if df[df['season'] == 2025].empty:
            print("  Warning: 2025 season empty (likely filtered). Using 2024 as validation.")
            train_df = df[df['season'] < 2024].copy()
            val_df = df[df['season'] == 2024].copy()
            val_seasons = [2024]
        else:
            train_df = df[df['season'] != 2025].copy()
            val_df = df[df['season'] == 2025].copy()
            val_seasons = args.val_seasons

        train_seasons = sorted(train_df["season"].unique().tolist())
        print(f"\nTrain seasons: {train_seasons} ({len(train_df):,} rows)")
        print(f"Val seasons: {val_seasons} ({len(val_df):,} rows)")
    
    print(f"\nFeatures ({len(features)}): {features}")
    
    # Train
    if args.regularized:
        print("Using REGULARIZED_PARAMS (stronger regularization for limited data)")
        params = REGULARIZED_PARAMS.copy()
    else:
        params = BASE_PARAMS.copy()
        params["num_leaves"] = args.num_leaves
        params["learning_rate"] = args.learning_rate
    params["target_transform"] = args.target_transform
    params["sample_weighting"] = args.sample_weighting
    params["chalk_weight"] = args.chalk_weight
    params["source_weight"] = args.source_weight
    params["chalk_classifier"] = args.chalk_classifier
    params["seed"] = args.seed
    params["feature_fraction_seed"] = args.seed
    params["bagging_seed"] = args.seed
    params["data_random_seed"] = args.seed
    params["deterministic"] = True
    params["force_row_wise"] = True
    params["num_threads"] = args.num_threads
    
    print("\nTraining model...")
    model, metrics = train_model(train_df, val_df, features, params=params)
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\n--- Feature Importance ---")
    print(importance.to_string(index=False))
    
    # Write artifacts
    run_dir = data_path() / "artifacts" / "ownership_v1" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    if not val_df.empty:
        print("\nSaving validation predictions...")
        val_preds = model.predict(val_df[features], num_iteration=model.best_iteration)
        
        # Apply inverse transform if needed
        if params.get("target_transform") == "logit":
            val_preds = 1 / (1 + np.exp(-val_preds))
            val_preds *= 100.0
            
        val_df = val_df.copy()
        val_df["pred_own_pct"] = val_preds
        val_df.to_csv(run_dir / "val_predictions.csv", index=False)

    if args.fit_calibrator:
        print("\nFitting softmax calibrator on training set...")
        train_scores = model.predict(train_df[features], num_iteration=model.best_iteration)
        if params.get("target_transform") == "logit":
            train_scores = 1 / (1 + np.exp(-train_scores))
            train_scores *= 100.0

        train_scores = np.clip(train_scores, 0.0, 100.0)
        calib_df = train_df[["slate_id", "actual_own_pct"]].copy()
        calib_df["pred_own_pct"] = train_scores

        calibrator = SoftmaxCalibrator().fit(
            calib_df,
            score_col="pred_own_pct",
            target_col="actual_own_pct",
            slate_id_col="slate_id",
            R=float(args.target_sum_pct) / 100.0,
            verbose=True,
        )
        cal_path = run_dir / "calibrator.json"
        calibrator.save(cal_path)
        print(f"[calibration] Saved calibrator to: {cal_path}")
    
    write_artifacts(
        run_dir=run_dir,
        model=model,
        features=features,
        metrics=metrics,
        params=params,
        train_seasons=train_seasons,
        val_seasons=val_seasons, # Use the corrected val_seasons
    )
    
    print(f"\n✅ Training complete: {args.run_id}")


if __name__ == "__main__":
    main()
