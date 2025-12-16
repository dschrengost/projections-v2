#!/usr/bin/env python3
"""
Predict ownership for a slate using the production ownership_v1 model.

Usage:
    python predict_ownership_live.py --slate-path /path/to/slate.csv

Input: CSV with columns: player_name, team, pos, salary, proj_fpts
       Optional: player_is_questionable, team_outs_count

Output: Same CSV with added 'pred_own_pct' column
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

# Paths
PROJECTIONS_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECTIONS_ROOT / "models" / "ownership_v1"
HISTORICAL_OWN_PATH = Path.home() / "projections-data" / "bronze" / "dk_contests" / "ownership_by_slate" / "all_ownership.parquet"

# Feature list (must match training)
FEATURE_COLS = [
    "value_per_k",
    "salary_rank",
    "proj_fpts_rank",
    "proj_fpts_zscore",
    "salary",
    "is_value_tier",
    "is_mid_tier",
    "is_high_tier",
    "pos_PG",
    "pos_SG",
    "pos_SF",
    "pos_PF",
    "pos_C",
    "player_is_questionable",
    "team_outs_count",
    "player_own_avg_10",
    "slate_size",
    "salary_pct_of_max",
    "is_min_salary",
    "slate_near_min_count",
]


def load_model() -> lgb.Booster:
    """Load production model."""
    model_path = MODEL_DIR / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return lgb.Booster(model_file=str(model_path))


def load_historical_ownership() -> Optional[pd.DataFrame]:
    """Load historical ownership data for player_own_avg_10 feature."""
    if HISTORICAL_OWN_PATH.exists():
        return pd.read_parquet(HISTORICAL_OWN_PATH)
    return None


def compute_player_own_avg(df: pd.DataFrame, historical: Optional[pd.DataFrame]) -> pd.Series:
    """Compute rolling average ownership for each player."""
    if historical is None:
        return pd.Series(0.0, index=df.index)
    
    # Get avg ownership per player from historical data
    player_avgs = historical.groupby('Player')['own_pct'].mean()
    
    # Map to current slate
    return df['player_name'].map(player_avgs).fillna(0.0)


def prepare_features(df: pd.DataFrame, historical_own: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute all features required for inference."""
    result = df.copy()
    
    # Value features
    result["value_per_k"] = result["proj_fpts"] / (result["salary"] / 1000.0)
    result["salary_rank"] = result["salary"].rank(ascending=False, method="min")
    result["proj_fpts_rank"] = result["proj_fpts"].rank(ascending=False, method="min")
    
    # Z-score within slate
    mean_fpts = result["proj_fpts"].mean()
    std_fpts = result["proj_fpts"].std()
    result["proj_fpts_zscore"] = (result["proj_fpts"] - mean_fpts) / (std_fpts if std_fpts > 0 else 1)
    
    # Salary tiers
    result["is_value_tier"] = (result["salary"] < 5000).astype(int)
    result["is_mid_tier"] = ((result["salary"] >= 5000) & (result["salary"] < 7500)).astype(int)
    result["is_high_tier"] = (result["salary"] >= 7500).astype(int)
    
    # Position one-hot
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        result[f"pos_{pos}"] = result["pos"].str.contains(pos, na=False).astype(int)
    
    # Injury features (default to 0 if not provided)
    if "player_is_questionable" not in result.columns:
        result["player_is_questionable"] = 0
    if "team_outs_count" not in result.columns:
        result["team_outs_count"] = 0
    
    # Historical ownership
    result["player_own_avg_10"] = compute_player_own_avg(result, historical_own)
    
    # Slate structure features
    result["slate_size"] = len(result)
    max_salary = result["salary"].max()
    min_salary = result["salary"].min()
    result["salary_pct_of_max"] = result["salary"] / max_salary
    result["is_min_salary"] = (result["salary"] == min_salary).astype(int)
    result["slate_near_min_count"] = (result["salary"] <= min_salary + 200).sum()
    
    return result


def predict(df: pd.DataFrame, model: lgb.Booster) -> np.ndarray:
    """Run inference on prepared features."""
    X = df[FEATURE_COLS].values
    return model.predict(X)


def main():
    parser = argparse.ArgumentParser(description="Predict ownership for a slate")
    parser.add_argument("--slate-path", type=str, required=True, help="Path to slate CSV")
    parser.add_argument("--output", type=str, help="Output path (default: adds _with_own to input)")
    args = parser.parse_args()
    
    # Load slate
    slate_path = Path(args.slate_path)
    if not slate_path.exists():
        print(f"Slate not found: {slate_path}")
        return
    
    df = pd.read_csv(slate_path)
    print(f"Loaded {len(df)} players from {slate_path.name}")
    
    # Load model and historical data
    print("Loading model...")
    model = load_model()
    historical = load_historical_ownership()
    print(f"Historical ownership: {len(historical) if historical is not None else 0} records")
    
    # Prepare features
    print("Computing features...")
    df_features = prepare_features(df, historical)
    
    # Predict
    print("Predicting ownership...")
    df["pred_own_pct"] = predict(df_features, model)
    
    # Clip to valid range
    df["pred_own_pct"] = df["pred_own_pct"].clip(0, 100)
    
    # Output
    output_path = args.output or slate_path.with_name(f"{slate_path.stem}_with_own.csv")
    df.to_csv(output_path, index=False)
    print(f"\nOutput: {output_path}")
    
    # Show preview
    print("\n=== Top 10 Predicted Ownership ===")
    preview = df.nlargest(10, "pred_own_pct")[["player_name", "salary", "proj_fpts", "pred_own_pct"]]
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
