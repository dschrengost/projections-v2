#!/usr/bin/env python3
"""
Train dupe penalty model from training data.

Estimates λ (expected other duplicates) per feature bin using entry-weighted
statistics with Bayesian shrinkage. Outputs E[1/K] lookup table.

Input:  gold/dupe_training_data.parquet
Output: gold/dupe_model.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from projections.api.contest_service import get_data_root


# Binning configuration
FIELD_BUCKETS = [
    (0, 3000, "<3k"),
    (3000, 10000, "3k-10k"),
    (10000, 50000, "10k-50k"),
    (50000, float("inf"), "50k+"),
]

ENTRY_MAX_BUCKETS = [
    (1, 1, "single"),
    (2, 20, "low_multi"),
    (21, 150, "high_multi"),
    (151, float("inf"), "max_multi"),
]

SUM_OWN_BIN_SIZE = 10  # 10% buckets


def get_field_bucket(field_size: int) -> str:
    """Get field size bucket label."""
    for low, high, label in FIELD_BUCKETS:
        if low <= field_size < high:
            return label
    return "50k+"


def get_entry_max_bucket(entry_max: int) -> str:
    """Get entry max bucket label."""
    for low, high, label in ENTRY_MAX_BUCKETS:
        if low <= entry_max <= high:
            return label
    return "max_multi"


def get_sum_own_bin(sum_own: float) -> int:
    """Get sum ownership bin start (e.g., 80 for 80-90%)."""
    return int(sum_own / SUM_OWN_BIN_SIZE) * SUM_OWN_BIN_SIZE


def compute_e_inv_k(lambda_hat: float) -> float:
    """Compute E[1/K] = (1 - e^(-λ)) / λ for Poisson model."""
    if lambda_hat <= 0:
        return 1.0
    if lambda_hat < 1e-6:
        # Taylor expansion for small λ: 1 - λ/2 + λ²/6 - ...
        return 1.0 - lambda_hat / 2
    return (1 - math.exp(-lambda_hat)) / lambda_hat


def train_model(df: pd.DataFrame, shrinkage_k: int = 300) -> dict:
    """
    Train dupe model with binning and Bayesian shrinkage.

    Args:
        df: Training data with dupe_count, sum_own, field_size, entry_max
        shrinkage_k: Pseudo-entries for shrinkage (higher = more shrinkage)

    Returns:
        Model dict with bin -> stats mapping
    """
    # Apply binning
    df = df.copy()
    df["field_bucket"] = df["field_size"].apply(get_field_bucket)
    df["entry_max_bucket"] = df["entry_max"].apply(get_entry_max_bucket)
    df["sum_own_bin"] = df["sum_own"].apply(get_sum_own_bin)

    # Compute D = K - 1 (other duplicates)
    df["other_dupes"] = df["dupe_count"] - 1

    # Global lambda (for shrinkage fallback)
    global_lambda = df["other_dupes"].mean()
    print(f"Global λ: {global_lambda:.3f}")

    # Compute per-bin stats
    model = {
        "_meta": {
            "global_lambda": global_lambda,
            "global_e_inv_k": compute_e_inv_k(global_lambda),
            "shrinkage_k": shrinkage_k,
            "total_entries": len(df),
            "bin_size": SUM_OWN_BIN_SIZE,
        }
    }

    # Group by (field_bucket, entry_max_bucket, sum_own_bin)
    grouped = df.groupby(["field_bucket", "entry_max_bucket", "sum_own_bin"])

    for (field_bucket, entry_max_bucket, sum_own_bin), group in grouped:
        n = len(group)
        lambda_raw = group["other_dupes"].mean()

        # Bayesian shrinkage toward global
        lambda_post = (n * lambda_raw + shrinkage_k * global_lambda) / (n + shrinkage_k)

        bin_key = f"{field_bucket}|{entry_max_bucket}|{sum_own_bin}"

        model[bin_key] = {
            "lambda": round(lambda_post, 4),
            "lambda_raw": round(lambda_raw, 4),
            "n_entries": n,
            "e_k": round(1 + lambda_post, 4),  # Expected K
            "e_inv_k": round(compute_e_inv_k(lambda_post), 4),  # Expected split factor
        }

    print(f"Trained {len(model) - 1} bins")

    return model


def analyze_model(model: dict, df: pd.DataFrame):
    """Print analysis of trained model."""
    print("\n=== MODEL ANALYSIS ===")
    print(f"Total bins: {len(model) - 1}")

    # Convert to sorted list for analysis
    bins = [(k, v) for k, v in model.items() if k != "_meta"]
    bins.sort(key=lambda x: (x[1]["lambda"], -x[1]["n_entries"]))

    print("\n--- Lowest λ bins (most unique) ---")
    for bin_key, stats in bins[:5]:
        print(f"  {bin_key}: λ={stats['lambda']:.3f}, E[1/K]={stats['e_inv_k']:.3f}, n={stats['n_entries']:,}")

    print("\n--- Highest λ bins (most duped) ---")
    for bin_key, stats in bins[-5:]:
        print(f"  {bin_key}: λ={stats['lambda']:.3f}, E[1/K]={stats['e_inv_k']:.3f}, n={stats['n_entries']:,}")

    # Analyze by sum_own
    print("\n--- E[1/K] by sum_own (all buckets combined) ---")
    df = df.copy()
    df["sum_own_bin"] = df["sum_own"].apply(get_sum_own_bin)
    df["other_dupes"] = df["dupe_count"] - 1
    by_own = df.groupby("sum_own_bin").agg({
        "dupe_count": ["mean", "count"],
        "other_dupes": "mean",
    }).round(3)
    by_own.columns = ["mean_K", "n_entries", "mean_D"]
    by_own["implied_E_inv_K"] = by_own["mean_D"].apply(compute_e_inv_k).round(3)
    print(by_own.to_string())


def main():
    parser = argparse.ArgumentParser(description="Train dupe penalty model")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input training data path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output model path",
    )
    parser.add_argument(
        "--shrinkage-k",
        type=int,
        default=300,
        help="Pseudo-entries for Bayesian shrinkage (default: 300)",
    )
    args = parser.parse_args()

    data_root = get_data_root()
    input_path = args.input or (data_root / "gold" / "dupe_training_data.parquet")
    output_path = args.output or (data_root / "gold" / "dupe_model.json")

    if not input_path.exists():
        print(f"Training data not found: {input_path}")
        print("Run build_dupe_training_data.py first.")
        sys.exit(1)

    print(f"Loading training data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} entries")

    # Train model
    print("\nTraining model...")
    model = train_model(df, shrinkage_k=args.shrinkage_k)

    # Analyze
    analyze_model(model, df)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(model, f, indent=2)

    print(f"\n=== OUTPUT ===")
    print(f"Model saved to: {output_path}")
    print(f"Global E[1/K]: {model['_meta']['global_e_inv_k']:.4f}")


if __name__ == "__main__":
    main()
