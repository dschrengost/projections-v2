import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from scipy import stats
from projections.paths import data_path


def compute_chalk_recall_at_k(df: pd.DataFrame, k: int = 5) -> float:
    """Compute Chalk Recall@K: of top K actually owned players per slate, how many in top K predictions?

    Args:
        df: DataFrame with slate_id, actual_own_pct, pred_own_pct
        k: Number of top players to consider

    Returns:
        Recall@K averaged across slates
    """
    recalls = []
    for _, group in df.groupby("slate_id"):
        if len(group) < k:
            continue

        # Top K actually owned
        actual_top_k = set(group.nlargest(k, "actual_own_pct").index)

        # Top K predicted
        pred_top_k = set(group.nlargest(k, "pred_own_pct").index)

        # Overlap
        recall = len(actual_top_k & pred_top_k) / k
        recalls.append(recall)

    return np.mean(recalls) if recalls else float("nan")


def compute_per_slate_spearman(df: pd.DataFrame) -> tuple[float, float]:
    """Compute per-slate Spearman rank correlation.

    Returns:
        (mean_spearman, std_spearman)
    """
    spearmans = []
    for _, group in df.groupby("slate_id"):
        if len(group) < 3:
            continue
        if group["actual_own_pct"].std() == 0 or group["pred_own_pct"].std() == 0:
            continue

        rho, _ = stats.spearmanr(group["actual_own_pct"], group["pred_own_pct"])
        if not np.isnan(rho):
            spearmans.append(rho)

    if not spearmans:
        return float("nan"), float("nan")

    return np.mean(spearmans), np.std(spearmans)


def analyze_run(run_id: str):
    artifacts_dir = data_path() / "artifacts" / "ownership_v1" / "runs" / run_id
    pred_file = artifacts_dir / "val_predictions.csv"

    if not pred_file.exists():
        print(f"Error: Predictions file not found at {pred_file}")
        return

    df = pd.read_csv(pred_file)
    print(f"Loaded {len(df):,} predictions from {run_id}")

    # Basic Metrics
    mae = np.mean(np.abs(df["actual_own_pct"] - df["pred_own_pct"]))
    rmse = np.sqrt(np.mean((df["actual_own_pct"] - df["pred_own_pct"]) ** 2))
    overall_corr = df["actual_own_pct"].corr(df["pred_own_pct"])
    print(f"\n=== Overall Metrics ===")
    print(f"MAE: {mae:.3f}%")
    print(f"RMSE: {rmse:.3f}%")
    print(f"Pearson Correlation: {overall_corr:.3f}")

    # Per-slate Spearman correlation
    spearman_mean, spearman_std = compute_per_slate_spearman(df)
    print(f"Per-Slate Spearman: {spearman_mean:.3f} (+/- {spearman_std:.3f})")

    # Chalk Recall@K
    print(f"\n=== Chalk Recall@K (per slate) ===")
    for k in [3, 5, 10]:
        recall = compute_chalk_recall_at_k(df, k=k)
        print(f"Recall@{k}: {recall:.1%}")

    # Slate correlation (Pearson)
    corrs = []
    for _, group in df.groupby("slate_id"):
        if len(group) > 1 and group["actual_own_pct"].std() > 0:
            corrs.append(group["actual_own_pct"].corr(group["pred_own_pct"]))
    print(f"Per-Slate Pearson: {np.mean(corrs):.3f}")
    
    # Chalk Analysis
    print("\n--- Chalk Prediction Accuracy ---")
    thresholds = [20, 30, 40, 50]
    for t in thresholds:
        actual_chalk = df[df['actual_own_pct'] >= t]
        pred_chalk = df[df['pred_own_pct'] >= t]
        
        # Recall: Of the players who were actually chalk, how many did we predict?
        # But we care more about: "Did we predict high ownership for them?"
        # Let's check the average prediction for actual chalk
        avg_pred_for_chalk = actual_chalk['pred_own_pct'].mean()
        
        # Count how many of actual chalk were predicted >= t
        caught = len(actual_chalk[actual_chalk['pred_own_pct'] >= t])
        total = len(actual_chalk)
        recall = caught / total if total > 0 else 0
        
        print(f"Actual > {t}% (n={total}): Avg Pred = {avg_pred_for_chalk:.1f}%, Caught (>={t}%) = {recall:.1%}")

    # Bucket Analysis
    print("\n--- Error by Actual Ownership Bucket ---")
    df['bucket'] = pd.cut(df['actual_own_pct'], bins=[0, 5, 15, 30, 50, 100], labels=['0-5', '5-15', '15-30', '30-50', '50+'])
    grouped = df.groupby('bucket', observed=True).agg({
        'actual_own_pct': 'mean',
        'pred_own_pct': 'mean',
        'player_name': 'count'
    })
    grouped['mae'] = df.groupby('bucket', observed=True).apply(
        lambda x: np.mean(np.abs(x['actual_own_pct'] - x['pred_own_pct'])),
        include_groups=False
    )
    grouped['bias'] = grouped['pred_own_pct'] - grouped['actual_own_pct']
    print(grouped.round(2))
    
    # Top Misses (Under-predicted)
    print("\n--- Top 10 Under-Predicted (Missed Chalk) ---")
    df['error'] = df['pred_own_pct'] - df['actual_own_pct']
    misses = df.sort_values('error').head(10)
    print(misses[['game_date', 'player_name', 'salary', 'actual_own_pct', 'pred_own_pct', 'error']].to_string(index=False))

    # Top Misses (Over-predicted)
    print("\n--- Top 10 Over-Predicted (False Chalk) ---")
    over = df.sort_values('error', ascending=False).head(10)
    print(over[['game_date', 'player_name', 'salary', 'actual_own_pct', 'pred_own_pct', 'error']].to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    analyze_run(args.run_id)
