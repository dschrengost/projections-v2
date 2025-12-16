"""
Shared metrics computation for usage shares models.

This module is imported by both train_lgbm.py and train_nn.py to ensure
consistent baseline computation and evaluation across backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from projections.usage_shares_v1.features import GROUP_COLS


# =============================================================================
# Metrics Dataclass
# =============================================================================


@dataclass
class TargetMetrics:
    """Metrics for a single target."""
    share_MAE: float
    share_MAE_baseline: float
    KL: float
    top1_acc: float
    H_mean_pred: float
    H_mean_true: float
    n_groups: int
    n_rows: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "share_MAE": round(self.share_MAE, 6),
            "share_MAE_baseline": round(self.share_MAE_baseline, 6),
            "KL": round(self.KL, 6),
            "top1_acc": round(self.top1_acc, 4),
            "H_mean_pred": round(self.H_mean_pred, 4),
            "H_mean_true": round(self.H_mean_true, 4),
            "n_groups": self.n_groups,
            "n_rows": self.n_rows,
            "beats_baseline": self.share_MAE < self.share_MAE_baseline,
        }


# =============================================================================
# Baseline Computation
# =============================================================================


def compute_baseline_log_weights(
    df: pd.DataFrame,
    target: str,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Compute rate-weighted baseline log-weights.
    
    Baseline: log(season_{target}_per_min * minutes_pred_p50 + alpha)
    
    This is the SAME baseline used by both LGBM and NN backends.
    
    Args:
        df: DataFrame with feature columns
        target: Target name (fga/fta/tov)
        alpha: Smoothing constant for log transformation
        
    Returns:
        Log-weights array of shape (n_rows,)
    """
    season_rate_col = f"season_{target}_per_min"
    
    if season_rate_col in df.columns and "minutes_pred_p50" in df.columns:
        rate = df[season_rate_col].fillna(0.0).values
        mins = df["minutes_pred_p50"].fillna(0.0).values
        baseline_raw = rate * mins
    else:
        # Fallback: use minutes_pred_p50 directly
        baseline_raw = df["minutes_pred_p50"].fillna(1.0).values if "minutes_pred_p50" in df.columns else np.ones(len(df))
    
    return np.log(baseline_raw.clip(min=0.01) + alpha)


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_metrics(
    df: pd.DataFrame,
    log_weights_pred: np.ndarray,
    target: str,
    alpha: float = 0.5,
) -> TargetMetrics:
    """
    Compute evaluation metrics for predictions.
    
    Uses the SAME baseline computation as compute_baseline_log_weights.
    
    Args:
        df: DataFrame with true shares and group columns
        log_weights_pred: Predicted log-weights from model
        target: Target name (fga/fta/tov)
        alpha: Alpha used for log transformation
        
    Returns:
        TargetMetrics with MAE, KL, top1, Herfindahl stats
    """
    share_col = f"share_{target}"
    eps = 1e-9
    
    # Compute baseline using shared function
    log_weights_baseline = compute_baseline_log_weights(df, target, alpha)
    
    # Convert log-weights to weights
    weights_pred = np.exp(log_weights_pred)
    weights_baseline = np.exp(log_weights_baseline)
    
    working = df[GROUP_COLS + [share_col]].copy()
    working["weight_pred"] = weights_pred
    working["weight_baseline"] = weights_baseline
    
    # Normalize within groups to get share predictions
    group_sums_pred = working.groupby(GROUP_COLS)["weight_pred"].transform("sum")
    group_sums_baseline = working.groupby(GROUP_COLS)["weight_baseline"].transform("sum")
    
    working["share_pred"] = working["weight_pred"] / group_sums_pred.clip(lower=eps)
    working["share_baseline"] = working["weight_baseline"] / group_sums_baseline.clip(lower=eps)
    working["share_true"] = working[share_col]
    
    # Per-row metrics
    working["abs_err"] = (working["share_pred"] - working["share_true"]).abs()
    working["abs_err_baseline"] = (working["share_baseline"] - working["share_true"]).abs()
    working["share_pred_sq"] = working["share_pred"] ** 2
    working["share_true_sq"] = working["share_true"] ** 2
    
    # Group-level aggregations
    group_agg = working.groupby(GROUP_COLS).agg(
        mae=("abs_err", "mean"),
        mae_baseline=("abs_err_baseline", "mean"),
        H_pred=("share_pred_sq", "sum"),
        H_true=("share_true_sq", "sum"),
    ).reset_index()
    
    # KL divergence per group: sum_i share_true_i * log((share_true_i + eps) / (share_pred_i + eps))
    def kl_per_group(g: pd.DataFrame) -> float:
        s_true = g["share_true"].values + eps
        s_pred = g["share_pred"].values + eps
        return float(np.sum(s_true * np.log(s_true / s_pred)))
    
    kl_per_group_series = working.groupby(GROUP_COLS).apply(kl_per_group, include_groups=False)
    
    # Top-1 accuracy: is the predicted top player actually the true top player?
    def top1_match(g: pd.DataFrame) -> float:
        pred_top = g["share_pred"].idxmax()
        true_top = g["share_true"].idxmax()
        return 1.0 if pred_top == true_top else 0.0
    
    top1_per_group = working.groupby(GROUP_COLS).apply(top1_match, include_groups=False)
    
    return TargetMetrics(
        share_MAE=float(group_agg["mae"].mean()),
        share_MAE_baseline=float(group_agg["mae_baseline"].mean()),
        KL=float(kl_per_group_series.mean()),
        top1_acc=float(top1_per_group.mean()),
        H_mean_pred=float(group_agg["H_pred"].mean()),
        H_mean_true=float(group_agg["H_true"].mean()),
        n_groups=len(group_agg),
        n_rows=len(working),
    )


# =============================================================================
# Leakage Check
# =============================================================================


def check_odds_leakage(df: pd.DataFrame) -> tuple[int, int, float]:
    """
    Check for potential odds leakage (odds_as_of_ts > tip_ts).
    
    Args:
        df: DataFrame with odds_as_of_ts and tip_ts columns
        
    Returns:
        (n_leaky, n_checked, missing_frac) - count of leaky rows, rows checked, fraction missing timestamps
    """
    if "odds_as_of_ts" not in df.columns or "tip_ts" not in df.columns:
        return 0, 0, 1.0
    
    odds_ts = pd.to_datetime(df["odds_as_of_ts"], utc=True, errors="coerce")
    tip_ts = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce")
    
    both_present = odds_ts.notna() & tip_ts.notna()
    n_checked = both_present.sum()
    missing_frac = 1.0 - (n_checked / len(df)) if len(df) > 0 else 1.0
    
    if n_checked == 0:
        return 0, 0, missing_frac
    
    # Check for leakage: odds_as_of_ts > tip_ts
    leaky = (odds_ts > tip_ts) & both_present
    n_leaky = leaky.sum()
    
    return int(n_leaky), int(n_checked), float(missing_frac)


__all__ = [
    "TargetMetrics",
    "compute_baseline_log_weights",
    "compute_metrics",
    "check_odds_leakage",
]
