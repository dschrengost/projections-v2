"""Scoring/inference functions for ownership_v1 model."""

from __future__ import annotations

import pandas as pd

from projections.ownership_v1.loader import OwnershipBundle


def predict_ownership(
    features: pd.DataFrame, 
    bundle: OwnershipBundle
) -> pd.Series:
    """
    Predict ownership percentage for each player in the slate.
    
    Args:
        features: DataFrame with required feature columns (from bundle.feature_cols).
            Should be prepared with compute_ownership_features() first.
        bundle: Loaded ownership model bundle.
    
    Returns:
        Series with predicted ownership percentages (0-100 scale), 
        indexed same as input features.
    """
    missing = [c for c in bundle.feature_cols if c not in features.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    
    X = features[bundle.feature_cols]
    preds = bundle.model.predict(X, num_iteration=bundle.model.best_iteration)
    
    # Clip to valid range [0, 100]
    preds = preds.clip(0, 100)
    
    return pd.Series(preds, index=features.index, name="predicted_own_pct")


def compute_ownership_features(
    df: pd.DataFrame,
    *,
    proj_fpts_col: str = "proj_fpts",
    salary_col: str = "salary",
    pos_col: str = "pos",
    slate_id_col: str | None = "slate_id",
) -> pd.DataFrame:
    """
    Compute ownership features from raw projection data.
    
    Args:
        df: DataFrame with player projections for one or more slates.
        proj_fpts_col: Column name for projected fantasy points.
        salary_col: Column name for salary.
        pos_col: Column name for position(s).
        slate_id_col: Column to group by for within-slate ranks. 
            If None, treats entire df as one slate.
    
    Returns:
        DataFrame with computed features ready for model inference.
    """
    result = df.copy()
    
    # Value per $1k
    result["value_per_k"] = result[proj_fpts_col] / (result[salary_col] / 1000.0)
    
    # Salary tiers
    result["is_value_tier"] = (result[salary_col] < 5000).astype(int)
    result["is_mid_tier"] = ((result[salary_col] >= 5000) & (result[salary_col] < 7500)).astype(int)
    result["is_high_tier"] = (result[salary_col] >= 7500).astype(int)
    
    # Position one-hot encoding
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        result[f"pos_{pos}"] = result[pos_col].str.contains(pos, na=False).astype(int)
    
    # Within-slate ranks and z-scores
    if slate_id_col and slate_id_col in result.columns:
        # Rank within slate (1 = highest)
        result["salary_rank"] = result.groupby(slate_id_col)[salary_col].rank(
            ascending=False, method="min"
        )
        result["proj_fpts_rank"] = result.groupby(slate_id_col)[proj_fpts_col].rank(
            ascending=False, method="min"
        )
        
        # Z-score within slate
        slate_means = result.groupby(slate_id_col)[proj_fpts_col].transform("mean")
        slate_stds = result.groupby(slate_id_col)[proj_fpts_col].transform("std")
        result["proj_fpts_zscore"] = (result[proj_fpts_col] - slate_means) / slate_stds.replace(0, 1)
    else:
        # Single slate - compute across all rows
        result["salary_rank"] = result[salary_col].rank(ascending=False, method="min")
        result["proj_fpts_rank"] = result[proj_fpts_col].rank(ascending=False, method="min")
        mean = result[proj_fpts_col].mean()
        std = result[proj_fpts_col].std()
        result["proj_fpts_zscore"] = (result[proj_fpts_col] - mean) / (std if std > 0 else 1)
    
    return result


__all__ = ["predict_ownership", "compute_ownership_features"]
