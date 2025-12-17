"""Scoring/inference functions for ownership_v1 model."""

from __future__ import annotations

import numpy as np
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

    target_transform = (
        bundle.meta.get("params", {}).get("target_transform", "none")
        if isinstance(bundle.meta, dict)
        else "none"
    )
    if target_transform == "logit":
        preds = 1.0 / (1.0 + np.exp(-preds))
        preds = preds * 100.0
    elif target_transform != "none":
        raise ValueError(f"Unknown target_transform: {target_transform}")

    # Clip to valid range [0, 100]
    preds = np.clip(preds, 0.0, 100.0)
    
    return pd.Series(preds, index=features.index, name="predicted_own_pct")


def compute_ownership_features(
    df: pd.DataFrame,
    *,
    proj_fpts_col: str = "proj_fpts",
    salary_col: str = "salary",
    pos_col: str = "pos",
    slate_id_col: str | None = "slate_id",
    team_col: str | None = "team",
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
        group = result.groupby(slate_id_col)
        # Rank within slate (1 = highest)
        result["salary_rank"] = group[salary_col].rank(
            ascending=False, method="min"
        )
        result["proj_fpts_rank"] = group[proj_fpts_col].rank(
            ascending=False, method="min"
        )
        
        # Z-score within slate
        slate_means = group[proj_fpts_col].transform("mean")
        slate_stds = group[proj_fpts_col].transform("std")
        result["proj_fpts_zscore"] = (result[proj_fpts_col] - slate_means) / slate_stds.replace(0, 1)

        # Slate structure
        result["slate_size"] = group[salary_col].transform("size").astype(int)
        slate_max_salary = group[salary_col].transform("max").replace(0, 1)
        slate_min_salary = group[salary_col].transform("min")
        result["salary_pct_of_max"] = result[salary_col] / slate_max_salary
        result["is_min_salary"] = (result[salary_col] == slate_min_salary).astype(int)
        near_min = (result[salary_col] <= slate_min_salary + 200).astype(int)
        result["slate_near_min_count"] = near_min.groupby(result[slate_id_col]).transform("sum").astype(int)

        # Value leverage (relative to slate)
        slate_value_avg = group["value_per_k"].transform("mean")
        result["value_vs_slate_avg"] = result["value_per_k"] - slate_value_avg
        slate_salary_median = group[salary_col].transform("median").replace(0, 1)
        result["salary_vs_median"] = (result[salary_col] - slate_salary_median) / slate_salary_median

        # Is min-priced by position (cheapest at any of a player's positions on the slate)
        pos_cols = [f"pos_{p}" for p in ["PG", "SG", "SF", "PF", "C"]]
        has_pos_cols = all(c in result.columns for c in pos_cols)
        if has_pos_cols:
            def _min_priced_by_pos(g: pd.DataFrame) -> pd.Series:
                mins = {}
                for p in ["PG", "SG", "SF", "PF", "C"]:
                    mask = g[f"pos_{p}"].astype(int) == 1
                    mins[p] = float(g.loc[mask, salary_col].min()) if mask.any() else float("inf")
                flags = np.zeros(len(g), dtype=bool)
                for p in ["PG", "SG", "SF", "PF", "C"]:
                    flags |= (g[f"pos_{p}"].astype(int) == 1).to_numpy() & (g[salary_col].to_numpy() == mins[p])
                return pd.Series(flags.astype(int), index=g.index)

            result["is_min_priced_by_pos"] = result.groupby(slate_id_col, group_keys=False).apply(
                _min_priced_by_pos, include_groups=False
            )
        else:
            result["is_min_priced_by_pos"] = 0

        # Game count on slate (proxy from team count)
        if team_col and team_col in result.columns:
            team_counts = group[team_col].transform("nunique")
            result["game_count_on_slate"] = (team_counts // 2).astype(int)
        else:
            result["game_count_on_slate"] = 0
    else:
        # Single slate - compute across all rows
        result["salary_rank"] = result[salary_col].rank(ascending=False, method="min")
        result["proj_fpts_rank"] = result[proj_fpts_col].rank(ascending=False, method="min")
        mean = result[proj_fpts_col].mean()
        std = result[proj_fpts_col].std()
        result["proj_fpts_zscore"] = (result[proj_fpts_col] - mean) / (std if std > 0 else 1)

        # Slate structure
        result["slate_size"] = int(len(result))
        max_salary = float(result[salary_col].max()) if len(result) else 1.0
        min_salary = float(result[salary_col].min()) if len(result) else 0.0
        result["salary_pct_of_max"] = result[salary_col] / (max_salary if max_salary > 0 else 1.0)
        result["is_min_salary"] = (result[salary_col] == min_salary).astype(int)
        result["slate_near_min_count"] = int((result[salary_col] <= min_salary + 200).sum())

        # Value leverage
        result["value_vs_slate_avg"] = result["value_per_k"] - float(result["value_per_k"].mean())
        salary_median = float(result[salary_col].median()) if len(result) else 1.0
        result["salary_vs_median"] = (result[salary_col] - salary_median) / (salary_median if salary_median != 0 else 1.0)

        # Is min-priced by position
        mins = {}
        for p in ["PG", "SG", "SF", "PF", "C"]:
            col = f"pos_{p}"
            if col in result.columns:
                mask = result[col].astype(int) == 1
                mins[p] = float(result.loc[mask, salary_col].min()) if mask.any() else float("inf")
            else:
                mins[p] = float("inf")
        flags = np.zeros(len(result), dtype=bool)
        for p in ["PG", "SG", "SF", "PF", "C"]:
            col = f"pos_{p}"
            if col not in result.columns:
                continue
            flags |= (result[col].astype(int).to_numpy() == 1) & (result[salary_col].to_numpy() == mins[p])
        result["is_min_priced_by_pos"] = flags.astype(int)

        # Game count on slate
        if team_col and team_col in result.columns:
            result["game_count_on_slate"] = int(result[team_col].nunique() // 2)
        else:
            result["game_count_on_slate"] = 0

    # Interaction features (safe defaults if enrichment missing)
    result["value_x_value_tier"] = result["value_per_k"] * result["is_value_tier"]
    outs = result["team_outs_count"].astype(float) if "team_outs_count" in result.columns else 0.0
    result["outs_x_salary_rank"] = outs * result["salary_rank"].astype(float)
    
    return result


__all__ = ["predict_ownership", "compute_ownership_features"]
