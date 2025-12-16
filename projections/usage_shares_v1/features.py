"""Feature definitions for usage shares v1 models.

This module defines the canonical feature sets used by both LightGBM and NN backends
for predicting within-team opportunity shares (FGA/FTA/TOV).

Feature provenance verified:
- Season priors use .shift(1) to exclude current game (rates/build_training_base.py L397-404, L751-770)
- Vacancy features are computed from pre-tip injury snapshots
- Odds features come from pre-tip snapshots with odds_lead_time_minutes tracking
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# Feature Constants (v1 - intentionally slim to avoid overfitting)
# =============================================================================

NUMERIC_COLS: list[str] = [
    # Minutes predictions (primary signal)
    "minutes_pred_p50",
    "minutes_pred_play_prob",
    "minutes_pred_p50_team_scaled",  # Scaled to sum to 240 within team
    "minutes_pred_team_sum_invalid",  # 1 if team sum <= eps, else 0
    "minutes_pred_team_rank",  # Rank within team (1 = highest minutes)
    # Role
    "is_starter",
    # Position flags
    "position_flags_PG",
    "position_flags_SG",
    "position_flags_SF",
    "position_flags_PF",
    "position_flags_C",
    # Vegas (with lead time for horizon awareness)
    "spread_close",
    "total_close",
    "team_itt",
    "opp_itt",
    "has_odds",
    "odds_lead_time_minutes",  # tip_ts - odds_as_of_ts in minutes
    # Vacancy aggregates
    "vac_min_szn",
    "vac_fga_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
    # Vacancy interactions (high-leverage for injury behavior)
    "vac_min_szn_x_is_starter",
    "vac_min_szn_x_minutes_rank",
    "vac_fga_szn_x_is_starter",
    # Season priors (verified: uses shift(1) to exclude current game)
    "season_fga_per_min",
    "season_fta_per_min",
    "season_tov_per_min",
]

CATEGORICAL_COLS: list[str] = [
    "track_role_cluster",  # Single categorical for v1
]

FEATURE_COLS: list[str] = NUMERIC_COLS + CATEGORICAL_COLS

# Label columns used for training
LABEL_COLS: list[str] = [
    "fga",
    "fta",
    "tov",
    "share_fga",
    "share_fta",
    "share_tov",
]

# Validity flags per target
VALIDITY_COLS: list[str] = [
    "share_fga_valid",
    "share_fta_valid",
    "share_tov_valid",
]

# Grouping columns for normalization
GROUP_COLS: list[str] = ["game_id", "team_id"]

# Key columns needed in training data
KEY_COLS: list[str] = [
    "season",
    "game_id",
    "game_date",
    "team_id",
    "player_id",
]


# =============================================================================
# Feature Preparation Utilities
# =============================================================================


def compute_odds_lead_time_minutes(df: pd.DataFrame) -> pd.Series:
    """
    Compute odds lead time in minutes: (tip_ts - odds_as_of_ts).
    
    Positive values mean odds were captured before tip (safe).
    Negative values indicate potential leak (odds after tip).
    
    Returns NaN if either timestamp is missing.
    """
    if "tip_ts" not in df.columns or "odds_as_of_ts" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    
    tip_ts = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce")
    odds_ts = pd.to_datetime(df["odds_as_of_ts"], utc=True, errors="coerce")
    
    # Lead time = tip - odds (positive = odds before tip)
    delta_seconds = (tip_ts - odds_ts).dt.total_seconds()
    return delta_seconds / 60.0  # Convert to minutes


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    fill_missing_numeric: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Prepare feature arrays from a dataframe.
    
    Args:
        df: Source dataframe with feature columns
        feature_cols: Ordered list of all feature columns (default: FEATURE_COLS)
        categorical_cols: List of categorical columns (default: CATEGORICAL_COLS)
        fill_missing_numeric: Value to fill missing numeric features
        
    Returns:
        X_num: Numeric features array (n_rows, n_numeric)
        X_cat: Categorical features array (n_rows, n_cat) as int codes
        numeric_cols: List of numeric column names in order
        cat_cols: List of categorical column names in order
    """
    feature_cols = feature_cols or FEATURE_COLS
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    cat_cols = [c for c in feature_cols if c in categorical_cols]
    
    # Extract numeric features
    X_num_list = []
    for col in numeric_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(fill_missing_numeric)
        else:
            vals = pd.Series(fill_missing_numeric, index=df.index)
        X_num_list.append(vals.values)
    X_num = np.column_stack(X_num_list) if X_num_list else np.empty((len(df), 0))
    
    # Extract categorical features as integer codes
    X_cat_list = []
    for col in cat_cols:
        if col in df.columns:
            # Convert to codes, treating NaN as -1
            vals = df[col].fillna(-1)
            if hasattr(vals, "cat"):
                codes = vals.cat.codes.values
            else:
                # For non-category columns, use raw values as codes
                codes = pd.to_numeric(vals, errors="coerce").fillna(-1).astype(int).values
        else:
            codes = np.full(len(df), -1, dtype=int)
        X_cat_list.append(codes)
    X_cat = np.column_stack(X_cat_list) if X_cat_list else np.empty((len(df), 0), dtype=int)
    
    return X_num, X_cat, numeric_cols, cat_cols


def compute_minutes_team_scaled(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Compute team-scaled minutes prediction.
    
    minutes_pred_p50_team_scaled = minutes_pred_p50 * (240 / sum_team(minutes_pred_p50))
    
    This matches sim behavior where predicted minutes are normalized to sum to 240.
    
    Args:
        df: DataFrame with minutes_pred_p50, game_id, team_id columns
        
    Returns:
        (scaled_minutes, invalid_flag) where:
        - scaled_minutes: Team-scaled minutes prediction
        - invalid_flag: 1 if team sum <= eps (edge case), else 0
    """
    eps = 1e-6
    
    if "minutes_pred_p50" not in df.columns:
        return pd.Series(np.nan, index=df.index), pd.Series(1.0, index=df.index)
    
    if "game_id" not in df.columns or "team_id" not in df.columns:
        # Can't group - return raw values
        return df["minutes_pred_p50"].fillna(0.0), pd.Series(0.0, index=df.index)
    
    raw_mins = df["minutes_pred_p50"].fillna(0.0)
    
    # Compute team sum
    team_sum = df.groupby(["game_id", "team_id"])["minutes_pred_p50"].transform("sum")
    team_sum = team_sum.fillna(0.0)
    
    # Invalid flag: 1 if sum is tiny/zero
    invalid_flag = (team_sum <= eps).astype(float)
    
    # Scale to 240 (typical game minutes)
    # If sum <= eps, just use raw minutes_pred_p50 (avoid division by zero)
    scaled = np.where(
        team_sum > eps,
        raw_mins * (240.0 / team_sum),
        raw_mins  # Fallback: use raw value
    )
    
    return pd.Series(scaled, index=df.index), invalid_flag


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the dataframe.
    
    Currently adds:
    - odds_lead_time_minutes: Lead time of odds snapshot before tip
    - minutes_pred_p50_team_scaled: Minutes normalized to sum to 240 within team
    - minutes_pred_team_sum_invalid: Flag for edge cases with tiny team sum
    - minutes_pred_team_rank: Player's rank within team by predicted minutes (1=highest)
    - vac_min_szn_x_is_starter: Interaction of vacancy with starter flag
    - vac_min_szn_x_minutes_rank: Interaction of vacancy with minutes rank
    - vac_fga_szn_x_is_starter: Interaction of FGA vacancy with starter flag
    
    Returns a copy with new columns added.
    """
    df = df.copy()
    
    if "odds_lead_time_minutes" not in df.columns:
        df["odds_lead_time_minutes"] = compute_odds_lead_time_minutes(df)
    
    if "minutes_pred_p50_team_scaled" not in df.columns:
        scaled, invalid = compute_minutes_team_scaled(df)
        df["minutes_pred_p50_team_scaled"] = scaled
        df["minutes_pred_team_sum_invalid"] = invalid
    
    # Compute team rank (1 = highest minutes)
    if "minutes_pred_team_rank" not in df.columns:
        if "minutes_pred_p50" in df.columns and "game_id" in df.columns and "team_id" in df.columns:
            df["minutes_pred_team_rank"] = df.groupby(["game_id", "team_id"])["minutes_pred_p50"].rank(
                ascending=False, method="min"
            )
        else:
            df["minutes_pred_team_rank"] = 5.0  # Default to middle rank
    
    # Vacancy x is_starter interactions
    if "vac_min_szn_x_is_starter" not in df.columns:
        vac_min = df["vac_min_szn"].fillna(0.0) if "vac_min_szn" in df.columns else 0.0
        is_starter = df["is_starter"].fillna(0.0) if "is_starter" in df.columns else 0.0
        df["vac_min_szn_x_is_starter"] = vac_min * is_starter
    
    # Vacancy x minutes_rank interaction (higher rank = lower number = more benefit)
    if "vac_min_szn_x_minutes_rank" not in df.columns:
        vac_min = df["vac_min_szn"].fillna(0.0) if "vac_min_szn" in df.columns else 0.0
        # Invert rank so higher value = higher minutes player
        rank_inv = 10.0 - df["minutes_pred_team_rank"].fillna(5.0).clip(1, 10)
        df["vac_min_szn_x_minutes_rank"] = vac_min * rank_inv
    
    # FGA vacancy x is_starter interaction
    if "vac_fga_szn_x_is_starter" not in df.columns:
        vac_fga = df["vac_fga_szn"].fillna(0.0) if "vac_fga_szn" in df.columns else 0.0
        is_starter = df["is_starter"].fillna(0.0) if "is_starter" in df.columns else 0.0
        df["vac_fga_szn_x_is_starter"] = vac_fga * is_starter
    
    return df


def build_category_maps(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
) -> dict[str, dict[Any, int]]:
    """
    Build vocabulary maps for categorical columns.
    
    Index 0 is reserved for UNK (unknown/unseen categories).
    
    Args:
        df: Training dataframe
        categorical_cols: List of categorical columns (default: CATEGORICAL_COLS)
        
    Returns:
        Dictionary mapping column name -> {value: index}
    """
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    category_maps: dict[str, dict[Any, int]] = {}
    
    for col in categorical_cols:
        if col not in df.columns:
            category_maps[col] = {}
            continue
            
        # Get unique values, excluding NaN
        unique_vals = df[col].dropna().unique().tolist()
        # Sort for determinism
        unique_vals = sorted(unique_vals, key=lambda x: (str(type(x)), x))
        # Index 0 reserved for UNK
        category_maps[col] = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    
    return category_maps


def encode_categoricals(
    df: pd.DataFrame,
    category_maps: dict[str, dict[Any, int]],
    categorical_cols: list[str] | None = None,
) -> np.ndarray:
    """
    Encode categorical features using pre-built vocabulary maps.
    
    Unknown values map to index 0 (UNK).
    
    Args:
        df: Dataframe to encode
        category_maps: Vocabulary maps from build_category_maps()
        categorical_cols: List of categorical columns (default: CATEGORICAL_COLS)
        
    Returns:
        Encoded categorical array (n_rows, n_cat) as int
    """
    categorical_cols = categorical_cols or CATEGORICAL_COLS
    
    X_cat_list = []
    for col in categorical_cols:
        if col not in df.columns:
            codes = np.zeros(len(df), dtype=int)
        else:
            col_map = category_maps.get(col, {})
            # Map values, defaulting to 0 (UNK) for unknown
            codes = df[col].map(lambda x: col_map.get(x, 0)).fillna(0).astype(int).values
        X_cat_list.append(codes)
    
    return np.column_stack(X_cat_list) if X_cat_list else np.empty((len(df), 0), dtype=int)


__all__ = [
    "NUMERIC_COLS",
    "CATEGORICAL_COLS", 
    "FEATURE_COLS",
    "LABEL_COLS",
    "VALIDITY_COLS",
    "GROUP_COLS",
    "KEY_COLS",
    "compute_odds_lead_time_minutes",
    "compute_minutes_team_scaled",
    "prepare_features",
    "add_derived_features",
    "build_category_maps",
    "encode_categoricals",
]
