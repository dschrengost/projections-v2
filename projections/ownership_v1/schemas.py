"""Schema validation for ownership_v1 model inputs.

Ensures strict parity between training and inference data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class OwnershipInputSchema:
    """Schema definition for ownership model inputs."""
    
    # Required raw input columns
    REQUIRED_RAW_COLS: tuple[str, ...] = (
        "salary",
        "proj_fpts", 
        "pos",
    )
    
    # Features computed during inference
    COMPUTED_FEATURES: tuple[str, ...] = (
        "value_per_k",
        "salary_rank",
        "proj_fpts_rank",
        "proj_fpts_zscore",
        "is_value_tier",
        "is_mid_tier",
        "is_high_tier",
        "pos_PG",
        "pos_SG",
        "pos_SF",
        "pos_PF",
        "pos_C",
        # V5/V6 computed features (safe to compute at inference time)
        "value_vs_slate_avg",
        "salary_vs_median",
        "is_min_priced_by_pos",
        "game_count_on_slate",
        "value_x_value_tier",
        "outs_x_salary_rank",
    )
    
    # Optional enrichment columns (filled with defaults if missing)
    OPTIONAL_ENRICHMENT_COLS: tuple[str, ...] = (
        "player_is_questionable",  # Default: 0
        "team_outs_count",          # Default: 0
        "player_own_avg_10",        # Default: 0
        # V6 player popularity features
        "player_own_median",        # Default: 0
        "player_own_variance",      # Default: 0
        "player_chalk_rate",        # Default: 0
    )
    
    # Slate-level features (computed from data)
    SLATE_FEATURES: tuple[str, ...] = (
        "slate_size",
        "salary_pct_of_max",
        "is_min_salary",
        "slate_near_min_count",
    )


# Singleton schema instance
SCHEMA = OwnershipInputSchema()


def validate_raw_input(df: pd.DataFrame) -> list[str]:
    """
    Validate that raw input DataFrame has required columns.
    
    Returns:
        List of missing column names (empty if valid).
    """
    return [col for col in SCHEMA.REQUIRED_RAW_COLS if col not in df.columns]


def fill_optional_columns(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """
    Fill optional columns with defaults if missing.
    
    Args:
        df: Input DataFrame.
        copy: Whether to copy DataFrame before modifying.
    
    Returns:
        DataFrame with optional columns filled.
    """
    if copy:
        df = df.copy()
    
    defaults = {
        "player_is_questionable": 0,
        "team_outs_count": 0,
        "player_own_avg_10": 0.0,
        "player_own_median": 0.0,
        "player_own_variance": 0.0,
        "player_chalk_rate": 0.0,
    }
    
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    
    return df


def validate_feature_set(
    df: pd.DataFrame, 
    expected_features: Sequence[str],
) -> list[str]:
    """
    Validate that DataFrame has exactly the expected features.
    
    Args:
        df: DataFrame with computed features.
        expected_features: List of feature column names expected by model.
    
    Returns:
        List of missing feature names.
    """
    return [col for col in expected_features if col not in df.columns]


def prepare_model_input(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Prepare final model input with only the expected feature columns.
    
    Ensures model sees ONLY the features it was trained on.
    
    Args:
        df: DataFrame with all computed features.
        feature_cols: Ordered list of feature columns from model bundle.
    
    Returns:
        DataFrame with exactly feature_cols in order.
    
    Raises:
        KeyError: If any feature columns are missing.
    """
    missing = validate_feature_set(df, feature_cols)
    if missing:
        raise KeyError(f"Missing required features for model: {missing}")
    
    return df[list(feature_cols)]


__all__ = [
    "SCHEMA",
    "OwnershipInputSchema",
    "validate_raw_input",
    "fill_optional_columns",
    "validate_feature_set",
    "prepare_model_input",
]
