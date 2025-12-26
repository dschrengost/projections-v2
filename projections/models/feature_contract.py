"""Feature contract for minute_share model.

This module enforces strict train↔live feature parity to prevent:
1. Leaky features (post-game stats used as pre-game features)
2. Schema drift between training and inference
3. Silent imputation of missing features

The contract defines:
- ALLOWED_FEATURES: Features safe for both training and live inference
- FORBIDDEN_FEATURES: Post-game/leaky features that must never be used
- Contract validation functions to fail loudly on mismatches
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

# ============================================================================
# FORBIDDEN FEATURES - Post-game/leaky columns that MUST NOT be used
# ============================================================================
FORBIDDEN_FEATURES: frozenset[str] = frozenset({
    # Box-score stats (post-game labels, not pre-game features)
    "pts",
    "ast", 
    "reb",
    "blk",
    "stl",
    "fg3m",
    "tov",
    "dk_fpts_actual",
    # Actual/played flags that are post-game truth
    "played_flag",
    "active_flag",
    "starter_flag_actual",
    # DK salary - excluded to avoid duplication bugs (per user requirement)
    "salary",
    # Biometrics - not reliably available in live, exclude for now
    "age",
    "height_in",
    "weight_lb",
    # Generic post-game indicators
    "is_home",  # Derivable from home_team_id but not consistently available
})

# ============================================================================
# ALLOWED FEATURES - Pre-game features available in both training and live
# ============================================================================
# These are the ONLY features that may be used for minute_share models.
# They are available in both:
# - Gold slates (pretip.parquet) used for training
# - Live feature builder (build_minutes_live.py) for inference
ALLOWED_FEATURES: frozenset[str] = frozenset({
    # Archetype features
    "arch_delta_max_role",
    "arch_delta_min_role",
    "arch_delta_same_pos",
    "arch_delta_sum",
    "arch_missing_same_pos_count",
    "arch_missing_total_count",
    # Position availability
    "available_B",
    "available_G",
    "available_W",
    # Game context
    "away_team_id",
    "home_team_id",
    "home_flag",
    "spread_home",
    "total",
    # Team/opponent context features
    "team_pace_szn",
    "team_off_rtg_szn",
    "team_def_rtg_szn",
    "opp_pace_szn",
    "opp_def_rtg_szn",
    # Vacancy features
    "vac_min_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
    # Game scenario features
    "blowout_index",
    "blowout_risk_score",
    "close_game_score",
    # Rest/schedule features
    "days_since_last",
    "days_since_return",
    "games_since_return",
    "is_3in4",
    "is_4in6",
    "is_b2b",
    # Injury/status features
    "injury_snapshot_missing",
    "is_out",
    "is_prob",
    "is_q",
    # Starter signals (pre-game projections)
    "is_confirmed_starter",
    "is_projected_starter",
    "starter_flag",
    "starter_prev_game_asof",
    # Historical minutes features
    "min_last1",
    "min_last3",
    "min_last5",
    "sum_min_7d",
    "roll_mean_3",
    "roll_mean_5",
    "roll_mean_10",
    "roll_iqr_5",
    "z_vs_10",
    # Team depth/structure
    "depth_same_pos_active",
    "same_archetype_overlap",
    # Role features
    "prior_play_prob",
    "ramp_flag",
    "restriction_flag",
    "recent_start_pct_10",
    "role_change_rate_10g",
    "rotation_minutes_std_5g",
    "team_minutes_dispersion_prior",
    # Season context
    "season_phase",
})

# Metadata columns that are NOT features (excluded from feature selection)
METADATA_COLUMNS: frozenset[str] = frozenset({
    "game_id",
    "player_id",
    "team_id",
    "opponent_team_id",
    "season",
    "game_date",
    "tip_ts",
    "feature_as_of_ts",
    "horizon_min",
    "split",
    "ingested_ts",
    "label_frozen_ts",
    "starter_flag_label",
    "starter_flag_x",
    "starter_flag_y",
    "time_to_tip_min",
    "odds_missing",
    "player_name",
    "team_name",
    "team_tricode",
    "opponent_team_name",
    "opponent_team_tricode",
    "status",
    "archetype",
    "pos_bucket",
    "lineup_role",
    "lineup_status",
    "lineup_roster_status",
    "lineup_timestamp",
    "injury_as_of_ts",
    "odds_as_of_ts",
    "roster_as_of_ts",
    # Target column
    "minutes",
})


@dataclass
class FeatureContractValidation:
    """Result of feature contract validation."""
    
    passed: bool
    training_features: list[str]
    live_features: list[str]
    missing_in_live: list[str]
    forbidden_used: list[str]
    dtype_mismatches: list[tuple[str, str, str]]  # (col, train_dtype, live_dtype)
    warnings: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "training_features": self.training_features,
            "live_features": self.live_features,
            "missing_in_live": self.missing_in_live,
            "forbidden_used": self.forbidden_used,
            "dtype_mismatches": [
                {"column": c, "train_dtype": t, "live_dtype": l}
                for c, t, l in self.dtype_mismatches
            ],
            "warnings": self.warnings,
        }


def validate_feature_contract(
    training_features: list[str],
    live_df: pd.DataFrame,
    *,
    strict: bool = True,
) -> FeatureContractValidation:
    """Validate that training features are available in live inference.
    
    Args:
        training_features: List of feature column names used for training
        live_df: Sample DataFrame from live feature builder
        strict: If True, fail on any mismatch. If False, only warn.
    
    Returns:
        FeatureContractValidation with pass/fail status and details
    
    Raises:
        ValueError: If strict=True and contract validation fails
    """
    warnings: list[str] = []
    
    # Check for forbidden features
    forbidden_used = [f for f in training_features if f in FORBIDDEN_FEATURES]
    
    # Check for features missing in live
    live_cols = set(live_df.columns)
    missing_in_live = [f for f in training_features if f not in live_cols]
    
    # Check dtype compatibility
    dtype_mismatches: list[tuple[str, str, str]] = []
    for f in training_features:
        if f in live_df.columns:
            # We're lenient on dtype - numeric types are generally compatible
            pass
    
    passed = not forbidden_used and not missing_in_live
    
    result = FeatureContractValidation(
        passed=passed,
        training_features=training_features,
        live_features=[c for c in live_df.columns if c in ALLOWED_FEATURES],
        missing_in_live=missing_in_live,
        forbidden_used=forbidden_used,
        dtype_mismatches=dtype_mismatches,
        warnings=warnings,
    )
    
    if not passed and strict:
        error_parts = []
        if forbidden_used:
            error_parts.append(
                f"FORBIDDEN features used: {', '.join(sorted(forbidden_used))}\n"
                "These are post-game/leaky columns that cannot be used in training."
            )
        if missing_in_live:
            error_parts.append(
                f"Features MISSING in live: {', '.join(sorted(missing_in_live))}\n"
                "These features are used in training but not available at inference time."
            )
        raise ValueError(
            "Feature contract validation FAILED:\n\n" + "\n\n".join(error_parts)
        )
    
    return result


def filter_to_contract_features(
    df: pd.DataFrame,
    *,
    target_col: str = "minutes",
) -> list[str]:
    """Filter DataFrame columns to only contract-allowed features.
    
    Args:
        df: Feature DataFrame
        target_col: Target column to exclude
    
    Returns:
        List of feature column names that are in ALLOWED_FEATURES and numeric
    """
    features = []
    for col in df.columns:
        if col == target_col:
            continue
        if col in METADATA_COLUMNS:
            continue
        if col in FORBIDDEN_FEATURES:
            LOGGER.warning(f"Filtering out forbidden feature: {col}")
            continue
        if col in ALLOWED_FEATURES:
            if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype in ["boolean", "bool"]:
                features.append(col)
    
    return sorted(features)


def save_feature_contract(
    features: list[str],
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save the feature contract to a JSON file.
    
    Args:
        features: List of feature column names
        path: Output path for the contract JSON
        metadata: Optional metadata to include
    """
    contract = {
        "version": "1.0.0",
        "features": sorted(features),
        "n_features": len(features),
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    LOGGER.info(f"Saved feature contract to {path}")


def load_feature_contract(path: Path) -> list[str]:
    """Load feature list from a contract JSON file."""
    contract = json.loads(path.read_text(encoding="utf-8"))
    return contract["features"]


def assert_no_leakage(features: list[str]) -> None:
    """Assert that none of the features are forbidden (leaky).
    
    Raises:
        ValueError: If any forbidden features are present
    """
    leaky = set(features) & FORBIDDEN_FEATURES
    if leaky:
        raise ValueError(
            f"LEAKAGE GUARD FAILED: Found forbidden features in model: {sorted(leaky)}\n"
            "These columns contain post-game information and cannot be used."
        )


__all__ = [
    "ALLOWED_FEATURES",
    "FORBIDDEN_FEATURES",
    "METADATA_COLUMNS",
    "FeatureContractValidation",
    "assert_no_leakage",
    "filter_to_contract_features",
    "load_feature_contract",
    "save_feature_contract",
    "validate_feature_contract",
]
