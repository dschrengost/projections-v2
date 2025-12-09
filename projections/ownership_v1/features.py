"""Feature definitions for ownership_v1 model."""

from __future__ import annotations

# Core value-based features (normalized within slate)
VALUE_FEATURES = [
    "value_per_k",        # proj_fpts / (salary / 1000)
    "salary_rank",        # rank within slate (1 = highest)
    "proj_fpts_rank",     # rank within slate (1 = highest)
    "proj_fpts_zscore",   # z-score within slate
]

# Salary tier indicators
SALARY_TIER_FEATURES = [
    "salary",
    "is_value_tier",      # salary < 5000
    "is_mid_tier",        # 5000 <= salary < 7500
    "is_high_tier",       # salary >= 7500
]

# Position one-hot encoding
POSITION_FEATURES = [
    "pos_PG",
    "pos_SG",
    "pos_SF",
    "pos_PF",
    "pos_C",
]

# Injury context features
INJURY_FEATURES = [
    "player_is_questionable",  # player on injury report (Q/GTD/PROB)
    "team_outs_count",         # how many teammates are OUT
]

# Historical ownership (player's typical chalk level)
HISTORICAL_FEATURES = [
    "player_own_avg_10",       # player's avg ownership over last 10 slates
]

# Slate structure features
SLATE_FEATURES = [
    "slate_size",              # number of players on slate
    "salary_pct_of_max",       # player's salary / max salary on slate
    "is_min_salary",           # player is at slate minimum salary
    "slate_near_min_count",    # count of punt options ($200 of min)
]

# Vegas context (optional - may be missing)
VEGAS_FEATURES = [
    "total_close",        # game total
    "spread_close",       # team spread (positive = underdog)
    "team_implied_total", # implied team total
]

# Base feature set for training
OWNERSHIP_FEATURES = (
    VALUE_FEATURES 
    + SALARY_TIER_FEATURES 
    + POSITION_FEATURES
)

# Features with injury context
OWNERSHIP_FEATURES_V2 = (
    OWNERSHIP_FEATURES
    + INJURY_FEATURES
)

# Features with injury + historical context
OWNERSHIP_FEATURES_V3 = (
    OWNERSHIP_FEATURES_V2
    + HISTORICAL_FEATURES
)

# Features with slate structure
OWNERSHIP_FEATURES_V4 = (
    OWNERSHIP_FEATURES_V3
    + SLATE_FEATURES
)

# Extended features including Vegas (for when available)
OWNERSHIP_FEATURES_EXTENDED = (
    OWNERSHIP_FEATURES_V4 
    + VEGAS_FEATURES
)


__all__ = [
    "VALUE_FEATURES",
    "SALARY_TIER_FEATURES", 
    "POSITION_FEATURES",
    "INJURY_FEATURES",
    "HISTORICAL_FEATURES",
    "SLATE_FEATURES",
    "VEGAS_FEATURES",
    "OWNERSHIP_FEATURES",
    "OWNERSHIP_FEATURES_V2",
    "OWNERSHIP_FEATURES_V3",
    "OWNERSHIP_FEATURES_V4",
    "OWNERSHIP_FEATURES_EXTENDED",
]
