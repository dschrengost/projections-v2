"""
Shared feature set definitions for rates_v1 models.

These mirror the stage configurations used by rates training and are reused
by downstream consumers (e.g., fpts_v2) to keep context columns consistent.

Stage progression:
- Stage 0: Leaky baseline with minutes_actual
- Stage 1: Non-leaky with minutes_pred_* from minutes model
- Stage 2: + tracking features (touches, drives, role cluster)
- Stage 3: + context features (vacancy, pace, efficiency)
- Stage 4: + recency features (last 1/3/5/10 game rolling averages)
- Stage 5: + extended FTA tracking features (drive FTA, paint touches, foul-drawing)
"""

from __future__ import annotations

from projections.features.action_props import ACTION_MARKET_FEATURE_COLUMNS

STAGE0_FEATURES = [
    "minutes_actual",
    "is_starter",
    "home_flag",
    "days_rest",
    "position_flags_PG",
    "position_flags_SG",
    "position_flags_SF",
    "position_flags_PF",
    "position_flags_C",
    "season_fga_per_min",
    "season_3pa_per_min",
    "season_fta_per_min",
    "season_ast_per_min",
    "season_tov_per_min",
    "season_reb_per_min",
    "season_stl_per_min",
    "season_blk_per_min",
]

STAGE1_FEATURES = [
    "minutes_pred_p50",
    "minutes_pred_spread",
    "minutes_pred_play_prob",
    "is_starter",
    "home_flag",
    "days_rest",
    "position_flags_PG",
    "position_flags_SG",
    "position_flags_SF",
    "position_flags_PF",
    "position_flags_C",
    "season_fga_per_min",
    "season_3pa_per_min",
    "season_fta_per_min",
    "season_ast_per_min",
    "season_tov_per_min",
    "season_reb_per_min",
    "season_stl_per_min",
    "season_blk_per_min",
    "spread_close",
    "total_close",
    "team_itt",
    "opp_itt",
    "has_odds",
]

TRACKING_FEATURES = [
    "track_touches_per_min_szn",
    "track_sec_per_touch_szn",
    "track_pot_ast_per_min_szn",
    "track_drives_per_min_szn",
    "track_role_cluster",
    "track_role_is_low_minutes",
]

# Extended tracking features for FTA prediction
TRACKING_FEATURES_FTA = [
    "track_drive_fta_per_min_szn",     # FTA from drives per minute
    "track_drive_pf_per_min_szn",      # Fouls drawn on drives per minute
    "track_paint_touches_per_min_szn", # Paint touches per minute
    "track_fta_per_drive_szn",         # FTA per drive (foul-drawing skill)
]

# Extended tracking features for 3PA prediction
TRACKING_FEATURES_3PA = [
    "track_catch_shoot_fg3a_per_min_szn",  # Catch-and-shoot 3PA volume per minute
    "track_pull_up_fg3a_per_min_szn",      # Pull-up 3PA volume per minute
    "track_pull_up_3pa_share_szn",         # Pull-up share of tracked 3PA profile
]

TRACKING_FEATURES_EXTENDED = TRACKING_FEATURES + TRACKING_FEATURES_FTA + TRACKING_FEATURES_3PA

FEATURES_STAGE2_TRACKING = STAGE1_FEATURES + TRACKING_FEATURES

CONTEXT_FEATURES = [
    "vac_min_szn",
    "vac_fga_szn",
    "vac_ast_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
    "season_fg2_pct",
    "season_fg3_pct",
    "season_ft_pct",
    "team_pace_szn",
    "team_off_rtg_szn",
    "team_def_rtg_szn",
    "opp_pace_szn",
    "opp_def_rtg_szn",
]

# Additional season-to-date per-minute features (split from total FGA)
ADDITIONAL_SEASON_FEATURES = [
    "season_fga2_per_min",
    "season_oreb_per_min",
    "season_dreb_per_min",
]

# Recency features: rolling windows of last 1/3/5/10 games
RECENCY_FEATURES = [
    "last1_minutes_sum",
    "last3_minutes_sum",
    "last5_minutes_sum",
    "last10_minutes_sum",
    "last1_fga2_per_min",
    "last1_fga3_per_min",
    "last1_fta_per_min",
    "last1_ast_per_min",
    "last1_tov_per_min",
    "last1_oreb_per_min",
    "last1_dreb_per_min",
    "last1_stl_per_min",
    "last1_blk_per_min",
    "last3_fga2_per_min",
    "last3_fga3_per_min",
    "last3_fta_per_min",
    "last3_ast_per_min",
    "last3_tov_per_min",
    "last3_oreb_per_min",
    "last3_dreb_per_min",
    "last3_stl_per_min",
    "last3_blk_per_min",
    "last5_fga2_per_min",
    "last5_fga3_per_min",
    "last5_fta_per_min",
    "last5_ast_per_min",
    "last5_tov_per_min",
    "last5_oreb_per_min",
    "last5_dreb_per_min",
    "last5_stl_per_min",
    "last5_blk_per_min",
    "last10_fga2_per_min",
    "last10_fga3_per_min",
    "last10_fta_per_min",
    "last10_ast_per_min",
    "last10_tov_per_min",
    "last10_oreb_per_min",
    "last10_dreb_per_min",
    "last10_stl_per_min",
    "last10_blk_per_min",
]

# Sample size features for shrinkage calibration
SAMPLE_SIZE_FEATURES = [
    "n_games_season",      # Number of games played in season (for shrinkage calibration)
    "season_minutes_sum",  # Total minutes played in season (sample size for per-min rates)
]

# FTA-specific opponent context
FTA_CONTEXT_FEATURES = [
    "opp_fta_allowed_per_game",  # Opponent's defensive FTA allowed (how many FTA they give up)
]

FEATURES_STAGE3_CONTEXT = FEATURES_STAGE2_TRACKING + CONTEXT_FEATURES

# Stage 4: Stage 3 + recency features + additional season splits
FEATURES_STAGE4_RECENCY = (
    FEATURES_STAGE3_CONTEXT
    + ADDITIONAL_SEASON_FEATURES
    + RECENCY_FEATURES
)

# Stage 5: Stage 4 + extended FTA/3PA tracking + FTA context + sample size
# This is the full feature set with all available features
FEATURES_STAGE5_FTA_TRACKING = (
    FEATURES_STAGE4_RECENCY
    + TRACKING_FEATURES_FTA
    + TRACKING_FEATURES_3PA
    + FTA_CONTEXT_FEATURES
    + SAMPLE_SIZE_FEATURES
)

# Stage 6: Stage 5 + Action Network prop-market features (excluding as-of timestamp).
ACTION_PROPS_FEATURES = [
    col for col in ACTION_MARKET_FEATURE_COLUMNS if col != "action_props_as_of_ts"
]
FEATURES_STAGE6_ACTION_PROPS = FEATURES_STAGE5_FTA_TRACKING + ACTION_PROPS_FEATURES


def get_rates_feature_sets() -> dict[str, list[str]]:
    """
    Return a mapping of feature set keys to their column lists.

    Keys: stage0, stage1, stage2_tracking, stage3_context, stage4_recency,
    stage5_fta_tracking, stage6_action_props
    """
    return {
        "stage0": STAGE0_FEATURES,
        "stage1": STAGE1_FEATURES,
        "stage2_tracking": FEATURES_STAGE2_TRACKING,
        "stage3_context": FEATURES_STAGE3_CONTEXT,
        "stage4_recency": FEATURES_STAGE4_RECENCY,
        "stage5_fta_tracking": FEATURES_STAGE5_FTA_TRACKING,
        "stage6_action_props": FEATURES_STAGE6_ACTION_PROPS,
    }


__all__ = [
    "get_rates_feature_sets",
    "STAGE0_FEATURES",
    "STAGE1_FEATURES",
    "TRACKING_FEATURES",
    "TRACKING_FEATURES_FTA",
    "TRACKING_FEATURES_3PA",
    "TRACKING_FEATURES_EXTENDED",
    "FEATURES_STAGE2_TRACKING",
    "CONTEXT_FEATURES",
    "ADDITIONAL_SEASON_FEATURES",
    "RECENCY_FEATURES",
    "SAMPLE_SIZE_FEATURES",
    "FTA_CONTEXT_FEATURES",
    "FEATURES_STAGE3_CONTEXT",
    "FEATURES_STAGE4_RECENCY",
    "FEATURES_STAGE5_FTA_TRACKING",
    "ACTION_PROPS_FEATURES",
    "FEATURES_STAGE6_ACTION_PROPS",
]
