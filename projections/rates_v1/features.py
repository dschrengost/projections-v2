from __future__ import annotations

"""
Shared feature set definitions for rates_v1 models.

These mirror the stage configurations used by rates training and are reused
by downstream consumers (e.g., fpts_v2) to keep context columns consistent.
"""

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

FEATURES_STAGE2_TRACKING = STAGE1_FEATURES + TRACKING_FEATURES

CONTEXT_FEATURES = [
    "vac_min_szn",
    "vac_fga_szn",
    "vac_ast_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
    "team_pace_szn",
    "team_off_rtg_szn",
    "team_def_rtg_szn",
    "opp_pace_szn",
    "opp_def_rtg_szn",
]

FEATURES_STAGE3_CONTEXT = FEATURES_STAGE2_TRACKING + CONTEXT_FEATURES


def get_rates_feature_sets() -> dict[str, list[str]]:
    """
    Return a mapping of feature set keys to their column lists.

    Keys: stage0, stage1, stage2_tracking, stage3_context
    """
    return {
        "stage0": STAGE0_FEATURES,
        "stage1": STAGE1_FEATURES,
        "stage2_tracking": FEATURES_STAGE2_TRACKING,
        "stage3_context": FEATURES_STAGE3_CONTEXT,
    }


__all__ = [
    "get_rates_feature_sets",
    "STAGE0_FEATURES",
    "STAGE1_FEATURES",
    "TRACKING_FEATURES",
    "FEATURES_STAGE2_TRACKING",
    "CONTEXT_FEATURES",
    "FEATURES_STAGE3_CONTEXT",
]
