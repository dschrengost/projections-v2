"""Schema definitions for rates_v1 features and outputs.

Follows the TableSchema + Pandera pattern from minutes_v1.schemas.
"""

from __future__ import annotations

from projections.minutes_v1.schemas import (
    FLOAT_DTYPE,
    INT_DTYPE,
    STRING_DTYPE,
    BOOL_DTYPE,
    UTC_TS,
    NAIVE_TS,
    TableSchema,
    enforce_schema,
    validate_with_pandera,
)
from projections.rates_v1.features import (
    STAGE0_FEATURES,
    STAGE1_FEATURES,
    FEATURES_STAGE2_TRACKING,
    FEATURES_STAGE3_CONTEXT,
    TRACKING_FEATURES,
    CONTEXT_FEATURES,
)

# Key columns for all rates feature tables
RATES_KEY_COLUMNS = ("game_id", "player_id", "team_id", "game_date")

# Dtype mapping for stage3 context features (the production feature set)
_STAGE3_DTYPES = {
    # Minutes predictions
    "minutes_pred_p50": FLOAT_DTYPE,
    "minutes_pred_spread": FLOAT_DTYPE,
    "minutes_pred_play_prob": FLOAT_DTYPE,
    # Player context
    "is_starter": INT_DTYPE,
    "home_flag": INT_DTYPE,
    "days_rest": INT_DTYPE,
    # Position flags
    "position_flags_PG": INT_DTYPE,
    "position_flags_SG": INT_DTYPE,
    "position_flags_SF": INT_DTYPE,
    "position_flags_PF": INT_DTYPE,
    "position_flags_C": INT_DTYPE,
    # Season per-minute stats
    "season_fga_per_min": FLOAT_DTYPE,
    "season_3pa_per_min": FLOAT_DTYPE,
    "season_fta_per_min": FLOAT_DTYPE,
    "season_ast_per_min": FLOAT_DTYPE,
    "season_tov_per_min": FLOAT_DTYPE,
    "season_reb_per_min": FLOAT_DTYPE,
    "season_stl_per_min": FLOAT_DTYPE,
    "season_blk_per_min": FLOAT_DTYPE,
    # Vegas context
    "spread_close": FLOAT_DTYPE,
    "total_close": FLOAT_DTYPE,
    "team_itt": FLOAT_DTYPE,
    "opp_itt": FLOAT_DTYPE,
    "has_odds": INT_DTYPE,
    # Tracking features
    "track_touches_per_min_szn": FLOAT_DTYPE,
    "track_sec_per_touch_szn": FLOAT_DTYPE,
    "track_pot_ast_per_min_szn": FLOAT_DTYPE,
    "track_drives_per_min_szn": FLOAT_DTYPE,
    "track_role_cluster": INT_DTYPE,
    "track_role_is_low_minutes": INT_DTYPE,
    # Vacancy features
    "vac_min_szn": FLOAT_DTYPE,
    "vac_fga_szn": FLOAT_DTYPE,
    "vac_ast_szn": FLOAT_DTYPE,
    "vac_min_guard_szn": FLOAT_DTYPE,
    "vac_min_wing_szn": FLOAT_DTYPE,
    "vac_min_big_szn": FLOAT_DTYPE,
    "season_fg2_pct": FLOAT_DTYPE,
    "season_fg3_pct": FLOAT_DTYPE,
    "season_ft_pct": FLOAT_DTYPE,
    # Team/opponent context
    "team_pace_szn": FLOAT_DTYPE,
    "team_off_rtg_szn": FLOAT_DTYPE,
    "team_def_rtg_szn": FLOAT_DTYPE,
    "opp_pace_szn": FLOAT_DTYPE,
    "opp_def_rtg_szn": FLOAT_DTYPE,
    # Keys
    "game_id": INT_DTYPE,
    "player_id": INT_DTYPE,
    "team_id": INT_DTYPE,
    "game_date": NAIVE_TS,
}

# Full column list for live rates features (keys + stage3 features)
FEATURES_RATES_V1_COLUMNS = (
    "game_id",
    "player_id",
    "team_id",
    "game_date",
    *FEATURES_STAGE3_CONTEXT,
)

FEATURES_RATES_V1_SCHEMA = TableSchema(
    name="features_rates_v1",
    columns=FEATURES_RATES_V1_COLUMNS,
    pandas_dtypes=_STAGE3_DTYPES,
    primary_key=("game_id", "player_id"),
    defaults={
        # Tracking features may be missing for some players
        "track_touches_per_min_szn": 0.0,
        "track_sec_per_touch_szn": 0.0,
        "track_pot_ast_per_min_szn": 0.0,
        "track_drives_per_min_szn": 0.0,
        "track_role_cluster": 0,
        "track_role_is_low_minutes": 0,
        # Vacancy features default to 0 when no absences
        "vac_min_szn": 0.0,
        "vac_fga_szn": 0.0,
        "vac_ast_szn": 0.0,
        "vac_min_guard_szn": 0.0,
        "vac_min_wing_szn": 0.0,
        "vac_min_big_szn": 0.0,
        "season_fg2_pct": 0.0,
        "season_fg3_pct": 0.0,
        "season_ft_pct": 0.0,
    },
)

# Output schema for rates predictions
RATES_PREDICTIONS_COLUMNS = (
    "game_id",
    "player_id",
    "team_id",
    "game_date",
    "pred_fga2_per_min",
    "pred_fga3_per_min",
    "pred_fta_per_min",
    "pred_ast_per_min",
    "pred_tov_per_min",
    "pred_oreb_per_min",
    "pred_dreb_per_min",
    "pred_stl_per_min",
    "pred_blk_per_min",
    "pred_fg2_pct",
    "pred_fg3_pct",
    "pred_ft_pct",
)

_RATES_PREDICTIONS_DTYPES = {
    "game_id": INT_DTYPE,
    "player_id": INT_DTYPE,
    "team_id": INT_DTYPE,
    "game_date": NAIVE_TS,
    "pred_fga2_per_min": FLOAT_DTYPE,
    "pred_fga3_per_min": FLOAT_DTYPE,
    "pred_fta_per_min": FLOAT_DTYPE,
    "pred_ast_per_min": FLOAT_DTYPE,
    "pred_tov_per_min": FLOAT_DTYPE,
    "pred_oreb_per_min": FLOAT_DTYPE,
    "pred_dreb_per_min": FLOAT_DTYPE,
    "pred_stl_per_min": FLOAT_DTYPE,
    "pred_blk_per_min": FLOAT_DTYPE,
    "pred_fg2_pct": FLOAT_DTYPE,
    "pred_fg3_pct": FLOAT_DTYPE,
    "pred_ft_pct": FLOAT_DTYPE,
}

RATES_PREDICTIONS_SCHEMA = TableSchema(
    name="rates_predictions",
    columns=RATES_PREDICTIONS_COLUMNS,
    pandas_dtypes=_RATES_PREDICTIONS_DTYPES,
    primary_key=("game_id", "player_id"),
)

# Efficiency label/target helpers
EFFICIENCY_TARGETS = [
    "fg2_pct",
    "fg3_pct",
    "ft_pct",
]


class FeatureSchemaMismatchError(ValueError):
    """Raised when features do not match expected schema."""


def validate_rates_features(
    df,
    *,
    schema: TableSchema = FEATURES_RATES_V1_SCHEMA,
    strict: bool = True,
):
    """Validate a rates features dataframe against the schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    schema : TableSchema
        Schema to validate against (default: FEATURES_RATES_V1_SCHEMA).
    strict : bool
        If True, raise FeatureSchemaMismatchError on validation failure.
        If False, return list of missing columns.

    Returns
    -------
    list[str]
        List of missing columns (empty if validation passes).

    Raises
    ------
    FeatureSchemaMismatchError
        If strict=True and validation fails.
    """
    expected = set(schema.columns)
    actual = set(df.columns)
    missing = expected - actual

    if missing:
        if strict:
            raise FeatureSchemaMismatchError(
                f"{schema.name} missing required columns: {sorted(missing)}"
            )
        return sorted(missing)

    return []


__all__ = [
    "FEATURES_RATES_V1_COLUMNS",
    "FEATURES_RATES_V1_SCHEMA",
    "FeatureSchemaMismatchError",
    "RATES_KEY_COLUMNS",
    "RATES_PREDICTIONS_COLUMNS",
    "RATES_PREDICTIONS_SCHEMA",
    "EFFICIENCY_TARGETS",
    "enforce_schema",
    "validate_rates_features",
    "validate_with_pandera",
]
