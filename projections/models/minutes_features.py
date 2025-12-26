"""Shared feature metadata for minutes models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

DEFAULT_MINUTES_ALPHAS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
MINUTES_TARGET_COL = "minutes"
MINUTES_STATUS_COL = "status"
INJURY_SNAPSHOT_MISSING_COL = "injury_snapshot_missing"

# Columns that should never be surfaced as trainable features.
EXCLUDED_FEATURE_COLUMNS = {
    # Key columns (identifiers, not features)
    "game_id",
    "player_id",
    "team_id",
    "opponent_team_id",
    "season",
    "game_date",
    "tip_ts",
    "feature_as_of_ts",
    # Horizonized training metadata (kept for evaluation, not as a model feature).
    "horizon_min",
    "split",
    "ingested_ts",
    "time_to_tip_min",
    "odds_missing",
    # Label columns - these are ground truth derived post-game, not pre-game features.
    "starter_flag_label",
    "starter_flag_x",
    "starter_flag_y",
    "label_frozen_ts",
    # =========================================================================
    # FORBIDDEN FEATURES - Post-game/leaky columns that MUST NOT be used
    # See projections/models/feature_contract.py for the authoritative list
    # =========================================================================
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
    # DK salary - excluded to avoid duplication bugs
    "salary",
    # Biometrics - not reliably available in live
    "age",
    "height_in",
    "weight_lb",
    # Derivable but inconsistently available
    "is_home",
    # Enrichment suffix columns - these are training-only artifacts
    # The live pipeline uses the non-suffix versions
    "vac_min_szn_enrich",
    "vac_min_guard_szn_enrich",
    "vac_min_wing_szn_enrich",
    "vac_min_big_szn_enrich",
    "team_pace_szn_enrich",
    "opp_pace_szn_enrich",
    "team_off_rtg_szn_enrich",
    "team_def_rtg_szn_enrich",
    "opp_def_rtg_szn_enrich",
    # Snapshot timestamp columns (metadata, not features)
    "frozen_at",
    "snapshot_ts",
    "snapshot_type",
}

# The legacy pipeline only exposes numeric features, so the default categorical list
# is empty. The helper below lets experiments override this when needed.
DEFAULT_MINUTES_CAT_COLS: tuple[str, ...] = ()


@dataclass(frozen=True)
class MinutesFeatureSpec:
    """Explicit split of continuous vs categorical feature columns."""

    continuous: list[str]
    categorical: list[str]

    def to_metadata(self) -> dict[str, Any]:
        return {"continuous": self.continuous, "categorical": self.categorical}


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    target_col: str = MINUTES_TARGET_COL,
    excluded: Iterable[str] | None = None,
) -> list[str]:
    """Infer usable feature columns by dtype, mirroring the LightGBM pipeline."""

    blocked = set(EXCLUDED_FEATURE_COLUMNS)
    if excluded is not None:
        blocked.update(excluded)

    candidates: list[str] = []
    for col in df.columns:
        if col == target_col or col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    if not candidates:
        raise ValueError("Unable to infer feature columns — provide them explicitly.")
    return sorted(candidates)


def build_feature_spec(
    feature_columns: Sequence[str],
    *,
    categorical_columns: Sequence[str] | None = None,
) -> MinutesFeatureSpec:
    """Split feature columns into continuous/categorical subsets."""

    categorical_list = list(categorical_columns or DEFAULT_MINUTES_CAT_COLS)
    cat_set = set(categorical_list)
    missing = cat_set.difference(feature_columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Categorical columns not found in feature set: {missing_cols}")
    continuous = [col for col in feature_columns if col not in cat_set]
    return MinutesFeatureSpec(continuous=continuous, categorical=categorical_list)


__all__ = [
    "DEFAULT_MINUTES_ALPHAS",
    "DEFAULT_MINUTES_CAT_COLS",
    "EXCLUDED_FEATURE_COLUMNS",
    "INJURY_SNAPSHOT_MISSING_COL",
    "MINUTES_STATUS_COL",
    "MINUTES_TARGET_COL",
    "MinutesFeatureSpec",
    "build_feature_spec",
    "infer_feature_columns",
]
