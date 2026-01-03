from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from projections.builders.features_builder import FeatureBuildConfig, SharedFeaturesBuilder


def test_shared_features_builder_ensures_required_scoring_columns() -> None:
    cfg = FeatureBuildConfig(
        data_root=Path("/tmp"),
        season=2025,
        as_of_ts=pd.Timestamp("2026-01-01T00:00:00Z"),
        target_day=date(2026, 1, 1),
    )
    builder = SharedFeaturesBuilder(cfg)

    df = pd.DataFrame(
        {
            "game_id": [22500476],
            "player_id": [202687],
            "team_id": [1610612737],
        }
    )
    result = builder._ensure_schema_columns(df)

    required = {
        "starter_flag",
        "team_pace_szn",
        "team_off_rtg_szn",
        "team_def_rtg_szn",
        "opp_pace_szn",
        "opp_def_rtg_szn",
        "vac_min_szn",
        "vac_min_guard_szn",
        "vac_min_wing_szn",
        "vac_min_big_szn",
    }
    assert required.issubset(result.columns)
