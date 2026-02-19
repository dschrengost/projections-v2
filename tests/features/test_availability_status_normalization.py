from __future__ import annotations

import pandas as pd

from projections.features.availability import attach_availability_features
from projections.minutes_v1.constants import AvailabilityStatus, STATUS_PRIORS


def test_attach_availability_features_normalizes_ava_status() -> None:
    base = pd.DataFrame(
        {
            "game_id": ["0022500505"],
            "player_id": [203500],
            "tip_ts": [pd.Timestamp("2026-01-05T01:00:00Z")],
        }
    )
    prepared_injuries = pd.DataFrame(
        {
            "game_id": ["0022500505"],
            "player_id": [203500],
            "status": ["Ava"],  # live feed variant
            "restriction_flag": [False],
            "ramp_flag": [False],
            "games_since_return": [0],
            "days_since_return": [0],
            "as_of_ts": [pd.Timestamp("2026-01-05T00:00:00Z")],
        }
    )

    out = attach_availability_features(base, prepared_injuries=prepared_injuries)

    assert out.loc[0, "status"] == AvailabilityStatus.AVAILABLE
    assert float(out.loc[0, "prior_play_prob"]) == STATUS_PRIORS[AvailabilityStatus.AVAILABLE]
    assert int(out.loc[0, "is_out"]) == 0
    assert int(out.loc[0, "is_q"]) == 0
    assert int(out.loc[0, "is_prob"]) == 0


def test_attach_availability_features_sets_available_prior_when_no_injury_row() -> None:
    base = pd.DataFrame(
        {
            "game_id": ["0022500506"],
            "player_id": [1631105],
            "tip_ts": [pd.Timestamp("2026-01-06T01:00:00Z")],
        }
    )

    out = attach_availability_features(base, injuries_snapshot=pd.DataFrame())

    assert out.loc[0, "status"] == "Ava"
    assert float(out.loc[0, "prior_play_prob"]) == STATUS_PRIORS[AvailabilityStatus.AVAILABLE]
    assert int(out.loc[0, "injury_snapshot_missing"]) == 1
    assert bool(out.loc[0, "injury_row_present"]) is False
