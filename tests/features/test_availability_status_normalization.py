from __future__ import annotations

import pandas as pd

from projections.features.availability import attach_availability_features, normalize_status
from projections.minutes_v1.constants import AvailabilityStatus, STATUS_PRIORS
from projections.minutes_v1.season_dataset import _status_from_raw


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


def test_normalize_status_maps_doubtful_variants_to_out() -> None:
    assert normalize_status("Doubtful") == AvailabilityStatus.OUT
    assert normalize_status("D") == AvailabilityStatus.OUT
    assert normalize_status("Doubtful (ankle)") == AvailabilityStatus.OUT


def test_attach_availability_features_clears_is_out_when_upgraded() -> None:
    base = pd.DataFrame(
        {
            "game_id": ["0022500505"],
            "player_id": [203500],
            "tip_ts": [pd.Timestamp("2026-01-05T01:00:00Z")],
        }
    )
    doubtful = pd.DataFrame(
        {
            "game_id": ["0022500505"],
            "player_id": [203500],
            "status": ["Doubtful (knee)"],
            "restriction_flag": [False],
            "ramp_flag": [False],
            "games_since_return": [0],
            "days_since_return": [0],
            "as_of_ts": [pd.Timestamp("2026-01-05T00:00:00Z")],
        }
    )
    available = doubtful.assign(status="Ava")

    out_d = attach_availability_features(base, prepared_injuries=doubtful)
    out_a = attach_availability_features(base, prepared_injuries=available)

    assert out_d.loc[0, "status"] == AvailabilityStatus.OUT
    assert int(out_d.loc[0, "is_out"]) == 1
    assert float(out_d.loc[0, "prior_play_prob"]) == STATUS_PRIORS[AvailabilityStatus.OUT]

    assert out_a.loc[0, "status"] == AvailabilityStatus.AVAILABLE
    assert int(out_a.loc[0, "is_out"]) == 0
    assert float(out_a.loc[0, "prior_play_prob"]) == STATUS_PRIORS[AvailabilityStatus.AVAILABLE]


def test_attach_availability_features_handles_mixed_enum_status_and_missing_rows() -> None:
    base = pd.DataFrame(
        {
            "game_id": ["0022500505", "0022500505"],
            "player_id": [203500, 203501],
            "tip_ts": [pd.Timestamp("2026-01-05T01:00:00Z")] * 2,
        }
    )
    prepared_injuries = pd.DataFrame(
        {
            "game_id": ["0022500505"],
            "player_id": [203500],
            "status": [AvailabilityStatus.OUT],
            "restriction_flag": [False],
            "ramp_flag": [False],
            "games_since_return": [0],
            "days_since_return": [0],
            "as_of_ts": [pd.Timestamp("2026-01-05T00:00:00Z")],
        }
    )

    out = attach_availability_features(base, prepared_injuries=prepared_injuries)

    assert out.loc[0, "status"] == AvailabilityStatus.OUT
    assert out.loc[1, "status"] == "Ava"
    assert bool(out.loc[0, "injury_row_present"]) is True
    assert bool(out.loc[1, "injury_row_present"]) is False


def test_status_from_raw_maps_doubtful_to_out() -> None:
    assert _status_from_raw("Doubtful") == AvailabilityStatus.OUT
    assert _status_from_raw("D") == AvailabilityStatus.OUT
