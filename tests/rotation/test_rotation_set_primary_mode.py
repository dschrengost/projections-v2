import numpy as np
import pandas as pd

from projections.cli.score_minutes_rotation_set_v1 import (
    _derive_gate_probs,
    _derive_out_mask,
    _extract_baseline_minutes,
    _scale_minutes_to_team_target,
)
from projections.minutes import PLAY_THRESHOLD_MINUTES, ROTATION_THRESHOLD_MINUTES


def test_primary_mode_out_zero_and_scale_to_240() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2],
            "team_id": [10, 10, 10, 20, 20],
            "player_id": [101, 102, 103, 201, 202],
            "status": ["OUT", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE"],
            "is_out": [1, 0, 0, 0, 0],
        }
    )
    minutes = pd.Series([30.0, 20.0, 10.0, 100.0, 20.0], index=df.index)
    out_mask = _derive_out_mask(df)
    minutes = minutes.where(~out_mask, 0.0)

    scaled, summary = _scale_minutes_to_team_target(df, minutes, team_target=240.0)
    df["scaled"] = scaled

    team_sums = df.groupby(["game_id", "team_id"])["scaled"].sum()
    assert (team_sums.round(6) == 240.0).all()
    assert float(df.loc[0, "scaled"]) == 0.0
    assert summary["zero_sum_teams"] == 0


def test_primary_mode_out_mask_respects_lineup_role_out() -> None:
    df = pd.DataFrame(
        {
            "status": ["ACTIVE", "ACTIVE", "ACTIVE"],
            "is_out": [0, 0, 0],
            "lineup_role": ["out", "projected_starter", pd.NA],
        }
    )

    out_mask = _derive_out_mask(df)
    assert out_mask.tolist() == [True, False, False]


def test_primary_mode_gate_probs_use_thresholds() -> None:
    minutes = pd.Series(
        [
            0.0,
            PLAY_THRESHOLD_MINUTES - 0.01,
            PLAY_THRESHOLD_MINUTES,
            ROTATION_THRESHOLD_MINUTES - 0.01,
            ROTATION_THRESHOLD_MINUTES,
        ]
    )
    out_mask = pd.Series([False, False, False, False, True])
    play_prob, rotation_prob = _derive_gate_probs(minutes, out_mask=out_mask)

    assert play_prob.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert rotation_prob.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_extract_baseline_minutes_from_minutes_p50_model() -> None:
    """Test that baseline_minutes_p50 is extracted from minutes_p50_model (priority 1)."""
    df = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "minutes_p50_model": [25.0, 30.0, 15.0],
            "minutes_final": [20.0, 25.0, 10.0],
            "minutes_p50": [0.0, 0.0, 0.0],  # Would be all zeros if used
        }
    )
    result = _extract_baseline_minutes(df)

    assert result.source_col == "minutes_p50_model"
    assert result.gt0_count == 3
    assert np.allclose(result.values, [25.0, 30.0, 15.0])
    assert result.min_val == 15.0
    assert result.max_val == 30.0


def test_extract_baseline_minutes_fallback_to_minutes_final() -> None:
    """Test fallback to minutes_final when minutes_p50_model is all zeros."""
    df = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "minutes_p50_model": [0.0, 0.0, 0.0],  # All zeros, skip
            "minutes_final": [20.0, 25.0, 10.0],
            "minutes_p50": [0.0, 0.0, 0.0],
        }
    )
    result = _extract_baseline_minutes(df)

    assert result.source_col == "minutes_final"
    assert result.gt0_count == 3
    assert np.allclose(result.values, [20.0, 25.0, 10.0])


def test_extract_baseline_minutes_fallback_to_minutes_p50() -> None:
    """Test fallback to minutes_p50 when both minutes_p50_model and minutes_final are missing."""
    df = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "minutes_p50": [18.0, 22.0, 8.0],
        }
    )
    result = _extract_baseline_minutes(df)

    assert result.source_col == "minutes_p50"
    assert result.gt0_count == 3
    assert np.allclose(result.values, [18.0, 22.0, 8.0])


def test_extract_baseline_minutes_returns_zeros_when_no_valid_column() -> None:
    """Test that zeros are returned when no valid baseline column exists."""
    df = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "some_other_col": [1, 2, 3],
        }
    )
    result = _extract_baseline_minutes(df)

    assert result.source_col == "none"
    assert result.gt0_count == 0
    assert np.allclose(result.values, [0.0, 0.0, 0.0])
    assert result.min_val == 0.0
    assert result.max_val == 0.0
