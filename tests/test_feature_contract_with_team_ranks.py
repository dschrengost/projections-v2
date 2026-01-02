"""Tests that within-team rank features are included in the feature contract."""

from __future__ import annotations

import pandas as pd

from projections.models.feature_contract import filter_to_contract_features


def test_filter_to_contract_features_includes_within_team_ranks() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "player_id": [10, 11],
            "team_id": [100, 100],
            "minutes": [30.0, 10.0],
            "roll_mean_10": [28.0, 8.0],
            "team_roll_mean_10_rank": [1, 2],
            "team_roll_mean_10_rank_pct": [0.0, 1.0],
            "team_roll_mean_10_gap_to_8th": [20.0, 0.0],
            "team_roll_mean_10_gap_to_10th": [20.0, 0.0],
            "team_roll_mean_10_is_top8": [1, 1],
            "team_roll_mean_10_is_top10": [1, 1],
        }
    )
    feats = filter_to_contract_features(df, target_col="minutes")
    assert "team_roll_mean_10_rank" in feats
    assert "team_roll_mean_10_rank_pct" in feats
    assert "team_roll_mean_10_gap_to_8th" in feats
    assert "team_roll_mean_10_gap_to_10th" in feats
    assert "team_roll_mean_10_is_top8" in feats
    assert "team_roll_mean_10_is_top10" in feats

