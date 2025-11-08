"""Tests for feature engineering helpers."""

import pandas as pd

from projections import features


def test_add_rolling_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "player_id": [1, 1, 1],
            "game_date": pd.date_range("2023-01-01", periods=3, freq="D"),
            "minutes": [20, 25, 30],
        }
    )
    enriched = features.add_rolling_features(
        df, group_cols=["player_id"], target_col="minutes", windows=(2,)
    )
    assert "minutes_rolling_2" in enriched.columns
    assert enriched.loc[2, "minutes_rolling_2"] == 27.5


def test_build_feature_target_split_returns_expected_shapes():
    df = pd.DataFrame(
        {
            "player_id": [1, 2],
            "game_date": pd.date_range("2023-01-01", periods=2, freq="D"),
            "minutes": [20, 25],
        }
    )
    X, y = features.build_feature_target_split(
        df, target_col="minutes", drop_cols=["player_id", "game_date"]
    )
    assert y.tolist() == [20, 25]
    assert list(X.columns) == []
