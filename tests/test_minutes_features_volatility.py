"""Tests for new volatility/regime features used by minutes models."""

from __future__ import annotations

import pandas as pd

from projections.features import depth as depth_features
from projections.features import game_env as game_env_features
from projections.features import trend as trend_features


def test_trend_features_include_volatility_signals() -> None:
    df = pd.DataFrame(
        {
            "player_id": [1] * 6 + [2] * 3,
            "team_id": [7] * 6 + [8] * 3,
            "season": [2025] * 9,
            "game_date": pd.date_range("2025-01-01", periods=9, freq="D"),
            "minutes": [24, 18, 30, 12, 20, 28, 16, 18, 20],
            "starter_flag": [True, True, False, False, True, True, False, True, False],
        }
    )

    enriched = trend_features.attach_trend_features(df)

    assert "rotation_minutes_std_5g" in enriched
    assert "role_change_rate_10g" in enriched
    assert "season_phase" in enriched

    first_player = enriched[enriched["player_id"] == 1]
    assert first_player["rotation_minutes_std_5g"].iloc[0] == 0.0
    assert first_player["rotation_minutes_std_5g"].iloc[-1] > 0
    assert first_player["role_change_rate_10g"].between(0.0, 1.0).all()
    assert first_player["season_phase"].between(0.0, 1.0).all()
    assert first_player["season_phase"].is_monotonic_increasing


def test_depth_same_pos_active_counts_active_teammates() -> None:
    base = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "player_id": [100, 101, 102],
            "game_date": pd.to_datetime(["2025-01-02"] * 3),
        }
    )
    roster = pd.DataFrame(
        {
            "team_id": [10, 10, 10],
            "game_date": pd.to_datetime(["2025-01-02"] * 3),
            "player_id": [100, 101, 102],
            "active_flag": [True, True, True],
            "listed_pos": ["G", "G", "C"],
            "as_of_ts": pd.Timestamp("2025-01-01", tz="UTC"),
        }
    )

    enriched = depth_features.attach_depth_features(base, roster)
    guard_depth = enriched.loc[enriched["archetype"].isin(["G"]), "depth_same_pos_active"]
    assert (guard_depth == 1).all()
    big_depth = enriched.loc[enriched["archetype"] == "B", "depth_same_pos_active"].iloc[0]
    assert big_depth == 0


def test_game_env_scores_from_spread() -> None:
    base = pd.DataFrame({"game_id": [1, 2]})
    odds = pd.DataFrame(
        {
            "game_id": [1, 2],
            "home_line": [-12, None],
            "total": [220, 215],
            "as_of_ts": ["2025-01-01", "2025-01-01"],
        }
    )

    enriched = game_env_features.attach_game_environment_features(base, odds)
    assert "blowout_risk_score" in enriched
    assert "close_game_score" in enriched
    assert enriched.loc[0, "blowout_risk_score"] > 0.0
    assert 0.0 <= enriched.loc[0, "close_game_score"] <= 1.0
    assert enriched.loc[1, ["blowout_risk_score", "close_game_score"]].eq(0.5).all()
