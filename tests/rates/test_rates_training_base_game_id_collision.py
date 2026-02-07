from __future__ import annotations

import pandas as pd
import pytest

from scripts.rates.build_training_base import _assert_unique_training_keys, build_features


def test_build_features_respects_game_date_when_game_id_reused() -> None:
    stats = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_id": 22500529,
                "game_date": pd.Timestamp("2026-01-08"),
                "tip_ts": pd.Timestamp("2026-01-09T01:00:00Z"),
                "team_id": 1610612741,
                "opponent_id": 1610612748,
                "home_flag": 1,
                "player_id": 202696,
                "minutes_played": 25.0,
                "points": 15.0,
                "fgm": 6.0,
                "three_pm": 1.0,
                "fga": 12.0,
                "three_pa": 4.0,
                "fta": 2.0,
                "ftm": 2.0,
                "assists": 5.0,
                "turnovers": 2.0,
                "oreb": 1.0,
                "dreb": 4.0,
                "steals": 1.0,
                "blocks": 0.0,
                "starter_flag_box": True,
            },
            {
                "season": 2025,
                "game_id": 22500529,
                "game_date": pd.Timestamp("2026-01-29"),
                "tip_ts": pd.Timestamp("2026-01-30T01:00:00Z"),
                "team_id": 1610612741,
                "opponent_id": 1610612748,
                "home_flag": 1,
                "player_id": 202696,
                "minutes_played": 32.0,
                "points": 22.0,
                "fgm": 8.0,
                "three_pm": 2.0,
                "fga": 15.0,
                "three_pa": 5.0,
                "fta": 4.0,
                "ftm": 4.0,
                "assists": 7.0,
                "turnovers": 3.0,
                "oreb": 1.0,
                "dreb": 5.0,
                "steals": 1.0,
                "blocks": 0.0,
                "starter_flag_box": True,
            },
        ]
    )

    labels = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_id": 22500529,
                "game_date": pd.Timestamp("2026-01-08"),
                "team_id": 1610612741,
                "player_id": 202696,
                "minutes_actual": 25.0,
                "starter_flag": 1,
                "listed_pos": "G",
            },
            {
                "season": 2025,
                "game_id": 22500529,
                "game_date": pd.Timestamp("2026-01-29"),
                "team_id": 1610612741,
                "player_id": 202696,
                "minutes_actual": 32.0,
                "starter_flag": 1,
                "listed_pos": "G",
            },
        ]
    )

    roster = pd.DataFrame(
        [
            {
                "game_id": 22500529,
                "game_date": pd.Timestamp("2026-01-08"),
                "team_id": 1610612741,
                "player_id": 202696,
                "starter_flag": 1,
                "listed_pos": "G",
                "as_of_ts": pd.Timestamp("2026-01-08T22:00:00Z"),
            },
            {
                "game_id": 22500529,
                "game_date": pd.Timestamp("2026-01-29"),
                "team_id": 1610612741,
                "player_id": 202696,
                "starter_flag": 1,
                "listed_pos": "G",
                "as_of_ts": pd.Timestamp("2026-01-29T22:00:00Z"),
            },
        ]
    )

    features = build_features(
        labels=labels,
        stats=stats,
        roster=roster,
        odds=pd.DataFrame(),
        minutes_preds=pd.DataFrame(),
        injuries=pd.DataFrame(),
    )

    # Regression guard: same game_id on different dates must not cross-join.
    assert len(features) == 2
    assert features.duplicated(["season", "game_date", "game_id", "team_id", "player_id"]).sum() == 0
    assert sorted(features["minutes_actual"].tolist()) == [25.0, 32.0]


def test_assert_unique_training_keys_raises_on_duplicates() -> None:
    df = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_date": pd.Timestamp("2026-01-29"),
                "game_id": 22500529,
                "team_id": 1610612741,
                "player_id": 202696,
            },
            {
                "season": 2025,
                "game_date": pd.Timestamp("2026-01-29"),
                "game_id": 22500529,
                "team_id": 1610612741,
                "player_id": 202696,
            },
        ]
    )

    with pytest.raises(RuntimeError, match="duplicate rows for key"):
        _assert_unique_training_keys(df)
