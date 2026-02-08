from __future__ import annotations

import pandas as pd

from scripts.rates.build_training_base import (
    _compute_team_context,
    _groupwise_cumsum_shift1,
    _season_history_start,
)


def test_groupwise_cumsum_shift1_resets_per_group() -> None:
    df = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "player_id": [1, 1, 2, 2],
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = _groupwise_cumsum_shift1(df, ["season", "player_id"], "value")
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 10.0
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] == 30.0


def test_compute_team_context_first_game_has_no_prior_context() -> None:
    # Single game => each team has 0 prior games and context features should be NaN.
    stats = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_id": 101,
                "team_id": 10,
                "opponent_id": 20,
                "game_date": pd.Timestamp("2025-10-25"),
                "tip_ts": pd.Timestamp("2025-10-25T23:00:00Z"),
                "points": 110.0,
                "fga": 88.0,
                "fta": 20.0,
                "turnovers": 12.0,
            },
            {
                "season": 2025,
                "game_id": 101,
                "team_id": 20,
                "opponent_id": 10,
                "game_date": pd.Timestamp("2025-10-25"),
                "tip_ts": pd.Timestamp("2025-10-25T23:00:00Z"),
                "points": 102.0,
                "fga": 84.0,
                "fta": 18.0,
                "turnovers": 11.0,
            },
        ]
    )

    ctx = _compute_team_context(stats)
    assert len(ctx) == 2
    assert ctx["team_pace_szn"].isna().all()
    assert ctx["team_def_rtg_szn"].isna().all()


def test_season_history_start_uses_august_anchor() -> None:
    assert _season_history_start(pd.Timestamp("2026-02-07")) == pd.Timestamp("2025-08-01")
    assert _season_history_start(pd.Timestamp("2025-10-15")) == pd.Timestamp("2025-08-01")
