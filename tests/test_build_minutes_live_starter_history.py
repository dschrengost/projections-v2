"""Regression tests for live minutes feature history starter flags.

Historically we've observed a failure mode where `starter_prev_game_asof`
became constant (e.g., 1 for every row) due to unreliable starter flags in
label history. This breaks both the minutes model and downstream injury-regime
logic that relies on previous-game starters.
"""

from __future__ import annotations

from datetime import timezone

import pandas as pd

from projections.cli.build_minutes_live import _load_label_history
from projections.features.role import attach_role_features


def test_load_label_history_derives_starters_from_minutes_and_role_shift_is_not_constant():
    target_day = pd.Timestamp("2025-12-30").normalize()
    run_as_of_ts = pd.Timestamp("2025-12-30T15:30:17Z").tz_convert(timezone.utc)

    # Two games worth of history for a single team with 6 players (need >=5).
    # Player 1005 is the non-starter in both games.
    labels = pd.DataFrame(
        {
            "game_id": [1] * 6 + [2] * 6,
            "team_id": [100] * 12,
            "player_id": [1000, 1001, 1002, 1003, 1004, 1005] * 2,
            "player_name": [f"p{i}" for i in range(6)] * 2,
            "season": ["2025-26"] * 12,
            "game_date": [pd.Timestamp("2025-12-28")] * 6 + [pd.Timestamp("2025-12-29")] * 6,
            "minutes": [34, 33, 32, 30, 28, 4, 35, 33, 31, 29, 27, 3],
            "source": ["test"] * 12,
            "label_frozen_ts": [pd.NaT] * 12,
            # Include bogus starter flags to ensure we overwrite them.
            "starter_flag": [1] * 12,
            "starter_flag_label": [1] * 12,
        }
    )

    history = _load_label_history(
        labels,
        target_day=target_day,
        history_days=None,
        run_as_of_ts=run_as_of_ts,
        label_source="test",
    )

    # Should have exactly 5 starters per team-game, derived from minutes.
    starter_sums = history.groupby(["game_id", "team_id"], sort=False)["starter_flag_label"].sum()
    assert (starter_sums == 5).all()
    assert (history["starter_flag"] == history["starter_flag_label"]).all()

    # Role feature shift should not become constant 1 for all players.
    history_with_tip = history.copy()
    # Provide a monotonic tip_ts within each game_date to satisfy attach_role_features.
    history_with_tip["tip_ts"] = pd.to_datetime(history_with_tip["game_date"]).dt.tz_localize("UTC")
    role = attach_role_features(history_with_tip[["player_id", "starter_flag", "tip_ts"]])
    prev = pd.to_numeric(role["starter_prev_game_asof"], errors="coerce").fillna(0).astype(int)
    assert prev.nunique() > 1

