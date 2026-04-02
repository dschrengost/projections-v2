from __future__ import annotations

import pandas as pd

from projections.etl.roster_nightly import _schedule_roster_game_id_mismatch


def test_schedule_roster_game_id_mismatch_detects_missing_and_stale() -> None:
    schedule_df = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "game_date": [pd.Timestamp("2026-04-01")] * 3,
        }
    )
    roster_rows = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 4],
            "game_date": [pd.Timestamp("2026-04-01")] * 4,
        }
    )

    missing_ids, stale_ids = _schedule_roster_game_id_mismatch(
        schedule_df,
        roster_rows,
        start_day=pd.Timestamp("2026-04-01"),
        end_day=pd.Timestamp("2026-04-01"),
    )

    assert missing_ids == [3]
    assert stale_ids == [4]


def test_schedule_roster_game_id_mismatch_returns_empty_for_match() -> None:
    schedule_df = pd.DataFrame(
        {
            "game_id": [10, 20],
            "game_date": [pd.Timestamp("2026-04-01")] * 2,
        }
    )
    roster_rows = pd.DataFrame(
        {
            "game_id": [10, 20, 20],
            "game_date": [pd.Timestamp("2026-04-01")] * 3,
        }
    )

    missing_ids, stale_ids = _schedule_roster_game_id_mismatch(
        schedule_df,
        roster_rows,
        start_day=pd.Timestamp("2026-04-01"),
        end_day=pd.Timestamp("2026-04-01"),
    )

    assert missing_ids == []
    assert stale_ids == []
