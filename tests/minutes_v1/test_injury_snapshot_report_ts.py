from __future__ import annotations

import pandas as pd

from projections.minutes_v1.snapshots import select_injury_snapshot


def test_select_injury_snapshot_uses_report_time_for_pre_tip_eligibility() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "player_id": [10, 10],
            "tip_ts": [
                pd.Timestamp("2025-01-02T00:00:00Z"),
                pd.Timestamp("2025-01-02T00:00:00Z"),
            ],
            # Raw scrape cadence can lag the actual report publication time.
            "as_of_ts": [
                pd.Timestamp("2025-01-02T00:30:00Z"),
                pd.Timestamp("2025-01-01T23:30:00Z"),
            ],
            "status": ["OUT", "Q"],
            "restriction_flag": [False, False],
            "ramp_flag": [False, False],
            "games_since_return": [pd.NA, pd.NA],
            "days_since_return": [pd.NA, pd.NA],
            "ingested_ts": [
                pd.Timestamp("2025-01-02T00:45:00Z"),
                pd.Timestamp("2025-01-01T23:45:00Z"),
            ],
            "source": [
                "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2025-01-01_07_00PM.pdf",
                "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2025-01-01_06_00PM.pdf",
            ],
        }
    )

    snapshot = select_injury_snapshot(df)

    assert len(snapshot) == 1
    row = snapshot.iloc[0]
    assert row["status"] == "OUT"
    # Canonical as_of_ts should reflect report time semantics once selected.
    assert row["as_of_ts"] == pd.Timestamp("2025-01-02T00:00:00Z")
    assert row["selection_rule"] == "latest_leq_tip"
    assert int(row["snapshot_missing"]) == 0
