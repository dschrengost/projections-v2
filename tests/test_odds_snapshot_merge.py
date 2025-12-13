from __future__ import annotations

import pandas as pd

from projections.etl.odds import _merge_odds_snapshots
from projections.minutes_v1.schemas import ODDS_SNAPSHOT_SCHEMA, enforce_schema


def test_merge_odds_snapshots_preserves_missing_games_and_fields() -> None:
    existing = pd.DataFrame(
        {
            "game_id": [1, 2],
            "as_of_ts": ["2025-01-01T10:00:00Z", "2025-01-01T10:00:00Z"],
            "spread_home": [-3.5, 2.0],
            "total": [220.0, 215.0],
            "book": ["old", "old"],
            "book_pref": [pd.NA, pd.NA],
            "ingested_ts": ["2025-01-01T10:01:00Z", "2025-01-01T10:01:00Z"],
            "source": ["oddstrader", "oddstrader"],
        }
    )
    incoming = pd.DataFrame(
        {
            "game_id": [1, 3],
            "as_of_ts": ["2025-01-01T10:10:00Z", "2025-01-01T10:05:00Z"],
            # Simulate a transient missing spread update for game 1.
            "spread_home": [pd.NA, -1.0],
            "total": [221.0, 218.0],
            "book": ["new", "new"],
            "book_pref": [pd.NA, pd.NA],
            "ingested_ts": ["2025-01-01T10:11:00Z", "2025-01-01T10:06:00Z"],
            "source": ["oddstrader", "oddstrader"],
        }
    )

    existing = enforce_schema(existing, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)
    incoming = enforce_schema(incoming, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)
    merged = _merge_odds_snapshots(existing, incoming)
    merged = enforce_schema(merged, ODDS_SNAPSHOT_SCHEMA, allow_missing_optional=True)

    merged = merged.sort_values("game_id").reset_index(drop=True)
    assert merged["game_id"].tolist() == [1, 2, 3]

    game1 = merged[merged["game_id"] == 1].iloc[0]
    assert game1["as_of_ts"] == pd.Timestamp("2025-01-01T10:10:00Z")
    assert float(game1["total"]) == 221.0
    # Preserve the previous spread when the new scrape lacks it.
    assert float(game1["spread_home"]) == -3.5

