from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.pbp.constants import PBP_V1_SCHEMA_VERSION
from projections.pbp.vendor_ingest import CANON_PBP_EVENTS_COLS, ingest_vendor_game_csv


def test_ingest_vendor_game_csv_minimal_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "game.csv"
    df = pd.DataFrame(
        [
            {
                "game_id": "22400061",
                "data_set": "NBA 2024-2025 Regular Season",
                "date": "2024-10-22",
                "period": "1",
                "away_score": "0",
                "home_score": "0",
                "remaining_time": "0:12:00",
                "elapsed": "0:00:00",
                "play_length": "0:00:00",
                "play_id": "1",
                "team": "",
                "event_type": "start of period",
                "description": "",
                "a1": "A One",
                "a2": "A Two",
                "a3": "A Three",
                "a4": "A Four",
                "a5": "A Five",
                "h1": "H One",
                "h2": "H Two",
                "h3": "H Three",
                "h4": "H Four",
                "h5": "H Five",
                "player": "",
            },
            {
                "game_id": "22400061",
                "data_set": "NBA 2024-2025 Regular Season",
                "date": "2024-10-22",
                "period": "1",
                "away_score": "0",
                "home_score": "2",
                "remaining_time": "0:11:50",
                "elapsed": "0:00:10",
                "play_length": "0:00:10",
                "play_id": "2",
                "team": "HOM",
                "event_type": "shot",
                "description": "H One makes 2pt",
                "a1": "A One",
                "a2": "A Two",
                "a3": "A Three",
                "a4": "A Four",
                "a5": None,
                "h1": "H One",
                "h2": "H Two",
                "h3": "H Three",
                "h4": "H Four",
                "h5": "H Five",
                "player": "H One",
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    result = ingest_vendor_game_csv(
        csv_path,
        season_id="2024-25",
        schema_version=PBP_V1_SCHEMA_VERSION,
        prev_players_dim=None,
    )

    out = result.pbp_events
    assert list(out.columns) == CANON_PBP_EVENTS_COLS
    assert out["game_id"].nunique() == 1
    assert out["game_id"].iloc[0] == "0022400061"
    assert out["event_index"].tolist() == [0, 1]
    assert not out[[f"away_p{i}" for i in range(1, 6)]].isna().any().any()
    assert not out[[f"home_p{i}" for i in range(1, 6)]].isna().any().any()
    assert out["away_lineup_key"].iloc[0].count("|") == 4
    assert out["home_lineup_key"].iloc[0].count("|") == 4
