import json
from pathlib import Path

import pandas as pd

from projections.api.optimizer_service import load_salaries_for_date
from projections.dk.salaries_schema import dk_salaries_gold_path


def test_load_salaries_for_date_falls_back_to_bronze_draftables(tmp_path: Path):
    data_root = tmp_path
    game_date = "2025-12-28"
    draft_group_id = 999

    bronze_path = data_root / "bronze" / "dk" / "draftables" / f"draftables_raw_{draft_group_id}.json"
    bronze_path.parent.mkdir(parents=True, exist_ok=True)
    bronze_path.write_text(
        json.dumps(
            {
                "competitions": [
                    {
                        "competitionId": 1,
                        "awayTeam": {"abbreviation": "AAA"},
                        "homeTeam": {"abbreviation": "BBB"},
                        "startTime": "2025-12-28T23:00:00.0000000Z",
                    }
                ],
                "draftables": [
                    {
                        "draftableId": 123,
                        "playerId": 1111,
                        "displayName": "Player One",
                        "position": "PG",
                        "salary": 5000,
                        "teamAbbreviation": "AAA",
                        "competition": {"competitionId": 1},
                    },
                    {
                        "draftableId": 456,
                        "playerId": 2222,
                        "displayName": "Player Two",
                        "position": "C",
                        "salary": 7000,
                        "teamAbbreviation": "BBB",
                        "competition": {"competitionId": 1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    gold_path = dk_salaries_gold_path(data_root, "dk", game_date, draft_group_id)
    assert not gold_path.exists()

    df = load_salaries_for_date(game_date, draft_group_id, site="dk", data_root=data_root)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"dk_player_id", "display_name", "positions", "salary", "team_abbrev"}.issubset(df.columns)
    assert gold_path.exists()

