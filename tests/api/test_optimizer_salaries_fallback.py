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


def test_load_salaries_for_date_falls_back_to_bronze_fd_payloads(tmp_path: Path):
    data_root = tmp_path
    game_date = "2026-01-10"
    draft_group_id = 555

    bronze_dir = (
        data_root
        / "bronze"
        / "fd"
        / "fixture_lists"
        / f"game_date={game_date}"
        / f"draft_group_id={draft_group_id}"
    )
    bronze_dir.mkdir(parents=True, exist_ok=True)
    (bronze_dir / "players.json").write_text(
        json.dumps(
            {
                "players": [
                    {
                        "id": "127611-84680",
                        "first_name": "Nikola",
                        "last_name": "Jokic",
                        "nickname": "Nikola Jokic",
                        "position": "C",
                        "salary": 12000,
                        "team": 1,
                        "fixture": 9001,
                    }
                ],
                "fixtures": [
                    {
                        "id": 9001,
                        "start_date": "2026-01-10T00:30:00Z",
                        "home_team": 2,
                        "away_team": 1,
                    }
                ],
                "teams": [
                    {"id": 1, "code": "DEN"},
                    {"id": 2, "code": "LAL"},
                ],
            }
        ),
        encoding="utf-8",
    )

    gold_path = dk_salaries_gold_path(data_root, "fd", game_date, draft_group_id)
    assert not gold_path.exists()

    df = load_salaries_for_date(game_date, draft_group_id, site="fd", data_root=data_root)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"fd_player_id", "site_player_id", "display_name", "positions", "salary", "team_abbrev"}.issubset(df.columns)
    assert gold_path.exists()
