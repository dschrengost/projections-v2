from __future__ import annotations

from pathlib import Path

from projections.fd.normalize import normalize_fd_players_to_salaries, players_json_to_df


def _sample_players_payload() -> dict:
    return {
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
                "injury_indicators": [{"code": "Q"}],
            },
            {
                "id": "127611-84681",
                "first_name": "Jamal",
                "last_name": "Murray",
                "nickname": "Jamal Murray",
                "position": "PG/SG",
                "salary": 7600,
                "team": 1,
                "fixture": 9001,
            },
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
            {"id": 1, "code": "DEN", "name": "Nuggets"},
            {"id": 2, "code": "LAL", "name": "Lakers"},
        ],
    }


def test_players_json_to_df_and_normalize(tmp_path: Path) -> None:
    raw_df = players_json_to_df(_sample_players_payload(), fixture_list_id=4444)

    assert len(raw_df) == 2
    assert set(raw_df["fd_player_id"].tolist()) == {"127611-84680", "127611-84681"}
    assert set(raw_df["game_matchup"].dropna().tolist()) == {"DEN@LAL"}

    salaries_df = normalize_fd_players_to_salaries(
        root=tmp_path,
        site="fd",
        game_date="2026-01-10",
        draft_group_id=4444,
        df=raw_df,
    )

    assert len(salaries_df) == 2
    assert set(salaries_df["fd_player_id"].tolist()) == {"127611-84680", "127611-84681"}
    assert set(salaries_df["site_player_id"].tolist()) == {"127611-84680", "127611-84681"}
    row = salaries_df.set_index("fd_player_id").loc["127611-84681"]
    assert set(row["positions"]) == {"PG", "SG"}
