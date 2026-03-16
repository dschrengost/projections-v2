from __future__ import annotations

import csv
import io

from projections.api import optimizer_api


def test_assign_fd_lineup_to_slots_happy_path() -> None:
    lineup = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    positions_by_player = {
        "1": {"PG"},
        "2": {"PG"},
        "3": {"SG"},
        "4": {"SG"},
        "5": {"SF"},
        "6": {"SF"},
        "7": {"PF"},
        "8": {"PF"},
        "9": {"C"},
    }

    assigned = optimizer_api._assign_fd_lineup_to_slots(lineup, positions_by_player)

    assert assigned == lineup


def test_export_lineups_to_site_csv_fd(monkeypatch) -> None:
    monkeypatch.setattr(
        optimizer_api,
        "build_player_pool",
        lambda **_: [
            {"player_id": "1", "name": "PG One", "positions": ["PG"], "fd_id": "101"},
            {"player_id": "2", "name": "PG Two", "positions": ["PG"], "fd_id": "102"},
            {"player_id": "3", "name": "SG One", "positions": ["SG"], "fd_id": "103"},
            {"player_id": "4", "name": "SG Two", "positions": ["SG"], "fd_id": "104"},
            {"player_id": "5", "name": "SF One", "positions": ["SF"], "fd_id": "105"},
            {"player_id": "6", "name": "SF Two", "positions": ["SF"], "fd_id": "106"},
            {"player_id": "7", "name": "PF One", "positions": ["PF"], "fd_id": "107"},
            {"player_id": "8", "name": "PF Two", "positions": ["PF"], "fd_id": "108"},
            {"player_id": "9", "name": "C One", "positions": ["C"], "fd_id": "109"},
        ],
    )

    csv_text = optimizer_api._export_lineups_to_site_csv(
        game_date="2026-01-10",
        draft_group_id=999,
        site="fd",
        lineups=[["1", "2", "3", "4", "5", "6", "7", "8", "9"]],
    )

    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[0] == optimizer_api.FD_NBA_SLOTS
    assert rows[1] == [
        "PG One (101)",
        "PG Two (102)",
        "SG One (103)",
        "SG Two (104)",
        "SF One (105)",
        "SF Two (106)",
        "PF One (107)",
        "PF Two (108)",
        "C One (109)",
    ]


def test_export_lineups_to_site_csv_dispatches_dk(monkeypatch) -> None:
    monkeypatch.setattr(optimizer_api, "_export_lineups_to_dk_csv", lambda **_: "dk-csv")

    out = optimizer_api._export_lineups_to_site_csv(
        game_date="2026-01-10",
        draft_group_id=123,
        site="dk",
        lineups=[["1"] * 8],
    )

    assert out == "dk-csv"
