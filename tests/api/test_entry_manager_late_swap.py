from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from projections import paths
from projections.api.entry_manager_api import EntryFileState
from projections.api.minutes_api import create_app


def test_late_swap_holds_when_all_slots_locked(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_root"

    def fake_data_path(*parts: Any) -> Path:
        return data_root.joinpath(*parts)

    monkeypatch.setattr(paths, "data_path", fake_data_path)

    import projections.api.entry_manager_api as entry_manager_api

    monkeypatch.setattr(
        entry_manager_api,
        "_refresh_draftables_for_late_swap",
        lambda *_args, **_kwargs: None,
    )

    players = [
        {"player_id": "1", "proj": 10.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T00:00:00Z"},
        {"player_id": "2", "proj": 11.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T00:00:00Z"},
        {"player_id": "3", "proj": 12.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T00:00:00Z"},
        {"player_id": "4", "proj": 13.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T00:00:00Z"},
        {"player_id": "5", "proj": 14.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T00:00:00Z"},
        {"player_id": "6", "proj": 15.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T00:00:00Z"},
        {"player_id": "7", "proj": 16.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T00:00:00Z"},
        {"player_id": "8", "proj": 17.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T00:00:00Z"},
    ]
    monkeypatch.setattr(entry_manager_api, "build_player_pool", lambda **_kwargs: players)

    monkeypatch.setattr(
        entry_manager_api,
        "_build_dk_maps",
        lambda *_args, **_kwargs: (
            {"1": 101, "2": 102, "3": 103, "4": 104, "5": 105, "6": 106, "7": 107, "8": 108},
            {"1": "P1", "2": "P2", "3": "P3", "4": "P4", "5": "P5", "6": "P6", "7": "P7", "8": "P8"},
            {
                101: {"PG": 11},
                102: {"SG": 12},
                103: {"SF": 13},
                104: {"PF": 14},
                105: {"C": 15},
                106: {"G": 16},
                107: {"F": 17},
                108: {"UTIL": 18},
            },
            {101: "P1", 102: "P2", 103: "P3", 104: "P4", 105: "P5", 106: "P6", 107: "P7", 108: "P8"},
        ),
    )

    past = datetime.now(timezone.utc) - timedelta(hours=1)
    monkeypatch.setattr(
        entry_manager_api,
        "_load_draftable_start_times",
        lambda *_args, **_kwargs: {11: past, 12: past, 13: past, 14: past, 15: past, 16: past, 17: past, 18: past},
    )

    solve_calls = {"count": 0}

    def _unexpected_solver_call(*_args, **_kwargs):
        solve_calls["count"] += 1
        raise AssertionError("solver should not run when all 8 slots are locked")

    monkeypatch.setattr(entry_manager_api, "solve_cpsat_iterative_counts", _unexpected_solver_call)

    game_date = "2026-03-03"
    contest_id = "188511762"
    state = EntryFileState(
        game_date=game_date,
        draft_group_id=142852,
        site="dk",
        contest_id=contest_id,
        contest_name="Contest",
        entry_fee="$1",
        created_at="2026-03-03T00:00:00Z",
        updated_at="2026-03-03T00:00:00Z",
        client_revision=1,
        header=["Entry ID", "Contest Name", "Contest ID", "Entry Fee", "PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"],
        entries=[
            {
                "entry_id": "e1",
                "entry_key": "e1",
                "contest_id": contest_id,
                "contest_name": "Contest",
                "entry_fee": "$1",
                "PG": "P1 (11)",
                "SG": "P2 (12)",
                "SF": "P3 (13)",
                "PF": "P4 (14)",
                "C": "P5 (15)",
                "G": "P6 (16)",
                "F": "P7 (17)",
                "UTIL": "P8 (18)",
            }
        ],
    )
    entry_path = fake_data_path("entries") / game_date / "dk" / f"{contest_id}.json"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    response = client.post(
        f"/api/entry-manager/entries/{contest_id}/late-swap",
        params={"date": game_date},
        json={},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["selection_summary"]["entries_total"] == 1
    assert body["selection_summary"]["entries_held"] == 1
    assert body["selection_summary"]["entries_swapped"] == 0
    assert body["updated_entries"] == 1
    assert body["entry_state"]["entries"][0]["PG"] == "P1 (11)"
    assert solve_calls["count"] == 0
