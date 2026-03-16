from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from projections import paths
from projections.api.entry_manager_api import EntryFileState
from projections.api.minutes_api import create_app


@dataclass
class _FakePlayer:
    player_id: str
    pos: str


@dataclass
class _FakeLineup:
    players: list[_FakePlayer]
    total_proj: float


def _write_entry_state(data_root: Path, game_date: str, contest_id: str) -> None:
    state = EntryFileState(
        game_date=game_date,
        draft_group_id=142852,
        site="dk",
        contest_id=contest_id,
        contest_name="Contest A",
        entry_fee="$1",
        created_at="2026-03-03T00:00:00Z",
        updated_at="2026-03-03T00:00:00Z",
        client_revision=1,
        header=[
            "Entry ID",
            "Contest Name",
            "Contest ID",
            "Entry Fee",
            "PG",
            "SG",
            "SF",
            "PF",
            "C",
            "G",
            "F",
            "UTIL",
        ],
        entries=[
            {
                "entry_id": "e1",
                "entry_key": "e1",
                "contest_id": contest_id,
                "contest_name": "Contest A",
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
    entry_path = data_root / "entries" / game_date / "dk" / f"{contest_id}.json"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def _write_fd_entry_state(data_root: Path, game_date: str, contest_id: str) -> None:
    state = EntryFileState(
        game_date=game_date,
        draft_group_id=127613,
        site="fd",
        contest_id=contest_id,
        contest_name="FD Contest A",
        entry_fee="0.25",
        created_at="2026-03-03T00:00:00Z",
        updated_at="2026-03-03T00:00:00Z",
        client_revision=1,
        header=["entry_id", "contest_id", "contest_name", "entry_fee", "PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C"],
        entries=[
            {
                "entry_id": "fd-e1",
                "entry_key": "fd-e1",
                "contest_id": contest_id,
                "contest_name": "FD Contest A",
                "entry_fee": "0.25",
                "PG1": "127613-1001",
                "PG2": "127613-1002",
                "SG1": "127613-1003",
                "SG2": "127613-1004",
                "SF1": "127613-1005",
                "SF2": "127613-1006",
                "PF1": "127613-1007",
                "PF2": "127613-1008",
                "C": "127613-1009",
            }
        ],
    )
    entry_path = data_root / "entries" / game_date / "fd" / f"{contest_id}.json"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def _patch_late_swap_dependencies(monkeypatch, *, data_root: Path, lock_pg: bool = False) -> None:
    import projections.api.entry_manager_api as entry_manager_api
    import projections.late_swap.candidate_generation as candidate_generation

    def fake_data_path(*parts: Any) -> Path:
        return data_root.joinpath(*parts)

    monkeypatch.setattr(paths, "data_path", fake_data_path)
    monkeypatch.setattr(entry_manager_api, "_refresh_draftables_for_late_swap", lambda *_a, **_k: None)
    monkeypatch.setattr(entry_manager_api, "_safe_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(entry_manager_api, "_resolve_latest_sim_v2_worlds", lambda **_: {})

    def fake_popen(*_args, **_kwargs):
        class DummyProc:
            pass

        return DummyProc()

    monkeypatch.setattr(entry_manager_api.subprocess, "Popen", fake_popen)

    players = [
        {"player_id": "1", "name": "P1", "team": "A", "matchup": "A@B", "proj": 20.0, "pred_own_pct": 28.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z"},
        {"player_id": "2", "name": "P2", "team": "A", "matchup": "A@B", "proj": 18.0, "pred_own_pct": 22.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z"},
        {"player_id": "3", "name": "P3", "team": "A", "matchup": "A@B", "proj": 17.0, "pred_own_pct": 20.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z"},
        {"player_id": "4", "name": "P4", "team": "B", "matchup": "A@B", "proj": 16.0, "pred_own_pct": 19.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z"},
        {"player_id": "5", "name": "P5", "team": "B", "matchup": "A@B", "proj": 15.0, "pred_own_pct": 18.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z"},
        {"player_id": "6", "name": "P6", "team": "B", "matchup": "A@B", "proj": 14.0, "pred_own_pct": 16.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z"},
        {"player_id": "7", "name": "P7", "team": "C", "matchup": "C@D", "proj": 13.0, "pred_own_pct": 15.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z"},
        {"player_id": "8", "name": "P8", "team": "C", "matchup": "C@D", "proj": 12.0, "pred_own_pct": 12.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z"},
        {"player_id": "9", "name": "P9", "team": "D", "matchup": "C@D", "proj": 21.0, "pred_own_pct": 8.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z"},
    ]
    monkeypatch.setattr(entry_manager_api, "build_player_pool", lambda **_kwargs: players)

    monkeypatch.setattr(
        entry_manager_api,
        "_build_dk_maps",
        lambda *_args, **_kwargs: (
            {
                "1": 101,
                "2": 102,
                "3": 103,
                "4": 104,
                "5": 105,
                "6": 106,
                "7": 107,
                "8": 108,
                "9": 109,
            },
            {str(i): f"P{i}" for i in range(1, 10)},
            {
                101: {"PG": 11},
                102: {"SG": 12},
                103: {"SF": 13},
                104: {"PF": 14},
                105: {"C": 15},
                106: {"G": 16},
                107: {"F": 17},
                108: {"UTIL": 18},
                109: {"UTIL": 19},
            },
            {
                101: "P1",
                102: "P2",
                103: "P3",
                104: "P4",
                105: "P5",
                106: "P6",
                107: "P7",
                108: "P8",
                109: "P9",
            },
        ),
    )

    future = datetime.now(timezone.utc) + timedelta(hours=5)
    past = datetime.now(timezone.utc) - timedelta(hours=3)
    monkeypatch.setattr(
        entry_manager_api,
        "_load_draftable_start_times",
        lambda *_args, **_kwargs: {
            11: past if lock_pg else future,
            12: future,
            13: future,
            14: future,
            15: future,
            16: future,
            17: future,
            18: future,
            19: future,
        },
    )

    def fake_solver(players_pool, constraints, seed, site):
        _ = players_pool, seed, site
        hold = _FakeLineup(
            players=[
                _FakePlayer("1", "PG"),
                _FakePlayer("2", "SG"),
                _FakePlayer("3", "SF"),
                _FakePlayer("4", "PF"),
                _FakePlayer("5", "C"),
                _FakePlayer("6", "G"),
                _FakePlayer("7", "F"),
                _FakePlayer("8", "UTIL"),
            ],
            total_proj=125.0,
        )
        alt = _FakeLineup(
            players=[
                _FakePlayer("1", "PG"),
                _FakePlayer("2", "SG"),
                _FakePlayer("3", "SF"),
                _FakePlayer("4", "PF"),
                _FakePlayer("5", "C"),
                _FakePlayer("6", "G"),
                _FakePlayer("7", "F"),
                _FakePlayer("9", "UTIL"),
            ],
            total_proj=130.0,
        )
        n = max(1, int(getattr(constraints, "N_lineups", 1)))
        base = [alt, hold]
        return (base[:n] if n <= len(base) else base + [hold] * (n - len(base))), {"status": "OPTIMAL"}

    monkeypatch.setattr(candidate_generation, "solve_cpsat_iterative_counts", fake_solver)


def _patch_late_swap_dependencies_fd(monkeypatch, *, data_root: Path) -> None:
    import projections.api.entry_manager_api as entry_manager_api
    import projections.late_swap.candidate_generation as candidate_generation

    def fake_data_path(*parts: Any) -> Path:
        return data_root.joinpath(*parts)

    monkeypatch.setattr(paths, "data_path", fake_data_path)
    monkeypatch.setattr(entry_manager_api, "_refresh_draftables_for_late_swap", lambda *_a, **_k: None)
    monkeypatch.setattr(entry_manager_api, "_safe_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(entry_manager_api, "_resolve_latest_sim_v2_worlds", lambda **_: {})

    def fake_popen(*_args, **_kwargs):
        class DummyProc:
            pass

        return DummyProc()

    monkeypatch.setattr(entry_manager_api.subprocess, "Popen", fake_popen)

    players = [
        {"player_id": "1", "name": "P1", "team": "A", "matchup": "A@B", "proj": 20.0, "pred_own_pct": 12.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z", "positions": ["PG"], "fd_id": "127613-1001"},
        {"player_id": "2", "name": "P2", "team": "A", "matchup": "A@B", "proj": 19.0, "pred_own_pct": 11.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z", "positions": ["PG"], "fd_id": "127613-1002"},
        {"player_id": "3", "name": "P3", "team": "A", "matchup": "A@B", "proj": 18.0, "pred_own_pct": 10.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z", "positions": ["SG"], "fd_id": "127613-1003"},
        {"player_id": "4", "name": "P4", "team": "B", "matchup": "A@B", "proj": 17.0, "pred_own_pct": 9.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z", "positions": ["SG"], "fd_id": "127613-1004"},
        {"player_id": "5", "name": "P5", "team": "B", "matchup": "A@B", "proj": 16.0, "pred_own_pct": 8.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z", "positions": ["SF"], "fd_id": "127613-1005"},
        {"player_id": "6", "name": "P6", "team": "C", "matchup": "C@D", "proj": 15.0, "pred_own_pct": 7.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z", "positions": ["SF"], "fd_id": "127613-1006"},
        {"player_id": "7", "name": "P7", "team": "C", "matchup": "C@D", "proj": 14.0, "pred_own_pct": 6.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z", "positions": ["PF"], "fd_id": "127613-1007"},
        {"player_id": "8", "name": "P8", "team": "D", "matchup": "C@D", "proj": 13.0, "pred_own_pct": 5.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z", "positions": ["PF"], "fd_id": "127613-1008"},
        {"player_id": "9", "name": "P9", "team": "D", "matchup": "C@D", "proj": 12.0, "pred_own_pct": 4.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z", "positions": ["C"], "fd_id": "127613-1009"},
        {"player_id": "10", "name": "P10", "team": "D", "matchup": "C@D", "proj": 25.0, "pred_own_pct": 3.0, "is_active": True, "is_out": False, "game_start_utc": "2026-03-03T23:30:00Z", "positions": ["C"], "fd_id": "127613-1010"},
    ]
    monkeypatch.setattr(entry_manager_api, "build_player_pool", lambda **_kwargs: players)

    def fake_solver(players_pool, constraints, seed, site):
        _ = players_pool, constraints, seed, site
        hold = _FakeLineup(
            players=[
                _FakePlayer("1", "PG"),
                _FakePlayer("2", "PG"),
                _FakePlayer("3", "SG"),
                _FakePlayer("4", "SG"),
                _FakePlayer("5", "SF"),
                _FakePlayer("6", "SF"),
                _FakePlayer("7", "PF"),
                _FakePlayer("8", "PF"),
                _FakePlayer("9", "C"),
            ],
            total_proj=140.0,
        )
        alt = _FakeLineup(
            players=[
                _FakePlayer("1", "PG"),
                _FakePlayer("2", "PG"),
                _FakePlayer("3", "SG"),
                _FakePlayer("4", "SG"),
                _FakePlayer("5", "SF"),
                _FakePlayer("6", "SF"),
                _FakePlayer("7", "PF"),
                _FakePlayer("8", "PF"),
                _FakePlayer("10", "C"),
            ],
            total_proj=152.0,
        )
        return [alt, hold], {"status": "OPTIMAL"}

    monkeypatch.setattr(candidate_generation, "solve_cpsat_iterative_counts", fake_solver)


def test_late_swap_v2_session_lifecycle(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_root"
    game_date = "2026-03-03"
    contest_id = "188511762"
    _write_entry_state(data_root, game_date, contest_id)
    _patch_late_swap_dependencies(monkeypatch, data_root=data_root, lock_pg=False)

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    create_resp = client.post(
        "/api/entry-manager/late-swap/sessions",
        json={"date": game_date, "contest_ids": [contest_id]},
    )
    assert create_resp.status_code == 200
    session = create_resp.json()
    session_id = session["session_id"]

    preview_resp = client.post(
        f"/api/entry-manager/late-swap/sessions/{session_id}/preview",
        params={"date": game_date},
        json={},
    )
    assert preview_resp.status_code == 200
    preview = preview_resp.json()
    assert preview["session"]["status"] in {"preview_ready", "failed"}
    assert preview["candidates_by_entry_id"]
    scoped_entry_id = next(iter(preview["candidates_by_entry_id"].keys()))
    candidate_ids = [c["candidate_id"] for c in preview["candidates_by_entry_id"][scoped_entry_id]]
    assert candidate_ids

    pin_resp = client.post(
        f"/api/entry-manager/late-swap/sessions/{session_id}/pin-candidates",
        params={"date": game_date},
        json={"pins": {scoped_entry_id: candidate_ids[-1]}},
    )
    assert pin_resp.status_code == 200
    pinned = pin_resp.json()
    assert pinned["session"]["pinned_candidates_by_entry_id"][scoped_entry_id] == candidate_ids[-1]

    commit_resp = client.post(
        f"/api/entry-manager/late-swap/sessions/{session_id}/commit",
        params={"date": game_date},
        json={},
    )
    assert commit_resp.status_code == 200
    committed_session = commit_resp.json()
    assert committed_session["status"] == "committed"

    state_path = data_root / "entries" / game_date / "dk" / f"{contest_id}.json"
    committed_state = EntryFileState.model_validate_json(state_path.read_text(encoding="utf-8"))
    assert committed_state.source_late_swap_session_id == session_id
    assert committed_state.source_late_swap_mode == committed_session["policy"]["mode"]

    export_resp = client.post(
        f"/api/entry-manager/late-swap/sessions/{session_id}/export",
        params={"date": game_date},
        json={"include_uncommitted_preview": True},
    )
    assert export_resp.status_code == 200
    assert "Entry ID" in export_resp.text

    preview_manifest = data_root / "late_swap" / game_date / "dk" / f"session={session_id}" / "preview_export_manifest.json"
    assert preview_manifest.exists()
    manifest = json.loads(preview_manifest.read_text(encoding="utf-8"))
    assert manifest["session_id"] == session_id


def test_late_swap_v2_forced_over_cap_by_locks(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_root"
    game_date = "2026-03-03"
    contest_id = "188511762"
    _write_entry_state(data_root, game_date, contest_id)
    _patch_late_swap_dependencies(monkeypatch, data_root=data_root, lock_pg=True)

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    create_resp = client.post(
        "/api/entry-manager/late-swap/sessions",
        json={
            "date": game_date,
            "contest_ids": [contest_id],
            "policy": {
                **{
                    "mode": "preserve_targets",
                    "target_source": "source_portfolio",
                    "exposure_bounds": {"1": {"max": 0.0}},
                    "team_exposure_bounds": {},
                    "game_exposure_bounds": {},
                    "min_uniques": 0,
                    "max_duplicate_lineups": 1,
                    "candidate_count_per_entry": 10,
                    "max_swaps_per_entry": None,
                    "max_total_swaps": None,
                    "min_gain_to_swap": 0.0,
                    "swap_cost_lambda": 0.15,
                    "target_deviation_lambda": 0.25,
                    "overlap_penalty_lambda": 0.0,
                    "ownership_penalty_lambda": 0.0,
                    "leverage_boost_lambda": 0.0,
                    "segment_mode": "global",
                    "segment_overrides": {},
                    "rerun_anchor": "source_portfolio",
                }
            },
        },
    )
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    preview_resp = client.post(
        f"/api/entry-manager/late-swap/sessions/{session_id}/preview",
        params={"date": game_date},
        json={},
    )
    assert preview_resp.status_code == 200
    body = preview_resp.json()
    warnings = body["session"]["diagnostics"]["warnings"]
    assert any("cap" in warning.lower() and "lock floor" in warning.lower() for warning in warnings)
    forced_rows = [
        row
        for row in body["session"]["diagnostics"]["exposure_states"]
        if row["player_id"] == "1"
    ]
    assert forced_rows
    assert forced_rows[0]["forced_over_cap_by_locks"] is True


def test_late_swap_v2_fd_session_lifecycle(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_root"
    game_date = "2026-03-03"
    contest_id = "127613-279208572"
    _write_fd_entry_state(data_root, game_date, contest_id)
    _patch_late_swap_dependencies_fd(monkeypatch, data_root=data_root)

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    create_resp = client.post(
        "/api/entry-manager/late-swap/sessions",
        json={"date": game_date, "site": "fd", "contest_ids": [contest_id]},
    )
    assert create_resp.status_code == 200
    session = create_resp.json()
    session_id = session["session_id"]
    assert session["site"] == "fd"

    preview_resp = client.post(
        f"/api/entry-manager/late-swap/sessions/{session_id}/preview",
        params={"date": game_date, "site": "fd"},
        json={},
    )
    assert preview_resp.status_code == 200
    preview = preview_resp.json()
    assert preview["candidates_by_entry_id"]
    scoped_entry_id = next(iter(preview["candidates_by_entry_id"].keys()))
    entry_candidates = preview["candidates_by_entry_id"][scoped_entry_id]
    candidate_ids = [c["candidate_id"] for c in entry_candidates]
    assert candidate_ids
    preferred_candidate = next(
        (c["candidate_id"] for c in entry_candidates if c.get("generated_by") != "hold"),
        candidate_ids[0],
    )

    pin_resp = client.post(
        f"/api/entry-manager/late-swap/sessions/{session_id}/pin-candidates",
        params={"date": game_date, "site": "fd"},
        json={"pins": {scoped_entry_id: preferred_candidate}},
    )
    assert pin_resp.status_code == 200

    commit_resp = client.post(
        f"/api/entry-manager/late-swap/sessions/{session_id}/commit",
        params={"date": game_date, "site": "fd"},
        json={},
    )
    assert commit_resp.status_code == 200
    committed_session = commit_resp.json()
    assert committed_session["status"] == "committed"

    state_path = data_root / "entries" / game_date / "fd" / f"{contest_id}.json"
    committed_state = EntryFileState.model_validate_json(state_path.read_text(encoding="utf-8"))
    assert committed_state.source_late_swap_session_id == session_id
    assert committed_state.entries[0]["C"] in {"127613-1009", "127613-1010"}

    export_resp = client.post(
        f"/api/entry-manager/late-swap/sessions/{session_id}/export",
        params={"date": game_date, "site": "fd"},
        json={"include_uncommitted_preview": True},
    )
    assert export_resp.status_code == 200
    assert "entry_id,contest_id,contest_name,entry_fee" in export_resp.text
