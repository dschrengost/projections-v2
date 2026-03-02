from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from projections import paths
from projections.api.entry_manager_api import EntryFileState
from projections.api.minutes_api import create_app


def test_entry_manager_export_writes_manifest_and_status(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_root"

    def fake_data_path(*parts: Any) -> Path:
        return data_root.joinpath(*parts)

    monkeypatch.setattr(paths, "data_path", fake_data_path)

    popen_calls: list[dict[str, Any]] = []

    def fake_popen(args, **kwargs):
        popen_calls.append({"args": args, "kwargs": kwargs})

        class DummyProc:
            pass

        return DummyProc()

    import projections.api.entry_manager_api as entry_manager_api

    monkeypatch.setattr(entry_manager_api.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(entry_manager_api, "_safe_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(entry_manager_api, "_resolve_latest_sim_v2_worlds", lambda **_: {})

    game_date = "2026-01-10"
    contest_id = "12345"
    dg = 999

    entry_state = EntryFileState(
        game_date=game_date,
        draft_group_id=dg,
        site="dk",
        contest_id=contest_id,
        contest_name="Contest",
        entry_fee="1",
        created_at="t",
        updated_at="t",
        client_revision=1,
        header=["Entry ID", "Contest Name", "Contest ID", "Entry Fee", "PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"],
        entries=[
            {
                "entry_id": "e1",
                "entry_key": "e1",
                "contest_id": contest_id,
                "contest_name": "Contest",
                "entry_fee": "1",
                "PG": "A (1)",
                "SG": "B (2)",
                "SF": "C (3)",
                "PF": "D (4)",
                "C": "E (5)",
                "G": "F (6)",
                "F": "G (7)",
                "UTIL": "H (8)",
            }
        ],
    )

    entry_path = fake_data_path("entries") / game_date / "dk" / f"{contest_id}.json"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(entry_state.model_dump_json(indent=2), encoding="utf-8")

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    resp = client.post(f"/api/entry-manager/entries/{contest_id}/export", params={"date": game_date})
    assert resp.status_code == 200
    export_id = resp.headers.get("X-Export-Id")
    assert export_id
    assert popen_calls, "expected a background job spawn"

    contest_root = fake_data_path("contests", "dk", f"game_date={game_date}", f"dg={dg}")
    export_csv_path = contest_root / "exports" / f"export_{export_id}.csv"
    manifest_path = contest_root / "exports" / f"export_{export_id}_manifest.json"
    status_path = contest_root / "eval_pre" / f"export_{export_id}" / "eval_status.json"

    assert export_csv_path.exists()
    assert manifest_path.exists()
    assert status_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["export_id"] == export_id
    assert manifest["draft_group_id"] == dg
    assert manifest["lineup_count"] == 1

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "PENDING"
    assert status["export_id"] == export_id

    status_resp = client.get(f"/api/entry-manager/exports/{export_id}/eval-status")
    assert status_resp.status_code == 200
    assert status_resp.json()["export_id"] == export_id


def test_entry_manager_export_filters_selected_entries(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_root"

    def fake_data_path(*parts: Any) -> Path:
        return data_root.joinpath(*parts)

    monkeypatch.setattr(paths, "data_path", fake_data_path)

    def fake_popen(*_args, **_kwargs):
        class DummyProc:
            pass

        return DummyProc()

    import projections.api.entry_manager_api as entry_manager_api

    monkeypatch.setattr(entry_manager_api.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(entry_manager_api, "_safe_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(entry_manager_api, "_resolve_latest_sim_v2_worlds", lambda **_: {})

    game_date = "2026-01-12"
    contest_id = "222"
    dg = 888

    entry_state = EntryFileState(
        game_date=game_date,
        draft_group_id=dg,
        site="dk",
        contest_id=contest_id,
        contest_name="Contest",
        entry_fee="1",
        created_at="t",
        updated_at="t",
        client_revision=1,
        header=["Entry ID", "Contest Name", "Contest ID", "Entry Fee", "PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"],
        entries=[
            {
                "entry_id": "e1",
                "entry_key": "e1",
                "contest_id": contest_id,
                "contest_name": "Contest",
                "entry_fee": "1",
                "PG": "A (1)",
                "SG": "B (2)",
                "SF": "C (3)",
                "PF": "D (4)",
                "C": "E (5)",
                "G": "F (6)",
                "F": "G (7)",
                "UTIL": "H (8)",
            },
            {
                "entry_id": "e2",
                "entry_key": "e2",
                "contest_id": contest_id,
                "contest_name": "Contest",
                "entry_fee": "1",
                "PG": "I (9)",
                "SG": "J (10)",
                "SF": "K (11)",
                "PF": "L (12)",
                "C": "M (13)",
                "G": "N (14)",
                "F": "O (15)",
                "UTIL": "P (16)",
            },
        ],
    )

    entry_path = fake_data_path("entries") / game_date / "dk" / f"{contest_id}.json"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(entry_state.model_dump_json(indent=2), encoding="utf-8")

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    resp = client.post(
        f"/api/entry-manager/entries/{contest_id}/export",
        params={"date": game_date},
        json={"entry_ids": ["e2"]},
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Entry-Count") == "1"
    csv_lines = resp.text.strip().splitlines()
    assert csv_lines[1].startswith("e2,")

    export_id = resp.headers.get("X-Export-Id")
    contest_root = fake_data_path("contests", "dk", f"game_date={game_date}", f"dg={dg}")
    manifest_path = contest_root / "exports" / f"export_{export_id}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["lineup_count"] == 1


def test_entry_manager_export_preserves_uploaded_dk_header(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_root"

    def fake_data_path(*parts: Any) -> Path:
        return data_root.joinpath(*parts)

    monkeypatch.setattr(paths, "data_path", fake_data_path)

    def fake_popen(*_args, **_kwargs):
        class DummyProc:
            pass

        return DummyProc()

    import projections.api.entry_manager_api as entry_manager_api

    monkeypatch.setattr(entry_manager_api.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(entry_manager_api, "_safe_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(entry_manager_api, "_resolve_latest_sim_v2_worlds", lambda **_: {})

    game_date = "2026-01-12"
    contest_id = "333"
    dg = 888
    uploaded_header = [
        "Entry ID", "Contest Name", "Contest ID", "Entry Fee",
        "PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "", "Instructions",
    ]

    entry_state = EntryFileState(
        game_date=game_date,
        draft_group_id=dg,
        site="dk",
        contest_id=contest_id,
        contest_name="Contest",
        entry_fee="1",
        created_at="t",
        updated_at="t",
        client_revision=1,
        header=uploaded_header,
        entries=[
            {
                "entry_id": "e1",
                "entry_key": "e1",
                "contest_id": contest_id,
                "contest_name": "Contest",
                "entry_fee": "1",
                "PG": "A (1)",
                "SG": "B (2)",
                "SF": "C (3)",
                "PF": "D (4)",
                "C": "E (5)",
                "G": "F (6)",
                "F": "G (7)",
                "UTIL": "H (8)",
            }
        ],
    )

    entry_path = fake_data_path("entries") / game_date / "dk" / f"{contest_id}.json"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(entry_state.model_dump_json(indent=2), encoding="utf-8")

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    resp = client.post(f"/api/entry-manager/entries/{contest_id}/export", params={"date": game_date})
    assert resp.status_code == 200
    header = resp.text.splitlines()[0]
    assert header == ",".join(uploaded_header)


def test_entry_manager_batch_export_combines_multiple_contests(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_root"

    def fake_data_path(*parts: Any) -> Path:
        return data_root.joinpath(*parts)

    monkeypatch.setattr(paths, "data_path", fake_data_path)

    def fake_popen(*_args, **_kwargs):
        class DummyProc:
            pass

        return DummyProc()

    import projections.api.entry_manager_api as entry_manager_api

    monkeypatch.setattr(entry_manager_api.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(entry_manager_api, "_safe_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(entry_manager_api, "_resolve_latest_sim_v2_worlds", lambda **_: {})

    game_date = "2026-01-13"
    dg = 777
    header = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee", "PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

    for contest_id, entry_prefix in (("111", "a"), ("222", "b")):
        entry_state = EntryFileState(
            game_date=game_date,
            draft_group_id=dg,
            site="dk",
            contest_id=contest_id,
            contest_name=f"Contest {contest_id}",
            entry_fee="1",
            created_at="t",
            updated_at="t",
            client_revision=1,
            header=header,
            entries=[
                {
                    "entry_id": f"{entry_prefix}1",
                    "entry_key": f"{entry_prefix}1",
                    "contest_id": contest_id,
                    "contest_name": f"Contest {contest_id}",
                    "entry_fee": "1",
                    "PG": "A (1)",
                    "SG": "B (2)",
                    "SF": "C (3)",
                    "PF": "D (4)",
                    "C": "E (5)",
                    "G": "F (6)",
                    "F": "G (7)",
                    "UTIL": "H (8)",
                },
                {
                    "entry_id": f"{entry_prefix}2",
                    "entry_key": f"{entry_prefix}2",
                    "contest_id": contest_id,
                    "contest_name": f"Contest {contest_id}",
                    "entry_fee": "1",
                    "PG": "I (9)",
                    "SG": "J (10)",
                    "SF": "K (11)",
                    "PF": "L (12)",
                    "C": "M (13)",
                    "G": "N (14)",
                    "F": "O (15)",
                    "UTIL": "P (16)",
                },
            ],
        )

        entry_path = fake_data_path("entries") / game_date / "dk" / f"{contest_id}.json"
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(entry_state.model_dump_json(indent=2), encoding="utf-8")

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/api/entry-manager/entries/export",
        params={"date": game_date},
        json={"contest_ids": ["111", "222"]},
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Entry-Count") == "4"
    csv_lines = resp.text.strip().splitlines()
    assert len(csv_lines) == 5
    assert csv_lines[1].startswith("a1,Contest 111,111,1,")
    assert csv_lines[4].startswith("b2,Contest 222,222,1,")
