from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from projections import paths
from projections.api.entry_manager_api import EntryFileState
from projections.api.minutes_api import create_app


def _fd_template_csv() -> str:
    return (
        "entry_id,contest_id,contest_name,entry_fee,PG,PG,SG,SG,SF,SF,PF,PF,C,\"\",\"Instructions\"\n"
        "\"3670015043\",\"127613-279208572\",\"FD Contest\",\"0.25\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"1) Fill slots\"\n"
        "\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"2) More instructions\"\n"
        "\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",Player ID + Player Name,Id,Position\n"
        "\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",127613-157808:Cade Cunningham,127613-157808,PG\n"
    )


def test_fd_upload_template_parses_and_persists(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data_root"

    def fake_data_path(*parts: Any) -> Path:
        return data_root.joinpath(*parts)

    monkeypatch.setattr(paths, "data_path", fake_data_path)

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/api/entry-manager/entries/upload",
        params={"date": "2026-03-15"},
        files={"file": ("fd_entries.csv", _fd_template_csv(), "text/csv")},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["contest_id"] == "127613-279208572"
    assert payload[0]["draft_group_id"] == 127613

    state_path = data_root / "entries" / "2026-03-15" / "fd" / "127613-279208572.json"
    assert state_path.exists()
    state = EntryFileState.model_validate_json(state_path.read_text(encoding="utf-8"))
    assert state.site == "fd"
    assert state.entries[0]["PG1"] == ""
    assert state.entries[0]["PG2"] == ""
    assert state.entries[0]["C"] == ""

    list_resp = client.get("/api/entry-manager/entries", params={"date": "2026-03-15", "site": "fd"})
    assert list_resp.status_code == 200
    listed = list_resp.json()
    assert len(listed) == 1
    assert listed[0]["contest_id"] == "127613-279208572"


def test_fd_export_preserves_duplicate_position_columns(tmp_path: Path, monkeypatch) -> None:
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

    game_date = "2026-03-15"
    contest_id = "127613-279208572"
    dg = 127613
    header = ["entry_id", "contest_id", "contest_name", "entry_fee", "PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C", "", "Instructions"]

    state = EntryFileState(
        game_date=game_date,
        draft_group_id=dg,
        site="fd",
        contest_id=contest_id,
        contest_name="FD Contest",
        entry_fee="0.25",
        created_at="t",
        updated_at="t",
        client_revision=1,
        header=header,
        entries=[
            {
                "entry_id": "3670015043",
                "entry_key": "3670015043",
                "contest_id": contest_id,
                "contest_name": "FD Contest",
                "entry_fee": "0.25",
                "PG1": "127613-157808",
                "PG2": "127613-9644",
                "SG1": "127613-66113",
                "SG2": "127613-228113",
                "SF1": "127613-145539",
                "SF2": "127613-157822",
                "PF1": "127613-40199",
                "PF2": "127613-157845",
                "C": "127613-49111",
            }
        ],
    )

    state_path = data_root / "entries" / game_date / "fd" / f"{contest_id}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    resp = client.post(
        f"/api/entry-manager/entries/{contest_id}/export",
        params={"date": game_date, "site": "fd"},
    )
    assert resp.status_code == 200
    rows = list(csv.reader(io.StringIO(resp.text)))
    assert rows[0] == header
    assert rows[1][:4] == ["3670015043", contest_id, "FD Contest", "0.25"]
    assert rows[1][4:13] == [
        "127613-157808",
        "127613-9644",
        "127613-66113",
        "127613-228113",
        "127613-145539",
        "127613-157822",
        "127613-40199",
        "127613-157845",
        "127613-49111",
    ]

    export_id = resp.headers.get("X-Export-Id")
    assert export_id
    manifest_path = (
        data_root
        / "contests"
        / "fd"
        / f"game_date={game_date}"
        / f"dg={dg}"
        / "exports"
        / f"export_{export_id}_manifest.json"
    )
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["site"] == "fd"
