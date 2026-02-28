from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from projections.api.live_status_api import _classify_run_status
from projections.api.minutes_api import create_app


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_classify_run_status_waiting_for_fresh_input(tmp_path: Path) -> None:
    report_dir = tmp_path / "run=20260228T145959Z"
    _write_json(
        report_dir / "skip_report.json",
        {
            "mode": "skip",
            "reason": "no_material_game_changes",
            "target_game_ids": [],
        },
    )

    status, reason = _classify_run_status(
        run_id="20260228T145959Z",
        published_run_id="20260228T141502Z",
        published_as_of_ts="2026-02-28T14:15:22Z",
        manifest={"as_of_ts": "2026-02-28T15:00:19Z"},
        report_dir=report_dir,
    )

    assert status == "waiting_for_fresh_input"
    assert reason == "no_material_game_changes"


def test_live_status_endpoint_returns_candidate_and_games(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    game_date = "2026-02-28"
    published_run_id = "20260228T141502Z"
    candidate_run_id = "20260228T145959Z"

    _write_json(
        data_root / "artifacts" / "projections" / game_date / "latest_run.json",
        {
            "run_id": published_run_id,
            "as_of_ts": "2026-02-28T14:15:22Z",
            "updated_at": "2026-02-28T14:22:35Z",
        },
    )

    published_manifest = {
        "run_id": published_run_id,
        "created_at": "2026-02-28T14:15:23Z",
        "as_of_ts": "2026-02-28T14:15:22Z",
        "source_freshness": {
            "per_game": {
                "22500863": {
                    "game_id": 22500863,
                    "tip_ts": "2026-02-28T18:00:00Z",
                    "injuries": {"source_used": "bronze", "latest_as_of_ts": "2026-02-28T14:15:21Z"},
                    "lineups": {"source_used": "rotowire", "latest_as_of_ts": "2026-02-28T14:15:20Z"},
                }
            }
        },
        "freshness_gates": {
            "lock_window": {"ok": True},
            "report_window": {"active": False, "blocking_games": []},
        },
        "input_change_set": {
            "changed_game_ids": [22500863],
            "changed_games": [{"game_id": 22500863, "changed_sources": ["injuries"], "tip_ts": "2026-02-28T18:00:00Z"}],
        },
        "rerun_plan": {"mode": "full_slate", "target_game_ids": [22500863]},
    }
    _write_json(
        data_root / "artifacts" / "runs" / "nba_live" / f"game_date={game_date}" / f"run={published_run_id}" / "manifest.json",
        published_manifest,
    )

    candidate_manifest = {
        "run_id": candidate_run_id,
        "created_at": "2026-02-28T15:00:20Z",
        "as_of_ts": "2026-02-28T15:00:19Z",
        "source_freshness": {
            "per_game": {
                "22500863": {
                    "game_id": 22500863,
                    "tip_ts": "2026-02-28T18:00:00Z",
                    "injuries": {"source_used": "bronze", "latest_as_of_ts": "2026-02-28T15:00:18Z"},
                    "lineups": {"source_used": "rotowire", "latest_as_of_ts": "2026-02-28T14:59:58Z"},
                    "odds": {"source_used": "silver", "latest_as_of_ts": "2026-02-28T15:00:10Z"},
                    "props": {"source_used": "rotowire", "latest_as_of_ts": "2026-02-28T15:00:18Z"},
                }
            }
        },
        "freshness_gates": {
            "lock_window": {"ok": True},
            "report_window": {"active": False, "blocking_games": []},
        },
        "input_change_set": {
            "changed_game_ids": [22500863],
            "changed_games": [{"game_id": 22500863, "changed_sources": ["injuries", "props"], "tip_ts": "2026-02-28T18:00:00Z"}],
        },
        "rerun_plan": {"mode": "game_scoped", "target_game_ids": [22500863]},
    }
    _write_json(
        data_root / "artifacts" / "runs" / "nba_live" / f"game_date={game_date}" / f"run={candidate_run_id}" / "manifest.json",
        candidate_manifest,
    )
    _write_json(
        data_root / "artifacts" / "runs" / "nba_live_v3" / f"game_date={game_date}" / f"run={candidate_run_id}" / "preflight_report.json",
        {"as_of_ts": "2026-02-28T15:00:19Z"},
    )

    monkeypatch.setattr("projections.api.live_status_api.paths.data_path", lambda *args, **kwargs: data_root)

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path, sim_root=tmp_path)
    client = TestClient(app)

    response = client.get("/api/live/status", params={"date": game_date})
    assert response.status_code == 200
    payload = response.json()

    assert payload["latest_published_run_id"] == published_run_id
    assert payload["candidate_run_id"] == candidate_run_id
    assert payload["candidate_status"] == "in_progress"
    assert payload["candidate_status_reason"] == "awaiting_publish_completion"

    assert len(payload["games"]) == 1
    game = payload["games"][0]
    assert game["game_id"] == "22500863"
    assert game["affected_by_change_set"] is True
    assert game["rerun_targeted"] is True
    assert game["changed_sources"] == ["injuries", "props"]
    assert "candidate-running" in game["warning_badges"]
