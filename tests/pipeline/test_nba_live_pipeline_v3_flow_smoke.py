from __future__ import annotations

import json
from pathlib import Path

import pytest

from projections.pipeline import control_plane


def test_nba_live_pipeline_v3_flow_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prefect_flows import live_nba_pipeline_v3
    from projections import paths

    game_date = "2026-01-18"
    monkeypatch.setenv("PROJECTIONS_ALLOW_DIRTY", "1")
    monkeypatch.setattr(paths, "get_data_root", lambda: tmp_path)

    result = live_nba_pipeline_v3.nba_live_pipeline_v3_flow(
        game_date=game_date,
        sim_worlds=64,
        placeholder_mode=True,
        promote_pointers=True,
    )

    run_id = result["run_id"]
    assert run_id

    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "input_change_set" in manifest

    projections_path = Path(result["projections_path"])
    assert projections_path.exists()

    v3_run_dir = (
        tmp_path
        / "artifacts"
        / "runs"
        / "nba_live_v3"
        / f"game_date={game_date}"
        / f"run={run_id}"
    )
    assert (v3_run_dir / "preflight_report.json").exists()
    assert (v3_run_dir / "postflight_report.json").exists()
    assert (v3_run_dir / "stale_publish_report.json").exists()
    assert (v3_run_dir / "input_change_set.json").exists()

    promoted_pointer = (
        tmp_path
        / "artifacts"
        / "projections"
        / game_date
        / control_plane.LATEST_DIRNAME
        / control_plane.CURRENT_POINTER_NAME
    )
    assert promoted_pointer.exists()
    payload = json.loads(promoted_pointer.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert "source_freshness_summary" in payload
