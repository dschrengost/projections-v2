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
    assert result["publish_status"] == "published"

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
    ownership_run_dir = (
        tmp_path
        / "silver"
        / "ownership_predictions"
        / game_date
        / f"run={run_id}"
    )
    assert ownership_run_dir.exists()
    assert list(ownership_run_dir.glob("*.parquet"))

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

    ownership_pointer = (
        tmp_path
        / "silver"
        / "ownership_predictions"
        / game_date
        / control_plane.LATEST_DIRNAME
        / control_plane.CURRENT_POINTER_NAME
    )
    assert ownership_pointer.exists()
    ownership_payload = json.loads(ownership_pointer.read_text(encoding="utf-8"))
    assert ownership_payload["run_id"] == run_id


def test_nba_live_pipeline_v3_flow_skips_gracefully_when_writer_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prefect_flows import live_nba_pipeline_v3
    from projections import paths

    class _BusyLock:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            raise RuntimeError(
                "[writer-guard] Another writer is active (lock held): busy"
            )

    monkeypatch.setenv("PROJECTIONS_ALLOW_DIRTY", "1")
    monkeypatch.setattr(paths, "get_data_root", lambda: tmp_path)
    monkeypatch.setattr(
        live_nba_pipeline_v3.writer_guard, "PipelineWriterLock", _BusyLock
    )

    result = live_nba_pipeline_v3.nba_live_pipeline_v3_flow(
        game_date="2026-01-18",
        sim_worlds=64,
        placeholder_mode=True,
        promote_pointers=True,
    )

    assert result["publish_status"] == "skipped_active_writer"
    assert result["pointer_count"] == "0"
    run_id = result["run_id"]
    report_path = (
        tmp_path
        / "artifacts"
        / "runs"
        / "nba_live_v3"
        / "game_date=2026-01-18"
        / f"run={run_id}"
        / "duplicate_run_report.json"
    )
    assert report_path.exists()
