from __future__ import annotations

import logging
from pathlib import Path


def test_rotation_priors_update_missing_vendor_url_skips_fetch(
    monkeypatch, tmp_path: Path
) -> None:
    import prefect_flows.rotation_priors_update as flow_mod

    calls: list[str] = []

    monkeypatch.setattr(
        flow_mod, "get_run_logger", lambda: logging.getLogger("rotation-priors-test")
    )
    monkeypatch.setattr(flow_mod.paths, "get_data_root", lambda: tmp_path)
    monkeypatch.setattr(
        flow_mod,
        "_resolve_pbp_input_glob",
        lambda **_: str(tmp_path / "bronze" / "pbp_vendor" / "*.csv"),
    )

    def _fetch(**kwargs) -> None:  # noqa: ANN003
        calls.append("fetch")

    def _ingest(**kwargs):  # noqa: ANN003
        calls.append("ingest")
        return tmp_path / "artifacts" / "pbp_v1" / "run-123"

    def _build_stints(**kwargs) -> None:  # noqa: ANN003
        calls.append("build_stints")

    def _qa(**kwargs) -> None:  # noqa: ANN003
        calls.append("qa")

    def _publish(**kwargs) -> None:  # noqa: ANN003
        calls.append("publish")

    def _build_rotation(**kwargs) -> None:  # noqa: ANN003
        calls.append("build_rotation")

    def _build_priors(**kwargs) -> None:  # noqa: ANN003
        calls.append("build_priors")

    def _tracking_backfill(**kwargs) -> None:  # noqa: ANN003
        calls.append("tracking_backfill")

    def _tracking_roles(**kwargs) -> None:  # noqa: ANN003
        calls.append("tracking_roles")

    def _tracking_guard(**kwargs):  # noqa: ANN003
        calls.append("tracking_guard")
        return {"status": "ok", "message": ""}

    monkeypatch.setattr(flow_mod, "pbp_vendor_fetch_daily_zip_task", _fetch)
    monkeypatch.setattr(flow_mod, "pbp_vendor_ingest_task", _ingest)
    monkeypatch.setattr(flow_mod, "pbp_build_stints_task", _build_stints)
    monkeypatch.setattr(flow_mod, "pbp_qa_task", _qa)
    monkeypatch.setattr(flow_mod, "pbp_publish_task", _publish)
    monkeypatch.setattr(flow_mod, "tracking_backfill_task", _tracking_backfill)
    monkeypatch.setattr(flow_mod, "tracking_build_roles_task", _tracking_roles)
    monkeypatch.setattr(flow_mod, "tracking_coverage_guard_task", _tracking_guard)
    monkeypatch.setattr(flow_mod, "build_rotation_v1_task", _build_rotation)
    monkeypatch.setattr(flow_mod, "build_rotation_priors_v1_task", _build_priors)

    result = flow_mod.rotation_priors_update_flow.fn(
        run_pbp_ingest=True,
        pbp_fetch_daily_zip=True,
        pbp_vendor_daily_url=None,
        pbp_run_id="run-123",
        pbp_season_id="2025-26",
    )

    assert result["status"] == "ok"
    assert "fetch" not in calls
    assert calls == [
        "ingest",
        "build_stints",
        "qa",
        "publish",
        "tracking_backfill",
        "tracking_roles",
        "tracking_guard",
        "build_rotation",
        "build_priors",
    ]


def test_rotation_priors_update_with_vendor_url_runs_fetch(
    monkeypatch, tmp_path: Path
) -> None:
    import prefect_flows.rotation_priors_update as flow_mod

    calls: list[str] = []

    monkeypatch.setattr(
        flow_mod, "get_run_logger", lambda: logging.getLogger("rotation-priors-test")
    )
    monkeypatch.setattr(flow_mod.paths, "get_data_root", lambda: tmp_path)
    monkeypatch.setattr(
        flow_mod,
        "_resolve_pbp_input_glob",
        lambda **_: str(tmp_path / "bronze" / "pbp_vendor" / "*.csv"),
    )

    def _fetch(**kwargs) -> None:  # noqa: ANN003
        calls.append("fetch")

    def _ingest(**kwargs):  # noqa: ANN003
        calls.append("ingest")
        return tmp_path / "artifacts" / "pbp_v1" / "run-123"

    def _tracking_backfill(**kwargs) -> None:  # noqa: ANN003
        calls.append("tracking_backfill")

    def _tracking_roles(**kwargs) -> None:  # noqa: ANN003
        calls.append("tracking_roles")

    def _tracking_guard(**kwargs):  # noqa: ANN003
        calls.append("tracking_guard")
        return {"status": "ok", "message": ""}

    monkeypatch.setattr(flow_mod, "pbp_vendor_fetch_daily_zip_task", _fetch)
    monkeypatch.setattr(flow_mod, "pbp_vendor_ingest_task", _ingest)
    monkeypatch.setattr(flow_mod, "pbp_build_stints_task", lambda **_: None)
    monkeypatch.setattr(flow_mod, "pbp_qa_task", lambda **_: None)
    monkeypatch.setattr(flow_mod, "pbp_publish_task", lambda **_: None)
    monkeypatch.setattr(flow_mod, "tracking_backfill_task", _tracking_backfill)
    monkeypatch.setattr(flow_mod, "tracking_build_roles_task", _tracking_roles)
    monkeypatch.setattr(flow_mod, "tracking_coverage_guard_task", _tracking_guard)
    monkeypatch.setattr(flow_mod, "build_rotation_v1_task", lambda **_: None)
    monkeypatch.setattr(flow_mod, "build_rotation_priors_v1_task", lambda **_: None)

    result = flow_mod.rotation_priors_update_flow.fn(
        run_pbp_ingest=True,
        pbp_fetch_daily_zip=True,
        pbp_vendor_daily_url="https://example.com/pbp.zip",
        pbp_run_id="run-123",
        pbp_season_id="2025-26",
    )

    assert result["status"] == "ok"
    assert calls == ["fetch", "ingest", "tracking_backfill", "tracking_roles", "tracking_guard"]
