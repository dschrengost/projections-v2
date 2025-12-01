from __future__ import annotations

import json

from fastapi.testclient import TestClient

from projections.api import pipeline_status_api
from projections.api.minutes_api import create_app
from projections.pipeline import status as pipeline_status


def test_get_status_reads_status_files(tmp_path) -> None:
    pipeline_status_api.STATUS_ROOT = tmp_path
    pipeline_status.STATUS_ROOT = tmp_path

    payload = {
        "job_name": "build_minutes_gold",
        "stage": "gold",
        "target_date": "2025-11-30",
        "run_ts": "2025-11-30T12:00:00Z",
        "status": "success",
        "rows_written": 42,
        "expected_rows": 50,
        "nan_rate_key_cols": 0.1,
        "schema_version": "v2",
        "message": "ok",
    }
    (tmp_path / "build_minutes_gold_2025-11-30.json").write_text(json.dumps(payload), encoding="utf-8")

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    response = client.get("/api/pipeline/status")
    assert response.status_code == 200
    assert response.json() == [payload]


def test_pipeline_summary_health_flag(tmp_path) -> None:
    pipeline_status_api.STATUS_ROOT = tmp_path
    pipeline_status.STATUS_ROOT = tmp_path

    success = {
        "job_name": "injuries_live",
        "stage": "silver",
        "target_date": "2025-11-30",
        "run_ts": "2025-11-30T12:00:00Z",
        "status": "success",
        "rows_written": 10,
    }
    error = {
        "job_name": "score_fpts_live",
        "stage": "projections",
        "target_date": "2025-11-30",
        "run_ts": "2025-11-30T12:05:00Z",
        "status": "error",
        "rows_written": 0,
    }
    (tmp_path / "injuries_live_2025-11-30.json").write_text(json.dumps(success), encoding="utf-8")
    (tmp_path / "score_fpts_live_2025-11-30.json").write_text(json.dumps(error), encoding="utf-8")

    app = create_app(daily_root=tmp_path, dashboard_dist=tmp_path, fpts_root=tmp_path)
    client = TestClient(app)

    response = client.get("/api/pipeline/summary", params={"target_date": "2025-11-30"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["target_date"] == "2025-11-30"
    assert len(payload["jobs"]) == 2
    assert payload["all_jobs_healthy"] is False

    # Mark the error job as success and expect healthy flag to flip.
    error["status"] = "success"
    (tmp_path / "score_fpts_live_2025-11-30.json").write_text(json.dumps(error), encoding="utf-8")
    response = client.get("/api/pipeline/summary", params={"target_date": "2025-11-30"})
    payload = response.json()
    assert payload["all_jobs_healthy"] is True
