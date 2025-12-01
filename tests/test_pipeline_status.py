from __future__ import annotations

import json
from datetime import datetime, timezone

from projections.pipeline import status as pipeline_status
from projections.pipeline.status import JobStatus


def test_write_status(tmp_path) -> None:
    pipeline_status.STATUS_ROOT = tmp_path

    job_status = JobStatus(
        job_name="scrape_injuries",
        stage="bronze",
        target_date="2025-11-30",
        run_ts=datetime.now(timezone.utc).isoformat(),
        status="success",
        rows_written=123,
        expected_rows=130,
        nan_rate_key_cols=0.01,
        schema_version="v1",
        message="ok",
    )

    pipeline_status.write_status(job_status)

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    with files[0].open() as f:
        data = json.load(f)

    assert data["job_name"] == job_status.job_name
    assert data["stage"] == job_status.stage
    assert data["target_date"] == job_status.target_date
    assert data["status"] == job_status.status
    assert data["rows_written"] == job_status.rows_written
    assert data["expected_rows"] == job_status.expected_rows
    assert data["nan_rate_key_cols"] == job_status.nan_rate_key_cols
    assert data["schema_version"] == job_status.schema_version
    assert data["message"] == job_status.message
