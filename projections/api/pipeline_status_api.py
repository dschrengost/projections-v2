from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Query

from projections.pipeline.status import STATUS_ROOT

from .models import JobStatusModel, PipelineJobSummary, PipelineSummary

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _read_all_status_files() -> List[dict]:
    items: List[dict] = []
    if not STATUS_ROOT.exists():
        return items
    for path in STATUS_ROOT.glob("*.json"):
        try:
            with path.open() as f:
                data = json.load(f)
            items.append(data)
        except Exception:
            continue
    return items


@router.get("/status", response_model=List[JobStatusModel])
def get_status(
    target_date: Optional[str] = Query(None, description="Filter by exact target_date"),
    stage: Optional[str] = Query(None, description="Filter by stage"),
):
    items = []
    for data in _read_all_status_files():
        if target_date and data.get("target_date") != target_date:
            continue
        if stage and data.get("stage") != stage:
            continue
        items.append(data)

    items.sort(key=lambda d: (d.get("target_date", ""), d.get("job_name", ""), d.get("run_ts", "")))
    return items


@router.get("/summary", response_model=PipelineSummary)
def get_pipeline_summary(
    target_date: Optional[str] = Query(None, description="Calendar date YYYY-MM-DD"),
):
    if target_date is None:
        target_date = datetime.now(timezone.utc).date().isoformat()

    all_statuses = _read_all_status_files()
    jobs = [s for s in all_statuses if s.get("target_date") == target_date]
    all_jobs_healthy = bool(jobs) and all(j.get("status") == "success" for j in jobs)

    job_summaries = [
        PipelineJobSummary(
            job_name=j.get("job_name", ""),
            stage=j.get("stage", ""),
            target_date=j.get("target_date", ""),
            status=j.get("status", ""),
            run_ts=j.get("run_ts", ""),
            rows_written=j.get("rows_written", 0),
            expected_rows=j.get("expected_rows"),
            nan_rate_key_cols=j.get("nan_rate_key_cols"),
            message=j.get("message"),
        )
        for j in jobs
    ]

    return PipelineSummary(
        target_date=target_date,
        slate_id=None,
        all_jobs_healthy=all_jobs_healthy,
        jobs=job_summaries,
    )
