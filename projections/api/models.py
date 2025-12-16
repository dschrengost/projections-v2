from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class JobStatusModel(BaseModel):
    job_name: str
    stage: str
    target_date: str
    run_ts: str
    status: str
    rows_written: int
    expected_rows: Optional[int] = None
    nan_rate_key_cols: Optional[float] = None
    schema_version: Optional[str] = None
    message: Optional[str] = None


class PipelineJobSummary(BaseModel):
    job_name: str
    stage: str
    target_date: str
    status: str
    run_ts: str
    rows_written: int
    expected_rows: Optional[int] = None
    nan_rate_key_cols: Optional[float] = None
    message: Optional[str] = None


class PipelineSummary(BaseModel):
    target_date: str
    slate_id: Optional[str] = None
    all_jobs_healthy: bool
    jobs: List[PipelineJobSummary]
