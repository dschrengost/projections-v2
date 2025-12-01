"""Helpers for writing pipeline job status snapshots."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

DATA_ROOT = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data"))
STATUS_ROOT = DATA_ROOT / "bronze" / "pipeline_status"


@dataclass
class JobStatus:
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


def write_status(status: JobStatus) -> None:
    """Persist a JobStatus snapshot for operators.

    Errors are logged but do not raise to avoid interrupting the main job flow.
    """

    try:
        STATUS_ROOT.mkdir(parents=True, exist_ok=True)
        safe_target = status.target_date.replace(":", "_")
        filename = f"{status.job_name}_{safe_target}.json"
        path = STATUS_ROOT / filename
        payload = asdict(status)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to write pipeline status for %s: %s", status.job_name, exc)
