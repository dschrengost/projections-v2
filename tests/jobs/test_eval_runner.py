from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from projections.jobs.eval_runner import _maybe_mark_stale_running, _try_acquire_lock


def test_try_acquire_lock_atomic(tmp_path: Path) -> None:
    lock_path = tmp_path / "lockfile.lock"
    assert _try_acquire_lock(lock_path, payload={"pid": 1}) is True
    assert lock_path.exists()
    assert _try_acquire_lock(lock_path, payload={"pid": 2}) is False


def test_maybe_mark_stale_running_marks_failed(tmp_path: Path) -> None:
    status_path = tmp_path / "eval_status.json"
    status_path.write_text(
        """
        {
          "status": "RUNNING",
          "export_id": "x",
          "created_at": "2026-01-10T00:00:00Z",
          "started_at": "2026-01-10T00:00:00Z",
          "finished_at": null,
          "pid": 999999,
          "return_code": null,
          "report_dir": "/tmp",
          "error_message": null,
          "warnings": []
        }
        """.strip(),
        encoding="utf-8",
    )
    # 0s threshold guarantees "stale" evaluation path, pid is not alive.
    updated = _maybe_mark_stale_running(status_path, stale_seconds=0)
    assert updated["status"] == "FAILED"
    assert updated["error_message"] == "stale job (server restart?)"
    assert updated["finished_at"]


def test_maybe_mark_stale_running_keeps_running_with_live_pid(tmp_path: Path) -> None:
    status_path = tmp_path / "eval_status.json"
    started_at = (datetime.now(tz=UTC) - timedelta(seconds=10)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    status_path.write_text(
        f"""
        {{
          "status": "RUNNING",
          "export_id": "x",
          "created_at": "{started_at}",
          "started_at": "{started_at}",
          "finished_at": null,
          "pid": {os.getpid()},
          "return_code": null,
          "report_dir": "/tmp",
          "error_message": null,
          "warnings": []
        }}
        """.strip(),
        encoding="utf-8",
    )
    updated = _maybe_mark_stale_running(status_path, stale_seconds=0)
    assert updated["status"] == "RUNNING"
    assert any("running longer than" in str(w) for w in (updated.get("warnings") or []))

