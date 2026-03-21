"""Shared helpers for retention tooling."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


def utc_now_compact() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_iso_date(raw: str | None) -> date | None:
    if raw is None:
        return None
    raw_eff = str(raw).strip()
    if not raw_eff:
        return None
    return date.fromisoformat(raw_eff)


def parse_run_id_ts(run_id: str | None) -> datetime | None:
    if not run_id:
        return None
    token = str(run_id).strip()
    try:
        return datetime.strptime(token, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def parse_any_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
    value = str(raw).strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def floor_bucket_utc(ts: datetime, minutes: int) -> datetime:
    bucket = max(1, int(minutes))
    floored_minute = (ts.minute // bucket) * bucket
    return ts.replace(minute=floored_minute, second=0, microsecond=0)


def dir_size_bytes(path: Path) -> tuple[int, int]:
    total_bytes = 0
    file_count = 0
    for root, _dirs, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            try:
                stat = file_path.stat()
            except OSError:
                continue
            total_bytes += int(stat.st_size)
            file_count += 1
    return total_bytes, file_count


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
