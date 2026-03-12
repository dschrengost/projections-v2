from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

LATEST_POINTER = "latest_run.json"
PINNED_POINTER = "pinned_run.json"
BLESSED_POINTER = "blessed_run.json"

PROJECTIONS_FILENAME = "projections.parquet"
SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True, slots=True)
class UnifiedRunContext:
    date: str
    resolved_run_id: str | None
    blessed_run_id: str | None
    pinned_run_id: str | None
    latest_run_id: str | None


def _read_pointer_run_id(pointer_path: Path) -> str | None:
    if not pointer_path.exists():
        return None
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    run_id = payload.get("run_id")
    return str(run_id) if run_id else None


def resolve_unified_run_dir(
    data_root: Path,
    slate_day: date,
    *,
    run_id: str | None = None,
) -> tuple[Path | None, UnifiedRunContext]:
    """Resolve the unified projections run dir with priority blessed > pinned > latest."""
    day_dir = data_root / "artifacts" / "projections" / slate_day.isoformat()
    context = UnifiedRunContext(
        date=slate_day.isoformat(),
        resolved_run_id=None,
        blessed_run_id=_read_pointer_run_id(day_dir / BLESSED_POINTER) if day_dir.exists() else None,
        pinned_run_id=_read_pointer_run_id(day_dir / PINNED_POINTER) if day_dir.exists() else None,
        latest_run_id=_read_pointer_run_id(day_dir / LATEST_POINTER) if day_dir.exists() else None,
    )

    if not day_dir.exists():
        return None, context

    if run_id:
        candidate = day_dir / f"run={run_id}"
        if (candidate / PROJECTIONS_FILENAME).exists():
            return candidate, context.__class__(
                date=context.date,
                resolved_run_id=str(run_id),
                blessed_run_id=context.blessed_run_id,
                pinned_run_id=context.pinned_run_id,
                latest_run_id=context.latest_run_id,
            )
        return None, context

    for desired in (context.blessed_run_id, context.pinned_run_id, context.latest_run_id):
        if not desired:
            continue
        candidate = day_dir / f"run={desired}"
        if (candidate / PROJECTIONS_FILENAME).exists():
            return candidate, context.__class__(
                date=context.date,
                resolved_run_id=str(desired),
                blessed_run_id=context.blessed_run_id,
                pinned_run_id=context.pinned_run_id,
                latest_run_id=context.latest_run_id,
            )

    run_dirs = sorted([p for p in day_dir.glob("run=*") if p.is_dir()], reverse=True)
    for candidate in run_dirs:
        if (candidate / PROJECTIONS_FILENAME).exists():
            resolved = candidate.name.split("=", 1)[1] if candidate.name.startswith("run=") else None
            return candidate, context.__class__(
                date=context.date,
                resolved_run_id=resolved,
                blessed_run_id=context.blessed_run_id,
                pinned_run_id=context.pinned_run_id,
                latest_run_id=context.latest_run_id,
            )

    return None, context


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / SUMMARY_FILENAME
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def load_projections_df(run_dir: Path) -> pd.DataFrame:
    parquet_path = run_dir / PROJECTIONS_FILENAME
    if not parquet_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(parquet_path)

