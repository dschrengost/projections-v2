"""Prune planner/executor for non-canonical run directories."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from projections.storage_retention.common import parse_any_ts, utc_now_compact, write_json
from projections.storage_retention.paths import retention_reports_dir

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PruneParams:
    execute: bool = False
    max_delete_files: int | None = None
    max_delete_bytes: int | None = None
    min_age_hours: float = 0.0


_LOCK_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("artifacts", "runs", "nba_live", "_locks", "nba_live_pipeline.lock"),
    ("artifacts", "runs", "nba_live_v3", "_locks", "nba_live_pipeline.lock"),
)


def _writer_lock_active(lock_path: Path) -> bool:
    if not lock_path.exists() or fcntl is None:
        return False
    try:
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    except OSError:
        return False
    return False


def assert_no_active_writer(*, data_root: Path) -> None:
    active: list[str] = []
    for rel in _LOCK_CANDIDATES:
        path = data_root.joinpath(*rel)
        if _writer_lock_active(path):
            active.append(str(path))
    if active:
        raise RuntimeError(
            "Refusing prune execute while live writer lock is active: " + ", ".join(active)
        )


def build_prune_plan(
    *,
    canonical_output: dict[str, Any],
    params: PruneParams,
) -> dict[str, Any]:
    now = datetime.now(tz=UTC)
    decisions = list(canonical_output.get("decisions") or [])

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    delete_files = 0
    delete_bytes = 0

    for row in decisions:
        run_path = Path(str(row.get("run_path") or "")).expanduser()
        protected = bool(row.get("protected"))
        classification = str(row.get("classification") or "")
        size_bytes = int(row.get("size_bytes") or 0)
        file_count = int(row.get("file_count") or 0)

        if protected:
            skipped.append({"run_path": str(run_path), "reason": "protected"})
            continue
        if classification != "noncanonical":
            skipped.append(
                {
                    "run_path": str(run_path),
                    "reason": f"classification={classification}",
                }
            )
            continue

        as_of = parse_any_ts(row.get("as_of_ts")) or parse_any_ts(row.get("run_ts"))
        if as_of is not None and params.min_age_hours > 0:
            age_hours = (now - as_of).total_seconds() / 3600.0
            if age_hours < float(params.min_age_hours):
                skipped.append(
                    {
                        "run_path": str(run_path),
                        "reason": f"age_hours<{params.min_age_hours}",
                    }
                )
                continue

        if params.max_delete_files is not None and (delete_files + file_count) > int(params.max_delete_files):
            skipped.append(
                {
                    "run_path": str(run_path),
                    "reason": "max_delete_files_cap",
                }
            )
            continue
        if params.max_delete_bytes is not None and (delete_bytes + size_bytes) > int(params.max_delete_bytes):
            skipped.append(
                {
                    "run_path": str(run_path),
                    "reason": "max_delete_bytes_cap",
                }
            )
            continue

        delete_files += file_count
        delete_bytes += size_bytes
        candidates.append(
            {
                "family": row.get("family"),
                "game_date": row.get("game_date"),
                "run_id": row.get("run_id"),
                "run_path": str(run_path),
                "size_bytes": size_bytes,
                "file_count": file_count,
            }
        )

    return {
        "generated_at": now.isoformat(),
        "execute": bool(params.execute),
        "summary": {
            "candidate_count": int(len(candidates)),
            "candidate_files": int(delete_files),
            "candidate_bytes": int(delete_bytes),
            "skipped_count": int(len(skipped)),
        },
        "candidates": candidates,
        "skipped": skipped,
    }


def execute_prune_plan(*, plan: dict[str, Any]) -> dict[str, Any]:
    deleted: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for row in list(plan.get("candidates") or []):
        path = Path(str(row.get("run_path") or "")).expanduser()
        if not path.exists():
            deleted.append({"run_path": str(path), "status": "missing"})
            continue
        try:
            shutil.rmtree(path)
        except OSError as exc:
            errors.append({"run_path": str(path), "error": str(exc)})
            continue
        deleted.append(
            {
                "run_path": str(path),
                "size_bytes": int(row.get("size_bytes") or 0),
                "file_count": int(row.get("file_count") or 0),
                "status": "deleted",
            }
        )

    return {
        "executed_at": datetime.now(tz=UTC).isoformat(),
        "deleted": deleted,
        "errors": errors,
        "summary": {
            "deleted_count": int(len([d for d in deleted if d.get("status") == "deleted"])),
            "deleted_bytes": int(sum(int(d.get("size_bytes") or 0) for d in deleted if d.get("status") == "deleted")),
            "deleted_files": int(sum(int(d.get("file_count") or 0) for d in deleted if d.get("status") == "deleted")),
            "error_count": int(len(errors)),
        },
    }


def write_prune_reports(*, hot_root: Path, plan: dict[str, Any], ledger: dict[str, Any] | None = None) -> dict[str, str]:
    ts = utc_now_compact()
    reports_dir = retention_reports_dir(hot_root=hot_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    plan_path = reports_dir / f"{ts}_storage_prune_plan.json"
    write_json(plan_path, plan)

    out = {"plan": str(plan_path)}
    if ledger is not None:
        ledger_path = reports_dir / f"{ts}_storage_prune_ledger.json"
        write_json(ledger_path, ledger)
        out["ledger"] = str(ledger_path)
    return out


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload
