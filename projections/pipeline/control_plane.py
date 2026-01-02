"""Control plane utilities for the live pipeline.

This module is responsible for:
- canonical run_id generation
- run manifest creation (written at run start)
- atomic pointer promotion (LATEST/current.json)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from projections import paths

LATEST_DIRNAME = "LATEST"
CURRENT_POINTER_NAME = "current.json"
LEGACY_POINTER_NAME = "latest_run.json"

ALLOW_UNPROMOTED_RUN_READS_ENV = "PROJECTIONS_ALLOW_UNPROMOTED_RUN_READS"

RUN_MANIFEST_NAME = "manifest.json"
RUN_MANIFEST_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def allow_unpromoted_run_reads() -> bool:
    """Return True when readers may scan run directories without a promoted pointer.

    Default is False to ensure production consumers only observe atomically-promoted outputs.
    """
    return os.environ.get(ALLOW_UNPROMOTED_RUN_READS_ENV, "").strip().lower() in {"1", "true", "yes", "y"}


def canonical_run_id(ts: datetime | None = None) -> str:
    """Return a canonical run_id (YYYYMMDDTHHMMSSZ)."""
    dt = ts or datetime.now(tz=UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC).replace(microsecond=0)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def resolve_git_sha(*, project_root: Path | None = None) -> str | None:
    """Return the current git SHA, or None if unavailable."""
    root = project_root or paths.get_project_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    sha = (result.stdout or "").strip()
    return sha if sha else None


def resolve_hostname() -> str:
    return socket.gethostname()


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Run-scoped manifest written at pipeline start."""

    run_id: str
    game_date: str
    as_of_ts: str
    sim_profile: str
    entrypoint: str

    git_sha: str | None = None
    hostname: str | None = None

    minutes_current_run_path: str = "config/minutes_current_run.json"
    rates_current_run_path: str = "config/rates_current_run.json"

    created_at: str = field(default_factory=_utc_now_iso)
    version: int = RUN_MANIFEST_VERSION

    slate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Drop nulls for compactness.
        return {k: v for k, v in payload.items() if v is not None}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{canonical_run_id()}.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def atomic_update_json(path: Path, patch: dict[str, Any]) -> dict[str, Any]:
    """Atomically update a JSON file with a shallow dict patch.

    Returns the updated payload.
    """
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        current = {}
    except (OSError, json.JSONDecodeError):
        current = {}

    if not isinstance(current, dict):
        current = {}
    current.update(patch)
    atomic_write_json(path, current)
    return current


def write_run_manifest_start(
    *,
    data_root: Path | None,
    game_date: str,
    run_id: str,
    as_of_ts: str,
    sim_profile: str,
    entrypoint: str,
    minutes_current_run_path: Path,
    rates_current_run_path: Path,
    slate: dict[str, Any] | None = None,
) -> Path:
    root = Path(data_root) if data_root is not None else paths.get_data_root()
    run_dir = root / "artifacts" / "runs" / "nba_live" / f"game_date={game_date}" / f"run={run_id}"
    manifest_path = run_dir / RUN_MANIFEST_NAME
    manifest = RunManifest(
        run_id=run_id,
        game_date=game_date,
        as_of_ts=as_of_ts,
        sim_profile=sim_profile,
        entrypoint=entrypoint,
        git_sha=resolve_git_sha(project_root=paths.get_project_root()),
        hostname=resolve_hostname(),
        minutes_current_run_path=str(minutes_current_run_path),
        rates_current_run_path=str(rates_current_run_path),
        slate=slate or {},
    )
    atomic_write_json(manifest_path, manifest.to_dict())
    return manifest_path


def copy_manifest_to_dir(manifest_path: Path, target_dir: Path) -> None:
    """Copy the run manifest into an artifact directory."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    atomic_write_json(Path(target_dir) / RUN_MANIFEST_NAME, payload)


def promote_run_pointer(
    *,
    dataset_dir: Path,
    run_id: str,
    manifest_path: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically promote a run as the latest published output for a dataset/day directory."""
    from projections.pipeline import writer_guard

    writer_guard.assert_can_write_pointers(purpose=f"promote {dataset_dir}")
    payload: dict[str, Any] = {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "updated_at": _utc_now_iso(),
        **(extra or {}),
    }
    current = dataset_dir / LATEST_DIRNAME / CURRENT_POINTER_NAME
    atomic_write_json(current, payload)
    # Back-compat: keep legacy pointer up to date for existing readers.
    atomic_write_json(dataset_dir / LEGACY_POINTER_NAME, payload)
    return current


def read_promoted_run_id(dataset_dir: Path) -> str | None:
    """Read promoted run_id from LATEST/current.json (preferred) or legacy latest_run.json."""
    preferred = dataset_dir / LATEST_DIRNAME / CURRENT_POINTER_NAME
    for candidate in (preferred, dataset_dir / LEGACY_POINTER_NAME):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("run_id"):
            return str(payload["run_id"])
    return None
