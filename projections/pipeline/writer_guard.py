"""Single-writer guard for production pipeline publishes.

This enforces that only the canonical Prefect flow (entrypoint="prefect")
can update mutable pointer files (e.g., latest_run.json, LATEST/current.json).

Implementation:
- A process-level advisory lock is held for the duration of a pipeline run.
- A writer token + lock path are exported via env vars so subprocess tasks can
  validate they're part of the active run before publishing.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from projections import paths

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

ENTRYPOINT_ENV = "PROJECTIONS_PIPELINE_ENTRYPOINT"
WRITER_TOKEN_ENV = "PROJECTIONS_PIPELINE_WRITER_TOKEN"
LOCK_PATH_ENV = "PROJECTIONS_PIPELINE_WRITER_LOCK_PATH"
ALLOW_UNSAFE_ENV = "PROJECTIONS_ALLOW_UNSAFE_POINTER_WRITES"

ENTRYPOINT_PREFECT = "prefect"


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y"}


def _read_lock_payload(lock_path: Path) -> dict[str, Any] | None:
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def assert_can_write_pointers(*, purpose: str) -> None:
    """Fail loudly if a caller attempts to publish pointers outside the guarded run."""
    if _truthy_env(ALLOW_UNSAFE_ENV):
        return

    entrypoint = os.environ.get(ENTRYPOINT_ENV)
    token = os.environ.get(WRITER_TOKEN_ENV)
    lock_path_raw = os.environ.get(LOCK_PATH_ENV)

    if entrypoint != ENTRYPOINT_PREFECT:
        raise RuntimeError(
            f"[writer-guard] Refusing to publish pointers for {purpose}: "
            f"{ENTRYPOINT_ENV}={entrypoint!r} (expected {ENTRYPOINT_PREFECT!r}). "
            f"Set {ALLOW_UNSAFE_ENV}=1 to override (not for production)."
        )

    if not token or not lock_path_raw:
        raise RuntimeError(
            f"[writer-guard] Refusing to publish pointers for {purpose}: missing "
            f"{WRITER_TOKEN_ENV} / {LOCK_PATH_ENV} in environment."
        )

    lock_path = Path(lock_path_raw)
    payload = _read_lock_payload(lock_path)
    if not payload:
        raise RuntimeError(
            f"[writer-guard] Refusing to publish pointers for {purpose}: "
            f"lock payload missing/unreadable at {lock_path}."
        )

    if str(payload.get("token") or "") != token:
        raise RuntimeError(
            f"[writer-guard] Refusing to publish pointers for {purpose}: "
            "writer token mismatch (possible concurrent writer)."
        )


@dataclass
class PipelineWriterLock:
    """Context manager holding the single-writer lock for a pipeline run."""

    data_root: Path | None = None
    run_id: str | None = None
    entrypoint: str = ENTRYPOINT_PREFECT

    lock_name: str = "nba_live_pipeline.lock"

    _lock_path: Path | None = None
    _token: str | None = None
    _fh: Any = None

    def __enter__(self) -> "PipelineWriterLock":
        root = Path(self.data_root) if self.data_root is not None else paths.get_data_root()
        lock_dir = root / "artifacts" / "runs" / "nba_live" / "_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        self._token = uuid.uuid4().hex
        self._lock_path = lock_dir / self.lock_name

        fh = open(self._lock_path, "a+", encoding="utf-8")  # noqa: SIM115
        self._fh = fh

        if fcntl is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"[writer-guard] Another writer is active (lock held): {self._lock_path}"
                ) from exc

        payload = {
            "entrypoint": self.entrypoint,
            "run_id": self.run_id,
            "token": self._token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": _utc_now_iso(),
        }
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(payload, indent=2, sort_keys=True))
        fh.flush()

        os.environ[ENTRYPOINT_ENV] = self.entrypoint
        os.environ[WRITER_TOKEN_ENV] = self._token
        os.environ[LOCK_PATH_ENV] = str(self._lock_path)

        return self

    @property
    def lock_path(self) -> Path:
        if self._lock_path is None:
            raise RuntimeError("Writer lock not initialized")
        return self._lock_path

    @property
    def token(self) -> str:
        if self._token is None:
            raise RuntimeError("Writer lock not initialized")
        return self._token

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        # Keep the file for post-mortem inspection; release the advisory lock.
        for name in (ENTRYPOINT_ENV, WRITER_TOKEN_ENV, LOCK_PATH_ENV):
            try:
                os.environ.pop(name, None)
            except Exception:
                pass
        try:
            if self._fh is not None:
                try:
                    self._fh.flush()
                except Exception:
                    pass
                try:
                    self._fh.close()
                except Exception:
                    pass
        finally:
            self._fh = None
