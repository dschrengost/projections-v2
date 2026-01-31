from __future__ import annotations

import json
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from projections.rotations.schemas import ROT_V1_SCHEMA_VERSION


def get_git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_latest_published_run_id(artifacts_root: Path) -> Optional[str]:
    p = artifacts_root / "LATEST_PUBLISHED"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() or None


def write_latest_published_run_id(artifacts_root: Path, run_id: str) -> None:
    p = artifacts_root / "LATEST_PUBLISHED"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(run_id, encoding="utf-8")


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    repo_root: Path,
    season_id: str,
    run_id: str,
    input_hashes_path: Path,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_sha(repo_root)
    input_hashes = read_json(input_hashes_path)
    return {
        "schema_version": ROT_V1_SCHEMA_VERSION,
        "git_sha": git_sha,
        "input_hashes": input_hashes.get("files", input_hashes),
        "season_id": season_id,
        "run_id": run_id,
        "created_at": created_at,
    }
