from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from projections.rotations.manifest import get_git_sha, sha256_file, write_json
from projections.rotations.schemas import ROT_EVAL_V1_SCHEMA_VERSION


def resolve_rot_bundle_dir(rot_bundle_path: Path) -> Path:
    """Resolve a rot bundle directory or a LATEST_PUBLISHED pointer file to a directory."""
    p = Path(rot_bundle_path)
    if p.is_dir():
        return p
    if p.is_file():
        run_id = p.read_text(encoding="utf-8").strip()
        if not run_id:
            raise ValueError(f"Empty rot bundle pointer: {p}")
        resolved = p.parent / run_id
        if not resolved.exists():
            raise FileNotFoundError(f"Pointer {p} -> {resolved} does not exist")
        return resolved
    raise FileNotFoundError(f"rot bundle not found: {p}")


def write_rot_eval_input_hashes(
    *,
    rot_bundle_dir: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Write file hashes for the rot_v1 bundle inputs used by rot_eval_v1."""
    events_path = rot_bundle_dir / "rotation_events.parquet"
    labels_path = rot_bundle_dir / "rotation_labels.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing rotation_events: {events_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing rotation_labels: {labels_path}")

    files = {
        str(events_path): sha256_file(events_path),
        str(labels_path): sha256_file(labels_path),
    }
    payload: dict[str, Any] = {"files": files}
    write_json(out_path, payload)
    return payload


def build_rot_eval_manifest(
    *,
    repo_root: Path,
    rot_bundle_path: Path,
    rot_bundle_dir: Path,
    run_id: str,
    n_worlds: int,
    seed: int,
    limit_team_games: int,
    sample_mode: str,
    use_truth_minutes_prior: bool,
    input_hashes_path: Path,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    git_sha = get_git_sha(repo_root)
    input_hashes = json.loads(input_hashes_path.read_text(encoding="utf-8")).get("files", {})
    return {
        "schema_version": ROT_EVAL_V1_SCHEMA_VERSION,
        "created_at": created_at,
        "git_sha": git_sha,
        "rot_bundle_path": str(Path(rot_bundle_path)),
        "rot_bundle_dir": str(Path(rot_bundle_dir)),
        "run_id": str(run_id),
        "n_worlds": int(n_worlds),
        "seed": int(seed),
        "limit_team_games": int(limit_team_games),
        "sample_mode": str(sample_mode),
        "use_truth_minutes_prior": bool(use_truth_minutes_prior),
        "input_hashes": input_hashes,
    }

