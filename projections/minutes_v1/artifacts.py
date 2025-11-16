"""Helpers for writing Minutes V1 artifacts to disk."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_run_directory(run_id: str, *, root: Path = Path("artifacts/minutes_v1")) -> Path:
    if not run_id:
        raise ValueError("run_id must be provided")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def compute_feature_hash(feature_columns: Iterable[str]) -> str:
    """Derive a deterministic hash for the ordered list of feature names."""

    normalized = "\n".join(str(col) for col in sorted(feature_columns))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_metadata(run_dir: Path, *, metrics: Mapping[str, Any], params: Mapping[str, Any], feature_hash: str) -> None:
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "meta.json",
        {
            "feature_hash": feature_hash,
            "params": params,
        },
    )

