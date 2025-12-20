"""Raw candidate lineup pool caching for representative field libraries.

We cache the expensive QuickBuild-generated candidate set so we can recompress
field libraries cheaply when projections/ownership change during the day.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from projections.paths import data_path

__all__ = [
    "CandidatePool",
    "candidate_pool_dir",
    "candidate_pool_path",
    "load_candidate_pool",
    "save_candidate_pool",
    "merge_candidates",
]


def candidate_pool_dir(
    game_date: str,
    draft_group_id: int,
    *,
    data_root: Optional[Path] = None,
) -> Path:
    root = data_root or data_path()
    return root / "field_libraries" / f"game_date={game_date}" / f"draft_group_id={int(draft_group_id)}"


def candidate_pool_path(
    game_date: str,
    draft_group_id: int,
    *,
    version: str = "v0",
    data_root: Optional[Path] = None,
) -> Path:
    return candidate_pool_dir(game_date, draft_group_id, data_root=data_root) / f"candidate_pool_{version}.jsonl.gz"


def _canonical_key(lineup: Sequence[object]) -> Tuple[str, ...]:
    return tuple(sorted(str(p).strip() for p in lineup if str(p).strip()))


def merge_candidates(
    existing: Sequence[Sequence[object]],
    new: Sequence[Sequence[object]],
    *,
    max_size: int,
) -> List[List[str]]:
    """Union-dedupe candidates by canonical key, capped to max_size."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    seen: set[Tuple[str, ...]] = set()
    merged: List[List[str]] = []

    for src in (existing, new):
        for lu in src:
            key = _canonical_key(lu)
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            merged.append(list(key))
            if len(merged) >= max_size:
                return merged

    return merged


@dataclass(frozen=True)
class CandidatePool:
    """A cached raw candidate pool for a single slate."""

    lineups: List[List[str]]
    meta: Dict[str, Any]

    def to_meta(self) -> Dict[str, Any]:
        return {
            **(self.meta or {}),
            "candidate_count": int(len(self.lineups)),
        }


def save_candidate_pool(pool: CandidatePool, path: Path) -> None:
    """Save as gzipped JSONL: first line is meta, remaining are lineup keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        **pool.to_meta(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}, sort_keys=True) + "\n")
        for lu in pool.lineups:
            key = ",".join(_canonical_key(lu))
            if not key:
                continue
            f.write(json.dumps({"key": key}) + "\n")


def load_candidate_pool(path: Path) -> CandidatePool:
    meta: Dict[str, Any] = {}
    lineups: List[List[str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0 and isinstance(obj, dict) and "_meta" in obj:
                meta = dict(obj.get("_meta") or {})
                continue
            key = str(obj.get("key") or "").strip()
            if not key:
                continue
            players = [p for p in key.split(",") if p]
            if players:
                lineups.append(players)
    return CandidatePool(lineups=lineups, meta=meta)
