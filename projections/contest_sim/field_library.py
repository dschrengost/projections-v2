"""Weighted opponent field libraries for contest simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from projections.paths import data_path

__all__ = [
    "FieldLibrary",
    "field_library_dir",
    "field_library_path",
    "load_field_library",
    "save_field_library",
    "list_field_library_paths",
]


@dataclass(frozen=True)
class FieldLibrary:
    """A compressed set of opponent lineups with integer weights."""

    lineups: List[List[str]]
    weights: List[int]
    meta: Dict[str, Any]

    def validate(self) -> None:
        if not self.lineups:
            raise ValueError("FieldLibrary.lineups must be non-empty")
        if len(self.lineups) != len(self.weights):
            raise ValueError("FieldLibrary.weights length must match FieldLibrary.lineups length")
        if any(int(w) < 0 for w in self.weights):
            raise ValueError("FieldLibrary.weights must be non-negative integers")

    def to_dict(self) -> Dict[str, Any]:
        return {"lineups": self.lineups, "weights": [int(w) for w in self.weights], "meta": self.meta}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FieldLibrary":
        lineups = payload.get("lineups")
        weights = payload.get("weights")
        meta = payload.get("meta")
        if not isinstance(lineups, list) or not isinstance(weights, list) or not isinstance(meta, dict):
            raise ValueError("Invalid FieldLibrary payload")
        library = cls(
            lineups=[[str(p) for p in lineup] for lineup in lineups],
            weights=[int(w) for w in weights],
            meta=dict(meta),
        )
        library.validate()
        return library


def field_library_dir(
    game_date: str,
    draft_group_id: int,
    data_root: Optional[Path] = None,
) -> Path:
    root = data_root or data_path()
    return root / "field_libraries" / f"game_date={game_date}" / f"draft_group_id={int(draft_group_id)}"


def field_library_path(
    game_date: str,
    draft_group_id: int,
    version: str = "v0",
    data_root: Optional[Path] = None,
) -> Path:
    return field_library_dir(game_date, draft_group_id, data_root) / f"field_library_{version}.json"


def save_field_library(library: FieldLibrary, path: Path) -> None:
    library.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = library.to_dict()
    payload.setdefault("meta", {})
    payload["meta"].setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_field_library(path: Path) -> FieldLibrary:
    with open(path) as f:
        payload = json.load(f)
    return FieldLibrary.from_dict(payload)


def list_field_library_paths(
    game_date: str,
    draft_group_id: int,
    data_root: Optional[Path] = None,
) -> List[Path]:
    root = field_library_dir(game_date, draft_group_id, data_root)
    if not root.exists():
        return []
    return sorted(root.glob("field_library_*.json"), reverse=True)

