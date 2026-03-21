"""Path resolution helpers for storage retention tooling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

from projections import paths

PROJECTIONS_HOT_ROOT_ENV = "PROJECTIONS_HOT_ROOT"
PROJECTIONS_ARCHIVE_ROOT_ENV = "PROJECTIONS_ARCHIVE_ROOT"

FAMILY_ROOTS: dict[str, tuple[str, ...]] = {
    "gtv2_worlds": ("artifacts", "gtv2_worlds"),
    "sim_v2_worlds_fpts_v2": ("artifacts", "sim_v2", "worlds_fpts_v2"),
}


@dataclass(frozen=True)
class StorageRoots:
    data_root: Path
    hot_root: Path
    archive_root: Path | None


def resolve_storage_roots(
    *,
    data_root: Path | None = None,
    hot_root: Path | None = None,
    archive_root: Path | None = None,
) -> StorageRoots:
    data_root_eff = Path(data_root) if data_root is not None else paths.get_data_root()
    hot_env = os.environ.get(PROJECTIONS_HOT_ROOT_ENV)
    hot_root_eff = (
        Path(hot_root)
        if hot_root is not None
        else (Path(hot_env).expanduser().resolve() if hot_env else data_root_eff)
    )
    archive_env = os.environ.get(PROJECTIONS_ARCHIVE_ROOT_ENV)
    archive_root_eff = (
        Path(archive_root)
        if archive_root is not None
        else (Path(archive_env).expanduser().resolve() if archive_env else None)
    )
    return StorageRoots(
        data_root=Path(data_root_eff).expanduser().resolve(),
        hot_root=Path(hot_root_eff).expanduser().resolve(),
        archive_root=(
            Path(archive_root_eff).expanduser().resolve()
            if archive_root_eff is not None
            else None
        ),
    )


def family_root(*, hot_root: Path, family: str) -> Path:
    rel = FAMILY_ROOTS.get(family)
    if rel is None:
        raise ValueError(f"Unsupported retention family: {family}")
    return hot_root.joinpath(*rel)


def retention_reports_dir(*, hot_root: Path) -> Path:
    return hot_root / "artifacts" / "retention" / "reports"


def retention_decision_dir(*, hot_root: Path, family: str, game_date: str, run_id: str) -> Path:
    return (
        hot_root
        / "artifacts"
        / "retention"
        / "v1"
        / family
        / f"game_date={game_date}"
        / f"run={run_id}"
    )


def parse_game_date_from_partition(name: str) -> date | None:
    if not name.startswith("game_date="):
        return None
    raw = name.split("=", 1)[1]
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def parse_run_id_from_partition(name: str) -> str | None:
    if not name.startswith("run="):
        return None
    run_id = name.split("=", 1)[1].strip()
    return run_id or None


def iter_game_date_partitions(
    *,
    family: str,
    hot_root: Path,
    start_date: date | None,
    end_date: date | None,
) -> Iterable[tuple[str, Path]]:
    root = family_root(hot_root=hot_root, family=family)
    if not root.exists():
        return []
    out: list[tuple[str, Path]] = []
    for path in sorted(root.glob("game_date=*")):
        if not path.is_dir():
            continue
        parsed = parse_game_date_from_partition(path.name)
        if parsed is None:
            continue
        if start_date is not None and parsed < start_date:
            continue
        if end_date is not None and parsed > end_date:
            continue
        out.append((parsed.isoformat(), path))
    return out
