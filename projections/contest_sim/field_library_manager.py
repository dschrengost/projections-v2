"""Field library caching and build orchestration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .field_library import (
    FieldLibrary,
    field_library_path,
    list_field_library_paths,
    load_field_library,
    save_field_library,
)
from .field_library_quickbuild import build_quickbuild_field_library

logger = logging.getLogger(__name__)

__all__ = ["load_or_build_field_library"]


def load_or_build_field_library(
    *,
    game_date: str,
    draft_group_id: int,
    version: str = "v0",
    k: int = 2500,
    candidate_pool_size: int = 40000,
    rebuild: bool = False,
    rebuild_candidates: bool = False,
    data_root: Optional[Path] = None,
) -> Tuple[FieldLibrary, Path, bool]:
    """Load a cached field library or build and persist a new one.

    Returns (library, path, built_now).
    """
    if version == "latest":
        paths = list_field_library_paths(game_date, draft_group_id, data_root=data_root)
        if paths and not rebuild:
            return load_field_library(paths[0]), paths[0], False
        version = "v0"

    path = field_library_path(game_date, draft_group_id, version=version, data_root=data_root)

    if path.exists() and not rebuild:
        return load_field_library(path), path, False

    logger.info(
        "Building field library: date=%s dg=%d version=%s k=%d candidates=%d",
        game_date,
        draft_group_id,
        version,
        k,
        candidate_pool_size,
    )
    library = build_quickbuild_field_library(
        game_date=game_date,
        draft_group_id=int(draft_group_id),
        version=version,
        k=k,
        candidate_pool_size=candidate_pool_size,
        rebuild_candidates=rebuild_candidates,
    )

    # Ensure required metadata keys are present.
    library.meta.setdefault("game_date", game_date)
    library.meta.setdefault("draft_group_id", int(draft_group_id))
    library.meta.setdefault("version", version)
    library.meta.setdefault("generated_at", datetime.now(timezone.utc).isoformat())

    save_field_library(library, path)
    return library, path, True
