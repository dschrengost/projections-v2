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
    site: str = "dk",
    version: str = "v0",
    k: int = 2500,
    candidate_pool_size: int = 40000,
    rebuild: bool = False,
    rebuild_candidates: bool = False,
    use_ownership_features: bool = True,
    data_root: Optional[Path] = None,
) -> Tuple[FieldLibrary, Path, bool]:
    """Load a cached field library or build and persist a new one.

    Returns (library, path, built_now).
    """
    site_norm = str(site or "dk").strip().lower()

    if version == "latest":
        paths = list_field_library_paths(game_date, draft_group_id, data_root=data_root)
        if paths and not rebuild:
            for candidate in paths:
                try:
                    library = load_field_library(candidate)
                except Exception:
                    continue
                lib_site = str(library.meta.get("site") or "dk").strip().lower()
                if lib_site == site_norm:
                    return library, candidate, False
        version = "v0"
    if site_norm == "fd" and not str(version).startswith("fd__"):
        # Keep FD caches/artifacts separate from DK under shared draft_group_id roots.
        version = f"fd__{version}"

    path = field_library_path(game_date, draft_group_id, version=version, data_root=data_root)

    if path.exists() and not rebuild:
        library = load_field_library(path)
        lib_site = str(library.meta.get("site") or "dk").strip().lower()
        if lib_site == site_norm:
            return library, path, False

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
        site=site,
        k=k,
        candidate_pool_size=candidate_pool_size,
        rebuild_candidates=rebuild_candidates,
        use_ownership_features=use_ownership_features,
    )

    # Ensure required metadata keys are present.
    library.meta.setdefault("game_date", game_date)
    library.meta.setdefault("draft_group_id", int(draft_group_id))
    library.meta.setdefault("version", version)
    library.meta.setdefault("site", site_norm)
    library.meta.setdefault("generated_at", datetime.now(timezone.utc).isoformat())

    save_field_library(library, path)
    return library, path, True
