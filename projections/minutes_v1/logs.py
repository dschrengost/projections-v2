"""Helpers for resolving minutes prediction log paths."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from projections import paths
from projections.minutes_v1.production import resolve_production_run_dir


def _normalize(path: Path) -> Path:
    return path.expanduser().resolve()


def prediction_logs_base(data_root: Path | None = None) -> Path:
    """Return the base directory for prediction logs (without run= partition)."""

    root = data_root.expanduser().resolve() if data_root else paths.get_data_root()
    return (root / "gold" / "prediction_logs_minutes").resolve()


def prediction_logs_candidates(
    *,
    run_id: str | None,
    data_root: Path | None = None,
    legacy: bool = True,
) -> list[Path]:
    """Return candidate directories (new + legacy) for minutes prediction logs."""

    candidates: list[Path] = []
    base = prediction_logs_base(data_root)

    def _add(path: Path) -> None:
        resolved = _normalize(path)
        if resolved not in candidates:
            candidates.append(resolved)

    if run_id:
        _add(base / f"run={run_id}")
    _add(base)
    if data_root is not None:
        local_legacy = (
            data_root.expanduser().resolve() / "gold" / "prediction_logs_minutes_v1"
        )
        _add(local_legacy)
    if legacy:
        legacy_root = (paths.get_data_root() / "gold" / "prediction_logs_minutes_v1").resolve()
        _add(legacy_root)
    return candidates


def default_minutes_run_id(config_path: Path | None = None) -> str | None:
    """Return the production minutes run_id if available."""

    try:
        _, run_id = resolve_production_run_dir(config_path)
        return run_id
    except Exception:
        return None
