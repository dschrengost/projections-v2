"""Resolve minutes/rates selector config paths for runtime and repo contexts.

Runtime selectors live under ``$PROJECTIONS_DATA_ROOT/control_plane/model_selectors``.
Repo-local config files remain the fallback/default source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from projections import paths

SelectorName = Literal["minutes", "rates"]

_RUNTIME_SELECTORS_REL = Path("control_plane") / "model_selectors"
_SELECTOR_FILENAMES: dict[SelectorName, str] = {
    "minutes": "minutes_current_run.json",
    "rates": "rates_current_run.json",
}


def repo_selector_path(name: SelectorName, *, project_root: Path | None = None) -> Path:
    root = (project_root or paths.get_project_root()).expanduser().resolve()
    return root / "config" / _SELECTOR_FILENAMES[name]


def runtime_selector_path(name: SelectorName, *, data_root: Path | None = None) -> Path:
    root = (data_root or paths.get_data_root()).expanduser().resolve()
    return root / _RUNTIME_SELECTORS_REL / _SELECTOR_FILENAMES[name]


def active_selector_path(
    name: SelectorName,
    *,
    data_root: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    runtime_path = runtime_selector_path(name, data_root=data_root)
    if runtime_path.exists():
        return runtime_path
    return repo_selector_path(name, project_root=project_root)


def repo_minutes_selector_path(*, project_root: Path | None = None) -> Path:
    return repo_selector_path("minutes", project_root=project_root)


def runtime_minutes_selector_path(*, data_root: Path | None = None) -> Path:
    return runtime_selector_path("minutes", data_root=data_root)


def active_minutes_selector_path(
    *,
    data_root: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    return active_selector_path("minutes", data_root=data_root, project_root=project_root)


def repo_rates_selector_path(*, project_root: Path | None = None) -> Path:
    return repo_selector_path("rates", project_root=project_root)


def runtime_rates_selector_path(*, data_root: Path | None = None) -> Path:
    return runtime_selector_path("rates", data_root=data_root)


def active_rates_selector_path(
    *,
    data_root: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    return active_selector_path("rates", data_root=data_root, project_root=project_root)
