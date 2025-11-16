"""Utilities for resolving the canonical data root for the project.

The helper checks the ``PROJECTIONS_DATA_ROOT`` environment variable first. When
it is set to a non-empty value, that directory becomes the base data root. When
the variable is missing, paths fall back to the repo-local ``./data`` directory.
All returned paths are absolute to avoid depending on the current working
directory.
"""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import Union

PROJECTIONS_DATA_ENV = "PROJECTIONS_DATA_ROOT"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

Pathish = Union[str, PathLike[str]]


def get_project_root() -> Path:
    """Return the absolute path to the repository root."""
    return _PROJECT_ROOT


def get_data_root() -> Path:
    """Return the absolute data root, honoring PROJECTIONS_DATA_ROOT when set."""
    env_value = os.environ.get(PROJECTIONS_DATA_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (_PROJECT_ROOT / "data").resolve()


def data_path(*parts: Pathish) -> Path:
    """Join the canonical data root with the provided path segments."""
    return get_data_root().joinpath(*parts)
