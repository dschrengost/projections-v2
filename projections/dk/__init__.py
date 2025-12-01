"""DraftKings NBA scraping utilities.

This package provides thin API wrappers and helpers for discovering DraftKings
NBA slates and fetching draftables. See ``projections.dk.api`` and
``projections.dk.slates`` for the primary entry points.
"""

from __future__ import annotations

__all__ = [
    "api",
    "slates",
    "normalize",
    "salaries_schema",
]
