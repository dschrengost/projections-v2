"""Helpers for archetype roles and deltas used by minutes features."""

from __future__ import annotations

from .roles import (
    POSITION_GROUP_MAP,
    RoleConfig,
    build_roles_table,
    infer_position_group,
)
from .deltas import (
    ArchetypeDeltaConfig,
    build_archetype_deltas,
    compute_team_role_counts,
    compute_team_missing_totals,
    load_config,
    prepare_injury_availability,
)

__all__ = [
    "POSITION_GROUP_MAP",
    "RoleConfig",
    "ArchetypeDeltaConfig",
    "build_roles_table",
    "build_archetype_deltas",
    "compute_team_role_counts",
    "compute_team_missing_totals",
    "infer_position_group",
    "load_config",
    "prepare_injury_availability",
]
