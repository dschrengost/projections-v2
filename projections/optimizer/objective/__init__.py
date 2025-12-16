from __future__ import annotations

# Re-export public API
from .own_penalty import (
    OwnershipPenaltyConfig,
    AggroSchedule,
    CurveKind,
    build_ownership_penalty,
)

# Lightweight global to pass config from the driver to the solver without
# changing intermediate adapters. This keeps wiring minimal and contained.
from typing import Optional, Tuple

_ACTIVE_OWN_CFG: Optional[OwnershipPenaltyConfig] = None
_ACTIVE_AGGRO: Optional[AggroSchedule] = None


def set_active_ownership_penalty(
    cfg: Optional[OwnershipPenaltyConfig],
    aggro: Optional[AggroSchedule] = None,
) -> None:
    global _ACTIVE_OWN_CFG, _ACTIVE_AGGRO
    _ACTIVE_OWN_CFG = cfg
    _ACTIVE_AGGRO = aggro


def get_active_ownership_penalty(
    ) -> Tuple[Optional[OwnershipPenaltyConfig], Optional[AggroSchedule]]:
    return _ACTIVE_OWN_CFG, _ACTIVE_AGGRO

