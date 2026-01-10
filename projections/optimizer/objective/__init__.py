from __future__ import annotations

# Re-export public API
from .own_penalty import (
    OwnershipPenaltyConfig as OwnershipPenaltyConfig,
    AggroSchedule as AggroSchedule,
    CurveKind as CurveKind,
    build_ownership_penalty as build_ownership_penalty,
)
from .late_swap_bonus import (
    LateSwapBonusConfig as LateSwapBonusConfig,
    build_late_swap_bonus as build_late_swap_bonus,
)

# Lightweight global to pass config from the driver to the solver without
# changing intermediate adapters. This keeps wiring minimal and contained.
from typing import Optional, Tuple

_ACTIVE_OWN_CFG: Optional[OwnershipPenaltyConfig] = None
_ACTIVE_AGGRO: Optional[AggroSchedule] = None
_ACTIVE_LATE_SWAP_CFG: Optional[LateSwapBonusConfig] = None


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


def set_active_late_swap_bonus(cfg: Optional[LateSwapBonusConfig]) -> None:
    global _ACTIVE_LATE_SWAP_CFG
    _ACTIVE_LATE_SWAP_CFG = cfg


def get_active_late_swap_bonus() -> Optional[LateSwapBonusConfig]:
    return _ACTIVE_LATE_SWAP_CFG

