from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class TeamContext:
    season_id: str
    game_id: str
    team_id: int
    opponent_team_id: int
    is_home: bool
    vegas_spread: Optional[float] = None
    vegas_total: Optional[float] = None
    candidate_player_ids: Optional[List[int]] = None
    starter_candidates: Optional[List[int]] = None
    minutes_prior: Optional[Dict[int, float]] = None
    minutes_p10_prior: Optional[Dict[int, float]] = None
    minutes_p90_prior: Optional[Dict[int, float]] = None
    play_prob_prior: Optional[Dict[int, float]] = None
    regime_label: Optional[str] = None
    n_worlds: int = 5000
    rng_seed: int = 0


@dataclass(frozen=True)
class RotationWorlds:
    minutes_by_player: Dict[int, np.ndarray]
    starter_by_player: Optional[Dict[int, np.ndarray]] = None
    diagnostics: Optional[Dict] = None


class RotationGenerator(Protocol):
    def generate(self, ctx: TeamContext) -> RotationWorlds: ...
