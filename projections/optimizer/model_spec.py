from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

SiteType = Literal["dk", "fd"]


@dataclass
class SpecPlayer:
    player_id: str
    name: str
    team: str
    positions: List[str]
    salary: int
    proj: float
    dk_id: Optional[str] = None
    own_proj: Optional[float] = (
        None  # PRP-16: Ownership percentage for penalty calculation
    )
    stddev: Optional[float] = None


@dataclass
class Spec:
    site: SiteType
    roster_slots: List[str]
    salary_cap: int
    min_salary: Optional[int] = None
    players: List[SpecPlayer] = field(default_factory=list)

    # Team limits
    team_max: Optional[int] = 4
    team_limits: Dict[str, int] = field(default_factory=dict)

    # Locks/Bans
    lock_ids: List[str] = field(default_factory=list)
    ban_ids: List[str] = field(default_factory=list)

    # Cardinality
    lineup_size: int = 8
    N_lineups: int = 20
    unique_players: int = 1

    # Exposure & groups (reserved for future)
    at_least: Dict[str, int] = field(default_factory=dict)
    at_most: Dict[str, int] = field(default_factory=dict)
    groups_at_least: Dict[str, int] = field(default_factory=dict)
    groups_at_most: Dict[str, int] = field(default_factory=dict)

    # Engine options
    engine: str = "cbc"
    cp_sat_params: Dict[str, float] = field(default_factory=dict)

    # PRP-16: Ownership penalty settings
    ownership_penalty: Optional[Dict] = None  # OwnershipPenaltySettings as dict

    # Optional lineup projection sum bounds (raw points, before jitter)
    min_proj_sum: Optional[float] = None
    max_proj_sum: Optional[float] = None
    # Optional variance-aware randomization (percent of per-player stddev)
    randomness_pct: Optional[float] = None
