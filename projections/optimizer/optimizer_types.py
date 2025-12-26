"""
Comprehensive data schemas for NBA DFS optimization with Streamlit integration
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Literal
from datetime import datetime

Mode = Literal["by_percent", "by_points"]
Curve = Literal["sigmoid", "linear", "power", "neglog"]


@dataclass
class GroupRule:
    """Player group constraint rule (e.g., at_least 2 from ['Player1', 'Player2'])"""

    count: int
    players: List[str]


@dataclass
class OwnershipPenaltySettings:
    """Ownership penalty settings for contrarian lineup optimization (PRP-15)"""

    enabled: bool = True
    mode: Mode = "by_percent"  # "by_percent" | "by_points"

    # By % off optimal controls
    target_offoptimal_pct: float = 0.08  # 8%
    tol_offoptimal_pct: float = 0.003  # ±0.3%

    # By points legacy (kept for back-compat & A/B)
    weight_lambda: float = 1.0

    # Curve selector
    curve_type: Curve = "sigmoid"  # "sigmoid" | "linear" | "power" | "neglog"
    power_k: float = 1.5  # used when curve_type == "power"

    # Sigmoid controls (default)
    pivot_p0: float = 0.20
    curve_alpha: float = 2.0

    # Shared controls
    clamp_min: float = 0.01
    clamp_max: float = 0.80
    shrink_gamma: float = 1.0


@dataclass
class Constraints:
    """Optimization constraints and parameters"""

    N_lineups: int = 20
    unique_players: int = 1
    proj_min: float = 0.0
    randomness_pct: float = 0.0
    min_salary: Optional[int] = None  # Site defaults: dk=49000, fd=59000
    max_salary: Optional[int] = None  # Site defaults: dk=50000, fd=60000
    global_team_limit: Optional[int] = None
    team_limits: Dict[str, int] = field(default_factory=dict)
    at_least: List[GroupRule] = field(default_factory=list)
    at_most: List[GroupRule] = field(default_factory=list)
    slate_id: Optional[str] = None
    require_dk_ids: bool = (
        False  # Fail-fast mode: require real DK IDs for export compatibility
    )
    min_dk_id_match_rate: float = (
        95.0  # Minimum percentage of players that must have real DK IDs
    )
    cp_sat_params: Dict[str, float] = field(
        default_factory=dict
    )  # CP-SAT solver parameters
    ownership_penalty: OwnershipPenaltySettings = field(
        default_factory=OwnershipPenaltySettings
    )  # PRP-15: Ownership penalty settings
    # Slot-level locks (DraftKings late swap): slot -> player_id
    lock_slots: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Constraints":
        """Create Constraints from dictionary (UI input format)"""
        # Convert GroupRule dictionaries to objects if needed
        at_least = []
        if "at_least" in data:
            for rule_data in data["at_least"]:
                if isinstance(rule_data, dict):
                    at_least.append(GroupRule(**rule_data))
                else:
                    at_least.append(rule_data)

        at_most = []
        if "at_most" in data:
            for rule_data in data["at_most"]:
                if isinstance(rule_data, dict):
                    at_most.append(GroupRule(**rule_data))
                else:
                    at_most.append(rule_data)

        # Convert OwnershipPenaltySettings dictionary to object if needed
        ownership_penalty = data.get("ownership_penalty")
        if ownership_penalty and isinstance(ownership_penalty, dict):
            ownership_penalty = OwnershipPenaltySettings(**ownership_penalty)

        # Create constraints with converted rules
        constraints_data = {**data}
        constraints_data["at_least"] = at_least
        constraints_data["at_most"] = at_most
        if ownership_penalty:
            constraints_data["ownership_penalty"] = ownership_penalty

        # Support alias 'num_lineups' from UI
        if "num_lineups" in constraints_data and "N_lineups" not in constraints_data:
            constraints_data["N_lineups"] = constraints_data.pop("num_lineups")

        return cls(**constraints_data)

    def validate(
        self, site: Literal["dk", "fd"], stddev_available: bool = True
    ) -> "Constraints":
        """
        Validate and coerce constraints with site-specific defaults

        Note: DraftKings automatically applies a 4-player team limit if global_team_limit is None.
        Set global_team_limit to 8 (or higher) to disable team limits entirely.
        FanDuel has a natural 4-player maximum due to roster construction.
        """
        # Set site-specific salary defaults
        if self.min_salary is None:
            self.min_salary = 49000 if site == "dk" else 59000
        if self.max_salary is None:
            self.max_salary = 50000 if site == "dk" else 60000

        # Validate ranges
        if self.N_lineups < 1 or self.N_lineups > 200000:
            raise ValueError("N_lineups must be between 1 and 200000")

        roster_size = 8 if site == "dk" else 9
        if not (1 <= self.unique_players <= roster_size):
            raise ValueError(
                f"unique_players must be between 1 and {roster_size} for {site}"
            )

        if self.randomness_pct < 0 or self.randomness_pct > 100:
            raise ValueError("randomness_pct must be between 0 and 100")

        if not stddev_available and self.randomness_pct > 0:
            raise ValueError("Randomness requires stddev column in projections")

        if self.lock_slots:
            if site != "dk":
                raise ValueError("lock_slots is only supported for DraftKings (dk)")
            allowed = {"PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"}
            invalid = sorted(set(self.lock_slots) - allowed)
            if invalid:
                raise ValueError(f"Invalid lock_slots keys: {invalid}")
            empty = sorted(k for k, v in self.lock_slots.items() if not str(v).strip())
            if empty:
                raise ValueError(f"lock_slots has empty player_id for: {empty}")

        # Ownership penalty validations
        s = self.ownership_penalty
        if s.mode not in ("by_percent", "by_points"):
            raise ValueError(
                "ownership_penalty.mode must be 'by_percent' or 'by_points'"
            )
        if not (0.0 <= s.target_offoptimal_pct <= 0.20):
            raise ValueError(
                "ownership_penalty.target_offoptimal_pct must be in [0.0, 0.20]"
            )
        if not (0.0 <= s.tol_offoptimal_pct <= 0.02):
            raise ValueError(
                "ownership_penalty.tol_offoptimal_pct must be in [0.0, 0.02]"
            )
        if s.curve_type not in ("sigmoid", "linear", "power", "neglog"):
            raise ValueError(
                "ownership_penalty.curve_type must be one of: sigmoid, linear, power, neglog"
            )
        if not (0.0 <= s.clamp_min < s.clamp_max <= 0.99):
            raise ValueError(
                "ownership_penalty clamp_min/max must satisfy 0 ≤ min < max ≤ 0.99"
            )
        if s.curve_type == "power" and not (1.0 <= s.power_k <= 3.0):
            raise ValueError("ownership_penalty.power_k must be in [1.0, 3.0]")

        return self

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @property
    def num_lineups(self) -> int:
        """Alias for compatibility with code expecting `num_lineups`."""
        return self.N_lineups


@dataclass
class Player:
    """Player model for lineup optimization"""

    player_id: str
    name: str
    pos: str  # Position assigned in lineup ("PG", "SG", etc.)
    team: str
    salary: int
    proj: float
    dk_id: Optional[str] = None  # DraftKings player ID for export
    own_proj: Optional[float] = None
    stddev: Optional[float] = None
    minutes: Optional[float] = None
    matchup: Optional[str] = None
    game_time: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime to ISO string if present
        if self.game_time:
            data["game_time"] = self.game_time.isoformat()
        return data


@dataclass
class Lineup:
    """Optimized lineup model"""

    lineup_id: int
    total_proj: float
    total_salary: int
    players: List[Player]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "lineup_id": self.lineup_id,
            "total_proj": self.total_proj,
            "total_salary": self.total_salary,
            "players": [p.to_dict() for p in self.players],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Lineup":
        """Create Lineup from dictionary"""
        players = []
        for p_data in data["players"]:
            # Convert datetime string back if present
            if "game_time" in p_data and p_data["game_time"]:
                p_data["game_time"] = datetime.fromisoformat(p_data["game_time"])
            players.append(Player(**p_data))

        return cls(
            lineup_id=data["lineup_id"],
            total_proj=data["total_proj"],
            total_salary=data["total_salary"],
            players=players,
        )


@dataclass
class LineupSet:
    """Collection of optimized lineups with metadata"""

    id: str
    created_at: datetime
    seed: int
    slate_id: Optional[str]
    constraints: Constraints
    engine: str
    version: str
    runtime_sec: float
    lineups: List[Lineup]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "seed": self.seed,
            "slate_id": self.slate_id,
            "constraints": self.constraints.to_dict(),
            "engine": self.engine,
            "version": self.version,
            "runtime_sec": self.runtime_sec,
            "lineups": [lineup.to_dict() for lineup in self.lineups],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LineupSet":
        """Create LineupSet from dictionary"""
        return cls(
            id=data["id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            seed=data["seed"],
            slate_id=data.get("slate_id"),
            constraints=Constraints.from_dict(data["constraints"]),
            engine=data["engine"],
            version=data["version"],
            runtime_sec=data["runtime_sec"],
            lineups=[Lineup.from_dict(lineup) for lineup in data["lineups"]],
        )


class OptimizerError(Exception):
    """Structured error for optimizer failures"""

    def __init__(
        self,
        code: str,
        message: str,
        user_message: str,
        details: Optional[dict] = None,
        severity: str = "error",
        retriable: bool = False,
    ):
        super().__init__(message)
        self.code = code
        self.user_message = user_message  # UI-safe message
        self.details = details or {}
        self.severity = severity
        self.retriable = retriable


# Error code constants
class ErrorCodes:
    MISSING_COLUMNS = "MISSING_COLUMNS"
    INVALID_CONSTRAINTS = "INVALID_CONSTRAINTS"
    INFEASIBLE = "INFEASIBLE"
    SOLVER_TIMEOUT = "SOLVER_TIMEOUT"
    DATA_MISMATCH_IDS = "DATA_MISMATCH_IDS"
    CONFIG_ERROR = "CONFIG_ERROR"
    INVALID_PROJECTIONS = "INVALID_PROJECTIONS"


# Type aliases
SiteType = Literal["dk", "fd"]
PositionType = Literal["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
