from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from projections.contest_sim.field_library import FieldLibrary
from projections.contest_sim.scoring_models import ContestSimResult


@dataclass(frozen=True)
class ContestReplayMeta:
    game_date: str
    contest_id: str
    contest_name: str
    draft_group_id: Optional[int]
    entry_fee: float
    field_size: int
    results_path: Optional[str] = None
    source: str = "raw_results_csv"
    source_mode: str = "exact_replay"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_date": self.game_date,
            "contest_id": self.contest_id,
            "contest_name": self.contest_name,
            "draft_group_id": self.draft_group_id,
            "entry_fee": self.entry_fee,
            "field_size": self.field_size,
            "results_path": self.results_path,
            "source": self.source,
            "source_mode": self.source_mode,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class ContestReplayEntry:
    entry_id: str
    entry_name: str
    rank: Optional[int]
    points: Optional[float]
    lineup_names: List[str]
    raw_lineup: str
    lineup_key: str
    prize: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entry_name": self.entry_name,
            "rank": self.rank,
            "points": self.points,
            "lineup_names": list(self.lineup_names),
            "raw_lineup": self.raw_lineup,
            "lineup_key": self.lineup_key,
            "prize": self.prize,
        }


@dataclass(frozen=True)
class ResolvedContestReplayEntry(ContestReplayEntry):
    player_ids: List[str] = field(default_factory=list)
    unresolved_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["player_ids"] = list(self.player_ids)
        payload["unresolved_names"] = list(self.unresolved_names)
        return payload


@dataclass(frozen=True)
class PreparedReplayContext:
    meta: ContestReplayMeta
    entries: List[ContestReplayEntry]
    resolved_entries: List[ResolvedContestReplayEntry]
    user_entries: List[ResolvedContestReplayEntry]
    user_lineups: List[List[str]]
    user_weights: List[int]
    opponent_field_library: FieldLibrary
    resolution_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "entries": [entry.to_dict() for entry in self.entries],
            "resolved_entries": [entry.to_dict() for entry in self.resolved_entries],
            "user_entries": [entry.to_dict() for entry in self.user_entries],
            "user_lineups": [list(lineup) for lineup in self.user_lineups],
            "user_weights": [int(weight) for weight in self.user_weights],
            "opponent_field_library": self.opponent_field_library.to_dict(),
            "resolution_stats": dict(self.resolution_stats),
        }


@dataclass(frozen=True)
class ContestReplayRun:
    prepared: PreparedReplayContext
    simulation: ContestSimResult
    run_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prepared": self.prepared.to_dict(),
            "simulation": self.simulation.to_dict(),
            "run_meta": dict(self.run_meta),
        }
