"""Data models for contest simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class PayoutTier:
    """Single payout tier defining payout for a range of places."""
    start_place: int
    end_place: int
    payout: float

    def __post_init__(self):
        if self.start_place <= 0 or self.end_place <= 0:
            raise ValueError("Payout tiers must use positive ranks")
        if self.end_place < self.start_place:
            raise ValueError("end_place must be >= start_place")


@dataclass
class ContestConfig:
    """Contest configuration for simulation."""
    field_size: int
    entry_fee: float
    archetype: str
    rake: float = 0.12
    ceiling_percentile: int = 90

    @property
    def prize_pool(self) -> float:
        """Total prize pool after rake."""
        return self.field_size * self.entry_fee * (1 - self.rake)


@dataclass
class LineupEVResult:
    """Per-lineup simulation results with extended metrics."""
    lineup_id: int
    player_ids: List[str]

    # Score distribution
    mean: float
    std: float
    p90: float
    p95: float

    # EV metrics
    expected_payout: float
    expected_value: float  # expected_payout - entry_fee
    roi: float  # (expected_payout - entry_fee) / entry_fee

    # Rate metrics (all as decimals, e.g., 0.01 = 1%)
    win_rate: float  # P(1st place)
    top_1pct_rate: float  # P(top 1% finish)
    top_5pct_rate: float  # P(top 5% finish)
    top_10pct_rate: float  # P(top 10% finish)
    cash_rate: float  # P(ITM / any payout)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "lineup_id": self.lineup_id,
            "player_ids": self.player_ids,
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "p90": round(self.p90, 2),
            "p95": round(self.p95, 2),
            "expected_payout": round(self.expected_payout, 4),
            "expected_value": round(self.expected_value, 4),
            "roi": round(self.roi, 4),
            "win_rate": round(self.win_rate, 6),
            "top_1pct_rate": round(self.top_1pct_rate, 6),
            "top_5pct_rate": round(self.top_5pct_rate, 6),
            "top_10pct_rate": round(self.top_10pct_rate, 6),
            "cash_rate": round(self.cash_rate, 6),
        }


@dataclass
class SummaryStats:
    """Summary statistics for a contest simulation run."""
    lineup_count: int
    worlds_count: int
    avg_ev: float
    avg_roi: float
    positive_ev_count: int
    best_ev_lineup_id: int
    best_win_rate_lineup_id: int
    best_top1pct_lineup_id: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "lineup_count": self.lineup_count,
            "worlds_count": self.worlds_count,
            "avg_ev": round(self.avg_ev, 4),
            "avg_roi": round(self.avg_roi, 4),
            "positive_ev_count": self.positive_ev_count,
            "best_ev_lineup_id": self.best_ev_lineup_id,
            "best_win_rate_lineup_id": self.best_win_rate_lineup_id,
            "best_top1pct_lineup_id": self.best_top1pct_lineup_id,
        }


@dataclass
class ContestSimResult:
    """Complete result from a contest simulation."""
    results: List[LineupEVResult]
    config: ContestConfig
    stats: SummaryStats

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "config": {
                "field_size": self.config.field_size,
                "entry_fee": self.config.entry_fee,
                "archetype": self.config.archetype,
                "rake": self.config.rake,
                "prize_pool": round(self.config.prize_pool, 2),
            },
            "stats": self.stats.to_dict(),
        }
