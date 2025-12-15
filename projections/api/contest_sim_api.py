"""FastAPI router for contest simulation endpoints."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from projections.contest_sim.contest_sim_service import run_contest_simulation
from projections.contest_sim.payout_generator import load_config

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class ContestSimRequest(BaseModel):
    """Request to run a contest simulation."""

    game_date: str = Field(..., description="Game date in YYYY-MM-DD format")
    lineups: List[List[str]] = Field(..., description="List of lineups (each a list of player_ids)")
    archetype: str = Field(default="medium", description="Payout archetype: top_heavy, medium, flat")
    field_size_bucket: str = Field(default="medium", description="Field size bucket: small, medium, massive")
    field_size_override: Optional[int] = Field(default=None, description="Exact field size (overrides bucket)")
    entry_fee: float = Field(default=3.0, description="Entry fee per lineup")
    weights: Optional[List[int]] = Field(default=None, description="Entry counts per lineup")


class LineupEVResultResponse(BaseModel):
    """Per-lineup simulation results."""

    lineup_id: int
    player_ids: List[str]
    mean: float
    std: float
    p90: float
    p95: float
    expected_payout: float
    expected_value: float
    roi: float
    win_rate: float
    top_1pct_rate: float
    top_5pct_rate: float
    top_10pct_rate: float
    cash_rate: float


class ContestConfigResponse(BaseModel):
    """Contest configuration."""

    field_size: int
    entry_fee: float
    archetype: str
    rake: float
    prize_pool: float


class SummaryStatsResponse(BaseModel):
    """Summary statistics."""

    lineup_count: int
    worlds_count: int
    avg_ev: float
    avg_roi: float
    positive_ev_count: int
    best_ev_lineup_id: int
    best_win_rate_lineup_id: int
    best_top1pct_lineup_id: int


class ContestSimResponse(BaseModel):
    """Complete contest simulation response."""

    results: List[LineupEVResultResponse]
    config: ContestConfigResponse
    stats: SummaryStatsResponse


class FieldSizeOption(BaseModel):
    """Field size bucket option."""

    key: str
    label: str
    default: int
    range: List[int]


class PayoutArchetypeOption(BaseModel):
    """Payout archetype option."""

    key: str
    label: str
    first_place_pct: float
    itm_pct: float


class ConfigResponse(BaseModel):
    """Available configuration options."""

    field_sizes: List[FieldSizeOption]
    payout_archetypes: List[PayoutArchetypeOption]
    default_entry_fee: float
    default_archetype: str
    default_field_size_bucket: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run", response_model=ContestSimResponse)
async def run_simulation(request: ContestSimRequest):
    """Run contest simulation for the given lineups.

    Lineups compete against each other (self-play mode) with the
    specified payout archetype and field size.
    """
    try:
        result = run_contest_simulation(
            lineups=request.lineups,
            game_date=request.game_date,
            archetype=request.archetype,
            field_size_bucket=request.field_size_bucket,
            field_size_override=request.field_size_override,
            entry_fee=request.entry_fee,
            weights=request.weights,
        )

        return ContestSimResponse(
            results=[
                LineupEVResultResponse(**r.to_dict())
                for r in result.results
            ],
            config=ContestConfigResponse(
                field_size=result.config.field_size,
                entry_fee=result.config.entry_fee,
                archetype=result.config.archetype,
                rake=result.config.rake,
                prize_pool=result.config.prize_pool,
            ),
            stats=SummaryStatsResponse(**result.stats.to_dict()),
        )

    except FileNotFoundError as e:
        logger.error(f"Worlds data not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Contest simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get available configuration options for contest simulation."""
    try:
        config = load_config()

        # field_sizes is a list of dicts with 'label' and 'value' keys
        field_sizes = []
        for i, fs in enumerate(config.get("field_sizes", [])):
            field_sizes.append(FieldSizeOption(
                key=str(fs.get("value", i)),
                label=fs.get("label", f"Size {i}"),
                default=fs.get("value", 25000),
                range=[0, fs.get("value", 25000) * 2],
            ))

        # payout_archetypes is a list of dicts with 'name', 'field_paid_pct', 'payout_table'
        archetypes = []
        for i, arch in enumerate(config.get("payout_archetypes", [])):
            archetypes.append(PayoutArchetypeOption(
                key=arch.get("name", f"archetype_{i}"),
                label=arch.get("name", f"Archetype {i}"),
                first_place_pct=arch.get("first_place_pct", 0.2),
                itm_pct=arch.get("field_paid_pct", 20.0) / 100.0,
            ))

        defaults = config.get("defaults", {})

        return ConfigResponse(
            field_sizes=field_sizes,
            payout_archetypes=archetypes,
            default_entry_fee=defaults.get("entry_fee", 3.0),
            default_archetype=defaults.get("archetype", "GPP Standard (20% paid)"),
            default_field_size_bucket=defaults.get("field_size_bucket", "5000"),
        )

    except Exception as e:
        logger.exception(f"Failed to load config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

