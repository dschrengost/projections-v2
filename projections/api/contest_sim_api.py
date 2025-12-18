"""FastAPI router for contest simulation endpoints."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from projections import paths
from projections.contest_sim.contest_sim_service import run_contest_simulation
from projections.contest_sim.payout_generator import load_config

logger = logging.getLogger(__name__)
router = APIRouter()


def _load_player_ownership(game_date: str) -> Dict[str, float]:
    """Load player_id -> ownership % mapping from projections or ownership predictions.
    
    Tries unified projections first, falls back to silver/ownership_predictions.
    Returns empty dict if not available (dupe penalties disabled).
    """
    data_root = paths.data_path()
    
    # Try unified projections artifact first
    unified_root = data_root / "artifacts" / "projections" / game_date
    for run_dir in sorted(unified_root.glob("run=*"), reverse=True) if unified_root.exists() else []:
        proj_path = run_dir / "projections.parquet"
        if proj_path.exists():
            try:
                df = pd.read_parquet(proj_path)
                if "player_id" in df.columns and "pred_own_pct" in df.columns:
                    ownership = df.dropna(subset=["pred_own_pct"])
                    result = dict(zip(ownership["player_id"].astype(str), ownership["pred_own_pct"]))
                    logger.info(f"Loaded ownership for {len(result)} players from unified projections")
                    return result
            except Exception as e:
                logger.warning(f"Failed to load unified projections: {e}")
                continue
    
    # Fall back to silver/ownership_predictions
    slate_dir = data_root / "silver" / "ownership_predictions" / game_date
    if slate_dir.exists():
        slate_files = [p for p in slate_dir.glob("*.parquet") if not p.name.endswith("_locked.parquet")]
        if slate_files:
            own_path = max(slate_files, key=lambda p: p.stat().st_size)
            try:
                df = pd.read_parquet(own_path)
                if "player_id" in df.columns and "pred_own_pct" in df.columns:
                    ownership = df.dropna(subset=["pred_own_pct"])
                    result = dict(zip(ownership["player_id"].astype(str), ownership["pred_own_pct"]))
                    logger.info(
                        f"Loaded ownership for {len(result)} players from silver/ownership_predictions/{game_date}/"
                    )
                    return result
            except Exception as e:
                logger.warning(f"Failed to load per-slate ownership predictions: {e}")

    own_path = data_root / "silver" / "ownership_predictions" / f"{game_date}.parquet"
    if own_path.exists():
        try:
            df = pd.read_parquet(own_path)
            if "player_id" in df.columns and "pred_own_pct" in df.columns:
                ownership = df.dropna(subset=["pred_own_pct"])
                result = dict(zip(ownership["player_id"].astype(str), ownership["pred_own_pct"]))
                logger.info(f"Loaded ownership for {len(result)} players from silver/ownership_predictions")
                return result
        except Exception as e:
            logger.warning(f"Failed to load ownership predictions: {e}")
    
    logger.info("No ownership data available, dupe penalties disabled")
    return {}


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
    entry_max: int = Field(default=150, description="Max entries per user (for dupe penalty)")


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
    dupe_penalty: float = 1.0  # E[1/K], 1.0 = no penalty
    adjusted_expected_payout: Optional[float] = None  # expected_payout * dupe_penalty


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
    
    Dupe penalties are automatically applied if ownership data is available.
    """
    try:
        # Load ownership data for dupe penalty calculation
        player_ownership = _load_player_ownership(request.game_date)
        
        result = run_contest_simulation(
            lineups=request.lineups,
            game_date=request.game_date,
            archetype=request.archetype,
            field_size_bucket=request.field_size_bucket,
            field_size_override=request.field_size_override,
            entry_fee=request.entry_fee,
            weights=request.weights,
            player_ownership=player_ownership if player_ownership else None,
            entry_max=request.entry_max,
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
