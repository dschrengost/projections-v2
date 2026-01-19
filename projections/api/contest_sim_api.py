"""FastAPI router for contest simulation endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from projections import paths
from projections.contest_sim.contest_sim_service import run_contest_simulation
from projections.contest_sim.field_library_manager import load_or_build_field_library
from projections.contest_sim.field_library import load_field_library, list_field_library_paths
from projections.contest_sim.payout_generator import load_config

logger = logging.getLogger(__name__)
router = APIRouter()


def _normalize_ownership_mode(mode: str | None) -> str:
    m = str(mode or "full").strip().lower()
    aliases = {
        "on": "full",
        "none": "off",
        "no": "off",
        "false": "off",
        "true": "full",
    }
    m = aliases.get(m, m)
    allowed = {"full", "off", "dupe_only", "field_only"}
    if m not in allowed:
        raise ValueError(f"Invalid ownership_mode: {mode!r} (allowed: {sorted(allowed)})")
    return m


def _normalize_rank_mode(mode: str | None) -> str:
    m = str(mode or "current").strip().lower()
    allowed = {"current", "tail_only", "tail_times_dupe"}
    if m not in allowed:
        raise ValueError(f"Invalid rank_mode: {mode!r} (allowed: {sorted(allowed)})")
    return m


def _sim_builds_dir() -> Path:
    """Get the contest sim builds directory under projections-data."""
    return paths.data_path() / "builds" / "contest_sim"


def _save_sim_build(game_date: str, build_data: Dict[str, object]) -> str:
    builds_root = _sim_builds_dir() / game_date
    builds_root.mkdir(parents=True, exist_ok=True)
    build_id = build_data.get("build_id") or str(uuid4())
    build_path = builds_root / f"{build_id}.json"
    build_data["build_id"] = build_id
    with open(build_path, "w") as f:
        import json
        json.dump(build_data, f, indent=2)
    return str(build_id)


def _load_sim_build(game_date: str, build_id: str) -> Optional[Dict[str, object]]:
    build_path = _sim_builds_dir() / game_date / f"{build_id}.json"
    if not build_path.exists():
        return None
    with open(build_path) as f:
        import json
        return json.load(f)


def _delete_sim_build(game_date: str, build_id: str) -> bool:
    build_path = _sim_builds_dir() / game_date / f"{build_id}.json"
    if build_path.exists():
        build_path.unlink()
        return True
    return False


def _list_sim_builds(game_date: str) -> List[Dict[str, object]]:
    builds_root = _sim_builds_dir() / game_date
    if not builds_root.exists():
        return []
    builds = []
    for build_file in sorted(builds_root.glob("*.json"), reverse=True):
        try:
            import json
            with open(build_file) as f:
                data = json.load(f)
            builds.append({
                "build_id": data.get("build_id", build_file.stem),
                "game_date": data.get("game_date"),
                "draft_group_id": data.get("draft_group_id"),
                "created_at": data.get("created_at"),
                "lineups_count": data.get("lineups_count"),
                "name": data.get("name"),
                "kind": data.get("kind", "run"),
                "stats": data.get("stats", {}),
            })
        except Exception as exc:
            logger.warning("Failed to read sim build %s: %s", build_file, exc)
            continue
    return builds


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
    draft_group_id: Optional[int] = Field(default=None, description="Draft group ID")
    lineups: List[List[str]] = Field(..., description="List of lineups (each a list of player_ids)")
    field_mode: str = Field(
        default="self_play",
        description="Field mode: self_play | generated_field",
    )
    field_library_version: str = Field(
        default="v0",
        description="Field library version to load/build (or 'latest')",
    )
    field_library_k: int = Field(
        default=2500,
        description="Target unique field lineups K for generated_field",
    )
    field_candidate_pool_size: int = Field(
        default=40000,
        description="QuickBuild candidate pool size for generated_field",
    )
    field_library_rebuild: bool = Field(
        default=False,
        description="Force rebuild field library even if cached",
    )
    field_library_rebuild_candidates: bool = Field(
        default=False,
        description="Force rebuild raw candidate pool for generated_field",
    )
    archetype: str = Field(
        default="GPP Standard (20% paid)",
        description="Payout archetype name from contest_sim.yaml",
    )
    field_size_bucket: str = Field(
        default="5000",
        description="Field size bucket (numeric string like '5000' or label from contest_sim.yaml)",
    )
    field_size_override: Optional[int] = Field(default=None, description="Exact field size (overrides bucket)")
    entry_fee: float = Field(default=3.0, description="Entry fee per lineup")
    weights: Optional[List[int]] = Field(default=None, description="Entry counts per lineup")
    entry_max: int = Field(default=150, description="Max entries per user (for dupe penalty)")
    ownership_mode: str = Field(
        default="full",
        description="Ownership usage: full | off | dupe_only | field_only",
    )
    rank_mode: str = Field(
        default="current",
        description="Ranking mode for select_score: current | tail_only | tail_times_dupe",
    )


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
    unadjusted_expected_payout: Optional[float] = None  # expected_payout before dupe penalty
    adjusted_expected_payout: Optional[float] = None  # expected_payout * dupe_penalty
    # Tail / upside selection metrics
    ucv90: Optional[float] = None  # Upper CVaR at 90th pctile (mean of top 10% scores)
    tail_score: Optional[float] = None  # Weighted combo: 0.6*p90 + 0.4*ucv90
    select_score: Optional[float] = None  # tail_score - dupe penalty impact


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
    debug: Dict[str, object] = Field(default_factory=dict)


class ContestSimResponse(BaseModel):
    """Complete contest simulation response."""

    results: List[LineupEVResultResponse]
    config: ContestConfigResponse
    stats: SummaryStatsResponse
    build_id: Optional[str] = None


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


class SavedSimBuildSummary(BaseModel):
    build_id: str
    game_date: str
    draft_group_id: Optional[int] = None
    created_at: str
    lineups_count: int
    name: Optional[str] = None
    kind: str = "run"  # run | lineups
    stats: Dict[str, object] = {}


class SavedSimBuildDetail(BaseModel):
    build_id: str
    game_date: str
    draft_group_id: Optional[int] = None
    created_at: str
    lineups_count: int
    name: Optional[str] = None
    kind: str = "run"
    config: Optional[Dict[str, object]] = None
    stats: Dict[str, object]
    results: Optional[List[LineupEVResultResponse]] = None
    lineups: List[List[str]]
    request: Optional[Dict[str, object]] = None


class SaveSimLineupsRequest(BaseModel):
    game_date: str
    draft_group_id: Optional[int] = None
    name: str
    lineups: List[List[str]]
    results: Optional[List[LineupEVResultResponse]] = None
    config: Optional[ContestConfigResponse] = None
    stats: Optional[SummaryStatsResponse] = None


class FieldLibrarySummaryResponse(BaseModel):
    """Summary of a cached field library."""

    version: str
    path: str
    game_date: str
    draft_group_id: int
    method: Optional[str] = None
    generated_at: Optional[str] = None
    selected_k: int
    weights_sum: int
    meta: Dict[str, object] = Field(default_factory=dict)


class BuildFieldLibraryRequest(BaseModel):
    game_date: str
    draft_group_id: int
    version: str = "v0"
    k: int = 2500
    candidate_pool_size: int = 40000
    rebuild: bool = False
    rebuild_candidates: bool = False
    ownership_mode: str = Field(
        default="full",
        description="Ownership usage: full | off | dupe_only | field_only",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run", response_model=ContestSimResponse)
async def run_simulation(request: ContestSimRequest):
    """Run contest simulation for the given lineups.

    User lineups compete against a modeled opponent field.
    
    Dupe penalties and field generation can be toggled via `ownership_mode`.
    """
    try:
        try:
            ownership_mode = _normalize_ownership_mode(request.ownership_mode)
            rank_mode = _normalize_rank_mode(request.rank_mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if ownership_mode == "off":
            rank_mode = "tail_only"

        use_dupe_ownership = ownership_mode in {"full", "dupe_only"}
        use_field_ownership = ownership_mode in {"full", "field_only"}

        # Load ownership data for dupe penalty calculation (only when enabled)
        player_ownership = _load_player_ownership(request.game_date) if use_dupe_ownership else {}

        field_lineups = None
        field_weights = None
        field_library_info: Dict[str, object] = {}
        if request.field_mode not in {"self_play", "generated_field"}:
            raise HTTPException(status_code=400, detail=f"Invalid field_mode: {request.field_mode}")

        if request.field_mode == "generated_field":
            if request.draft_group_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="draft_group_id is required when field_mode=generated_field",
                )
            # Cache separation: avoid overwriting ownership-aware field libraries with ownership-free builds.
            version = request.field_library_version
            if not use_field_ownership:
                version = "v0" if version == "latest" else version
                if not str(version).endswith("_noown"):
                    version = f"{version}_noown"

            library, lib_path, built_now = load_or_build_field_library(
                game_date=request.game_date,
                draft_group_id=int(request.draft_group_id),
                version=version,
                k=int(request.field_library_k),
                candidate_pool_size=int(request.field_candidate_pool_size),
                rebuild=bool(request.field_library_rebuild),
                rebuild_candidates=bool(request.field_library_rebuild_candidates),
                use_ownership_features=use_field_ownership,
            )
            field_lineups = library.lineups
            field_weights = library.weights
            field_library_info = {
                "field_mode": request.field_mode,
                "field_library_path": str(lib_path),
                "field_library_built_now": built_now,
                "field_library_version": library.meta.get("version", request.field_library_version),
                "field_library_method": library.meta.get("method"),
                "field_library_selected_k": len(library.lineups),
                "field_library_weights_sum": int(sum(library.weights)),
                "ownership_mode": ownership_mode,
                "rank_mode": rank_mode,
                "field_library_use_ownership": bool(use_field_ownership),
            }
        
        result = run_contest_simulation(
            user_lineups=request.lineups,
            game_date=request.game_date,
            archetype=request.archetype,
            field_size_bucket=request.field_size_bucket,
            field_size_override=request.field_size_override,
            entry_fee=request.entry_fee,
            user_weights=request.weights,
            field_lineups=field_lineups,
            field_weights=field_weights,
            player_ownership=player_ownership if (use_dupe_ownership and player_ownership) else None,
            entry_max=request.entry_max,
            ownership_mode=ownership_mode,
            rank_mode=rank_mode,
        )
        if field_library_info:
            result.stats.debug.update(field_library_info)
        else:
            result.stats.debug.update({"ownership_mode": ownership_mode, "rank_mode": rank_mode})

        build_data = {
            "build_id": str(uuid4()),
            "game_date": request.game_date,
            "draft_group_id": request.draft_group_id,
            "created_at": datetime.utcnow().isoformat(),
            "lineups_count": len(request.lineups),
            "kind": "run",
            "name": None,
            "config": {
                "field_size": result.config.field_size,
                "entry_fee": result.config.entry_fee,
                "archetype": result.config.archetype,
                "rake": result.config.rake,
                "prize_pool": result.config.prize_pool,
            },
            "stats": result.stats.to_dict(),
            "results": [r.to_dict() for r in result.results],
            "lineups": request.lineups,
            "request": request.model_dump(),
        }
        build_id = _save_sim_build(request.game_date, build_data)

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
            build_id=build_id,
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


@router.get("/saved-builds", response_model=List[SavedSimBuildSummary])
async def list_saved_sim_builds(
    date: str,
    kind: Optional[str] = None,
):
    """List saved contest sim builds for a date."""
    builds = _list_sim_builds(date)
    if kind:
        builds = [b for b in builds if b.get("kind") == kind]
    return builds


def _backfill_tail_metrics(results: List[Dict]) -> List[Dict]:
    """Backfill ucv90/tail_score/select_score for legacy saved builds."""
    tail_weight_p90 = 0.6
    tail_weight_ucv = 0.4
    for r in results:
        # Skip if already has tail metrics
        if r.get("ucv90") is not None:
            continue
        # Compute from p90 - for legacy builds we estimate UCVaR as ~p90 * 1.05
        # (a rough heuristic since we don't have the full score distribution)
        p90 = r.get("p90")
        mean = r.get("mean")
        dupe_penalty = r.get("dupe_penalty", 1.0)
        if p90 is not None and mean is not None:
            # Estimate UCVaR90 as p90 + 0.5 * (p90 - mean) based on typical distributions
            ucv90_est = p90 + 0.5 * max(0, p90 - mean)
            r["ucv90"] = round(ucv90_est, 2)
            tail_score = tail_weight_p90 * p90 + tail_weight_ucv * ucv90_est
            r["tail_score"] = round(tail_score, 2)
            penalty_impact = (1.0 - dupe_penalty) * mean
            r["select_score"] = round(tail_score - penalty_impact, 2)
    return results


@router.get("/saved-builds/{build_id}", response_model=SavedSimBuildDetail)
async def load_saved_sim_build(build_id: str, date: str):
    """Load a saved contest sim build with lineups/results."""
    data = _load_sim_build(date, build_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Sim build {build_id} not found for date {date}")
    results = data.get("results")
    # Backfill tail metrics for legacy builds
    if results:
        results = _backfill_tail_metrics(results)
    return SavedSimBuildDetail(
        build_id=data.get("build_id", build_id),
        game_date=data.get("game_date", date),
        draft_group_id=data.get("draft_group_id"),
        created_at=data.get("created_at", datetime.utcnow().isoformat()),
        lineups_count=data.get("lineups_count", 0),
        name=data.get("name"),
        kind=data.get("kind", "run"),
        config=data.get("config"),
        stats=data.get("stats", {}),
        results=[LineupEVResultResponse(**r) for r in results] if results else None,
        lineups=data.get("lineups", []),
        request=data.get("request"),
    )


@router.post("/saved-lineups", response_model=SavedSimBuildSummary)
async def save_sim_lineups(request: SaveSimLineupsRequest):
    """Save a named lineup set derived from contest sim results."""
    if not request.lineups:
        raise HTTPException(status_code=400, detail="No lineups provided")
    if not request.results or not request.config or not request.stats:
        raise HTTPException(status_code=400, detail="Snapshot results/config/stats are required to save sim lineups")
    build_data = {
        "build_id": str(uuid4()),
        "game_date": request.game_date,
        "draft_group_id": request.draft_group_id,
        "created_at": datetime.utcnow().isoformat(),
        "lineups_count": len(request.lineups),
        "kind": "lineups",
        "name": request.name,
        "stats": request.stats.model_dump() if request.stats else {},
        "config": request.config.model_dump() if request.config else None,
        "results": [r.model_dump() for r in request.results] if request.results else None,
        "lineups": request.lineups,
    }
    build_id = _save_sim_build(request.game_date, build_data)
    return SavedSimBuildSummary(
        build_id=build_id,
        game_date=request.game_date,
        draft_group_id=request.draft_group_id,
        created_at=build_data["created_at"],
        lineups_count=len(request.lineups),
        name=request.name,
        kind="lineups",
        stats=build_data["stats"] or {},
    )


@router.delete("/saved-builds/{build_id}")
async def delete_saved_sim_build(build_id: str, date: str):
    """Delete a saved contest sim build."""
    deleted = _delete_sim_build(date, build_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Sim build {build_id} not found for date {date}")
    return {"status": "deleted", "build_id": build_id}


@router.get("/field-libraries", response_model=List[FieldLibrarySummaryResponse])
async def list_field_libraries(date: str, draft_group_id: int):
    """List cached field libraries for a slate."""
    paths = list_field_library_paths(date, int(draft_group_id))
    summaries: List[FieldLibrarySummaryResponse] = []
    for path in paths:
        try:
            library = load_field_library(path)
            version = path.stem.replace("field_library_", "")
            summaries.append(
                FieldLibrarySummaryResponse(
                    version=version,
                    path=str(path),
                    game_date=str(library.meta.get("game_date", date)),
                    draft_group_id=int(library.meta.get("draft_group_id", draft_group_id)),
                    method=str(library.meta.get("method", "")) or None,
                    generated_at=str(library.meta.get("generated_at", "")) or None,
                    selected_k=len(library.lineups),
                    weights_sum=int(sum(library.weights)),
                    meta={k: v for k, v in library.meta.items() if k not in {"lineups", "weights"}},
                )
            )
        except Exception as exc:
            logger.warning("Failed to load field library %s: %s", path, exc)
            continue
    return summaries


@router.post("/field-libraries/build", response_model=FieldLibrarySummaryResponse)
async def build_field_library(request: BuildFieldLibraryRequest):
    """Build (or rebuild) a cached field library for a slate."""
    try:
        ownership_mode = _normalize_ownership_mode(request.ownership_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    use_field_ownership = ownership_mode in {"full", "field_only"}
    version = request.version
    if not use_field_ownership:
        version = "v0" if version == "latest" else version
        if not str(version).endswith("_noown"):
            version = f"{version}_noown"
    library, path, _built_now = load_or_build_field_library(
        game_date=request.game_date,
        draft_group_id=int(request.draft_group_id),
        version=version,
        k=int(request.k),
        candidate_pool_size=int(request.candidate_pool_size),
        rebuild=bool(request.rebuild),
        rebuild_candidates=bool(request.rebuild_candidates),
        use_ownership_features=use_field_ownership,
    )
    version = Path(path).stem.replace("field_library_", "")
    return FieldLibrarySummaryResponse(
        version=version,
        path=str(path),
        game_date=str(library.meta.get("game_date", request.game_date)),
        draft_group_id=int(library.meta.get("draft_group_id", request.draft_group_id)),
        method=str(library.meta.get("method", "")) or None,
        generated_at=str(library.meta.get("generated_at", "")) or None,
        selected_k=len(library.lineups),
        weights_sum=int(sum(library.weights)),
        meta=dict(library.meta),
    )
