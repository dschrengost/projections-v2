"""FastAPI router for QuickBuild optimizer endpoints.

Endpoints:
    GET  /api/optimizer/slates         - List available draft groups for a date
    GET  /api/optimizer/pool           - Get merged player pool (projections + salaries)
    POST /api/optimizer/build          - Start async QuickBuild job
    GET  /api/optimizer/build/{job_id} - Poll job status
    GET  /api/optimizer/build/{job_id}/lineups - Get completed lineups
    GET  /api/optimizer/build/{job_id}/export  - Export lineups as CSV
    GET  /api/optimizer/jobs           - List recent jobs

    # User Override Endpoints
    GET  /api/optimizer/overrides      - Get overrides for a slate
    PUT  /api/optimizer/overrides      - Create/update overrides (batch)
    DELETE /api/optimizer/overrides/{player_id} - Remove player override
    DELETE /api/optimizer/overrides    - Clear all overrides for slate
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response
from pydantic import BaseModel, Field

from .optimizer_service import (
    build_player_pool,
    get_data_root,
    get_job_store,
    get_slates_for_date,
    load_optimizer_config,
    run_quick_build,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class GameInfo(BaseModel):
    """Game in a slate."""
    matchup: str
    start_time: Optional[str] = None


class SlateInfo(BaseModel):
    """Draft group / slate info."""

    game_date: str
    slate_type: str
    draft_group_id: int
    n_contests: int
    earliest_start: Optional[str] = None
    latest_start: Optional[str] = None
    example_contest_name: Optional[str] = None
    games: List[GameInfo] = []


class PoolPlayer(BaseModel):
    """Player in the optimizer pool."""

    player_id: str
    name: str
    team: str
    positions: List[str]
    salary: int
    proj: float
    dk_id: Optional[str] = None
    own_proj: Optional[float] = None
    stddev: Optional[float] = None
    p90: Optional[float] = None
    game_matchup: Optional[str] = None
    game_start_utc: Optional[str] = None
    # Override metadata (present when use_user_overrides=true)
    model_proj: Optional[float] = None
    model_minutes: Optional[float] = None
    model_own: Optional[float] = None
    effective_proj: Optional[float] = None
    effective_minutes: Optional[float] = None
    effective_own: Optional[float] = None
    override_minutes: Optional[float] = None
    override_fpts: Optional[float] = None
    override_own: Optional[float] = None
    is_out: Optional[bool] = None
    has_override: Optional[bool] = None
    used_fppm_fallback: Optional[bool] = None
    fppm: Optional[float] = None
    is_active: Optional[bool] = None


class QuickBuildRequest(BaseModel):
    """Request to start a QuickBuild job."""

    date: str = Field(..., description="Game date in YYYY-MM-DD format")
    draft_group_id: int = Field(..., description="DraftKings draft group ID")
    site: str = Field(default="dk", description="Site: dk or fd")
    run_id: Optional[str] = Field(default=None, description="Projections run ID to use")

    # Pool settings
    max_pool: int = Field(default=10000, ge=100, le=200000, description="Max lineups in pool")
    builds: int = Field(default=22, ge=1, le=24, description="Number of parallel workers")
    per_build: int = Field(default=3000, ge=100, le=50000, description="Target lineups per worker")
    min_uniq: int = Field(default=1, ge=1, le=8, description="Min unique players between lineups")
    jitter: float = Field(default=0.0005, ge=0.0, le=0.1, description="Projection jitter")
    near_dup_jaccard: float = Field(default=0.0, ge=0.0, le=1.0, description="Near-dup threshold")
    enum_enable: bool = Field(default=True, description="Enable enumeration phase")

    # Constraints
    min_salary: Optional[int] = Field(default=None, description="Minimum salary")
    max_salary: Optional[int] = Field(default=None, description="Maximum salary")
    global_team_limit: int = Field(default=4, ge=1, le=8, description="Max players per team")
    lock_ids: List[str] = Field(default_factory=list, description="Player IDs to lock")
    ban_ids: List[str] = Field(default_factory=list, description="Player IDs to ban")
    
    # Game filters
    include_games: List[str] = Field(default_factory=list, description="Only include players from these games (e.g., ['MIN@DAL'])")
    exclude_games: List[str] = Field(default_factory=list, description="Exclude players from these games")

    # Ownership penalty
    ownership_penalty_enabled: bool = Field(default=False, description="Enable ownership penalty")
    ownership_lambda: float = Field(default=1.0, ge=0.0, le=10.0, description="Ownership lambda")
    ownership_curve: str = Field(default="sigmoid", description="Curve: sigmoid/linear/power/neglog")

    # Randomness
    randomness_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Randomness %")

    # User overrides
    use_user_overrides: bool = Field(default=False, description="Use user projection overrides ('My Proj')")
    ownership_mode: str = Field(default="renormalize", description="Ownership mode: raw or renormalize")


class JobStatus(BaseModel):
    """QuickBuild job status."""

    job_id: str
    status: str
    created_at: str
    game_date: str
    draft_group_id: int
    site: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = 0
    target: int = 0
    lineups_count: int = 0
    wall_time_sec: Optional[float] = None
    error: Optional[str] = None


class LineupRow(BaseModel):
    """A single lineup from QuickBuild results."""

    lineup_id: int
    player_ids: List[str]
    mean: Optional[float] = None
    p10: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None
    stdev: Optional[float] = None
    ceiling_upside: Optional[float] = None


class LineupsResponse(BaseModel):
    """Response containing lineups from a completed job."""

    job_id: str
    lineups_count: int
    lineups: List[LineupRow]
    stats: Dict[str, Any]


class ExportLineupsRequest(BaseModel):
    """Request to export arbitrary lineups as DK-uploadable CSV."""

    date: str = Field(..., description="Game date YYYY-MM-DD")
    draft_group_id: int = Field(..., description="DraftKings draft group ID")
    site: str = Field(default="dk", description="Site: dk")
    filename_prefix: Optional[str] = Field(default=None, description="Optional filename prefix")
    lineups: List[List[str]] = Field(..., description="List of lineups (each a list of player_ids)")


# ---------------------------------------------------------------------------
# DK CSV Export Helpers
# ---------------------------------------------------------------------------


DK_NBA_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

# DraftKings rosterSlotId → DK CSV slot (NBA Classic)
DK_NBA_ROSTER_SLOT_ID_TO_SLOT = {
    458: "PG",
    459: "SG",
    460: "SF",
    461: "PF",
    462: "C",
    463: "F",
    464: "G",
    465: "UTIL",
    # Note: Single-game slates (rosterSlotId 569/570) use position-based fallback
    # in _load_dk_nba_draftable_ids_by_player() instead of this mapping
}


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "lineups"


def _load_dk_nba_draftable_ids_by_player(
    draft_group_id: int,
) -> tuple[Dict[int, Dict[str, int]], Dict[int, str]]:
    """Return {dk_player_id -> {slot -> draftableId}}, plus {dk_player_id -> displayName}.
    
    For classic slates, uses rosterSlotId directly.
    For single-game slates (rosterSlotId 569/570), derives slot eligibility from position field.
    """
    bronze_path = (
        get_data_root()
        / "bronze"
        / "dk"
        / "draftables"
        / f"draftables_raw_{draft_group_id}.json"
    )
    if not bronze_path.exists():
        raise FileNotFoundError(f"Draftables not found: {bronze_path}")

    with open(bronze_path) as f:
        payload = json.load(f)

    draftables = payload.get("draftables", [])
    if not isinstance(draftables, list):
        raise RuntimeError("Draftables payload missing 'draftables' list")

    # Position to slot eligibility mapping for NBA classic format
    POSITION_TO_SLOTS = {
        "PG": ["PG", "G", "UTIL"],
        "SG": ["SG", "G", "UTIL"],
        "SF": ["SF", "F", "UTIL"],
        "PF": ["PF", "F", "UTIL"],
        "C": ["C", "UTIL"],
    }

    ids_by_player: Dict[int, Dict[str, int]] = {}
    names_by_player: Dict[int, str] = {}
    for d in draftables:
        if not isinstance(d, dict):
            continue
        dk_player_id = d.get("playerId")
        draftable_id = d.get("draftableId") or d.get("id")
        roster_slot_id = d.get("rosterSlotId")
        position = d.get("position", "")
        if dk_player_id is None or draftable_id is None:
            continue
        try:
            dk_player_id_i = int(dk_player_id)
            draftable_id_i = int(draftable_id)
            roster_slot_id_i = int(roster_slot_id) if roster_slot_id else None
        except (TypeError, ValueError):
            continue

        # Try classic slot mapping first
        slot = DK_NBA_ROSTER_SLOT_ID_TO_SLOT.get(roster_slot_id_i) if roster_slot_id_i else None
        
        if slot:
            # Classic format - single slot from rosterSlotId
            ids_by_player.setdefault(dk_player_id_i, {})[slot] = draftable_id_i
        else:
            # Single-game/showdown format - derive slots from position field
            # Handle multi-position like "PF/C" or "SG/SF"
            pos_parts = [p.strip() for p in position.split("/") if p.strip()]
            eligible_slots: set[str] = set()
            for pos in pos_parts:
                eligible_slots.update(POSITION_TO_SLOTS.get(pos, []))
            
            # Map this draftable_id to all eligible slots
            for slot in eligible_slots:
                ids_by_player.setdefault(dk_player_id_i, {})[slot] = draftable_id_i

        display_name = d.get("displayName")
        if isinstance(display_name, str) and display_name.strip() and dk_player_id_i not in names_by_player:
            names_by_player[dk_player_id_i] = display_name.strip()

    return ids_by_player, names_by_player


def _export_lineups_to_dk_csv(
    game_date: str,
    draft_group_id: int,
    site: str,
    lineups: List[List[str]],
) -> str:
    """Render DK-uploadable CSV content for the given lineups.

    Uses DK 'draftableId' (roster IDs) per slot, not playerId.
    """
    pool = build_player_pool(
        game_date=game_date,
        draft_group_id=draft_group_id,
        site=site,
    )

    internal_to_dk_player_id: Dict[str, int] = {}
    internal_to_name: Dict[str, str] = {}
    for p in pool:
        pid = str(p.get("player_id"))
        internal_to_name[pid] = str(p.get("name") or pid)
        dk_id_raw = p.get("dk_id")
        if not dk_id_raw:
            continue
        try:
            internal_to_dk_player_id[pid] = int(dk_id_raw)
        except (TypeError, ValueError):
            continue

    draftable_ids_by_player, dk_names_by_player = _load_dk_nba_draftable_ids_by_player(draft_group_id)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(DK_NBA_SLOTS)

    def assign_lineup_to_slots(player_ids: List[str]) -> Dict[str, str]:
        """Assign internal player_ids to DK slots using draftableId availability."""
        pids = [str(pid) for pid in player_ids]
        if len(pids) != len(DK_NBA_SLOTS):
            return {}

        adj: Dict[str, List[str]] = {}
        for pid in pids:
            dk_pid = internal_to_dk_player_id.get(pid)
            if dk_pid is None:
                adj[pid] = []
                continue
            adj[pid] = list(draftable_ids_by_player.get(dk_pid, {}).keys())

        match_r: Dict[str, Optional[str]] = {s: None for s in DK_NBA_SLOTS}
        match_l: Dict[str, Optional[str]] = {pid: None for pid in pids}

        def dfs(pid: str, seen: set[str]) -> bool:
            for s in adj.get(pid, []):
                if s in seen:
                    continue
                seen.add(s)
                if match_r[s] is None or dfs(match_r[s], seen):
                    match_r[s] = pid
                    match_l[pid] = s
                    return True
            return False

        changed = True
        while changed:
            changed = False
            for pid in pids:
                if match_l[pid] is None and dfs(pid, set()):
                    changed = True

        return {slot: pid for slot, pid in match_r.items() if pid is not None}

    for lineup in lineups:
        slot_to_internal = assign_lineup_to_slots(list(lineup))
        row: List[str] = []
        for slot in DK_NBA_SLOTS:
            internal_pid = slot_to_internal.get(slot)
            if not internal_pid:
                row.append("")
                continue
            dk_player_id = internal_to_dk_player_id.get(internal_pid)
            if dk_player_id is None:
                row.append("")
                continue
            draftable_id = draftable_ids_by_player.get(dk_player_id, {}).get(slot)
            if not draftable_id:
                row.append("")
                continue
            name = dk_names_by_player.get(dk_player_id) or internal_to_name.get(internal_pid) or str(internal_pid)
            row.append(f"{name} ({draftable_id})")
        writer.writerow(row)

    return output.getvalue()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/slates", response_model=List[SlateInfo])
async def get_slates(
    date: str = Query(..., description="Game date YYYY-MM-DD"),
    slate_type: str = Query(default="all", description="Filter: main/night/turbo/early/showdown/all"),
):
    """List available draft groups for a date."""
    try:
        slates = get_slates_for_date(date, slate_type=slate_type)
        # Convert datetime objects to strings
        for s in slates:
            for key in ["earliest_start", "latest_start", "game_date"]:
                if key in s and s[key] is not None:
                    if hasattr(s[key], "isoformat"):
                        s[key] = s[key].isoformat()
                    else:
                        s[key] = str(s[key])
        return slates
    except Exception as exc:
        logger.exception("Failed to get slates: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/pool", response_model=List[PoolPlayer])
async def get_player_pool(
    date: str = Query(..., description="Game date YYYY-MM-DD"),
    draft_group_id: int = Query(..., description="Draft group ID"),
    run_id: Optional[str] = Query(default=None, description="Projections run ID"),
    site: str = Query(default="dk", description="Site: dk or fd"),
    include_games: Optional[str] = Query(default=None, description="Comma-separated games to include (e.g., 'MIN@DAL,LAL@GSW')"),
    exclude_games: Optional[str] = Query(default=None, description="Comma-separated games to exclude"),
    use_user_overrides: bool = Query(default=False, description="Apply user projection overrides"),
    ownership_mode: str = Query(default="renormalize", description="Ownership mode: raw or renormalize"),
):
    """Get merged player pool (projections + salaries + positions).
    
    Use include_games or exclude_games to filter by specific games.
    Games are specified as 'AWAY@HOME' format (e.g., 'MIN@DAL').
    
    Set use_user_overrides=true to include user overrides with effective values.
    """
    try:
        # Parse comma-separated game lists
        include_list = [g.strip() for g in include_games.split(",")] if include_games else None
        exclude_list = [g.strip() for g in exclude_games.split(",")] if exclude_games else None
        
        pool = build_player_pool(
            game_date=date,
            draft_group_id=draft_group_id,
            site=site,
            run_id=run_id,
            include_games=include_list,
            exclude_games=exclude_list,
            use_user_overrides=use_user_overrides,
            ownership_mode=ownership_mode,
        )
        return pool
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to build player pool: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/build", response_model=JobStatus)
async def start_build(
    request: QuickBuildRequest,
    background_tasks: BackgroundTasks,
):
    """Start an async QuickBuild job.

    Returns immediately with job_id. Poll /build/{job_id} for status.
    """
    store = get_job_store()

    # Check concurrent job limit
    config = load_optimizer_config()
    max_concurrent = config.get("jobs", {}).get("max_concurrent", 4)
    running = [j for j in store.list_jobs(50) if j.status == "running"]
    if len(running) >= max_concurrent:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent jobs ({len(running)}/{max_concurrent}). Try again later.",
        )

    # Build player pool first (fail fast if data missing)
    try:
        # Parse game filters (convert empty lists to None)
        include_games = request.include_games if request.include_games else None
        exclude_games = request.exclude_games if request.exclude_games else None
        
        player_pool = build_player_pool(
            game_date=request.date,
            draft_group_id=request.draft_group_id,
            site=request.site,
            run_id=request.run_id,
            include_games=include_games,
            exclude_games=exclude_games,
            use_user_overrides=request.use_user_overrides,
            ownership_mode=request.ownership_mode,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to build player pool for job: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if len(player_pool) < 8:
        raise HTTPException(
            status_code=400,
            detail=f"Player pool too small ({len(player_pool)} players). Need at least 8.",
        )

    # Create job
    job = store.create(
        game_date=request.date,
        draft_group_id=request.draft_group_id,
        site=request.site,
        config=request.model_dump(),
        target=request.max_pool,
    )

    # Start background task
    background_tasks.add_task(run_quick_build, job, player_pool)

    logger.info(
        "Started QuickBuild job %s: date=%s, dg=%d, max_pool=%d",
        job.job_id,
        request.date,
        request.draft_group_id,
        request.max_pool,
    )

    return JobStatus(**job.to_dict())


@router.get("/build/{job_id}", response_model=JobStatus)
async def get_build_status(job_id: str):
    """Poll job status and progress."""
    job = get_job_store().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatus(**job.to_dict())


@router.get("/build/{job_id}/lineups", response_model=LineupsResponse)
async def get_build_lineups(job_id: str):
    """Get lineups from a completed job."""
    job = get_job_store().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status == "pending":
        raise HTTPException(status_code=400, detail="Job has not started yet")
    if job.status == "running":
        raise HTTPException(status_code=400, detail="Job is still running")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")

    lineups: List[LineupRow] = []
    for i, lineup in enumerate(job.lineups):
        extra: Dict[str, Any] = {}
        if job.lineup_stats and i < len(job.lineup_stats):
            extra = job.lineup_stats[i] or {}
        lineups.append(LineupRow(lineup_id=i, player_ids=list(lineup), **extra))

    return LineupsResponse(
        job_id=job.job_id,
        lineups_count=len(lineups),
        lineups=lineups,
        stats=job.stats,
    )


@router.get("/build/{job_id}/export")
async def export_lineups_csv(
    job_id: str,
    date: Optional[str] = Query(default=None, description="Game date for pool lookup"),
):
    """Export lineups as DraftKings-uploadable CSV.
    
    Format: Each cell is "Player Name (draftableId)" in positional order.
    """
    job = get_job_store().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed (status: {job.status})",
        )

    game_date = date or job.game_date
    try:
        csv_content = _export_lineups_to_dk_csv(
            game_date=game_date,
            draft_group_id=job.draft_group_id,
            site=job.site,
            lineups=[list(lu) for lu in job.lineups],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to export DK CSV: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    filename = f"lineups_{job.game_date}_{job.job_id[:8]}.csv"

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/export")
async def export_custom_lineups_csv(request: ExportLineupsRequest):
    """Export arbitrary lineups as DraftKings-uploadable CSV (draftableIds)."""
    if not request.lineups:
        raise HTTPException(status_code=400, detail="No lineups provided")

    try:
        csv_content = _export_lineups_to_dk_csv(
            game_date=request.date,
            draft_group_id=request.draft_group_id,
            site=request.site,
            lineups=request.lineups,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to export DK CSV: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    prefix = _safe_filename_part(request.filename_prefix) if request.filename_prefix else "lineups"
    filename = f"{prefix}_{request.date}_{request.draft_group_id}_{len(request.lineups)}.csv"

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@router.get("/jobs", response_model=List[JobStatus])
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="Max jobs to return"),
):
    """List recent QuickBuild jobs."""
    jobs = get_job_store().list_jobs(limit)
    return [JobStatus(**j.to_dict()) for j in jobs]


# ---------------------------------------------------------------------------
# Saved Builds Endpoints
# ---------------------------------------------------------------------------


class SavedBuildSummary(BaseModel):
    """Summary of a saved build (no lineups)."""

    job_id: str
    game_date: str
    draft_group_id: int
    site: str
    created_at: str
    completed_at: Optional[str] = None
    lineups_count: int
    config: Dict[str, Any] = {}
    stats: Dict[str, Any] = {}


class SavedBuildFull(BaseModel):
    """Full saved build including lineups."""

    job_id: str
    game_date: str
    draft_group_id: int
    site: str
    created_at: str
    completed_at: Optional[str] = None
    lineups_count: int
    config: Dict[str, Any] = {}
    stats: Dict[str, Any] = {}
    lineups: List[LineupRow] = []


@router.get("/saved-builds", response_model=List[SavedBuildSummary])
async def list_saved_builds_endpoint(
    date: str = Query(..., description="Game date YYYY-MM-DD"),
    draft_group_id: Optional[int] = Query(default=None, description="Filter by draft group"),
):
    """List saved builds for a date."""
    from .optimizer_service import list_saved_builds
    
    builds = list_saved_builds(date, draft_group_id)
    return builds


@router.get("/saved-builds/{job_id}", response_model=SavedBuildFull)
async def get_saved_build_endpoint(
    job_id: str,
    date: str = Query(..., description="Game date YYYY-MM-DD"),
):
    """Load a saved build including lineups."""
    from .optimizer_service import load_saved_build
    
    build = load_saved_build(date, job_id)
    if not build:
        raise HTTPException(status_code=404, detail=f"Build {job_id} not found for date {date}")
    
    # Convert lineups format
    if "lineups" in build:
        build["lineups"] = [
            LineupRow(
                lineup_id=lu.get("lineup_id", i),
                player_ids=lu.get("player_ids", []),
                mean=lu.get("mean"),
                p10=lu.get("p10"),
                p50=lu.get("p50"),
                p75=lu.get("p75"),
                p90=lu.get("p90"),
                stdev=lu.get("stdev"),
                ceiling_upside=lu.get("ceiling_upside"),
            )
            for i, lu in enumerate(build["lineups"])
        ]
    
    return build

@router.delete("/saved-builds/{job_id}")
async def delete_saved_build_endpoint(
    job_id: str,
    date: str = Query(..., description="Game date YYYY-MM-DD"),
):
    """Delete a saved build."""
    from .optimizer_service import delete_saved_build
    
    deleted = delete_saved_build(date, job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Build {job_id} not found for date {date}")
    
    return {"status": "deleted", "job_id": job_id}


# ---------------------------------------------------------------------------
# User Override Endpoints
# ---------------------------------------------------------------------------


class PlayerOverrideRequest(BaseModel):
    """Request to set overrides for a single player."""
    
    player_id: str = Field(..., description="Player ID")
    minutes: Optional[float] = Field(default=None, ge=0, le=48, description="Override minutes")
    fpts: Optional[float] = Field(default=None, ge=0, description="Override FPTS")
    own: Optional[float] = Field(default=None, ge=0, le=100, description="Override ownership %")
    is_out: bool = Field(default=False, description="Mark player as out")


class BatchOverrideRequest(BaseModel):
    """Request to set multiple overrides at once."""
    
    date: str = Field(..., description="Game date YYYY-MM-DD")
    draft_group_id: int = Field(..., description="Draft group ID")
    overrides: List[PlayerOverrideRequest] = Field(..., description="List of player overrides")
    expected_revision: Optional[int] = Field(default=None, description="For conflict detection")


class PlayerOverrideResponse(BaseModel):
    """Override values for a single player."""
    
    player_id: str
    minutes: Optional[float] = None
    fpts: Optional[float] = None
    own: Optional[float] = None
    is_out: bool = False
    updated_at: str


class SlateOverridesResponse(BaseModel):
    """Response containing all overrides for a slate."""
    
    game_date: str
    draft_group_id: int
    client_revision: int
    updated_at: str
    overrides: List[PlayerOverrideResponse]


@router.get("/overrides", response_model=SlateOverridesResponse)
async def get_overrides(
    date: str = Query(..., description="Game date YYYY-MM-DD"),
    draft_group_id: int = Query(..., description="Draft group ID"),
):
    """Get all user overrides for a slate."""
    from .user_overrides import load_slate_overrides
    
    slate_overrides = load_slate_overrides(date, draft_group_id)
    
    return SlateOverridesResponse(
        game_date=slate_overrides.game_date,
        draft_group_id=slate_overrides.draft_group_id,
        client_revision=slate_overrides.client_revision,
        updated_at=slate_overrides.updated_at,
        overrides=[
            PlayerOverrideResponse(
                player_id=po.player_id,
                minutes=po.minutes,
                fpts=po.fpts,
                own=po.own,
                is_out=po.is_out,
                updated_at=po.updated_at,
            )
            for po in slate_overrides.overrides.values()
        ],
    )


@router.put("/overrides", response_model=SlateOverridesResponse)
async def set_overrides(request: BatchOverrideRequest):
    """Create or update overrides (batch). Merges with existing overrides."""
    from .user_overrides import (
        PlayerOverride,
        load_slate_overrides,
        save_slate_overrides,
    )
    
    # Load existing overrides
    slate_overrides = load_slate_overrides(request.date, request.draft_group_id)
    
    # Check for revision conflict
    if request.expected_revision is not None:
        if slate_overrides.client_revision != request.expected_revision:
            raise HTTPException(
                status_code=409,
                detail=f"Revision conflict: expected {request.expected_revision}, got {slate_overrides.client_revision}",
            )
    
    # Apply updates
    for override_req in request.overrides:
        # If all values are None/False, remove the override
        if (
            override_req.minutes is None
            and override_req.fpts is None
            and override_req.own is None
            and not override_req.is_out
        ):
            slate_overrides.remove_override(override_req.player_id)
        else:
            po = PlayerOverride(
                player_id=override_req.player_id,
                minutes=override_req.minutes,
                fpts=override_req.fpts,
                own=override_req.own,
                is_out=override_req.is_out,
            )
            slate_overrides.set_override(po)
    
    # Save
    try:
        save_slate_overrides(slate_overrides)
    except Exception as e:
        logger.exception("Failed to save overrides: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save: {e}")
    
    logger.info(
        "Saved %d overrides for %s/dg=%d (rev=%d)",
        len(slate_overrides.overrides),
        request.date,
        request.draft_group_id,
        slate_overrides.client_revision,
    )
    
    return SlateOverridesResponse(
        game_date=slate_overrides.game_date,
        draft_group_id=slate_overrides.draft_group_id,
        client_revision=slate_overrides.client_revision,
        updated_at=slate_overrides.updated_at,
        overrides=[
            PlayerOverrideResponse(
                player_id=po.player_id,
                minutes=po.minutes,
                fpts=po.fpts,
                own=po.own,
                is_out=po.is_out,
                updated_at=po.updated_at,
            )
            for po in slate_overrides.overrides.values()
        ],
    )


@router.delete("/overrides/{player_id}")
async def remove_player_override(
    player_id: str,
    date: str = Query(..., description="Game date YYYY-MM-DD"),
    draft_group_id: int = Query(..., description="Draft group ID"),
):
    """Remove override for a single player."""
    from .user_overrides import load_slate_overrides, save_slate_overrides
    
    slate_overrides = load_slate_overrides(date, draft_group_id)
    removed = slate_overrides.remove_override(player_id)
    
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"No override found for player {player_id}",
        )
    
    save_slate_overrides(slate_overrides)
    
    return {"status": "deleted", "player_id": player_id}


@router.delete("/overrides")
async def clear_all_overrides(
    date: str = Query(..., description="Game date YYYY-MM-DD"),
    draft_group_id: int = Query(..., description="Draft group ID"),
):
    """Clear all overrides for a slate."""
    from .user_overrides import load_slate_overrides, save_slate_overrides
    
    slate_overrides = load_slate_overrides(date, draft_group_id)
    count = len(slate_overrides.overrides)
    slate_overrides.clear_all()
    save_slate_overrides(slate_overrides)
    
    logger.info("Cleared %d overrides for %s/dg=%d", count, date, draft_group_id)
    
    return {"status": "cleared", "count": count}
