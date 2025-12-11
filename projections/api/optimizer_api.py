"""FastAPI router for QuickBuild optimizer endpoints.

Endpoints:
    GET  /api/optimizer/slates         - List available draft groups for a date
    GET  /api/optimizer/pool           - Get merged player pool (projections + salaries)
    POST /api/optimizer/build          - Start async QuickBuild job
    GET  /api/optimizer/build/{job_id} - Poll job status
    GET  /api/optimizer/build/{job_id}/lineups - Get completed lineups
    GET  /api/optimizer/build/{job_id}/export  - Export lineups as CSV
    GET  /api/optimizer/jobs           - List recent jobs
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response
from pydantic import BaseModel, Field

from .optimizer_service import (
    build_player_pool,
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


class SlateInfo(BaseModel):
    """Draft group / slate info."""

    game_date: str
    slate_type: str
    draft_group_id: int
    n_contests: int
    earliest_start: Optional[str] = None
    latest_start: Optional[str] = None
    example_contest_name: Optional[str] = None


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


class QuickBuildRequest(BaseModel):
    """Request to start a QuickBuild job."""

    date: str = Field(..., description="Game date in YYYY-MM-DD format")
    draft_group_id: int = Field(..., description="DraftKings draft group ID")
    site: str = Field(default="dk", description="Site: dk or fd")
    run_id: Optional[str] = Field(default=None, description="Projections run ID to use")

    # Pool settings
    max_pool: int = Field(default=10000, ge=100, le=200000, description="Max lineups in pool")
    builds: int = Field(default=4, ge=1, le=16, description="Number of parallel workers")
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

    # Ownership penalty
    ownership_penalty_enabled: bool = Field(default=False, description="Enable ownership penalty")
    ownership_lambda: float = Field(default=1.0, ge=0.0, le=10.0, description="Ownership lambda")
    ownership_curve: str = Field(default="sigmoid", description="Curve: sigmoid/linear/power/neglog")

    # Randomness
    randomness_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Randomness %")


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


class LineupsResponse(BaseModel):
    """Response containing lineups from a completed job."""

    job_id: str
    lineups_count: int
    lineups: List[LineupRow]
    stats: Dict[str, Any]


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
):
    """Get merged player pool (projections + salaries + positions)."""
    try:
        pool = build_player_pool(
            game_date=date,
            draft_group_id=draft_group_id,
            site=site,
            run_id=run_id,
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
        player_pool = build_player_pool(
            game_date=request.date,
            draft_group_id=request.draft_group_id,
            site=request.site,
            run_id=request.run_id,
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

    lineups = [
        LineupRow(lineup_id=i, player_ids=list(lineup))
        for i, lineup in enumerate(job.lineups)
    ]

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
    
    Format: Each cell is "Player Name (DK_ID)" in positional order.
    """
    from .optimizer_service import build_player_pool
    from dataclasses import dataclass
    
    job = get_job_store().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed (status: {job.status})",
        )

    # DK slot order
    DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    
    # Build player lookup from pool
    game_date = date or job.game_date
    try:
        pool = build_player_pool(
            game_date=game_date,
            draft_group_id=job.draft_group_id,
            site=job.site,
        )
    except Exception as e:
        logger.warning("Failed to load pool for export: %s", e)
        pool = []
    
    # Create lookup: player_id -> {name, dk_id, positions}
    @dataclass
    class PlayerInfo:
        name: str
        dk_id: str
        positions: List[str]
    
    pid2player: Dict[str, PlayerInfo] = {}
    for p in pool:
        pid2player[p["player_id"]] = PlayerInfo(
            name=p["name"],
            dk_id=p.get("dk_id") or p["player_id"],
            positions=p["positions"],
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(DK_SLOTS)

    def assign_lineup_to_slots(player_ids: List[str]) -> Dict[str, str]:
        """Assign players to DK slots using bipartite matching.
        
        Returns dict mapping slot -> player_id.
        """
        SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
        
        def eligible_for_slot(pid: str, slot: str) -> bool:
            if pid not in pid2player:
                return False
            pos = set(pid2player[pid].positions)
            if slot in {"PG", "SG", "SF", "PF", "C"}:
                return slot in pos
            if slot == "G":
                return "PG" in pos or "SG" in pos
            if slot == "F":
                return "SF" in pos or "PF" in pos
            if slot == "UTIL":
                return True
            return False
        
        pids = list(player_ids)
        if len(pids) != 8:
            return {}
        
        # Build adjacency: pid -> list of eligible slots
        adj = {pid: [s for s in SLOTS if eligible_for_slot(pid, s)] for pid in pids}
        
        # Hopcroft-Karp style: keep trying to find augmenting paths
        matchR: Dict[str, Optional[str]] = {s: None for s in SLOTS}
        matchL: Dict[str, Optional[str]] = {pid: None for pid in pids}
        
        def dfs(pid: str, seen: set) -> bool:
            for s in adj.get(pid, []):
                if s in seen:
                    continue
                seen.add(s)
                if matchR[s] is None or dfs(matchR[s], seen):
                    matchR[s] = pid
                    matchL[pid] = s
                    return True
            return False
        
        # Keep trying until no more augmenting paths
        changed = True
        while changed:
            changed = False
            for pid in pids:
                if matchL[pid] is None:  # Unmatched player
                    if dfs(pid, set()):
                        changed = True
        
        # Build result: slot -> pid
        return {slot: pid for slot, pid in matchR.items() if pid is not None}



    for lineup in job.lineups:
        player_ids = list(lineup)
        
        # Check if we have player info
        if not all(pid in pid2player for pid in player_ids):
            # Fallback: write raw IDs
            row = player_ids[:8]
            while len(row) < 8:
                row.append("")
            writer.writerow(row)
            continue
        # Assign to slots
        slot_to_pid = assign_lineup_to_slots(player_ids)
        
        # Build row in slot order
        row = []
        for slot in DK_SLOTS:
            pid = slot_to_pid.get(slot)
            if pid and pid in pid2player:
                p = pid2player[pid]
                row.append(f"{p.name} ({p.dk_id})")
            else:
                row.append("")
        writer.writerow(row)

    csv_content = output.getvalue()
    filename = f"lineups_{job.game_date}_{job.job_id[:8]}.csv"

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
            LineupRow(lineup_id=lu.get("lineup_id", i), player_ids=lu.get("player_ids", []))
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
