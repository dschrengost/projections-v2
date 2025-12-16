"""FastAPI router for contest analysis endpoints."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from projections.api.contest_service import (
    ContestDetail,
    ContestSummary,
    PlayerPair,
    UserLineupEntry,
    find_user_lineups,
    get_contest_detail,
    get_pairs_analysis,
    list_available_dates,
    list_contests_for_date,
)

router = APIRouter()


class ContestDatesResponse(BaseModel):
    dates: list[str]


class ContestListResponse(BaseModel):
    date: str
    contests: list[ContestSummary]
    total_contests: int


class PairsResponse(BaseModel):
    contest_id: str
    top_n_analyzed: int
    pairs: list[PlayerPair]


class UserLineupsResponse(BaseModel):
    date: str
    pattern_used: str
    entries_found: int
    entries: list[UserLineupEntry]
    avg_finish_pct: float | None
    best_finish_rank: int | None
    best_finish_pct: float | None


@router.get("/dates", response_model=ContestDatesResponse)
def get_contest_dates(limit: int = Query(default=30, ge=1, le=100)):
    """Return available contest dates (most recent first)."""
    dates = list_available_dates(limit=limit)
    return ContestDatesResponse(dates=dates)


@router.get("/contests", response_model=ContestListResponse)
def get_contests(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    """List all contests for a given date with metadata."""
    contests = list_contests_for_date(date)
    return ContestListResponse(
        date=date,
        contests=contests,
        total_contests=len(contests),
    )


@router.get("/contests/{contest_id}", response_model=ContestDetail)
def get_contest(
    contest_id: str,
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
):
    """Get detailed metrics for a specific contest."""
    detail = get_contest_detail(date, contest_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Contest not found")
    return detail


@router.get("/contests/{contest_id}/pairs", response_model=PairsResponse)
def get_contest_pairs(
    contest_id: str,
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    top_n: int = Query(default=10, ge=1, le=100, description="Top N lineups to analyze"),
):
    """Get player pair correlation analysis for top finishers."""
    pairs = get_pairs_analysis(date, contest_id, top_n=top_n)
    return PairsResponse(
        contest_id=contest_id,
        top_n_analyzed=top_n,
        pairs=pairs,
    )


@router.get("/user-lineups", response_model=UserLineupsResponse)
def get_user_lineups(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    pattern: str = Query(..., description="Username pattern to search for"),
):
    """Find user's lineups across all contests for a date."""
    if not pattern or len(pattern) < 2:
        raise HTTPException(
            status_code=400, detail="Pattern must be at least 2 characters"
        )

    entries = find_user_lineups(date, pattern)

    avg_finish_pct = None
    best_finish_rank = None
    best_finish_pct = None

    if entries:
        avg_finish_pct = sum(e.pct_finish for e in entries) / len(entries)
        best_entry = min(entries, key=lambda e: e.rank)
        best_finish_rank = best_entry.rank
        best_finish_pct = best_entry.pct_finish

    return UserLineupsResponse(
        date=date,
        pattern_used=pattern,
        entries_found=len(entries),
        entries=entries,
        avg_finish_pct=round(avg_finish_pct, 2) if avg_finish_pct else None,
        best_finish_rank=best_finish_rank,
        best_finish_pct=round(best_finish_pct, 2) if best_finish_pct else None,
    )
