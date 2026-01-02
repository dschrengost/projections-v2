"""Contest simulation job management.

Thread-safe job store for running contest simulations asynchronously,
following the pattern in optimizer_service.py.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .scoring_models import ContestSimResult

__all__ = [
    "SimJob",
    "SimJobStore",
    "get_sim_job_store",
]


@dataclass
class SimJob:
    """Tracks state of a contest simulation job."""

    job_id: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    game_date: str
    request: Dict[str, Any]

    # Progress tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results (set on completion)
    result: Optional[ContestSimResult] = None
    build_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-friendly dict."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "game_date": self.game_date,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "wall_time_sec": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at and self.started_at
                else None
            ),
            "error": self.error,
            "build_id": self.build_id,
        }


class SimJobStore:
    """Thread-safe in-memory job store for contest simulations."""

    def __init__(self, max_jobs: int = 50):
        self._jobs: Dict[str, SimJob] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def create(
        self,
        game_date: str,
        request: Dict[str, Any],
    ) -> SimJob:
        """Create a new pending job."""
        job = SimJob(
            job_id=str(uuid4()),
            status="pending",
            created_at=datetime.utcnow(),
            game_date=game_date,
            request=request,
        )
        with self._lock:
            # Evict oldest jobs if at capacity
            if len(self._jobs) >= self._max_jobs:
                oldest = min(self._jobs.values(), key=lambda j: j.created_at)
                del self._jobs[oldest.job_id]
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[SimJob]:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> Optional[SimJob]:
        """Update job attributes."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    if hasattr(job, k):
                        setattr(job, k, v)
            return job

    def list_jobs(self, limit: int = 20) -> List[SimJob]:
        """List recent jobs, newest first."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]


# Global job store singleton
_sim_job_store: Optional[SimJobStore] = None


def get_sim_job_store() -> SimJobStore:
    """Get the global sim job store instance."""
    global _sim_job_store
    if _sim_job_store is None:
        _sim_job_store = SimJobStore()
    return _sim_job_store
