"""Evaluation metrics API for prediction accuracy tracking."""

from fastapi import APIRouter, Query
from pathlib import Path
import json
import os
from typing import Any

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

EVAL_PATH = Path(
    os.environ.get(
        "EVAL_DATA_PATH",
        "/home/daniel/projections-data/reports/eval_latest.json",
    )
)


def _compute_summary(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate summary metrics from evaluation data."""
    if not data:
        return {}

    def safe_avg(key: str) -> float | None:
        values = [d[key] for d in data if d.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    def safe_sum(key: str) -> int:
        return sum(d.get(key, 0) or 0 for d in data)

    return {
        # FPTS metrics
        "avg_fpts_mae": safe_avg("fpts_mae"),
        "avg_minutes_mae": safe_avg("minutes_mae"),
        "avg_coverage_80": safe_avg("coverage_80"),
        "avg_coverage_90": safe_avg("coverage_90"),
        "avg_bias": safe_avg("bias"),
        "total_missed": safe_sum("missed"),
        "total_false_preds": safe_sum("false_preds"),
        # Ownership metrics
        "avg_own_mae": safe_avg("own_mae"),
        "avg_own_corr": safe_avg("own_corr"),
        "avg_chalk_top5_acc": safe_avg("chalk_top5_acc"),
        "avg_own_bias": safe_avg("own_bias"),
        # Counts
        "dates_evaluated": len(data),
        "total_players_matched": safe_sum("players_matched"),
    }


@router.get("")
def get_evaluation(days: int = Query(7, ge=1, le=30)):
    """Get evaluation metrics for the last N days."""
    if not EVAL_PATH.exists():
        return {"metrics": [], "summary": {}, "days": days, "end_date": None}

    try:
        data = json.loads(EVAL_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"metrics": [], "summary": {}, "days": days, "end_date": None}

    if not data:
        return {"metrics": [], "summary": {}, "days": days, "end_date": None}

    # Sort by date descending, take last N days
    sorted_data = sorted(data, key=lambda x: x.get("date", ""), reverse=True)[:days]

    # Compute summary aggregates
    summary = _compute_summary(sorted_data)

    # Return metrics in ascending date order for charts
    metrics = sorted(sorted_data, key=lambda x: x.get("date", ""))

    return {
        "days": days,
        "end_date": sorted_data[0].get("date") if sorted_data else None,
        "metrics": metrics,
        "summary": summary,
    }
