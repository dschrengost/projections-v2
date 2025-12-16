"""Fantasy scoring utilities for per-minute projections."""

from __future__ import annotations

from typing import Mapping


def _safe_value(data: Mapping[str, float | int | None], key: str) -> float:
    value = data.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def draftkings_fantasy_points(stats: Mapping[str, float | int | None]) -> float:
    """Compute DraftKings fantasy points for a single player stat line."""

    points = _safe_value(stats, "points")
    rebounds = _safe_value(stats, "rebounds_total")
    assists = _safe_value(stats, "assists")
    steals = _safe_value(stats, "steals")
    blocks = _safe_value(stats, "blocks")
    turnovers = _safe_value(stats, "turnovers")
    threes = _safe_value(stats, "three_pointers_made")

    base = (
        points
        + 1.25 * rebounds
        + 1.5 * assists
        + 2.0 * steals
        + 2.0 * blocks
        - 0.5 * turnovers
        + 0.5 * threes
    )

    counting_stats = [points, rebounds, assists, steals, blocks]
    qualifying = sum(value >= 10.0 for value in counting_stats)
    double_double = 1.5 if qualifying >= 2 else 0.0
    triple_double = 3.0 if qualifying >= 3 else 0.0
    return base + double_double + triple_double


SCORING_SYSTEMS: dict[str, callable[[Mapping[str, float | int | None]], float]] = {
    "dk": draftkings_fantasy_points,
}


__all__ = ["SCORING_SYSTEMS", "draftkings_fantasy_points"]
