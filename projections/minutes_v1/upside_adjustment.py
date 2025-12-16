"""Upside adjustment heuristics for DFS Monte Carlo simulation.

This module provides post-processing adjustments to quantile predictions
to improve P90 coverage for fantasy sports applications, where capturing
upside scenarios is critical.

The raw LightGBM quantile model tends to under-predict P90 (only ~66% coverage
vs 90% target) because:
1. Bench players have high variance not captured in features
2. Injury returns/lineup changes create regime shifts
3. Narrow intervals signal false confidence

The adjustments widen intervals asymmetrically (more on upside) with
stronger adjustments for bench players and uncertain situations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class UpsideConfig:
    """Configuration for upside adjustment heuristics."""

    # Base multipliers for upside half of interval (p90 - p50)
    starter_upside_mult: float = 1.15
    bench_upside_mult: float = 1.6

    # Extra multiplier for narrow intervals (<threshold)
    narrow_interval_threshold: float = 8.0
    narrow_starter_mult: float = 1.1
    narrow_bench_mult: float = 1.25

    # Injury ramp adjustment (games_since_return <= threshold)
    ramp_games_threshold: int = 5
    ramp_mult: float = 1.3

    # Minimum upside floors by predicted minutes bucket
    deep_bench_p50_max: float = 10.0
    deep_bench_min_upside: float = 8.0
    mid_bench_p50_min: float = 10.0
    mid_bench_p50_max: float = 20.0
    mid_bench_min_upside: float = 10.0

    # Global constraints
    min_upside_floor: float = 4.0  # P90 >= p50 + this
    max_minutes: float = 48.0

    # P10 adjustment for widened players
    p10_shrink_factor: float = 0.92
    p10_shrink_threshold: float = 1.1  # Only shrink if p90 widened by > 10%

    # Column names
    p10_col: str = "minutes_p10"
    p50_col: str = "minutes_p50"
    p90_col: str = "minutes_p90"
    starter_col: str = "starter_flag"
    games_since_return_col: str = "games_since_return"


def apply_upside_adjustment(
    df: pd.DataFrame,
    config: UpsideConfig | None = None,
) -> pd.DataFrame:
    """Apply upside adjustment heuristics to quantile predictions.

    This function widens the P90 predictions asymmetrically to improve
    P90 coverage for Monte Carlo simulation, particularly for bench players
    and uncertain situations.

    Args:
        df: DataFrame with quantile predictions (minutes_p10, minutes_p50, minutes_p90)
        config: Configuration for adjustment parameters

    Returns:
        DataFrame with adjusted columns: minutes_p10_adj, minutes_p50_adj, minutes_p90_adj
    """
    cfg = config or UpsideConfig()
    out = df.copy()

    # Get base predictions
    p10 = pd.to_numeric(out[cfg.p10_col], errors="coerce").fillna(0.0).to_numpy()
    p50 = pd.to_numeric(out[cfg.p50_col], errors="coerce").fillna(0.0).to_numpy()
    p90 = pd.to_numeric(out[cfg.p90_col], errors="coerce").fillna(0.0).to_numpy()

    # Derived quantities
    interval_width = p90 - p10
    upside_half = p90 - p50

    # Starter mask
    starter_series = out.get(cfg.starter_col)
    if starter_series is not None:
        is_starter = starter_series.fillna(0).astype(bool).to_numpy()
    else:
        is_starter = np.zeros(len(out), dtype=bool)
    is_bench = ~is_starter

    # Initialize adjusted values
    p90_adj = p90.copy()
    p10_adj = p10.copy()

    # Step 1: Apply base multiplier (different for starters vs bench)
    starter_upside = p50[is_starter] + upside_half[is_starter] * cfg.starter_upside_mult
    bench_upside = p50[is_bench] + upside_half[is_bench] * cfg.bench_upside_mult
    p90_adj[is_starter] = starter_upside
    p90_adj[is_bench] = bench_upside

    # Step 2: Extra widening for narrow intervals
    narrow_mask = interval_width < cfg.narrow_interval_threshold
    narrow_bench = narrow_mask & is_bench
    narrow_starter = narrow_mask & is_starter
    p90_adj[narrow_bench] *= cfg.narrow_bench_mult
    p90_adj[narrow_starter] *= cfg.narrow_starter_mult

    # Step 3: Injury ramp adjustment
    ramp_series = out.get(cfg.games_since_return_col)
    if ramp_series is not None:
        games_since = pd.to_numeric(ramp_series, errors="coerce").fillna(9999).to_numpy()
        ramp_mask = games_since <= cfg.ramp_games_threshold
        p90_adj[ramp_mask] = p50[ramp_mask] + (p90_adj[ramp_mask] - p50[ramp_mask]) * cfg.ramp_mult

    # Step 4: Deep bench minimum upside
    deep_bench = (p50 < cfg.deep_bench_p50_max) & is_bench
    p90_adj[deep_bench] = np.maximum(
        p90_adj[deep_bench],
        p50[deep_bench] + cfg.deep_bench_min_upside,
    )

    # Step 5: Mid bench minimum upside
    mid_bench = (p50 >= cfg.mid_bench_p50_min) & (p50 < cfg.mid_bench_p50_max) & is_bench
    p90_adj[mid_bench] = np.maximum(
        p90_adj[mid_bench],
        p50[mid_bench] + cfg.mid_bench_min_upside,
    )

    # Step 6: Global floor
    p90_adj = np.maximum(p90_adj, p50 + cfg.min_upside_floor)

    # Step 7: Cap at max minutes
    p90_adj = np.minimum(p90_adj, cfg.max_minutes)

    # Step 8: Slight P10 shrink for heavily widened players (asymmetry)
    widened_mask = p90_adj > p90 * cfg.p10_shrink_threshold
    p10_adj[widened_mask] = p10[widened_mask] * cfg.p10_shrink_factor
    p10_adj = np.maximum(p10_adj, 0.0)

    # Ensure monotonicity: p10 <= p50 <= p90
    p10_adj = np.minimum(p10_adj, p50)
    p90_adj = np.maximum(p90_adj, p50)

    # Write adjusted columns
    out["minutes_p10_adj"] = p10_adj
    out["minutes_p50_adj"] = p50  # P50 unchanged
    out["minutes_p90_adj"] = p90_adj

    return out


def compute_adjustment_stats(df: pd.DataFrame, actual_col: str = "actual_minutes") -> dict:
    """Compute coverage statistics before/after adjustment.

    Args:
        df: DataFrame with original and adjusted predictions
        actual_col: Column name for actual minutes

    Returns:
        Dictionary with coverage and width statistics
    """
    actual = pd.to_numeric(df[actual_col], errors="coerce").to_numpy()
    valid = ~np.isnan(actual)
    actual = actual[valid]

    stats = {}
    for suffix, p10_c, p90_c in [
        ("raw", "minutes_p10", "minutes_p90"),
        ("adj", "minutes_p10_adj", "minutes_p90_adj"),
    ]:
        if p10_c not in df.columns or p90_c not in df.columns:
            continue
        p10 = df.loc[valid, p10_c].to_numpy()
        p90 = df.loc[valid, p90_c].to_numpy()

        stats[f"p10_coverage_{suffix}"] = float((actual < p10).mean())
        stats[f"p90_coverage_{suffix}"] = float((actual <= p90).mean())
        stats[f"inside_coverage_{suffix}"] = float(((actual >= p10) & (actual <= p90)).mean())
        stats[f"interval_width_{suffix}"] = float((p90 - p10).mean())

    return stats
