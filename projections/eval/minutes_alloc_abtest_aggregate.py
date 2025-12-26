"""Aggregation module for multi-slate A/B/C test results.

This module aggregates per-slate A/B/C test results to produce:
- Overall metrics across all slates
- Per-bucket accuracy breakdowns
- Per-slate comparison table
- Stability and consistency analysis
- Stratified aggregation by quality tier (clean/degraded/skipped)

Quality Tiers:
- clean: passes integrity + missing_feature_frac <= 0.02
- degraded: passes integrity + 0.02 < missing_feature_frac <= 0.10
- skipped: didn't run; only counted in skip report
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Quality tier thresholds
MISSING_FEATURE_CLEAN_THRESHOLD = 0.02
MISSING_FEATURE_SKIP_THRESHOLD = 0.10


@dataclass
class SlateMetrics:
    """Per-slate metrics extracted from summary.json."""

    game_date: str
    n_players: int = 0
    n_teams: int = 0
    n_games: int = 0

    # Quality tier
    quality_tier: str = "unknown"  # clean, degraded, skipped
    missing_feature_frac: float = 0.0
    n_expected_features: int = 0
    n_missing_features: int = 0
    skip_reason: str | None = None

    # Accuracy - A, B, C, D, E, F, M, N
    mae_A: float | None = None
    mae_B: float | None = None
    mae_C: float | None = None
    mae_D: float | None = None  # D with default alpha=0.7
    mae_E: float | None = None  # E (FRINGE_ONLY_ALPHA)
    mae_F: float | None = None  # F (POWER_POSTPROCESS)
    mae_M: float | None = None  # M (MIXTURE_SHARES_SCALED)
    mae_N: float | None = None  # N (MIXTURE_BENCHPOOL)
    mae_D_best: float | None = None  # D with best alpha
    best_D_alpha: float | None = None
    mae_F_best: float | None = None  # F with best p
    best_F_p: float | None = None
    mae_N_best: float | None = None  # N with best params
    best_N_bench_pool: float | None = None
    best_N_core_k: int | None = None
    rmse_A: float | None = None
    rmse_B: float | None = None
    rmse_C: float | None = None
    rmse_D: float | None = None
    rmse_E: float | None = None
    rmse_F: float | None = None
    rmse_M: float | None = None
    rmse_N: float | None = None
    mae_P: float | None = None  # P (PRODUCTION BASELINE)
    rmse_P: float | None = None

    # Per-bucket MAE - A, B, C, D, E, F, M, N, P
    mae_0_10_A: float | None = None
    mae_0_10_B: float | None = None
    mae_0_10_C: float | None = None
    mae_0_10_D: float | None = None
    mae_0_10_E: float | None = None
    mae_0_10_F: float | None = None
    mae_0_10_M: float | None = None
    mae_0_10_N: float | None = None
    mae_10_20_A: float | None = None
    mae_10_20_B: float | None = None
    mae_10_20_C: float | None = None
    mae_10_20_D: float | None = None
    mae_10_20_E: float | None = None
    mae_10_20_F: float | None = None
    mae_10_20_M: float | None = None
    mae_10_20_N: float | None = None
    mae_20_30_A: float | None = None
    mae_20_30_B: float | None = None
    mae_20_30_C: float | None = None
    mae_20_30_D: float | None = None
    mae_20_30_E: float | None = None
    mae_20_30_F: float | None = None
    mae_20_30_M: float | None = None
    mae_20_30_N: float | None = None
    mae_30_plus_A: float | None = None
    mae_30_plus_B: float | None = None
    mae_30_plus_C: float | None = None
    mae_30_plus_D: float | None = None
    mae_30_plus_E: float | None = None
    mae_30_plus_F: float | None = None
    mae_30_plus_M: float | None = None
    mae_30_plus_N: float | None = None
    mae_0_10_P: float | None = None
    mae_10_20_P: float | None = None
    mae_20_30_P: float | None = None
    mae_30_plus_P: float | None = None

    # Realism - A, B, C, D, E, F, M, N, P
    top5_sum_A: float = 0.0
    top5_sum_B: float = 0.0
    top5_sum_C: float = 0.0
    top5_sum_D: float = 0.0
    top5_sum_E: float = 0.0
    top5_sum_F: float = 0.0
    top5_sum_M: float | None = None  # None when allocator_M_status != "success"
    top5_sum_N: float | None = None  # None when allocator_N_status != "success"
    top5_sum_P: float | None = None  # None when allocator_P_status != "success"
    max_minutes_A: float = 0.0
    max_minutes_B: float = 0.0
    max_minutes_C: float = 0.0
    max_minutes_D: float = 0.0
    max_minutes_E: float = 0.0
    max_minutes_F: float = 0.0
    max_minutes_M: float | None = None  # None when allocator_M_status != "success"
    max_minutes_N: float | None = None  # None when allocator_N_status != "success"
    max_minutes_P: float | None = None  # None when allocator_P_status != "success"
    gini_A: float = 0.0
    gini_B: float = 0.0
    gini_C: float = 0.0
    gini_D: float = 0.0
    gini_E: float = 0.0
    gini_F: float = 0.0
    gini_M: float | None = None  # None when allocator_M_status != "success"
    gini_N: float | None = None  # None when allocator_N_status != "success"
    gini_P: float | None = None  # None when allocator_P_status != "success"
    hhi_A: float = 0.0
    hhi_B: float = 0.0
    hhi_C: float = 0.0
    hhi_D: float = 0.0
    hhi_E: float = 0.0
    hhi_F: float = 0.0
    hhi_M: float | None = None  # None when allocator_M_status != "success"
    hhi_N: float | None = None  # None when allocator_N_status != "success"
    hhi_P: float | None = None  # None when allocator_P_status != "success"
    top6_share_A: float = 0.0
    top6_share_B: float = 0.0
    top6_share_C: float = 0.0
    top6_share_D: float = 0.0
    top6_share_E: float = 0.0
    top6_share_F: float = 0.0
    top6_share_M: float | None = None  # None when allocator_M_status != "success"
    top6_share_N: float | None = None  # None when allocator_N_status != "success"
    top6_share_P: float | None = None  # None when allocator_P_status != "success"
    top8_share_A: float = 0.0
    top8_share_B: float = 0.0
    top8_share_C: float = 0.0
    top8_share_D: float = 0.0
    top8_share_E: float = 0.0
    top8_share_F: float = 0.0
    top8_share_M: float | None = None  # None when allocator_M_status != "success"
    top8_share_N: float | None = None  # None when allocator_N_status != "success"
    top8_share_P: float | None = None  # None when allocator_P_status != "success"
    sixth_man_mae_A: float | None = None
    sixth_man_mae_B: float | None = None
    sixth_man_mae_C: float | None = None
    sixth_man_mae_D: float | None = None
    sixth_man_mae_E: float | None = None
    sixth_man_mae_F: float | None = None
    sixth_man_mae_M: float | None = None  # None when allocator_M_status != "success"
    sixth_man_mae_N: float | None = None  # None when allocator_N_status != "success"
    sixth_man_mae_P: float | None = None  # None when allocator_P_status != "success"
    roster_size_A: float = 0.0
    roster_size_B: float = 0.0
    roster_size_C: float = 0.0
    roster_size_D: float = 0.0
    roster_size_E: float = 0.0
    roster_size_F: float = 0.0
    roster_size_M: float | None = None  # None when allocator_M_status != "success"
    roster_size_N: float | None = None  # None when allocator_N_status != "success"
    roster_size_P: float | None = None  # None when allocator_P_status != "success"

    # Pathology
    bench_crush_A: float = 0.0
    bench_crush_B: float = 0.0
    bench_crush_C: float = 0.0
    bench_crush_D: float = 0.0
    bench_crush_E: float = 0.0
    bench_crush_F: float = 0.0
    bench_crush_M: float | None = None  # None when allocator_M_status != "success"
    bench_crush_N: float | None = None  # None when allocator_N_status != "success"
    bench_crush_P: float | None = None  # None when allocator_P_status != "success"

    # Sum compliance
    team_sum_error_max_A: float = 0.0
    team_sum_error_max_B: float = 0.0
    team_sum_error_max_C: float = 0.0
    team_sum_error_max_D: float = 0.0
    team_sum_error_max_E: float = 0.0
    team_sum_error_max_F: float = 0.0
    team_sum_error_max_M: float | None = None  # None when allocator_M_status != "success"
    team_sum_error_max_N: float | None = None  # None when allocator_N_status != "success"
    team_sum_error_max_P: float | None = None  # None when allocator_P_status != "success"

    # Ordering accuracy for [10, 30] range
    ordering_10_30_A: float | None = None
    ordering_10_30_B: float | None = None
    ordering_10_30_C: float | None = None
    ordering_10_30_D: float | None = None
    ordering_10_30_E: float | None = None
    ordering_10_30_F: float | None = None
    ordering_10_30_M: float | None = None
    ordering_10_30_N: float | None = None
    ordering_10_30_P: float | None = None

    # Status
    has_labels: bool = False
    error: str | None = None


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all slates."""

    # Coverage
    total_slates: int = 0
    processed_slates: int = 0
    clean_slates: int = 0
    degraded_slates: int = 0
    skipped_slates: int = 0
    slates_with_labels: int = 0

    # Skip reasons breakdown
    skip_reasons: dict[str, int] = field(default_factory=dict)

    # Overall accuracy (mean across slates) - A, B, C, D, E, F, M, N
    mae_A_mean: float = float("nan")
    mae_B_mean: float = float("nan")
    mae_C_mean: float = float("nan")
    mae_D_mean: float = float("nan")
    mae_E_mean: float = float("nan")
    mae_F_mean: float = float("nan")
    mae_M_mean: float = float("nan")
    mae_N_mean: float = float("nan")
    mae_D_best_mean: float = float("nan")
    best_D_alpha_mode: float | None = None  # Most common best alpha
    mae_F_best_mean: float = float("nan")
    best_F_p_mode: float | None = None  # Most common best p
    mae_N_best_mean: float = float("nan")
    best_N_bench_pool_mode: float | None = None  # Most common best bench_pool
    best_N_core_k_mode: int | None = None  # Most common best core_k
    mae_A_median: float = float("nan")
    mae_B_median: float = float("nan")
    mae_C_median: float = float("nan")
    mae_D_median: float = float("nan")
    mae_E_median: float = float("nan")
    mae_F_median: float = float("nan")
    mae_M_median: float = float("nan")
    mae_N_median: float = float("nan")
    rmse_A_mean: float = float("nan")
    rmse_B_mean: float = float("nan")
    rmse_C_mean: float = float("nan")
    rmse_D_mean: float = float("nan")
    rmse_E_mean: float = float("nan")
    rmse_F_mean: float = float("nan")
    rmse_M_mean: float = float("nan")
    rmse_N_mean: float = float("nan")
    mae_P_mean: float = float("nan")
    mae_P_median: float = float("nan")
    rmse_P_mean: float = float("nan")

    # Per-bucket MAE (mean across slates)
    mae_by_bucket_A: dict[str, float] = field(default_factory=dict)
    mae_by_bucket_B: dict[str, float] = field(default_factory=dict)
    mae_by_bucket_C: dict[str, float] = field(default_factory=dict)
    mae_by_bucket_D: dict[str, float] = field(default_factory=dict)
    mae_by_bucket_E: dict[str, float] = field(default_factory=dict)
    mae_by_bucket_F: dict[str, float] = field(default_factory=dict)
    mae_by_bucket_M: dict[str, float] = field(default_factory=dict)
    mae_by_bucket_N: dict[str, float] = field(default_factory=dict)
    mae_by_bucket_P: dict[str, float] = field(default_factory=dict)

    # Realism (mean / p90) - A, B, C, D, E, F, M, N, P
    top5_sum_A_mean: float = float("nan")
    top5_sum_B_mean: float = float("nan")
    top5_sum_C_mean: float = float("nan")
    top5_sum_D_mean: float = float("nan")
    top5_sum_E_mean: float = float("nan")
    top5_sum_F_mean: float = float("nan")
    top5_sum_M_mean: float = float("nan")
    top5_sum_N_mean: float = float("nan")
    top5_sum_P_mean: float = float("nan")
    top5_sum_A_p90: float = float("nan")
    top5_sum_B_p90: float = float("nan")
    top5_sum_C_p90: float = float("nan")
    top5_sum_D_p90: float = float("nan")
    top5_sum_E_p90: float = float("nan")
    top5_sum_F_p90: float = float("nan")
    top5_sum_M_p90: float = float("nan")
    top5_sum_N_p90: float = float("nan")
    top5_sum_P_p90: float = float("nan")
    max_minutes_A_mean: float = float("nan")
    max_minutes_B_mean: float = float("nan")
    max_minutes_C_mean: float = float("nan")
    max_minutes_D_mean: float = float("nan")
    max_minutes_E_mean: float = float("nan")
    max_minutes_F_mean: float = float("nan")
    max_minutes_M_mean: float = float("nan")
    max_minutes_N_mean: float = float("nan")
    max_minutes_P_mean: float = float("nan")
    max_minutes_A_p90: float = float("nan")
    max_minutes_B_p90: float = float("nan")
    max_minutes_C_p90: float = float("nan")
    max_minutes_D_p90: float = float("nan")
    max_minutes_E_p90: float = float("nan")
    max_minutes_F_p90: float = float("nan")
    max_minutes_M_p90: float = float("nan")
    max_minutes_N_p90: float = float("nan")
    max_minutes_P_p90: float = float("nan")
    gini_A_mean: float = float("nan")
    gini_B_mean: float = float("nan")
    gini_C_mean: float = float("nan")
    gini_D_mean: float = float("nan")
    gini_E_mean: float = float("nan")
    gini_F_mean: float = float("nan")
    gini_M_mean: float = float("nan")
    gini_N_mean: float = float("nan")
    gini_P_mean: float = float("nan")
    hhi_A_mean: float = float("nan")
    hhi_B_mean: float = float("nan")
    hhi_C_mean: float = float("nan")
    hhi_D_mean: float = float("nan")
    hhi_E_mean: float = float("nan")
    hhi_F_mean: float = float("nan")
    hhi_M_mean: float = float("nan")
    hhi_N_mean: float = float("nan")
    hhi_P_mean: float = float("nan")
    top6_share_A_mean: float = float("nan")
    top6_share_B_mean: float = float("nan")
    top6_share_C_mean: float = float("nan")
    top6_share_D_mean: float = float("nan")
    top6_share_E_mean: float = float("nan")
    top6_share_F_mean: float = float("nan")
    top6_share_M_mean: float = float("nan")
    top6_share_N_mean: float = float("nan")
    top6_share_P_mean: float = float("nan")
    top8_share_A_mean: float = float("nan")
    top8_share_B_mean: float = float("nan")
    top8_share_C_mean: float = float("nan")
    top8_share_D_mean: float = float("nan")
    top8_share_E_mean: float = float("nan")
    top8_share_F_mean: float = float("nan")
    top8_share_M_mean: float = float("nan")
    top8_share_N_mean: float = float("nan")
    top8_share_P_mean: float = float("nan")
    sixth_man_mae_A_mean: float = float("nan")
    sixth_man_mae_B_mean: float = float("nan")
    sixth_man_mae_C_mean: float = float("nan")
    sixth_man_mae_D_mean: float = float("nan")
    sixth_man_mae_E_mean: float = float("nan")
    sixth_man_mae_F_mean: float = float("nan")
    sixth_man_mae_M_mean: float = float("nan")
    sixth_man_mae_N_mean: float = float("nan")
    sixth_man_mae_P_mean: float = float("nan")
    roster_size_A_mean: float = float("nan")
    roster_size_B_mean: float = float("nan")
    roster_size_C_mean: float = float("nan")
    roster_size_D_mean: float = float("nan")
    roster_size_E_mean: float = float("nan")
    roster_size_F_mean: float = float("nan")
    roster_size_M_mean: float = float("nan")
    roster_size_N_mean: float = float("nan")
    roster_size_P_mean: float = float("nan")

    # Pathology - A, B, C, D, E, F, M, N, P
    bench_crush_A_mean: float = float("nan")
    bench_crush_B_mean: float = float("nan")
    bench_crush_C_mean: float = float("nan")
    bench_crush_D_mean: float = float("nan")
    bench_crush_E_mean: float = float("nan")
    bench_crush_F_mean: float = float("nan")
    bench_crush_M_mean: float = float("nan")
    bench_crush_N_mean: float = float("nan")
    bench_crush_P_mean: float = float("nan")
    bench_crush_A_p90: float = float("nan")
    bench_crush_B_p90: float = float("nan")
    bench_crush_C_p90: float = float("nan")
    bench_crush_D_p90: float = float("nan")
    bench_crush_E_p90: float = float("nan")
    bench_crush_F_p90: float = float("nan")
    bench_crush_M_p90: float = float("nan")
    bench_crush_N_p90: float = float("nan")
    bench_crush_P_p90: float = float("nan")
    pct_teams_top5_lt_150_A: float = float("nan")
    pct_teams_top5_lt_150_B: float = float("nan")
    pct_teams_top5_lt_150_C: float = float("nan")
    pct_teams_top5_lt_150_D: float = float("nan")
    pct_teams_top5_lt_150_E: float = float("nan")
    pct_teams_top5_lt_150_F: float = float("nan")
    pct_teams_top5_lt_150_M: float = float("nan")
    pct_teams_top5_lt_150_N: float = float("nan")
    pct_teams_max_lt_30_A: float = float("nan")
    pct_teams_max_lt_30_B: float = float("nan")
    pct_teams_max_lt_30_C: float = float("nan")
    pct_teams_max_lt_30_D: float = float("nan")
    pct_teams_max_lt_30_E: float = float("nan")
    pct_teams_max_lt_30_F: float = float("nan")
    pct_teams_max_lt_30_M: float = float("nan")
    pct_teams_max_lt_30_N: float = float("nan")
    pct_teams_roster_ge_14_A: float = float("nan")
    pct_teams_roster_ge_14_B: float = float("nan")
    pct_teams_roster_ge_14_C: float = float("nan")
    pct_teams_roster_ge_14_D: float = float("nan")
    pct_teams_roster_ge_14_E: float = float("nan")
    pct_teams_roster_ge_14_F: float = float("nan")
    pct_teams_roster_ge_14_M: float = float("nan")
    pct_teams_roster_ge_14_N: float = float("nan")

    # Stability / consistency - A, C, D, E, F, M, N vs B
    mae_delta_AB_mean: float = float("nan")  # MAE_A - MAE_B
    mae_delta_AB_std: float = float("nan")
    mae_delta_CB_mean: float = float("nan")  # MAE_C - MAE_B
    mae_delta_CB_std: float = float("nan")
    mae_delta_DB_mean: float = float("nan")  # MAE_D - MAE_B
    mae_delta_DB_std: float = float("nan")
    mae_delta_EB_mean: float = float("nan")  # MAE_E - MAE_B
    mae_delta_EB_std: float = float("nan")
    mae_delta_FB_mean: float = float("nan")  # MAE_F - MAE_B
    mae_delta_FB_std: float = float("nan")
    mae_delta_MB_mean: float = float("nan")  # MAE_M - MAE_B
    mae_delta_MB_std: float = float("nan")
    mae_delta_NB_mean: float = float("nan")  # MAE_N - MAE_B
    mae_delta_NB_std: float = float("nan")
    pct_slates_A_wins_mae: float = float("nan")
    pct_slates_C_wins_mae: float = float("nan")
    pct_slates_D_wins_mae: float = float("nan")
    pct_slates_E_wins_mae: float = float("nan")
    pct_slates_F_wins_mae: float = float("nan")
    pct_slates_M_wins_mae: float = float("nan")
    pct_slates_N_wins_mae: float = float("nan")
    pct_slates_A_wins_top5: float = float("nan")
    pct_slates_C_wins_top5: float = float("nan")
    pct_slates_D_wins_top5: float = float("nan")
    pct_slates_E_wins_top5: float = float("nan")
    pct_slates_F_wins_top5: float = float("nan")
    pct_slates_M_wins_top5: float = float("nan")
    pct_slates_N_wins_top5: float = float("nan")
    pct_slates_A_wins_both: float = float("nan")
    pct_slates_C_wins_both: float = float("nan")
    pct_slates_D_wins_both: float = float("nan")
    pct_slates_E_wins_both: float = float("nan")
    pct_slates_F_wins_both: float = float("nan")
    pct_slates_M_wins_both: float = float("nan")
    pct_slates_N_wins_both: float = float("nan")
    pct_slates_P_wins_mae: float = float("nan")
    pct_slates_P_wins_top5: float = float("nan")
    pct_slates_P_wins_both: float = float("nan")

    # Stability vs P (Production Baseline)
    mae_delta_AP_mean: float = float("nan")
    mae_delta_DP_mean: float = float("nan")
    mae_delta_EP_mean: float = float("nan")
    mae_delta_FP_mean: float = float("nan")
    mae_delta_MP_mean: float = float("nan")
    mae_delta_NP_mean: float = float("nan")

    # Ordering accuracy for [10, 30] range (mean across slates)
    ordering_10_30_A_mean: float = float("nan")
    ordering_10_30_B_mean: float = float("nan")
    ordering_10_30_C_mean: float = float("nan")
    ordering_10_30_D_mean: float = float("nan")
    ordering_10_30_E_mean: float = float("nan")
    ordering_10_30_F_mean: float = float("nan")
    ordering_10_30_M_mean: float = float("nan")
    ordering_10_30_N_mean: float = float("nan")
    ordering_10_30_P_mean: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_slate_metrics(summary_path: Path) -> SlateMetrics:
    """Load metrics from a single slate's summary.json."""
    data = json.loads(summary_path.read_text())

    metrics = SlateMetrics(game_date=data.get("game_date", "unknown"))

    # Quality tier and integrity
    metrics.quality_tier = data.get("quality_tier", "unknown")
    metrics.missing_feature_frac = data.get("missing_feature_frac", 0.0)
    metrics.n_expected_features = data.get("n_expected_features", 0)
    metrics.n_missing_features = data.get("n_missing_features", 0)
    metrics.skip_reason = data.get("skip_reason")

    # Integrity counts
    integrity = data.get("integrity_counts", {})
    metrics.n_games = integrity.get("n_games", 0)
    metrics.n_teams = integrity.get("n_teams", 0)
    metrics.n_players = integrity.get("n_players", 0)

    # Check allocator M status - if not success, M metrics stay as None
    allocator_M_success = data.get("allocator_M_status") == "success"

    # Check allocator N status - if not success, N metrics stay as None
    allocator_N_success = data.get("allocator_N_status") == "success"

    # Check allocator P status - if not success, P metrics stay as None
    allocator_P_success = data.get("allocator_P_status") == "success"

    # Core metrics from 'metrics' section
    m = data.get("metrics", {})
    if not m:
        # Legacy format fallback
        m = data

    metrics.n_players = m.get("n_players", metrics.n_players)
    metrics.n_teams = m.get("n_teams", metrics.n_teams)

    # Accuracy - A, B, C, D, E, F, M, N, P
    metrics.mae_A = m.get("mae_A")
    metrics.mae_B = m.get("mae_B")
    metrics.mae_C = m.get("mae_C")
    metrics.mae_D = m.get("mae_D")  # D with alpha=0.7
    metrics.mae_E = m.get("mae_E")  # E (FRINGE_ONLY_ALPHA)
    metrics.mae_F = m.get("mae_F")  # F (POWER_POSTPROCESS)
    metrics.mae_M = m.get("mae_M")  # M (MIXTURE_SHARES_SCALED)
    metrics.mae_N = m.get("mae_N")  # N (MIXTURE_BENCHPOOL)
    metrics.mae_P = m.get("mae_P")  # P (PRODUCTION BASELINE)
    metrics.mae_D_best = m.get("mae_D_best")
    metrics.best_D_alpha = m.get("best_D_alpha")
    metrics.mae_F_best = m.get("mae_F_best")
    metrics.best_F_p = m.get("best_F_p")
    metrics.mae_N_best = m.get("mae_N_best")
    metrics.best_N_bench_pool = m.get("best_N_bench_pool")
    metrics.best_N_core_k = m.get("best_N_core_k")
    metrics.rmse_A = m.get("rmse_A")
    metrics.rmse_B = m.get("rmse_B")
    metrics.rmse_C = m.get("rmse_C")
    metrics.rmse_D = m.get("rmse_D")
    metrics.rmse_E = m.get("rmse_E")
    metrics.rmse_F = m.get("rmse_F")
    metrics.rmse_M = m.get("rmse_M")
    metrics.rmse_N = m.get("rmse_N")
    metrics.rmse_P = m.get("rmse_P")
    metrics.has_labels = metrics.mae_A is not None

    # Per-bucket MAE - A, B, C, D, E, F, M, N, P
    for alloc in ["A", "B", "C", "D", "E", "F", "M", "N", "P"]:
        bucket_key = f"mae_by_bucket_{alloc}"
        bucket_data = m.get(bucket_key) or {}
        setattr(metrics, f"mae_0_10_{alloc}", bucket_data.get("0-10"))
        setattr(metrics, f"mae_10_20_{alloc}", bucket_data.get("10-20"))
        setattr(metrics, f"mae_20_30_{alloc}", bucket_data.get("20-30"))
        setattr(metrics, f"mae_30_plus_{alloc}", bucket_data.get("30+"))

    # Gini and roster size - A, B, C, D, E, F (M handled separately)
    for alloc in ["A", "B", "C", "D", "E", "F"]:
        setattr(metrics, f"gini_{alloc}", m.get(f"gini_mean_{alloc}", 0.0))
        setattr(metrics, f"hhi_{alloc}", m.get(f"hhi_mean_{alloc}", 0.0))
        setattr(metrics, f"top6_share_{alloc}", m.get(f"top6_share_mean_{alloc}", 0.0))
        setattr(metrics, f"top8_share_{alloc}", m.get(f"top8_share_mean_{alloc}", 0.0))
        setattr(metrics, f"sixth_man_mae_{alloc}", m.get(f"sixth_man_mae_{alloc}"))
        setattr(metrics, f"roster_size_{alloc}", m.get(f"roster_size_mean_{alloc}", 0.0))
        setattr(metrics, f"bench_crush_{alloc}", m.get(f"bench_max_mean_{alloc}", 0.0))
        setattr(metrics, f"team_sum_error_max_{alloc}", m.get(f"team_sum_error_max_{alloc}", 0.0))

    # M realism metrics - only populate if allocator M succeeded
    if allocator_M_success:
        metrics.gini_M = m.get("gini_mean_M")
        metrics.hhi_M = m.get("hhi_mean_M")
        metrics.top6_share_M = m.get("top6_share_mean_M")
        metrics.top8_share_M = m.get("top8_share_mean_M")
        metrics.sixth_man_mae_M = m.get("sixth_man_mae_M")
        metrics.roster_size_M = m.get("roster_size_mean_M")
        metrics.bench_crush_M = m.get("bench_max_mean_M")
        metrics.team_sum_error_max_M = m.get("team_sum_error_max_M")

    # Starter realism from dedicated sections - A, B, C
    for alloc in ["A", "B", "C"]:
        starter = data.get(f"starter_realism_{alloc}", {})
        setattr(metrics, f"top5_sum_{alloc}", starter.get("top5_sum_mean", 0.0))
        setattr(metrics, f"max_minutes_{alloc}", starter.get("max_minutes_mean", 0.0))

    # D starter realism (from starter_realism_D dict keyed by alpha)
    starter_D = data.get("starter_realism_D", {})
    # Use alpha=0.7 for default D realism, or first available
    if "0.7" in starter_D:
        metrics.top5_sum_D = starter_D["0.7"].get("top5_sum_mean", 0.0)
        metrics.max_minutes_D = starter_D["0.7"].get("max_minutes_mean", 0.0)
    elif starter_D:
        # Take first available alpha
        first_key = next(iter(starter_D))
        metrics.top5_sum_D = starter_D[first_key].get("top5_sum_mean", 0.0)
        metrics.max_minutes_D = starter_D[first_key].get("max_minutes_mean", 0.0)

    # E starter realism
    starter_E = data.get("starter_realism_E", {})
    metrics.top5_sum_E = starter_E.get("top5_sum_mean", 0.0)
    metrics.max_minutes_E = starter_E.get("max_minutes_mean", 0.0)

    # F starter realism (from starter_realism_F dict keyed by p)
    starter_F = data.get("starter_realism_F", {})
    # Use p=1.2 for default F realism, or first available
    if "1.2" in starter_F:
        metrics.top5_sum_F = starter_F["1.2"].get("top5_sum_mean", 0.0)
        metrics.max_minutes_F = starter_F["1.2"].get("max_minutes_mean", 0.0)
    elif starter_F:
        # Take first available p
        first_key = next(iter(starter_F))
        if isinstance(starter_F[first_key], dict):
            metrics.top5_sum_F = starter_F[first_key].get("top5_sum_mean", 0.0)
            metrics.max_minutes_F = starter_F[first_key].get("max_minutes_mean", 0.0)

    # M starter realism - only populate if allocator M succeeded
    if allocator_M_success:
        starter_M = data.get("starter_realism_M", {})
        if starter_M:
            metrics.top5_sum_M = starter_M.get("top5_sum_mean")
            metrics.max_minutes_M = starter_M.get("max_minutes_mean")

    # N starter realism - only populate if allocator N succeeded
    if allocator_N_success:
        starter_N = data.get("starter_realism_N", {})
        if starter_N:
            metrics.top5_sum_N = starter_N.get("top5_sum_mean")
            metrics.max_minutes_N = starter_N.get("max_minutes_mean")
        # N realism metrics from main metrics section
        metrics.gini_N = m.get("gini_mean_N")
        metrics.hhi_N = m.get("hhi_mean_N")
        metrics.top6_share_N = m.get("top6_share_mean_N")
        metrics.top8_share_N = m.get("top8_share_mean_N")
        metrics.sixth_man_mae_N = m.get("sixth_man_mae_N")
        metrics.roster_size_N = m.get("roster_size_mean_N")
        metrics.bench_crush_N = m.get("bench_max_mean_N")
        metrics.team_sum_error_max_N = m.get("team_sum_error_max_N")

    # P starter realism - only populate if allocator P succeeded
    if allocator_P_success:
        starter_P = data.get("starter_realism_P", {})
        if starter_P:
            metrics.top5_sum_P = starter_P.get("top5_sum_mean")
            metrics.max_minutes_P = starter_P.get("max_minutes_mean")
        # P realism metrics from main metrics section
        metrics.gini_P = m.get("gini_mean_P")
        metrics.hhi_P = m.get("hhi_mean_P")
        metrics.top6_share_P = m.get("top6_share_mean_P")
        metrics.top8_share_P = m.get("top8_share_mean_P")
        metrics.sixth_man_mae_P = m.get("sixth_man_mae_P")
        metrics.roster_size_P = m.get("roster_size_mean_P")
        metrics.bench_crush_P = m.get("bench_max_mean_P")
        metrics.team_sum_error_max_P = m.get("team_sum_error_max_P")

    # Ordering accuracy for [10, 30] range
    ordering = m.get("ordering_10_30", {})
    metrics.ordering_10_30_A = ordering.get("A")
    metrics.ordering_10_30_B = ordering.get("B")
    metrics.ordering_10_30_C = ordering.get("C")
    metrics.ordering_10_30_D = ordering.get("D")
    metrics.ordering_10_30_E = ordering.get("E")
    metrics.ordering_10_30_F = ordering.get("F")
    metrics.ordering_10_30_M = ordering.get("M")
    metrics.ordering_10_30_N = ordering.get("N")
    metrics.ordering_10_30_P = ordering.get("P")

    return metrics


def aggregate_slate_metrics(
    slate_metrics: list[SlateMetrics],
    *,
    quality_filter: str | None = None,  # None = all, "clean", "degraded"
) -> AggregateMetrics:
    """Aggregate metrics across slates, optionally filtering by quality tier."""
    agg = AggregateMetrics()

    if not slate_metrics:
        return agg

    # Filter by quality tier if specified
    if quality_filter:
        slate_metrics = [s for s in slate_metrics if s.quality_tier == quality_filter]

    # Count tiers
    all_slates = slate_metrics
    agg.total_slates = len(all_slates)
    agg.clean_slates = len([s for s in all_slates if s.quality_tier == "clean"])
    agg.degraded_slates = len([s for s in all_slates if s.quality_tier == "degraded"])
    agg.skipped_slates = len([s for s in all_slates if s.quality_tier == "skipped"])
    agg.processed_slates = agg.clean_slates + agg.degraded_slates

    # Count skip reasons
    for s in all_slates:
        if s.skip_reason:
            reason = s.skip_reason
            agg.skip_reasons[reason] = agg.skip_reasons.get(reason, 0) + 1

    # Filter to processed slates (not skipped)
    processed = [s for s in all_slates if s.quality_tier in ("clean", "degraded")]

    # Filter to slates with labels for accuracy metrics
    with_labels = [s for s in processed if s.has_labels]
    agg.slates_with_labels = len(with_labels)

    if not with_labels:
        return agg

    # Extract arrays for A, B, C, D, E, F, M, N
    def extract_metric(attr: str) -> np.ndarray:
        return np.array([getattr(s, attr) for s in with_labels if getattr(s, attr) is not None])

    for alloc in ["A", "B", "C", "D", "E", "F", "M", "N", "P"]:
        mae = extract_metric(f"mae_{alloc}")
        rmse = extract_metric(f"rmse_{alloc}")

        if len(mae) > 0:
            setattr(agg, f"mae_{alloc}_mean", float(np.mean(mae)))
            setattr(agg, f"mae_{alloc}_median", float(np.median(mae)))
        if len(rmse) > 0:
            setattr(agg, f"rmse_{alloc}_mean", float(np.mean(rmse)))

    # D best (alpha tuned per-slate)
    mae_D_best = extract_metric("mae_D_best")
    if len(mae_D_best) > 0:
        agg.mae_D_best_mean = float(np.mean(mae_D_best))

    # Find mode of best_D_alpha
    best_alphas = [s.best_D_alpha for s in with_labels if s.best_D_alpha is not None]
    if best_alphas:
        from collections import Counter
        alpha_counts = Counter(best_alphas)
        agg.best_D_alpha_mode = alpha_counts.most_common(1)[0][0]

    # F best (p tuned per-slate)
    mae_F_best = extract_metric("mae_F_best")
    if len(mae_F_best) > 0:
        agg.mae_F_best_mean = float(np.mean(mae_F_best))

    # Find mode of best_F_p
    best_ps = [s.best_F_p for s in with_labels if s.best_F_p is not None]
    if best_ps:
        from collections import Counter
        p_counts = Counter(best_ps)
        agg.best_F_p_mode = p_counts.most_common(1)[0][0]

    # N best (params tuned per-slate)
    mae_N_best = extract_metric("mae_N_best")
    if len(mae_N_best) > 0:
        agg.mae_N_best_mean = float(np.mean(mae_N_best))

    # Find mode of best_N_bench_pool and best_N_core_k
    best_bench_pools = [s.best_N_bench_pool for s in with_labels if s.best_N_bench_pool is not None]
    if best_bench_pools:
        from collections import Counter
        bp_counts = Counter(best_bench_pools)
        agg.best_N_bench_pool_mode = bp_counts.most_common(1)[0][0]

    best_core_ks = [s.best_N_core_k for s in with_labels if s.best_N_core_k is not None]
    if best_core_ks:
        from collections import Counter
        ck_counts = Counter(best_core_ks)
        agg.best_N_core_k_mode = ck_counts.most_common(1)[0][0]

    # Per-bucket MAE - A, B, C, D, E, F, M, N
    buckets = ["0-10", "10-20", "20-30", "30+"]
    bucket_attrs = ["mae_0_10", "mae_10_20", "mae_20_30", "mae_30_plus"]

    for alloc in ["A", "B", "C", "D", "E", "F", "M", "N", "P"]:
        bucket_dict = {}
        for bucket, attr in zip(buckets, bucket_attrs):
            vals = [getattr(s, f"{attr}_{alloc}") for s in with_labels
                    if getattr(s, f"{attr}_{alloc}") is not None]
            if vals:
                bucket_dict[bucket] = float(np.mean(vals))
        setattr(agg, f"mae_by_bucket_{alloc}", bucket_dict)

    # Realism metrics (use all processed slates) - A, B, C, D, E, F, M, N
    # For M/N metrics, filter out None values (from failed allocator runs)
    for alloc in ["A", "B", "C", "D", "E", "F", "M", "N", "P"]:
        top5_raw = [getattr(s, f"top5_sum_{alloc}") for s in processed]
        max_min_raw = [getattr(s, f"max_minutes_{alloc}") for s in processed]
        gini_raw = [getattr(s, f"gini_{alloc}") for s in processed]
        hhi_raw = [getattr(s, f"hhi_{alloc}") for s in processed]
        top6_raw = [getattr(s, f"top6_share_{alloc}") for s in processed]
        top8_raw = [getattr(s, f"top8_share_{alloc}") for s in processed]
        roster_raw = [getattr(s, f"roster_size_{alloc}") for s in processed]
        bench_raw = [getattr(s, f"bench_crush_{alloc}") for s in processed]

        # Filter out None values (important for allocator M which may fail on some slates)
        top5 = np.array([v for v in top5_raw if v is not None])
        max_min = np.array([v for v in max_min_raw if v is not None])
        gini = np.array([v for v in gini_raw if v is not None])
        hhi = np.array([v for v in hhi_raw if v is not None])
        top6 = np.array([v for v in top6_raw if v is not None])
        top8 = np.array([v for v in top8_raw if v is not None])
        roster = np.array([v for v in roster_raw if v is not None])
        bench = np.array([v for v in bench_raw if v is not None])

        if len(top5) > 0:
            setattr(agg, f"top5_sum_{alloc}_mean", float(np.mean(top5)))
            setattr(agg, f"top5_sum_{alloc}_p90", float(np.percentile(top5, 90)))
        if len(max_min) > 0:
            setattr(agg, f"max_minutes_{alloc}_mean", float(np.mean(max_min)))
            setattr(agg, f"max_minutes_{alloc}_p90", float(np.percentile(max_min, 90)))
        if len(gini) > 0:
            setattr(agg, f"gini_{alloc}_mean", float(np.mean(gini)))
        if len(hhi) > 0:
            setattr(agg, f"hhi_{alloc}_mean", float(np.mean(hhi)))
        if len(top6) > 0:
            setattr(agg, f"top6_share_{alloc}_mean", float(np.mean(top6)))
        if len(top8) > 0:
            setattr(agg, f"top8_share_{alloc}_mean", float(np.mean(top8)))
        if len(roster) > 0:
            setattr(agg, f"roster_size_{alloc}_mean", float(np.mean(roster)))
        if len(bench) > 0:
            setattr(agg, f"bench_crush_{alloc}_mean", float(np.mean(bench)))
            setattr(agg, f"bench_crush_{alloc}_p90", float(np.percentile(bench, 90)))

        # Pathology percentages (only if we have data)
        if len(top5) > 0:
            setattr(agg, f"pct_teams_top5_lt_150_{alloc}", float((top5 < 150).mean()))
        if len(max_min) > 0:
            setattr(agg, f"pct_teams_max_lt_30_{alloc}", float((max_min < 30).mean()))
        if len(roster) > 0:
            setattr(agg, f"pct_teams_roster_ge_14_{alloc}", float((roster >= 14).mean()))

    # Ordering accuracy (only for slates with labels) - A, B, C, D, E, F, M, N
    for alloc in ["A", "B", "C", "D", "E", "F", "M", "N", "P"]:
        ordering = extract_metric(f"ordering_10_30_{alloc}")
        if len(ordering) > 0:
            setattr(agg, f"ordering_10_30_{alloc}_mean", float(np.mean(ordering)))

    # Sixth-man minutes error (only for slates with labels; None when unavailable)
    for alloc in ["A", "B", "C", "D", "E", "F", "M", "N", "P"]:
        sixth = extract_metric(f"sixth_man_mae_{alloc}")
        if len(sixth) > 0:
            setattr(agg, f"sixth_man_mae_{alloc}_mean", float(np.mean(sixth)))

    # Stability (only for slates with labels)
    mae_A = extract_metric("mae_A")
    mae_B = extract_metric("mae_B")
    mae_C = extract_metric("mae_C")

    if len(mae_A) > 0 and len(mae_B) > 0 and len(mae_A) == len(mae_B):
        delta_AB = mae_A - mae_B
        agg.mae_delta_AB_mean = float(np.mean(delta_AB))
        agg.mae_delta_AB_std = float(np.std(delta_AB))

        a_wins_mae = (mae_A < mae_B)
        top5_A = np.array([s.top5_sum_A for s in with_labels])
        top5_B = np.array([s.top5_sum_B for s in with_labels])
        a_wins_top5 = top5_A > top5_B
        a_wins_both = a_wins_mae & a_wins_top5

        agg.pct_slates_A_wins_mae = float(a_wins_mae.mean())
        agg.pct_slates_A_wins_top5 = float(a_wins_top5.mean())
        agg.pct_slates_A_wins_both = float(a_wins_both.mean())

    if len(mae_C) > 0 and len(mae_B) > 0 and len(mae_C) == len(mae_B):
        delta_CB = mae_C - mae_B
        agg.mae_delta_CB_mean = float(np.mean(delta_CB))
        agg.mae_delta_CB_std = float(np.std(delta_CB))

        c_wins_mae = (mae_C < mae_B)
        top5_C = np.array([s.top5_sum_C for s in with_labels])
        top5_B = np.array([s.top5_sum_B for s in with_labels])
        c_wins_top5 = top5_C > top5_B
        c_wins_both = c_wins_mae & c_wins_top5

        agg.pct_slates_C_wins_mae = float(c_wins_mae.mean())
        agg.pct_slates_C_wins_top5 = float(c_wins_top5.mean())
        agg.pct_slates_C_wins_both = float(c_wins_both.mean())

    # D vs B stability
    mae_D = extract_metric("mae_D")
    if len(mae_D) > 0 and len(mae_B) > 0 and len(mae_D) == len(mae_B):
        delta_DB = mae_D - mae_B
        agg.mae_delta_DB_mean = float(np.mean(delta_DB))
        agg.mae_delta_DB_std = float(np.std(delta_DB))

        d_wins_mae = (mae_D < mae_B)
        top5_D = np.array([s.top5_sum_D for s in with_labels])
        top5_B = np.array([s.top5_sum_B for s in with_labels])
        d_wins_top5 = top5_D > top5_B
        d_wins_both = d_wins_mae & d_wins_top5

        agg.pct_slates_D_wins_mae = float(d_wins_mae.mean())
        agg.pct_slates_D_wins_top5 = float(d_wins_top5.mean())
        agg.pct_slates_D_wins_both = float(d_wins_both.mean())

    # E vs B stability
    mae_E = extract_metric("mae_E")
    if len(mae_E) > 0 and len(mae_B) > 0 and len(mae_E) == len(mae_B):
        delta_EB = mae_E - mae_B
        agg.mae_delta_EB_mean = float(np.mean(delta_EB))
        agg.mae_delta_EB_std = float(np.std(delta_EB))

        e_wins_mae = (mae_E < mae_B)
        top5_E = np.array([s.top5_sum_E for s in with_labels])
        top5_B = np.array([s.top5_sum_B for s in with_labels])
        e_wins_top5 = top5_E > top5_B
        e_wins_both = e_wins_mae & e_wins_top5

        agg.pct_slates_E_wins_mae = float(e_wins_mae.mean())
        agg.pct_slates_E_wins_top5 = float(e_wins_top5.mean())
        agg.pct_slates_E_wins_both = float(e_wins_both.mean())

    # F vs B stability
    mae_F = extract_metric("mae_F")
    if len(mae_F) > 0 and len(mae_B) > 0 and len(mae_F) == len(mae_B):
        delta_FB = mae_F - mae_B
        agg.mae_delta_FB_mean = float(np.mean(delta_FB))
        agg.mae_delta_FB_std = float(np.std(delta_FB))

        f_wins_mae = (mae_F < mae_B)
        top5_F = np.array([s.top5_sum_F for s in with_labels])
        top5_B = np.array([s.top5_sum_B for s in with_labels])
        f_wins_top5 = top5_F > top5_B
        f_wins_both = f_wins_mae & f_wins_top5

        agg.pct_slates_F_wins_mae = float(f_wins_mae.mean())
        agg.pct_slates_F_wins_top5 = float(f_wins_top5.mean())
        agg.pct_slates_F_wins_both = float(f_wins_both.mean())

    # M vs B stability - only consider slates where M succeeded (non-None metrics)
    slates_with_M = [s for s in with_labels if s.mae_M is not None and s.top5_sum_M is not None]
    if len(slates_with_M) > 0:
        mae_M = np.array([s.mae_M for s in slates_with_M])
        mae_B_for_M = np.array([s.mae_B for s in slates_with_M])

        delta_MB = mae_M - mae_B_for_M
        agg.mae_delta_MB_mean = float(np.mean(delta_MB))
        agg.mae_delta_MB_std = float(np.std(delta_MB))

        m_wins_mae = (mae_M < mae_B_for_M)
        top5_M = np.array([s.top5_sum_M for s in slates_with_M])
        top5_B = np.array([s.top5_sum_B for s in slates_with_M])
        m_wins_top5 = top5_M > top5_B
        m_wins_both = m_wins_mae & m_wins_top5

        agg.pct_slates_M_wins_mae = float(m_wins_mae.mean())
        agg.pct_slates_M_wins_top5 = float(m_wins_top5.mean())
        agg.pct_slates_M_wins_both = float(m_wins_both.mean())

    # N vs B stability - only consider slates where N succeeded (non-None metrics)
    slates_with_N = [s for s in with_labels if s.mae_N is not None and s.top5_sum_N is not None]
    if len(slates_with_N) > 0:
        mae_N = np.array([s.mae_N for s in slates_with_N])
        mae_B_for_N = np.array([s.mae_B for s in slates_with_N])

        delta_NB = mae_N - mae_B_for_N
        agg.mae_delta_NB_mean = float(np.mean(delta_NB))
        agg.mae_delta_NB_std = float(np.std(delta_NB))

        n_wins_mae = (mae_N < mae_B_for_N)
        top5_N = np.array([s.top5_sum_N for s in slates_with_N])
        top5_B = np.array([s.top5_sum_B for s in slates_with_N])
        n_wins_top5 = top5_N > top5_B
        n_wins_both = n_wins_mae & n_wins_top5

        agg.pct_slates_N_wins_mae = float(n_wins_mae.mean())
        agg.pct_slates_N_wins_top5 = float(n_wins_top5.mean())
        agg.pct_slates_N_wins_both = float(n_wins_both.mean())

    return agg


def build_slate_dataframe(slate_metrics: list[SlateMetrics]) -> pd.DataFrame:
    """Build a DataFrame with one row per slate."""
    rows = []
    for s in slate_metrics:
        row = {
            "game_date": s.game_date,
            "quality_tier": s.quality_tier,
            "skip_reason": s.skip_reason,
            "missing_feature_frac": s.missing_feature_frac,
            "n_games": s.n_games,
            "n_teams": s.n_teams,
            "n_players": s.n_players,
            "has_labels": s.has_labels,
        }
        # A/B/C/D/E/F/M/N metrics
        for alloc in ["A", "B", "C", "D", "E", "F", "M", "N", "P"]:
            row[f"mae_{alloc}"] = getattr(s, f"mae_{alloc}")
            row[f"rmse_{alloc}"] = getattr(s, f"rmse_{alloc}")
            row[f"top5_sum_{alloc}"] = getattr(s, f"top5_sum_{alloc}")
            row[f"max_minutes_{alloc}"] = getattr(s, f"max_minutes_{alloc}")
            row[f"gini_{alloc}"] = getattr(s, f"gini_{alloc}")
            row[f"hhi_{alloc}"] = getattr(s, f"hhi_{alloc}")
            row[f"top6_share_{alloc}"] = getattr(s, f"top6_share_{alloc}")
            row[f"top8_share_{alloc}"] = getattr(s, f"top8_share_{alloc}")
            row[f"sixth_man_mae_{alloc}"] = getattr(s, f"sixth_man_mae_{alloc}")
            row[f"roster_size_{alloc}"] = getattr(s, f"roster_size_{alloc}")
            row[f"bench_crush_{alloc}"] = getattr(s, f"bench_crush_{alloc}")

        # D-specific
        row["mae_D_best"] = s.mae_D_best
        row["best_D_alpha"] = s.best_D_alpha

        # F-specific
        row["mae_F_best"] = s.mae_F_best
        row["best_F_p"] = s.best_F_p

        # N-specific
        row["mae_N_best"] = s.mae_N_best
        row["best_N_bench_pool"] = s.best_N_bench_pool
        row["best_N_core_k"] = s.best_N_core_k

        # Win flags
        if s.mae_A and s.mae_B:
            row["A_wins_mae"] = s.mae_A < s.mae_B
        if s.mae_C and s.mae_B:
            row["C_wins_mae"] = s.mae_C < s.mae_B
        if s.mae_D and s.mae_B:
            row["D_wins_mae"] = s.mae_D < s.mae_B
        if s.mae_E and s.mae_B:
            row["E_wins_mae"] = s.mae_E < s.mae_B
        if s.mae_F and s.mae_B:
            row["F_wins_mae"] = s.mae_F < s.mae_B
        if s.mae_M and s.mae_B:
            row["M_wins_mae"] = s.mae_M < s.mae_B
        if s.mae_N and s.mae_B:
            row["N_wins_mae"] = s.mae_N < s.mae_B
        row["A_wins_top5"] = s.top5_sum_A > s.top5_sum_B
        row["C_wins_top5"] = s.top5_sum_C > s.top5_sum_B
        row["D_wins_top5"] = s.top5_sum_D > s.top5_sum_B
        row["E_wins_top5"] = s.top5_sum_E > s.top5_sum_B
        row["F_wins_top5"] = s.top5_sum_F > s.top5_sum_B
        if s.top5_sum_M is not None:
            row["M_wins_top5"] = s.top5_sum_M > s.top5_sum_B
        if s.top5_sum_N is not None:
            row["N_wins_top5"] = s.top5_sum_N > s.top5_sum_B

        rows.append(row)
    return pd.DataFrame(rows)


def build_bucket_dataframe(slate_metrics: list[SlateMetrics]) -> pd.DataFrame:
    """Build a DataFrame with per-bucket MAE comparison (A, B, C, D, E, F)."""
    processed = [s for s in slate_metrics if s.quality_tier in ("clean", "degraded") and s.has_labels]

    if not processed:
        return pd.DataFrame()

    buckets = ["0-10", "10-20", "20-30", "30+"]
    bucket_attrs = ["mae_0_10", "mae_10_20", "mae_20_30", "mae_30_plus"]
    rows = []

    for bucket, attr in zip(buckets, bucket_attrs):
        row = {"bucket": bucket, "n_slates": 0}
        for alloc in ["A", "B", "C", "D", "E", "F", "M", "N", "P"]:
            vals = [getattr(s, f"{attr}_{alloc}") for s in processed
                    if getattr(s, f"{attr}_{alloc}") is not None]
            row[f"mae_{alloc}_mean"] = np.mean(vals) if vals else None
            row[f"mae_{alloc}_median"] = np.median(vals) if vals else None
            row[f"mae_{alloc}_std"] = np.std(vals) if len(vals) > 1 else None
            if alloc == "A":
                row["n_slates"] = len(vals)

        # Determine bucket winner (lowest MAE)
        mae_values = {
            "A": row.get("mae_A_mean"),
            "B": row.get("mae_B_mean"),
            "C": row.get("mae_C_mean"),
            "D": row.get("mae_D_mean"),
            "E": row.get("mae_E_mean"),
            "F": row.get("mae_F_mean"),
            "M": row.get("mae_M_mean"),
            "N": row.get("mae_N_mean"),
        }
        row["winner"] = determine_winner(mae_values, lower_is_better=True)

        # Also add individual vs B comparison flags for backward compat
        for alloc in ["A", "C", "D", "E", "F", "M", "N", "P"]:
            if row[f"mae_{alloc}_mean"] is not None and row["mae_B_mean"] is not None:
                row[f"{alloc}_better"] = row[f"mae_{alloc}_mean"] < row["mae_B_mean"]
            else:
                row[f"{alloc}_better"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def build_skips_dataframe(slate_metrics: list[SlateMetrics]) -> pd.DataFrame:
    """Build a DataFrame with skip reasons for skipped slates."""
    skipped = [s for s in slate_metrics if s.quality_tier == "skipped"]
    
    rows = []
    for s in skipped:
        rows.append({
            "game_date": s.game_date,
            "skip_reason": s.skip_reason or "unknown",
            "missing_feature_frac": s.missing_feature_frac,
            "n_expected_features": s.n_expected_features,
            "n_missing_features": s.n_missing_features,
            "n_games": s.n_games,
            "n_teams": s.n_teams,
            "n_players": s.n_players,
        })
    
    return pd.DataFrame(rows)


def _safe_fmt(val: float, fmt: str = ".2f", default: str = "N/A") -> str:
    """Format a float safely, returning default for NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return f"{val:{fmt}}"


def determine_winner(
    values: dict[str, float],
    lower_is_better: bool = True,
    tolerance: float = 1e-6,
) -> str:
    """Determine the winner among allocators based on their values.

    Args:
        values: Dict mapping allocator name to its metric value
        lower_is_better: If True, lowest value wins (MAE, RMSE). If False, highest wins (top5, max).
        tolerance: Values within this tolerance are considered tied

    Returns:
        Winner name, or "–" if tied or all NaN
    """
    # Filter out NaN values
    valid = {k: v for k, v in values.items() if not (v is None or (isinstance(v, float) and np.isnan(v)))}
    if not valid:
        return "–"

    if lower_is_better:
        best_val = min(valid.values())
        winners = [k for k, v in valid.items() if abs(v - best_val) < tolerance]
    else:
        best_val = max(valid.values())
        winners = [k for k, v in valid.items() if abs(v - best_val) < tolerance]

    if len(winners) == 1:
        return winners[0]
    elif len(winners) > 1:
        return "–"  # Tie
    return "–"


def generate_readme(
    agg: AggregateMetrics,
    agg_clean: AggregateMetrics,
    slate_df: pd.DataFrame,
    out_dir: Path,
) -> str:
    """Generate README.md with interpretation and recommendations."""

    def safe_gt(a: float, b: float) -> bool:
        if np.isnan(a) or np.isnan(b):
            return False
        return a > b

    def safe_lt(a: float, b: float) -> bool:
        if np.isnan(a) or np.isnan(b):
            return False
        return a < b

    # Use clean slates for recommendations
    a_better_mae = safe_lt(agg_clean.mae_A_mean, agg_clean.mae_B_mean)
    c_better_mae = safe_lt(agg_clean.mae_C_mean, agg_clean.mae_B_mean)
    a_better_top5 = safe_gt(agg_clean.top5_sum_A_mean, agg_clean.top5_sum_B_mean)
    c_better_top5 = safe_gt(agg_clean.top5_sum_C_mean, agg_clean.top5_sum_B_mean)

    # Compute proper winners using the new function (A, B, C, D, E, F, M)
    mae_winner = determine_winner(
        {"A": agg_clean.mae_A_mean, "B": agg_clean.mae_B_mean, "C": agg_clean.mae_C_mean, "D": agg_clean.mae_D_mean, "E": agg_clean.mae_E_mean, "F": agg_clean.mae_F_mean, "M": agg_clean.mae_M_mean},
        lower_is_better=True,
    )
    rmse_winner = determine_winner(
        {"A": agg_clean.rmse_A_mean, "B": agg_clean.rmse_B_mean, "C": agg_clean.rmse_C_mean, "D": agg_clean.rmse_D_mean, "E": agg_clean.rmse_E_mean, "F": agg_clean.rmse_F_mean, "M": agg_clean.rmse_M_mean},
        lower_is_better=True,
    )
    top5_winner = determine_winner(
        {"A": agg_clean.top5_sum_A_mean, "B": agg_clean.top5_sum_B_mean, "C": agg_clean.top5_sum_C_mean, "D": agg_clean.top5_sum_D_mean, "E": agg_clean.top5_sum_E_mean, "F": agg_clean.top5_sum_F_mean, "M": agg_clean.top5_sum_M_mean},
        lower_is_better=False,
    )

    # Determine recommendation
    if agg.processed_slates == 0:
        recommendation = "**No data**: No slates were successfully processed."
    elif c_better_mae and c_better_top5:
        recommendation = "**Use Allocator C**: SHARE_WITH_ROTALLOC_ELIGIBILITY wins on both accuracy and realism."
    elif a_better_mae and a_better_top5:
        recommendation = "**Use Allocator A**: SCALE_SHARES wins on both accuracy and realism."
    elif c_better_top5 and not c_better_mae:
        recommendation = "**Hybrid approach**: C has better realism but B has better accuracy. Consider tuning C's caps or weights."
    elif a_better_top5 and not a_better_mae:
        recommendation = "**Keep RotAlloc**: B has better accuracy. A's realism gains don't compensate."
    else:
        recommendation = "**Keep RotAlloc**: It outperforms on accuracy metrics."

    # Skip reason summary
    skip_lines = []
    for reason, count in sorted(agg.skip_reasons.items(), key=lambda x: -x[1]):
        skip_lines.append(f"  - {reason}: {count}")
    skip_summary = "\n".join(skip_lines) if skip_lines else "  (none)"

    # Pre-compute bucket winners (avoid dict-in-f-string issues)
    bucket_0_10_winner = determine_winner(
        {"A": agg_clean.mae_by_bucket_A.get('0-10'), "B": agg_clean.mae_by_bucket_B.get('0-10'),
         "C": agg_clean.mae_by_bucket_C.get('0-10'), "D": agg_clean.mae_by_bucket_D.get('0-10'),
         "E": agg_clean.mae_by_bucket_E.get('0-10'), "F": agg_clean.mae_by_bucket_F.get('0-10'),
         "M": agg_clean.mae_by_bucket_M.get('0-10')},
        lower_is_better=True
    )
    bucket_10_20_winner = determine_winner(
        {"A": agg_clean.mae_by_bucket_A.get('10-20'), "B": agg_clean.mae_by_bucket_B.get('10-20'),
         "C": agg_clean.mae_by_bucket_C.get('10-20'), "D": agg_clean.mae_by_bucket_D.get('10-20'),
         "E": agg_clean.mae_by_bucket_E.get('10-20'), "F": agg_clean.mae_by_bucket_F.get('10-20'),
         "M": agg_clean.mae_by_bucket_M.get('10-20')},
        lower_is_better=True
    )
    bucket_20_30_winner = determine_winner(
        {"A": agg_clean.mae_by_bucket_A.get('20-30'), "B": agg_clean.mae_by_bucket_B.get('20-30'),
         "C": agg_clean.mae_by_bucket_C.get('20-30'), "D": agg_clean.mae_by_bucket_D.get('20-30'),
         "E": agg_clean.mae_by_bucket_E.get('20-30'), "F": agg_clean.mae_by_bucket_F.get('20-30'),
         "M": agg_clean.mae_by_bucket_M.get('20-30')},
        lower_is_better=True
    )
    bucket_30plus_winner = determine_winner(
        {"A": agg_clean.mae_by_bucket_A.get('30+'), "B": agg_clean.mae_by_bucket_B.get('30+'),
         "C": agg_clean.mae_by_bucket_C.get('30+'), "D": agg_clean.mae_by_bucket_D.get('30+'),
         "E": agg_clean.mae_by_bucket_E.get('30+'), "F": agg_clean.mae_by_bucket_F.get('30+'),
         "M": agg_clean.mae_by_bucket_M.get('30+')},
        lower_is_better=True
    )

    readme = f"""# Minutes Allocator A/B/C/D/E/F/M Test Results

Generated: {datetime.now().isoformat()}

## Allocators

| Allocator | Description |
|-----------|-------------|
| **A** | SCALE_SHARES: Share model predictions scaled to 240 per team |
| **B** | ROTALLOC: Production allocator with rotation classifier + conditional means |
| **C** | SHARE_WITH_ROTALLOC_ELIGIBILITY: Share predictions scaled within RotAlloc's eligible set |
| **D** | BLEND_WITHIN_ELIGIBLE: Blends share weights with RotAlloc proxy weights within eligible set |
| **E** | FRINGE_ONLY_ALPHA: Two-tier blend - core players (top k by w_rot) use alpha_core, fringe use alpha_fringe |
| **F** | POWER_POSTPROCESS: Power transform on RotAlloc minutes to fix flat-top (increases concentration) |
| **M** | MIXTURE_SHARES_SCALED: Mixture model expected minutes → shares → scaled to 240 |

### Allocator D Details

D blends two weight vectors within RotAlloc's eligible set:
- w_share = share_pred ^ gamma (realism from share model)
- w_rot = (p_rot ^ a) * (mu_cond ^ mu_power) (bench ordering from RotAlloc)
- Final: w = alpha * w_share + (1-alpha) * w_rot

Default alpha = 0.7 (70% share, 30% RotAlloc proxy). Grid search tests alpha in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0].

### Allocator E Details

E improves on D by using different alpha values for core vs fringe players:
- Core players (top k_core by w_rot): use alpha_core (default 0.8 = lean shares)
- Fringe players (remaining eligible): use alpha_fringe (default 0.3 = lean RotAlloc proxy)

This targets improved MAE in the 10-30 minute buckets where fringe rotation players benefit from RotAlloc ordering.

### Allocator F Details

F applies a power transform to RotAlloc (B) minutes to fix the flat-top pathology:
- m_raw = m_B ** p (where p >= 1.0)
- m_F = 240 * m_raw / sum(m_raw)

Higher p increases top-end concentration while preserving B's bench ordering. Default p = 1.2.

### Allocator M Details

M uses a trained mixture model to predict expected minutes directly:
- Predict expected minutes from features using mixture distribution
- Mask inactive players (OUT, OFS, NWT status)
- Convert to shares within team
- Scale to 240 per team with iterative cap/redistribution

Requires --mixture-bundle path to enable.

## Quality Tiers

| Tier | Criteria |
|------|----------|
| **clean** | Passes integrity + missing_feature_frac ≤ 2% |
| **degraded** | Passes integrity + 2% < missing_feature_frac ≤ 10% |
| **skipped** | Failed integrity or missing_feature_frac > 10% |

## Coverage

| Metric | Count |
|--------|-------|
| Total slates | {agg.total_slates} |
| Processed | {agg.processed_slates} |
| Clean | {agg.clean_slates} |
| Degraded | {agg.degraded_slates} |
| Skipped | {agg.skipped_slates} |
| With labels | {agg.slates_with_labels} |

### Skip Reasons
{skip_summary}

## Summary (Clean Slates Only)

| Metric | A | B | C | D | E | M | Winner |
|--------|---|---|---|---|---|---|--------|
| MAE (mean) | {_safe_fmt(agg_clean.mae_A_mean)} | {_safe_fmt(agg_clean.mae_B_mean)} | {_safe_fmt(agg_clean.mae_C_mean)} | {_safe_fmt(agg_clean.mae_D_mean)} | {_safe_fmt(agg_clean.mae_E_mean)} | {_safe_fmt(agg_clean.mae_M_mean)} | {mae_winner} |
| RMSE (mean) | {_safe_fmt(agg_clean.rmse_A_mean)} | {_safe_fmt(agg_clean.rmse_B_mean)} | {_safe_fmt(agg_clean.rmse_C_mean)} | {_safe_fmt(agg_clean.rmse_D_mean)} | {_safe_fmt(agg_clean.rmse_E_mean)} | {_safe_fmt(agg_clean.rmse_M_mean)} | {rmse_winner} |
| Top-5 sum | {_safe_fmt(agg_clean.top5_sum_A_mean, '.1f')} | {_safe_fmt(agg_clean.top5_sum_B_mean, '.1f')} | {_safe_fmt(agg_clean.top5_sum_C_mean, '.1f')} | {_safe_fmt(agg_clean.top5_sum_D_mean, '.1f')} | {_safe_fmt(agg_clean.top5_sum_E_mean, '.1f')} | {_safe_fmt(agg_clean.top5_sum_M_mean, '.1f')} | {top5_winner} |
| Ordering 10-30 | {_safe_fmt(agg_clean.ordering_10_30_A_mean, '.3f')} | {_safe_fmt(agg_clean.ordering_10_30_B_mean, '.3f')} | {_safe_fmt(agg_clean.ordering_10_30_C_mean, '.3f')} | {_safe_fmt(agg_clean.ordering_10_30_D_mean, '.3f')} | {_safe_fmt(agg_clean.ordering_10_30_E_mean, '.3f')} | {_safe_fmt(agg_clean.ordering_10_30_M_mean, '.3f')} | – |
| Max minutes | {_safe_fmt(agg_clean.max_minutes_A_mean, '.1f')} | {_safe_fmt(agg_clean.max_minutes_B_mean, '.1f')} | {_safe_fmt(agg_clean.max_minutes_C_mean, '.1f')} | {_safe_fmt(agg_clean.max_minutes_D_mean, '.1f')} | {_safe_fmt(agg_clean.max_minutes_E_mean, '.1f')} | {_safe_fmt(agg_clean.max_minutes_M_mean, '.1f')} | – |
| Gini | {_safe_fmt(agg_clean.gini_A_mean, '.3f')} | {_safe_fmt(agg_clean.gini_B_mean, '.3f')} | {_safe_fmt(agg_clean.gini_C_mean, '.3f')} | {_safe_fmt(agg_clean.gini_D_mean, '.3f')} | {_safe_fmt(agg_clean.gini_E_mean, '.3f')} | {_safe_fmt(agg_clean.gini_M_mean, '.3f')} | – |
| Roster size | {_safe_fmt(agg_clean.roster_size_A_mean, '.1f')} | {_safe_fmt(agg_clean.roster_size_B_mean, '.1f')} | {_safe_fmt(agg_clean.roster_size_C_mean, '.1f')} | {_safe_fmt(agg_clean.roster_size_D_mean, '.1f')} | {_safe_fmt(agg_clean.roster_size_E_mean, '.1f')} | {_safe_fmt(agg_clean.roster_size_M_mean, '.1f')} | – |
| D best alpha (mode) | – | – | – | {_safe_fmt(agg_clean.best_D_alpha_mode, '.1f') if agg_clean.best_D_alpha_mode else '–'} | – | – | – |
| D best MAE | – | – | – | {_safe_fmt(agg_clean.mae_D_best_mean)} | – | – | – |

## Consistency (Clean Slates)

| Metric | A vs B | C vs B | D vs B | E vs B | M vs B |
|--------|--------|--------|--------|--------|--------|
| % slates wins MAE | {_safe_fmt(agg_clean.pct_slates_A_wins_mae * 100 if not np.isnan(agg_clean.pct_slates_A_wins_mae) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_C_wins_mae * 100 if not np.isnan(agg_clean.pct_slates_C_wins_mae) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_D_wins_mae * 100 if not np.isnan(agg_clean.pct_slates_D_wins_mae) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_E_wins_mae * 100 if not np.isnan(agg_clean.pct_slates_E_wins_mae) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_M_wins_mae * 100 if not np.isnan(agg_clean.pct_slates_M_wins_mae) else float('nan'), '.1f')}% |
| % slates wins top5 | {_safe_fmt(agg_clean.pct_slates_A_wins_top5 * 100 if not np.isnan(agg_clean.pct_slates_A_wins_top5) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_C_wins_top5 * 100 if not np.isnan(agg_clean.pct_slates_C_wins_top5) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_D_wins_top5 * 100 if not np.isnan(agg_clean.pct_slates_D_wins_top5) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_E_wins_top5 * 100 if not np.isnan(agg_clean.pct_slates_E_wins_top5) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_M_wins_top5 * 100 if not np.isnan(agg_clean.pct_slates_M_wins_top5) else float('nan'), '.1f')}% |
| % slates wins both | {_safe_fmt(agg_clean.pct_slates_A_wins_both * 100 if not np.isnan(agg_clean.pct_slates_A_wins_both) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_C_wins_both * 100 if not np.isnan(agg_clean.pct_slates_C_wins_both) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_D_wins_both * 100 if not np.isnan(agg_clean.pct_slates_D_wins_both) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_E_wins_both * 100 if not np.isnan(agg_clean.pct_slates_E_wins_both) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_slates_M_wins_both * 100 if not np.isnan(agg_clean.pct_slates_M_wins_both) else float('nan'), '.1f')}% |
| MAE delta mean±std | {_safe_fmt(agg_clean.mae_delta_AB_mean, '+.2f')} ± {_safe_fmt(agg_clean.mae_delta_AB_std)} | {_safe_fmt(agg_clean.mae_delta_CB_mean, '+.2f')} ± {_safe_fmt(agg_clean.mae_delta_CB_std)} | {_safe_fmt(agg_clean.mae_delta_DB_mean, '+.2f')} ± {_safe_fmt(agg_clean.mae_delta_DB_std)} | {_safe_fmt(agg_clean.mae_delta_EB_mean, '+.2f')} ± {_safe_fmt(agg_clean.mae_delta_EB_std)} | {_safe_fmt(agg_clean.mae_delta_MB_mean, '+.2f')} ± {_safe_fmt(agg_clean.mae_delta_MB_std)} |

## Per-Bucket Analysis (Clean Slates)

| Bucket | MAE (A) | MAE (B) | MAE (C) | MAE (D) | MAE (E) | MAE (M) | Winner |
|--------|---------|---------|---------|---------|---------|---------|--------|
| 0-10 min | {_safe_fmt(agg_clean.mae_by_bucket_A.get('0-10', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_B.get('0-10', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_C.get('0-10', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_D.get('0-10', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_E.get('0-10', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_M.get('0-10', float('nan')))} | {bucket_0_10_winner} |
| 10-20 min | {_safe_fmt(agg_clean.mae_by_bucket_A.get('10-20', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_B.get('10-20', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_C.get('10-20', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_D.get('10-20', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_E.get('10-20', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_M.get('10-20', float('nan')))} | {bucket_10_20_winner} |
| 20-30 min | {_safe_fmt(agg_clean.mae_by_bucket_A.get('20-30', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_B.get('20-30', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_C.get('20-30', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_D.get('20-30', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_E.get('20-30', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_M.get('20-30', float('nan')))} | {bucket_20_30_winner} |
| 30+ min | {_safe_fmt(agg_clean.mae_by_bucket_A.get('30+', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_B.get('30+', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_C.get('30+', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_D.get('30+', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_E.get('30+', float('nan')))} | {_safe_fmt(agg_clean.mae_by_bucket_M.get('30+', float('nan')))} | {bucket_30plus_winner} |

## Pathology Checks (Clean Slates)

| Metric | A | B | C | D | E | M |
|--------|---|---|---|---|---|---|
| % slates with top5 < 150 | {_safe_fmt(agg_clean.pct_teams_top5_lt_150_A * 100 if not np.isnan(agg_clean.pct_teams_top5_lt_150_A) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_teams_top5_lt_150_B * 100 if not np.isnan(agg_clean.pct_teams_top5_lt_150_B) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_teams_top5_lt_150_C * 100 if not np.isnan(agg_clean.pct_teams_top5_lt_150_C) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_teams_top5_lt_150_D * 100 if not np.isnan(agg_clean.pct_teams_top5_lt_150_D) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_teams_top5_lt_150_E * 100 if not np.isnan(agg_clean.pct_teams_top5_lt_150_E) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_teams_top5_lt_150_M * 100 if not np.isnan(agg_clean.pct_teams_top5_lt_150_M) else float('nan'), '.1f')}% |
| % slates with max < 30 | {_safe_fmt(agg_clean.pct_teams_max_lt_30_A * 100 if not np.isnan(agg_clean.pct_teams_max_lt_30_A) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_teams_max_lt_30_B * 100 if not np.isnan(agg_clean.pct_teams_max_lt_30_B) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_teams_max_lt_30_C * 100 if not np.isnan(agg_clean.pct_teams_max_lt_30_C) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_teams_max_lt_30_D * 100 if not np.isnan(agg_clean.pct_teams_max_lt_30_D) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_teams_max_lt_30_E * 100 if not np.isnan(agg_clean.pct_teams_max_lt_30_E) else float('nan'), '.1f')}% | {_safe_fmt(agg_clean.pct_teams_max_lt_30_M * 100 if not np.isnan(agg_clean.pct_teams_max_lt_30_M) else float('nan'), '.1f')}% |
| Bench crush (mean) | {_safe_fmt(agg_clean.bench_crush_A_mean, '.1f')} | {_safe_fmt(agg_clean.bench_crush_B_mean, '.1f')} | {_safe_fmt(agg_clean.bench_crush_C_mean, '.1f')} | {_safe_fmt(agg_clean.bench_crush_D_mean, '.1f')} | {_safe_fmt(agg_clean.bench_crush_E_mean, '.1f')} | {_safe_fmt(agg_clean.bench_crush_M_mean, '.1f')} |
| Bench crush (p90) | {_safe_fmt(agg_clean.bench_crush_A_p90, '.1f')} | {_safe_fmt(agg_clean.bench_crush_B_p90, '.1f')} | {_safe_fmt(agg_clean.bench_crush_C_p90, '.1f')} | {_safe_fmt(agg_clean.bench_crush_D_p90, '.1f')} | {_safe_fmt(agg_clean.bench_crush_E_p90, '.1f')} | {_safe_fmt(agg_clean.bench_crush_M_p90, '.1f')} |

## Recommendation

{recommendation}

## Next Steps

1. Examine C vs B bucket-level losses to understand where RotAlloc's conditional means help
2. Test different sharpen_exponent values for the share model
3. Consider hybrid: use share model weights within RotAlloc's probability framework
4. Investigate skipped slates for data pipeline fixes
5. Fine-tune E's alpha_core and alpha_fringe parameters for optimal 10-30 bucket performance

## Files

- `aggregate_summary.json` - All aggregated metrics (all processed)
- `aggregate_summary_clean.json` - Aggregated metrics (clean only)
- `aggregate_by_bucket.csv` - Per-bucket MAE breakdown
- `aggregate_by_slate.csv` - Per-slate comparison table
- `skips.csv` - Skipped slates with reasons
"""

    return readme


def run_aggregation(
    run_dir: Path,
    *,
    skip_missing: bool = True,
) -> tuple[AggregateMetrics, AggregateMetrics, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all slate results from run_dir and produce aggregates.
    
    Returns:
        Tuple of (agg_all, agg_clean, slate_df, bucket_df, skips_df)
    """
    slate_metrics: list[SlateMetrics] = []

    # Find all subdirectories (date folders)
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Skip aggregate output files
        if subdir.name.startswith("aggregate"):
            continue
            
        summary_path = subdir / "summary.json"
        if not summary_path.exists():
            if skip_missing:
                metrics = SlateMetrics(game_date=subdir.name, quality_tier="skipped", skip_reason="missing_summary")
                slate_metrics.append(metrics)
                continue
            else:
                raise FileNotFoundError(f"Missing summary.json in {subdir}")

        try:
            metrics = load_slate_metrics(summary_path)
            slate_metrics.append(metrics)
        except Exception as e:
            if skip_missing:
                metrics = SlateMetrics(game_date=subdir.name, quality_tier="skipped", skip_reason=f"load_error: {e}")
                slate_metrics.append(metrics)
            else:
                raise

    # Aggregate all processed
    agg_all = aggregate_slate_metrics(slate_metrics)
    
    # Aggregate clean only
    agg_clean = aggregate_slate_metrics(slate_metrics, quality_filter="clean")

    # Build DataFrames
    slate_df = build_slate_dataframe(slate_metrics)
    bucket_df = build_bucket_dataframe(slate_metrics)
    skips_df = build_skips_dataframe(slate_metrics)

    return agg_all, agg_clean, slate_df, bucket_df, skips_df


def save_aggregates(
    agg_all: AggregateMetrics,
    agg_clean: AggregateMetrics,
    slate_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    skips_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save all aggregate outputs to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. aggregate_summary.json (all processed)
    summary_path = out_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(agg_all.to_dict(), indent=2, default=str), encoding="utf-8")

    # 2. aggregate_summary_clean.json
    clean_path = out_dir / "aggregate_summary_clean.json"
    clean_path.write_text(json.dumps(agg_clean.to_dict(), indent=2, default=str), encoding="utf-8")

    # 3. aggregate_by_slate.csv
    slate_path = out_dir / "aggregate_by_slate.csv"
    slate_df.to_csv(slate_path, index=False)

    # 4. aggregate_by_bucket.csv
    bucket_path = out_dir / "aggregate_by_bucket.csv"
    bucket_df.to_csv(bucket_path, index=False)

    # 5. skips.csv
    skips_path = out_dir / "skips.csv"
    skips_df.to_csv(skips_path, index=False)

    # 6. README.md
    readme_content = generate_readme(agg_all, agg_clean, slate_df, out_dir)
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")


__all__ = [
    "SlateMetrics",
    "AggregateMetrics",
    "MISSING_FEATURE_CLEAN_THRESHOLD",
    "MISSING_FEATURE_SKIP_THRESHOLD",
    "load_slate_metrics",
    "aggregate_slate_metrics",
    "build_slate_dataframe",
    "build_bucket_dataframe",
    "build_skips_dataframe",
    "determine_winner",
    "generate_readme",
    "run_aggregation",
    "save_aggregates",
]
