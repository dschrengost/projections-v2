"""A/B test evaluation module for comparing minutes allocation methods.

This module provides utilities for comparing two minutes allocation methods:
- Allocator A (SCALE_SHARES): Scale predicted minute shares to sum to 240
- Allocator B (ROTALLOC): Use the production RotAlloc allocator

The primary functions are:
- scale_shares_to_240(): Apply shares-based allocation with caps
- compare_allocators(): Compute comparison metrics between two allocators
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

LOGGER = logging.getLogger(__name__)

# Constants
TEAM_TOTAL_MINUTES = 240.0
DEFAULT_CAP_MAX = 48.0


@dataclass
class ScaleSharesDiagnostics:
    """Diagnostics from scale_shares_to_240 allocation."""

    n_teams: int = 0
    n_players: int = 0
    n_eligible: int = 0
    fallback_count: int = 0
    fallback_teams: list[tuple[int, int]] = field(default_factory=list)
    cap_applied_count: int = 0
    redistribution_rounds_max: int = 0
    team_sum_dev_max: float = 0.0
    pre_cap_minutes_p95: float = float("nan")
    post_cap_minutes_p95: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AllocatorComparisonMetrics:
    """Metrics comparing two allocators."""

    # Team sum errors
    team_sum_error_max_A: float
    team_sum_error_mean_A: float
    team_sum_error_max_B: float
    team_sum_error_mean_B: float

    # Tail minutes distribution
    p95_minutes_A: float
    p99_minutes_A: float
    p95_minutes_B: float
    p99_minutes_B: float
    frac_ge_42_A: float
    frac_ge_44_A: float
    frac_ge_42_B: float
    frac_ge_44_B: float

    # Gini coefficient (distribution flatness, lower = flatter)
    gini_mean_A: float
    gini_mean_B: float

    # Concentration metrics (higher = more concentrated)
    hhi_mean_A: float
    hhi_mean_B: float
    top6_share_mean_A: float
    top6_share_mean_B: float
    top8_share_mean_A: float
    top8_share_mean_B: float

    # Effective roster size (players with >= 1 minute)
    roster_size_mean_A: float
    roster_size_mean_B: float

    # Bench crush proxy (max minutes among bottom-half)
    bench_max_mean_A: float
    bench_max_mean_B: float

    # Player-level delta statistics
    delta_mean: float
    delta_std: float
    delta_abs_mean: float
    delta_abs_max: float
    n_players: int
    n_teams: int

    # Actual minutes comparison (if available)
    mae_A: float | None = None
    rmse_A: float | None = None
    mae_B: float | None = None
    rmse_B: float | None = None
    mae_by_bucket_A: dict[str, float | None] | None = None
    mae_by_bucket_B: dict[str, float | None] | None = None

    # Sixth-man minutes error (absolute, actual 6th man by minutes)
    sixth_man_mae_A: float | None = None
    sixth_man_mae_B: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient for a distribution.

    0 = perfect equality, 1 = perfect inequality.
    """
    if len(values) < 2:
        return 0.0
    values = np.sort(values)
    n = len(values)
    total = values.sum()
    if total <= 0:
        return 0.0
    cumsum = np.cumsum(values)
    gini = (n + 1 - 2 * cumsum.sum() / total) / n
    return max(0.0, min(1.0, gini))

def _hhi(values: np.ndarray, *, total: float = TEAM_TOTAL_MINUTES) -> float:
    """Compute Herfindahl-Hirschman Index (HHI) on minute shares."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    shares = arr / float(total)
    return float(np.sum(shares**2))


def scale_shares_to_240(
    df: pd.DataFrame,
    shares_col: str,
    group_cols: tuple[str, str] = ("game_id", "team_id"),
    *,
    cap_max: float = DEFAULT_CAP_MAX,
    require_positive_share: bool = False,
    redistribute_after_cap: bool = True,
    max_redistribution_rounds: int = 10,
    status_col: str | None = "status",
    play_prob_col: str | None = "play_prob",
) -> tuple[pd.DataFrame, ScaleSharesDiagnostics]:
    """Scale per-player predicted minute shares to exactly 240 per team.

    Allocator A logic:
    1. Identify eligible players (not OUT, play_prob > 0 if exists)
    2. For each team: minutes_i = 240 * share_i / sum(share_eligible)
    3. Apply hard cap (cap_max) with optional redistribution
    4. Fallback: if sum(share_eligible) == 0, use top-1 by share or uniform

    Args:
        df: Input dataframe with player rows and shares column
        shares_col: Name of the column containing predicted minute shares
        group_cols: Tuple of (game_id, team_id) column names
        cap_max: Hard cap on minutes per player (default 48)
        require_positive_share: If True, only include players with share > 0
        redistribute_after_cap: If True, redistribute excess from capped players
        max_redistribution_rounds: Max iterations for cap redistribution
        status_col: Column name for player status (None to skip check)
        play_prob_col: Column name for play probability (None to skip check)

    Returns:
        Tuple of (output_df, diagnostics)
        output_df has columns: game_id, team_id, player_id, minutes_mean_A, share,
                               eligible_flag, team_minutes_sum
    """
    diag = ScaleSharesDiagnostics()
    working = df.copy()

    # Ensure shares column exists
    if shares_col not in working.columns:
        raise ValueError(f"Shares column '{shares_col}' not found in dataframe")

    # Auto-detect shares column if needed (try common names)
    shares = pd.to_numeric(working[shares_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Build eligibility mask
    eligible = pd.Series(True, index=working.index)

    if status_col and status_col in working.columns:
        status_upper = working[status_col].astype(str).str.upper().fillna("")
        eligible &= status_upper != "OUT"

    if play_prob_col and play_prob_col in working.columns:
        play_prob = pd.to_numeric(working[play_prob_col], errors="coerce").fillna(1.0)
        eligible &= play_prob > 0.0

    if require_positive_share:
        eligible &= shares > 0.0

    working["_share"] = shares
    working["_eligible"] = eligible.astype(bool)

    diag.n_players = len(working)
    diag.n_eligible = int(eligible.sum())

    # Allocate minutes per team
    minutes_out = np.zeros(len(working), dtype=np.float64)
    team_sum_out = np.zeros(len(working), dtype=np.float64)
    fallback_teams: list[tuple[int, int]] = []
    cap_counts = 0
    max_redist_rounds = 0
    max_dev = 0.0
    all_pre_cap: list[float] = []
    all_post_cap: list[float] = []

    for keys, g in working.groupby(list(group_cols), sort=False):
        game_id, team_id = keys
        idx = g.index.to_numpy()
        share_g = g["_share"].to_numpy(dtype=np.float64)
        elig_g = g["_eligible"].to_numpy(dtype=bool)

        # Compute eligible share sum
        share_eligible = np.where(elig_g, share_g, 0.0)
        share_sum = share_eligible.sum()

        # Fallback if no eligible shares
        if share_sum <= 1e-12:
            fallback_teams.append((int(game_id), int(team_id)))
            # Fallback: top-1 by raw share among non-OUT, or uniform among eligible
            if elig_g.any():
                # Uniform among eligible
                n_elig = elig_g.sum()
                minutes_g = np.where(elig_g, TEAM_TOTAL_MINUTES / n_elig, 0.0)
            else:
                # All players are OUT - pick top-1 by share anyway
                top_idx_local = np.argmax(share_g)
                minutes_g = np.zeros(len(idx), dtype=np.float64)
                minutes_g[top_idx_local] = TEAM_TOTAL_MINUTES
            LOGGER.warning(
                f"[scale_shares] Fallback used for game={game_id} team={team_id}: "
                f"share_sum={share_sum:.6f}, n_eligible={elig_g.sum()}"
            )
        else:
            # Standard allocation: minutes = 240 * share / sum(share)
            minutes_g = TEAM_TOTAL_MINUTES * share_eligible / share_sum

        all_pre_cap.extend(minutes_g[elig_g].tolist())

        # Apply cap with waterfill redistribution to preserve sum=240
        if redistribute_after_cap:
            # Waterfill redistribution: iteratively cap and redistribute
            for round_num in range(max_redistribution_rounds):
                # Check which players exceed cap
                over_cap = minutes_g > cap_max
                if not over_cap.any():
                    break
                
                # Calculate total excess from capped players
                excess = (minutes_g - cap_max)[over_cap].sum()
                if excess < 1e-9:
                    break
                
                # Cap the over-limit players
                minutes_g = np.where(over_cap, cap_max, minutes_g)
                
                # Find players who can receive redistribution (eligible, not capped)
                can_receive = elig_g & (minutes_g < cap_max - 0.1)
                receive_weights = np.where(can_receive, share_eligible, 0.0)
                receive_sum = receive_weights.sum()
                
                if receive_sum > 1e-12:
                    # Distribute excess proportionally to shares
                    minutes_g = np.where(
                        can_receive,
                        minutes_g + excess * receive_weights / receive_sum,
                        minutes_g,
                    )
                else:
                    # No one can receive - we're stuck (all at cap or ineligible)
                    break
                max_redist_rounds = max(max_redist_rounds, round_num + 1)
            
            # Final cap enforcement (shouldn't exceed by much after redistribution)
            minutes_g = np.minimum(minutes_g, cap_max)
            cap_counts += int((minutes_g >= cap_max - 0.1).sum())
        else:
            cap_counts += int((minutes_g >= cap_max).sum())
            minutes_g = np.minimum(minutes_g, cap_max)

        all_post_cap.extend(minutes_g[elig_g].tolist())

        # Validate sum
        team_total = float(minutes_g.sum())
        dev = abs(team_total - TEAM_TOTAL_MINUTES)
        if dev > max_dev:
            max_dev = dev

        minutes_out[idx] = minutes_g
        team_sum_out[idx] = team_total

    # Build output dataframe
    out_cols = [group_cols[0], group_cols[1], "player_id"]
    if "player_id" not in working.columns:
        out_cols = list(group_cols)
    out = working[out_cols].copy()
    out["minutes_mean_A"] = minutes_out
    out["share"] = working["_share"]
    out["eligible_flag"] = working["_eligible"].astype(int)
    out["team_minutes_sum"] = team_sum_out

    # Finalize diagnostics
    diag.n_teams = working.groupby(list(group_cols), sort=False).ngroups
    diag.fallback_count = len(fallback_teams)
    diag.fallback_teams = fallback_teams
    diag.cap_applied_count = cap_counts
    diag.redistribution_rounds_max = max_redist_rounds
    diag.team_sum_dev_max = max_dev

    if all_pre_cap:
        diag.pre_cap_minutes_p95 = float(np.percentile(all_pre_cap, 95))
    if all_post_cap:
        diag.post_cap_minutes_p95 = float(np.percentile(all_post_cap, 95))

    return out, diag


def compare_allocators(
    df_A: pd.DataFrame,
    df_B: pd.DataFrame,
    actual_minutes_df: pd.DataFrame | None = None,
    *,
    minutes_col_A: str = "minutes_mean_A",
    minutes_col_B: str = "minutes_mean",
    player_id_col: str = "player_id",
    group_cols: tuple[str, str] = ("game_id", "team_id"),
    actual_minutes_col: str = "minutes_actual",
) -> AllocatorComparisonMetrics:
    """Compare two allocator outputs and compute comparison metrics.

    Args:
        df_A: Output from Allocator A (scale_shares_to_240)
        df_B: Output from Allocator B (RotAlloc)
        actual_minutes_df: Optional dataframe with actual minutes for accuracy eval
        minutes_col_A: Column name for minutes in df_A
        minutes_col_B: Column name for minutes in df_B
        player_id_col: Column name for player ID
        group_cols: Tuple of column names for team grouping
        actual_minutes_col: Column name for actual minutes

    Returns:
        AllocatorComparisonMetrics with all comparison statistics
    """
    # Merge the two allocator outputs
    merge_cols = [player_id_col] + list(group_cols)
    merged = df_A.merge(
        df_B[[player_id_col, group_cols[0], group_cols[1], minutes_col_B]],
        on=merge_cols,
        how="inner",
        suffixes=("_A", "_B"),
    )

    if len(merged) == 0:
        raise ValueError("No overlapping players between allocator outputs")

    min_A = pd.to_numeric(merged[minutes_col_A], errors="coerce").fillna(0.0).to_numpy()
    min_B = pd.to_numeric(merged[minutes_col_B], errors="coerce").fillna(0.0).to_numpy()

    # Team sum errors
    team_sum_A = merged.groupby(list(group_cols))[minutes_col_A].transform("sum")
    team_sum_B = merged.groupby(list(group_cols))[minutes_col_B].transform("sum")
    team_sum_err_A = (team_sum_A - TEAM_TOTAL_MINUTES).abs()
    team_sum_err_B = (team_sum_B - TEAM_TOTAL_MINUTES).abs()

    # Tail minutes (computed on players with > 0 minutes only)
    played_A = min_A[min_A > 0]
    played_B = min_B[min_B > 0]

    p95_A = float(np.percentile(played_A, 95)) if len(played_A) > 0 else 0.0
    p99_A = float(np.percentile(played_A, 99)) if len(played_A) > 0 else 0.0
    p95_B = float(np.percentile(played_B, 95)) if len(played_B) > 0 else 0.0
    p99_B = float(np.percentile(played_B, 99)) if len(played_B) > 0 else 0.0

    frac_ge_42_A = float((min_A >= 42).mean())
    frac_ge_44_A = float((min_A >= 44).mean())
    frac_ge_42_B = float((min_B >= 42).mean())
    frac_ge_44_B = float((min_B >= 44).mean())

    # Gini coefficient per team
    gini_A_list = []
    gini_B_list = []
    hhi_A_list = []
    hhi_B_list = []
    top6_share_A_list = []
    top6_share_B_list = []
    top8_share_A_list = []
    top8_share_B_list = []
    roster_size_A_list = []
    roster_size_B_list = []
    bench_max_A_list = []
    bench_max_B_list = []

    for _, g in merged.groupby(list(group_cols), sort=False):
        mins_A_g = pd.to_numeric(g[minutes_col_A], errors="coerce").fillna(0.0).to_numpy()
        mins_B_g = pd.to_numeric(g[minutes_col_B], errors="coerce").fillna(0.0).to_numpy()

        # Gini on eligible players
        gini_A_list.append(_gini_coefficient(mins_A_g[mins_A_g > 0]))
        gini_B_list.append(_gini_coefficient(mins_B_g[mins_B_g > 0]))

        # Concentration metrics
        hhi_A_list.append(_hhi(mins_A_g[mins_A_g > 0]))
        hhi_B_list.append(_hhi(mins_B_g[mins_B_g > 0]))

        # Effective roster size
        roster_size_A_list.append(int((mins_A_g >= 1.0).sum()))
        roster_size_B_list.append(int((mins_B_g >= 1.0).sum()))

        # Bench crush: max minutes among bottom-half by minutes
        n = len(mins_A_g)
        half = n // 2
        sorted_A = np.sort(mins_A_g)[::-1]
        sorted_B = np.sort(mins_B_g)[::-1]
        bench_max_A_list.append(float(sorted_A[half:].max()) if half < n else 0.0)
        bench_max_B_list.append(float(sorted_B[half:].max()) if half < n else 0.0)

        # Top-k shares of minutes (by predicted minutes)
        top6_share_A_list.append(float(sorted_A[:6].sum() / TEAM_TOTAL_MINUTES) if sorted_A.size else 0.0)
        top6_share_B_list.append(float(sorted_B[:6].sum() / TEAM_TOTAL_MINUTES) if sorted_B.size else 0.0)
        top8_share_A_list.append(float(sorted_A[:8].sum() / TEAM_TOTAL_MINUTES) if sorted_A.size else 0.0)
        top8_share_B_list.append(float(sorted_B[:8].sum() / TEAM_TOTAL_MINUTES) if sorted_B.size else 0.0)

    # Delta statistics
    delta = min_A - min_B
    delta_abs = np.abs(delta)

    # Prepare metrics
    metrics = AllocatorComparisonMetrics(
        team_sum_error_max_A=float(team_sum_err_A.max()),
        team_sum_error_mean_A=float(team_sum_err_A.drop_duplicates().mean()),
        team_sum_error_max_B=float(team_sum_err_B.max()),
        team_sum_error_mean_B=float(team_sum_err_B.drop_duplicates().mean()),
        p95_minutes_A=p95_A,
        p99_minutes_A=p99_A,
        p95_minutes_B=p95_B,
        p99_minutes_B=p99_B,
        frac_ge_42_A=frac_ge_42_A,
        frac_ge_44_A=frac_ge_44_A,
        frac_ge_42_B=frac_ge_42_B,
        frac_ge_44_B=frac_ge_44_B,
        gini_mean_A=float(np.mean(gini_A_list)) if gini_A_list else 0.0,
        gini_mean_B=float(np.mean(gini_B_list)) if gini_B_list else 0.0,
        hhi_mean_A=float(np.mean(hhi_A_list)) if hhi_A_list else 0.0,
        hhi_mean_B=float(np.mean(hhi_B_list)) if hhi_B_list else 0.0,
        top6_share_mean_A=float(np.mean(top6_share_A_list)) if top6_share_A_list else 0.0,
        top6_share_mean_B=float(np.mean(top6_share_B_list)) if top6_share_B_list else 0.0,
        top8_share_mean_A=float(np.mean(top8_share_A_list)) if top8_share_A_list else 0.0,
        top8_share_mean_B=float(np.mean(top8_share_B_list)) if top8_share_B_list else 0.0,
        roster_size_mean_A=float(np.mean(roster_size_A_list)) if roster_size_A_list else 0.0,
        roster_size_mean_B=float(np.mean(roster_size_B_list)) if roster_size_B_list else 0.0,
        bench_max_mean_A=float(np.mean(bench_max_A_list)) if bench_max_A_list else 0.0,
        bench_max_mean_B=float(np.mean(bench_max_B_list)) if bench_max_B_list else 0.0,
        delta_mean=float(delta.mean()),
        delta_std=float(delta.std()),
        delta_abs_mean=float(delta_abs.mean()),
        delta_abs_max=float(delta_abs.max()),
        n_players=len(merged),
        n_teams=merged.groupby(list(group_cols)).ngroups,
    )

    # If actual minutes provided, compute accuracy metrics
    if actual_minutes_df is not None:
        actual = actual_minutes_df.copy()
        if actual_minutes_col not in actual.columns:
            LOGGER.warning(f"Actual minutes column '{actual_minutes_col}' not found")
        else:
            # Merge actuals
            eval_df = merged.merge(
                actual[[player_id_col, group_cols[0], group_cols[1], actual_minutes_col]],
                on=merge_cols,
                how="left",
            )
            act_min = pd.to_numeric(eval_df[actual_minutes_col], errors="coerce")
            valid = act_min.notna()
            if valid.sum() > 0:
                act = act_min[valid].to_numpy()
                pred_A = pd.to_numeric(eval_df.loc[valid, minutes_col_A], errors="coerce").fillna(0.0).to_numpy()
                pred_B = pd.to_numeric(eval_df.loc[valid, minutes_col_B], errors="coerce").fillna(0.0).to_numpy()

                metrics.mae_A = float(np.abs(act - pred_A).mean())
                metrics.rmse_A = float(np.sqrt(np.mean((act - pred_A) ** 2)))
                metrics.mae_B = float(np.abs(act - pred_B).mean())
                metrics.rmse_B = float(np.sqrt(np.mean((act - pred_B) ** 2)))

                # MAE by bucket
                buckets = {
                    "0-10": (0, 10),
                    "10-20": (10, 20),
                    "20-30": (20, 30),
                    "30+": (30, 100),
                }
                mae_A_bucket = {}
                mae_B_bucket = {}
                for name, (lo, hi) in buckets.items():
                    mask = (act >= lo) & (act < hi)
                    if mask.sum() > 0:
                        mae_A_bucket[name] = float(np.abs(act[mask] - pred_A[mask]).mean())
                        mae_B_bucket[name] = float(np.abs(act[mask] - pred_B[mask]).mean())
                    else:
                        mae_A_bucket[name] = None
                        mae_B_bucket[name] = None
                metrics.mae_by_bucket_A = mae_A_bucket
                metrics.mae_by_bucket_B = mae_B_bucket

                # Sixth-man minutes error: actual 6th man by actual minutes per team-game.
                sixth_abs_A: list[float] = []
                sixth_abs_B: list[float] = []
                eval_valid = eval_df.loc[valid].copy()
                for _, group in eval_valid.groupby(list(group_cols), sort=False):
                    actual_team = pd.to_numeric(group[actual_minutes_col], errors="coerce").fillna(0.0).to_numpy()
                    if actual_team.size < 6:
                        continue
                    order = np.argsort(-actual_team, kind="mergesort")
                    idx6 = int(order[5])
                    sixth_abs_A.append(abs(float(group.iloc[idx6][minutes_col_A]) - float(actual_team[idx6])))
                    sixth_abs_B.append(abs(float(group.iloc[idx6][minutes_col_B]) - float(actual_team[idx6])))
                if sixth_abs_A:
                    metrics.sixth_man_mae_A = float(np.mean(sixth_abs_A))
                    metrics.sixth_man_mae_B = float(np.mean(sixth_abs_B))

    return metrics


def build_players_report(
    df_A: pd.DataFrame,
    df_B: pd.DataFrame,
    features_df: pd.DataFrame | None = None,
    *,
    minutes_col_A: str = "minutes_mean_A",
    minutes_col_B: str = "minutes_mean",
    player_id_col: str = "player_id",
    group_cols: tuple[str, str] = ("game_id", "team_id"),
) -> pd.DataFrame:
    """Build a combined player-level report for debugging.

    Returns dataframe with:
    - player_id, game_id, team_id
    - minutes_mean_A, minutes_mean_B, delta
    - Key features: status, play_prob, share, p_rot, mu_cond, eligible_flag
    """
    merge_cols = [player_id_col] + list(group_cols)

    # Merge A and B
    report = df_A.merge(
        df_B[[player_id_col, group_cols[0], group_cols[1], minutes_col_B]].rename(
            columns={minutes_col_B: "minutes_mean_B"}
        ),
        on=merge_cols,
        how="outer",
    )

    # Compute delta
    report["delta"] = report[minutes_col_A].fillna(0) - report["minutes_mean_B"].fillna(0)
    report["delta_abs"] = report["delta"].abs()

    # Add features if provided
    if features_df is not None:
        feature_cols = [c for c in features_df.columns if c not in report.columns]
        feature_cols = [c for c in feature_cols if c in [
            "status", "play_prob", "p_rot", "mu_cond", "roll_mean_5", "is_starter",
            "player_name", "team_tricode",
        ]]
        if feature_cols:
            report = report.merge(
                features_df[merge_cols + feature_cols],
                on=merge_cols,
                how="left",
            )

    # Also include B's diagnostic columns if available
    for col in ["p_rot", "mu_cond", "eligible_flag"]:
        if col in df_B.columns and col not in report.columns:
            report = report.merge(
                df_B[[player_id_col, group_cols[0], group_cols[1], col]],
                on=merge_cols,
                how="left",
                suffixes=("", "_B"),
            )

    return report.sort_values("delta_abs", ascending=False)


def build_team_summary(
    df_A: pd.DataFrame,
    df_B: pd.DataFrame,
    metrics: AllocatorComparisonMetrics,
    *,
    minutes_col_A: str = "minutes_mean_A",
    minutes_col_B: str = "minutes_mean",
    group_cols: tuple[str, str] = ("game_id", "team_id"),
) -> pd.DataFrame:
    """Build team-level summary with per-team metrics."""
    merge_cols = list(group_cols)

    # Aggregate A
    team_A = df_A.groupby(merge_cols, as_index=False).agg(
        sum_A=(minutes_col_A, "sum"),
        n_eligible_A=("eligible_flag", "sum") if "eligible_flag" in df_A.columns else (minutes_col_A, lambda x: (x > 0).sum()),
        max_A=(minutes_col_A, "max"),
    )

    # Aggregate B
    agg_dict_B = {
        "sum_B": (minutes_col_B, "sum"),
        "max_B": (minutes_col_B, "max"),
    }
    if "eligible_flag" in df_B.columns:
        agg_dict_B["n_eligible_B"] = ("eligible_flag", "sum")
    else:
        agg_dict_B["n_eligible_B"] = (minutes_col_B, lambda x: (x > 0).sum())

    team_B = df_B.groupby(merge_cols, as_index=False).agg(**agg_dict_B)

    # Merge
    team_summary = team_A.merge(team_B, on=merge_cols, how="outer")
    team_summary["sum_error_A"] = (team_summary["sum_A"] - TEAM_TOTAL_MINUTES).abs()
    team_summary["sum_error_B"] = (team_summary["sum_B"] - TEAM_TOTAL_MINUTES).abs()

    return team_summary


__all__ = [
    "scale_shares_to_240",
    "compare_allocators",
    "build_players_report",
    "build_team_summary",
    "ScaleSharesDiagnostics",
    "AllocatorComparisonMetrics",
]
