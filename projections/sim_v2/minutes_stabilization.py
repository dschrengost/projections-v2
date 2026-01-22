"""Minutes stabilization: per-world noise with cheap team-240 projection.

This module provides:
1. Per-world minutes noise sampling (Normal or Student-t, configurable sigma by starter/bench)
2. Fast team-240 projection (not QP - just iterative redistribution)

The algorithm is designed to be deterministic (seeded RNG) and fast enough for live runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class MinutesNoiseStats:
    """Diagnostics from minutes noise application."""

    enabled: bool
    n_teams: int
    n_worlds: int
    max_delta_before_projection: float
    mean_delta_before_projection: float
    teams_requiring_residual_push: int
    frac_teams_residual_push: float
    max_residual_delta: float
    sum_240_violations: int  # teams with abs(sum - 240) > 1e-6


def _position_aware_target_cv(minutes_p50: float) -> float:
    """
    Target coefficient of variation (std/mean) as a function of expected minutes.

    This is a simple tiered heuristic used by tests and optional simulation modes.
    """
    m = float(minutes_p50)
    if m >= 32.0:
        return 0.10
    if m >= 26.0:
        return 0.11
    if m >= 20.0:
        return 0.13
    if m >= 14.0:
        return 0.18
    if m >= 8.0:
        return 0.28
    return 0.45


def sample_minutes_noise_per_world(
    *,
    minutes_reconciled: np.ndarray,  # (P,) reconciled minutes per player
    minutes_p10: np.ndarray,  # (P,) lower quantile
    minutes_p90: np.ndarray,  # (P,) upper quantile
    is_starter: np.ndarray,  # (P,) bool
    team_indices: np.ndarray,  # (P,) int, team code per player
    n_worlds: int,
    active_mask: np.ndarray | None = None,  # (W, P) bool - inactive players are fixed at 0 in that world
    sigma_starter: float = 2.0,
    sigma_bench: float = 3.0,
    use_position_aware_sigma: bool = False,
    min_minutes_for_noise: float = 8.0,
    min_minutes_for_noise_override: float | None = None,
    cap_abs: float = 6.0,
    use_student_t: bool = False,
    t_df: float = 8.0,
    include_tail_in_projection: bool = False,
    tail_min_adjustable_minutes: float = 0.0,
    lo_source: str = "zero",  # "zero" | "p10"
    hi_source: str = "p90",
    lo_pad: float = 0.0,
    hi_pad: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, MinutesNoiseStats]:
    """
    Sample per-world minutes with noise and project back to team total 240.

    Algorithm per world, per team:
    1. Start from reconciled minutes m_i (sum=240).
    2. For players with m_i >= min_minutes_for_noise, sample ε_i:
       - Normal(0, σ) or Student-t(0, σ) where σ depends on starter flag.
       - Hard cap ε_i to [-cap_abs, cap_abs].
    3. Apply m'_i = clamp(m_i + ε_i, lo_i, hi_i).
    4. Project back to 240 via fast redistribution.

    Returns:
        minutes_world: (W, P) array with team sums = 240
        stats: MinutesNoiseStats diagnostics
    """
    if rng is None:
        rng = np.random.default_rng()

    n_players = len(minutes_reconciled)
    if n_players == 0:
        return np.zeros((n_worlds, 0), dtype=float), MinutesNoiseStats(
            enabled=True, n_teams=0, n_worlds=n_worlds,
            max_delta_before_projection=0.0, mean_delta_before_projection=0.0,
            teams_requiring_residual_push=0, frac_teams_residual_push=0.0,
            max_residual_delta=0.0, sum_240_violations=0,
        )

    n_teams = int(team_indices.max()) + 1 if team_indices.size else 0

    # Broadcast baseline to all worlds
    m_base = np.broadcast_to(minutes_reconciled[None, :], (n_worlds, n_players)).copy()

    # Compute sigma per player (starter vs bench)
    if use_position_aware_sigma:
        cv = np.vectorize(_position_aware_target_cv, otypes=[float])(minutes_reconciled)
        sigma_per_player = np.maximum(np.asarray(minutes_reconciled, dtype=float) * cv, 0.0)
    else:
        is_starter_arr = np.asarray(is_starter, dtype=bool)
        sigma_per_player = np.where(is_starter_arr, sigma_starter, sigma_bench)

    noise_threshold = (
        float(min_minutes_for_noise_override)
        if min_minutes_for_noise_override is not None
        else float(min_minutes_for_noise)
    )
    noise_threshold = max(0.0, noise_threshold)

    # Mask: only apply noise to players with enough minutes.
    noise_mask = minutes_reconciled >= noise_threshold

    # Adjustable mask for projection:
    # - default: same as noise_mask (legacy behavior)
    # - when include_tail_in_projection=True: expand adjustable set to include tail.
    if include_tail_in_projection:
        adj_threshold = max(0.0, float(tail_min_adjustable_minutes))
        adjustable_mask = minutes_reconciled >= adj_threshold
    else:
        adjustable_mask = noise_mask.copy()

    # Sample noise
    if use_student_t:
        eps = rng.standard_t(df=t_df, size=(n_worlds, n_players))
        # Scale by sigma
        eps = eps * sigma_per_player[None, :]
    else:
        eps = rng.normal(loc=0.0, scale=sigma_per_player[None, :], size=(n_worlds, n_players))

    # Cap noise
    eps = np.clip(eps, -cap_abs, cap_abs)

    # Zero out noise for players below threshold
    eps = eps * noise_mask.astype(float)

    # Compute bounds
    if lo_source == "p10":
        lo = np.maximum(minutes_p10 - lo_pad, 0.0)
    else:  # "zero"
        lo = np.zeros_like(minutes_reconciled)

    if hi_source == "p90":
        hi = np.minimum(minutes_p90 + hi_pad, 48.0)
    else:
        hi = np.full_like(minutes_reconciled, 48.0)

    # Ensure hi >= lo
    hi = np.maximum(hi, lo)

    # Apply noise and clamp
    m_noisy = m_base + eps
    m_noisy = np.clip(m_noisy, lo[None, :], hi[None, :])

    # Project back to 240 per team per world
    m_final, stats = _project_team_240_fast(
        m_noisy,
        team_indices,
        lo,
        hi,
        adjustable_mask,
        n_worlds,
        n_teams,
        active_mask=active_mask,
    )

    return m_final, stats


def reconcile_team_minutes_active_softmax(
    m0: np.ndarray,  # (W, N)
    active_mask: np.ndarray,  # (W, N) bool
    *,
    total_minutes: float = 240.0,
    eps: float = 1e-6,
    cap_minutes: float | None = None,
    tol: float = 1e-6,
    cap_tol: float = 1e-9,
    max_cap_iters: int = 20,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """
    Reconcile minutes to `total_minutes` per row using active-only softmax.

    - Inactive players are forced to exactly 0.0 and excluded from the softmax denominator.
    - Active players get a non-negative allocation that sums to `total_minutes` (within `tol`).
    - Optional `cap_minutes` applies a hard cap with iterative proportional redistribution
      (weights proportional to pre-cap minutes, i.e. exp(logits)).
    """
    m0 = np.asarray(m0, dtype=float)
    active_mask = np.asarray(active_mask, dtype=bool)
    if m0.shape != active_mask.shape:
        raise ValueError(f"m0 and active_mask must have same shape; got {m0.shape} vs {active_mask.shape}")

    n_rows = int(m0.shape[0])
    if n_rows == 0:
        return m0.copy(), {
            "n_rows": 0,
            "n_all_inactive": 0,
            "n_cap_bind_rows": 0,
            "n_cap_infeasible_rows": 0,
        }

    all_inactive = ~active_mask.any(axis=1)
    n_all_inactive = int(all_inactive.sum())

    # Active-only softmax in log-space: minutes ∝ exp(log(max(m0, eps))) = max(m0, eps).
    logits = np.log(np.maximum(m0, float(eps)))
    logits = np.where(active_mask, logits, -np.inf)

    out = np.zeros_like(m0, dtype=float)
    if (~all_inactive).any():
        logits_valid = logits[~all_inactive]
        max_logits = np.max(logits_valid, axis=1, keepdims=True)
        exps = np.exp(logits_valid - max_logits)
        denom = exps.sum(axis=1, keepdims=True)
        probs = np.divide(exps, denom, out=np.zeros_like(exps), where=denom > 0.0)
        out_valid = float(total_minutes) * probs
        out[~all_inactive] = out_valid

    # Defense-in-depth: keep inactive exactly 0.
    out = np.where(active_mask, out, 0.0)

    if cap_minutes is None:
        return out, {
            "n_rows": n_rows,
            "n_all_inactive": n_all_inactive,
            "n_cap_bind_rows": 0,
            "n_cap_infeasible_rows": 0,
        }

    cap = float(cap_minutes)
    if cap <= 0.0:
        raise ValueError(f"cap_minutes must be > 0; got {cap_minutes}")

    # Cap is infeasible if active_count * cap < total_minutes.
    active_counts = active_mask.sum(axis=1).astype(int)
    cap_infeasible = (active_counts.astype(float) * cap) < (float(total_minutes) - float(tol))
    n_cap_infeasible = int(cap_infeasible.sum())

    # Only apply cap where feasible; keep softmax allocation otherwise.
    pre_cap = out.copy()
    capped = np.where(active_mask, np.minimum(out, cap), 0.0)
    capped[cap_infeasible] = pre_cap[cap_infeasible]

    n_cap_bind_rows = int(((active_mask & (pre_cap > (cap + cap_tol))).any(axis=1) & (~cap_infeasible)).sum())

    # Iterative proportional redistribution of remaining mass for feasible rows only.
    for _ in range(int(max_cap_iters)):
        feasible_idx = np.flatnonzero(~cap_infeasible)
        if feasible_idx.size == 0:
            break

        remaining = float(total_minutes) - capped[feasible_idx].sum(axis=1)
        if np.all(remaining <= tol):
            break

        eligible = active_mask[feasible_idx] & (capped[feasible_idx] < (cap - cap_tol))
        eligible_sum = eligible.sum(axis=1)
        can_add = (remaining > tol) & (eligible_sum > 0)
        if not can_add.any():
            break

        weights = np.where(eligible, pre_cap[feasible_idx], 0.0)
        weight_sum = weights.sum(axis=1)

        # Only allocate on rows that still need minutes and have eligible weight.
        rows = np.flatnonzero(can_add & (weight_sum > 0.0))
        for r in rows:
            row_idx = int(feasible_idx[r])
            add = weights[r] * (remaining[r] / weight_sum[r])
            capped_row = np.minimum(capped[row_idx] + add, cap)
            capped[row_idx] = capped_row

    # Final renormalization for feasible rows (keep caps and inactive zeros).
    feasible_rows = ~cap_infeasible
    if feasible_rows.any():
        row_sum = capped[feasible_rows].sum(axis=1)
        residual = float(total_minutes) - row_sum
        needs_fix = np.abs(residual) > tol
        if needs_fix.any():
            for r in np.flatnonzero(needs_fix):
                full_row_idx = np.flatnonzero(feasible_rows)[r]
                row_active = active_mask[full_row_idx]
                row = capped[full_row_idx]
                if residual[r] > 0:
                    eligible = row_active & (row < (cap - cap_tol))
                    if eligible.any():
                        weights = np.where(eligible, pre_cap[full_row_idx], 0.0)
                        wsum = float(weights.sum())
                        if wsum > 0.0:
                            add = weights * (residual[r] / wsum)
                            row = np.minimum(row + add, cap)
                else:
                    eligible = row_active & (row > 0.0)
                    if eligible.any():
                        weights = np.where(eligible, pre_cap[full_row_idx], 0.0)
                        wsum = float(weights.sum())
                        if wsum > 0.0:
                            sub = weights * ((-residual[r]) / wsum)
                            row = np.maximum(row - sub, 0.0)
                capped[full_row_idx] = np.where(row_active, row, 0.0)

    return capped, {
        "n_rows": n_rows,
        "n_all_inactive": n_all_inactive,
        "n_cap_bind_rows": n_cap_bind_rows,
        "n_cap_infeasible_rows": n_cap_infeasible,
    }


def _project_team_240_fast(
    m_noisy: np.ndarray,  # (W, P)
    team_indices: np.ndarray,  # (P,)
    lo: np.ndarray,  # (P,)
    hi: np.ndarray,  # (P,)
    adjustable_mask: np.ndarray,  # (P,) bool - which players can be adjusted
    n_worlds: int,
    n_teams: int,
    active_mask: np.ndarray | None = None,  # (W, P) bool - inactive are fixed at 0 in that world
) -> tuple[np.ndarray, MinutesNoiseStats]:
    """
    Fast projection to team=240 via iterative redistribution.

    NOT QP - this is a cheap O(W * T * P_per_team) algorithm.
    """
    out = m_noisy.copy()

    # Build team-to-player mapping
    team_to_players = [np.flatnonzero(team_indices == t) for t in range(n_teams)]

    # Diagnostics
    max_delta_before = 0.0
    sum_delta_before = 0.0
    teams_residual_push = 0
    max_residual = 0.0
    sum_240_violations = 0
    total_team_worlds = 0

    for team_players in team_to_players:
        if team_players.size == 0:
            continue

        lo_team = lo[team_players]
        hi_team = hi[team_players]
        adjustable_team_base = adjustable_mask[team_players]

        for w in range(n_worlds):
            total_team_worlds += 1
            m = out[w, team_players].copy()

            if active_mask is not None:
                active_team = np.asarray(active_mask[w, team_players], dtype=bool)
                if not active_team.any():
                    out[w, team_players] = 0.0
                    continue
                m = m * active_team.astype(float)
                adjustable_team = adjustable_team_base & active_team
            else:
                adjustable_team = adjustable_team_base

            current_sum = float(m.sum())
            delta = 240.0 - current_sum

            # Track pre-projection delta
            abs_delta = abs(delta)
            max_delta_before = max(max_delta_before, abs_delta)
            sum_delta_before += abs_delta

            if abs_delta < 1e-6:
                continue  # Already at 240

            # Iterative redistribution (max 3 passes)
            for _ in range(3):
                current_sum = float(m.sum())
                delta = 240.0 - current_sum
                if abs(delta) < 1e-6:
                    break

                if delta > 0:
                    # Need to add minutes: proportional to headroom
                    headroom = (hi_team - m) * adjustable_team.astype(float)
                    headroom = np.maximum(headroom, 0.0)
                    total_headroom = float(headroom.sum())
                    if total_headroom > 1e-6:
                        add = headroom * (delta / total_headroom)
                        m = np.minimum(m + add, hi_team)
                    else:
                        # No headroom in adjustable set - push to max-minute player
                        if active_mask is not None:
                            max_idx = int(np.argmax(np.where(active_team, m, -np.inf)))
                        else:
                            max_idx = int(np.argmax(m))
                        m[max_idx] = min(m[max_idx] + delta, 48.0)
                        teams_residual_push += 1
                        max_residual = max(max_residual, abs(delta))
                else:
                    # Need to remove minutes: proportional to removable
                    removable = (m - lo_team) * adjustable_team.astype(float)
                    removable = np.maximum(removable, 0.0)
                    total_removable = float(removable.sum())
                    if total_removable > 1e-6:
                        sub = removable * (-delta / total_removable)
                        m = np.maximum(m - sub, lo_team)
                    else:
                        # No removable in adjustable set - push to max-minute player
                        if active_mask is not None:
                            max_idx = int(np.argmax(np.where(active_team, m, -np.inf)))
                        else:
                            max_idx = int(np.argmax(m))
                        m[max_idx] = max(m[max_idx] + delta, 0.0)
                        teams_residual_push += 1
                        max_residual = max(max_residual, abs(delta))

            # Final correction to max-minute player if still off
            final_sum = float(m.sum())
            final_delta = 240.0 - final_sum
            if abs(final_delta) > 1e-6:
                if active_mask is not None:
                    max_idx = int(np.argmax(np.where(active_team, m, -np.inf)))
                else:
                    max_idx = int(np.argmax(m))
                m[max_idx] = np.clip(m[max_idx] + final_delta, 0.0, 48.0)
                if abs(final_delta) > 0.01:
                    teams_residual_push += 1
                    max_residual = max(max_residual, abs(final_delta))

            # Check final sum
            check_sum = float(m.sum())
            if abs(check_sum - 240.0) > 1e-6:
                sum_240_violations += 1

            out[w, team_players] = m

    mean_delta_before = sum_delta_before / max(total_team_worlds, 1)
    frac_residual = teams_residual_push / max(total_team_worlds, 1)

    stats = MinutesNoiseStats(
        enabled=True,
        n_teams=n_teams,
        n_worlds=n_worlds,
        max_delta_before_projection=max_delta_before,
        mean_delta_before_projection=mean_delta_before,
        teams_requiring_residual_push=teams_residual_push,
        frac_teams_residual_push=frac_residual,
        max_residual_delta=max_residual,
        sum_240_violations=sum_240_violations,
    )

    return out, stats


def apply_pre_sim_qp_reconcile(
    df: "pd.DataFrame",
    *,
    starter_weight: float = 2.0,
    minutes_weight_scale: float = 1.0,
) -> "pd.DataFrame":
    """
    Apply QP reconciliation to minutes before simulation.

    Uses the existing reconcile_minutes_p50_all from minutes_v1,
    with custom weights based on starter status and minutes.
    """
    from projections.minutes_v1.reconcile import (
        BoundsConfig,
        ReconcileConfig,
        TeamMinutesConfig,
        WeightsConfig,
        reconcile_minutes_p50_all,
    )

    # Build a config suitable for pre-sim reconciliation
    config = ReconcileConfig(
        team_minutes=TeamMinutesConfig(target=240.0, tolerance=0.0),
        p_play_min_rotation=0.0,  # Don't filter by play prob - trust upstream
        min_minutes_for_rotation=0.0,  # Don't filter by minutes - trust upstream
        max_rotation_size=None,  # No cap - trust upstream
        bounds=BoundsConfig(
            starter_floor=0.0,  # Don't enforce starter floor
            p90_cap_multiplier=1.10,
            max_extra_minutes_above_p50=12.0,  # Allow more headroom
            hard_cap=48.0,
        ),
        weights=WeightsConfig(
            starter_penalty=starter_weight,
            rotation_penalty=0.8,
            deep_penalty=0.2,
            spread_epsilon=0.5,
            scale_with_spread=True,
        ),
        clamp_tails=True,
    )

    # Ensure required columns exist
    import pandas as pd

    working = df.copy()
    if "minutes_p50" not in working.columns:
        for col in ("minutes_pred_p50", "minutes_mean"):
            if col in working.columns:
                working["minutes_p50"] = working[col]
                break
    if "minutes_p50" not in working.columns:
        LOGGER.warning("pre_sim_reconcile: no minutes_p50 column found, skipping")
        return df

    # Ensure minutes_p10 and minutes_p90 exist (needed for _compute_weights in reconcile.py)
    # If missing, derive from minutes_p50 with typical spread
    if "minutes_p10" not in working.columns:
        z90 = 1.28
        sigma = pd.to_numeric(working.get("sigma_minutes", 3.0), errors="coerce").fillna(3.0)
        working["minutes_p10"] = np.maximum(
            pd.to_numeric(working["minutes_p50"], errors="coerce") - z90 * sigma, 0.0
        )
    if "minutes_p90" not in working.columns:
        z90 = 1.28
        sigma = pd.to_numeric(working.get("sigma_minutes", 3.0), errors="coerce").fillna(3.0)
        working["minutes_p90"] = np.minimum(
            pd.to_numeric(working["minutes_p50"], errors="coerce") + z90 * sigma, 48.0
        )

    # Run reconciliation
    result = reconcile_minutes_p50_all(working, config)

    # Log summary
    if "minutes_p50_raw" in result.columns:
        raw = result["minutes_p50_raw"]
        reconciled = result["minutes_p50"]
        delta = (reconciled - raw).abs()
        LOGGER.info(
            "[pre_sim_reconcile] applied QP: max_delta=%.2f mean_delta=%.3f",
            delta.max(),
            delta.mean(),
        )

    return result


__all__ = [
    "MinutesNoiseStats",
    "_position_aware_target_cv",
    "apply_pre_sim_qp_reconcile",
    "reconcile_team_minutes_active_softmax",
    "sample_minutes_noise_per_world",
]
