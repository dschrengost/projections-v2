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


def _project_to_simplex_nonnegative(v: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection of v onto {x >= 0, sum(x) = total}."""
    v = np.asarray(v, dtype=float)
    n = int(v.size)
    if n == 0:
        return v
    total = float(total)
    if total <= 0.0:
        return np.zeros_like(v)

    # Standard simplex projection (Duchi et al. 2008).
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(1, n + 1)) > (cssv - total))[0]
    if rho.size == 0:
        theta = (cssv[-1] - total) / float(n)
    else:
        k = int(rho[-1])
        theta = (cssv[k] - total) / float(k + 1)
    return np.maximum(v - theta, 0.0)


def _project_to_simplex_nonnegative_batch(
    v: np.ndarray,
    active_mask: np.ndarray,
    total: float,
) -> np.ndarray:
    """
    Vectorized simplex projection for batch of rows with varying active sets.

    Projects each row of v onto {x >= 0, sum(x) = total} considering only active positions.
    Inactive positions are set to 0.

    Args:
        v: (W, N) array of values to project
        active_mask: (W, N) bool array indicating active positions per row
        total: target sum for each row

    Returns:
        (W, N) array with projected values (inactive positions = 0)
    """
    v = np.asarray(v, dtype=float)
    active_mask = np.asarray(active_mask, dtype=bool)
    w, n = v.shape
    total = float(total)

    if w == 0 or n == 0:
        return v.copy()
    if total <= 0.0:
        return np.zeros_like(v)

    # Set inactive positions to -inf for sorting (they'll end up last)
    v_masked = np.where(active_mask, v, -np.inf)

    # Sort descending along axis=1
    u = np.sort(v_masked, axis=1)[:, ::-1]

    # Count active per row
    active_counts = active_mask.sum(axis=1)  # (W,)

    # Cumsum of sorted values
    cssv = np.cumsum(u, axis=1)  # (W, N)

    # Build index array: 1, 2, 3, ... N
    idx = np.arange(1, n + 1, dtype=float)  # (N,)

    # Condition: u[k] * (k+1) > cssv[k] - total
    # We need to find the largest k where this holds for each row
    condition = u * idx[None, :] > (cssv - total)

    # Mask out positions beyond the active count for each row
    position_mask = idx[None, :] <= active_counts[:, None]
    condition = condition & position_mask

    # Find rho (largest valid index) per row
    # Use argmax on reversed condition to find last True
    has_valid = condition.any(axis=1)

    # For rows with valid indices, find the last True position
    # Trick: multiply condition by position index, take max
    rho_indices = (condition * idx[None, :]).max(axis=1).astype(int) - 1  # 0-indexed

    # Compute theta for each row
    # theta = (cssv[rho] - total) / (rho + 1)
    theta = np.zeros(w, dtype=float)

    # For rows with valid rho
    valid_rows = has_valid & (rho_indices >= 0)
    if valid_rows.any():
        theta[valid_rows] = (
            cssv[valid_rows, rho_indices[valid_rows]] - total
        ) / (rho_indices[valid_rows] + 1)

    # For rows without valid rho (all values too small), use fallback
    fallback_rows = ~valid_rows & (active_counts > 0)
    if fallback_rows.any():
        # Get the sum of active values for these rows
        active_sums = np.where(active_mask, v, 0.0).sum(axis=1)
        theta[fallback_rows] = (active_sums[fallback_rows] - total) / active_counts[fallback_rows]

    # Project: max(v - theta, 0), but only for active positions
    result = np.maximum(v - theta[:, None], 0.0)
    result = np.where(active_mask, result, 0.0)

    return result


def _project_to_capped_simplex_batch(
    v: np.ndarray,
    active_mask: np.ndarray,
    total: float,
    cap: float,
) -> np.ndarray:
    """
    Vectorized capped simplex projection for batch of rows.

    Projects each row of v onto {0 <= x <= cap, sum(x) = total} considering only active positions.

    Args:
        v: (W, N) array of values to project
        active_mask: (W, N) bool array indicating active positions per row
        total: target sum for each row
        cap: maximum value per element

    Returns:
        (W, N) array with projected values (inactive positions = 0)
    """
    v = np.asarray(v, dtype=float)
    active_mask = np.asarray(active_mask, dtype=bool)
    w, n = v.shape
    total = float(total)
    cap = float(cap)

    if w == 0 or n == 0:
        return v.copy()
    if total <= 0.0:
        return np.zeros_like(v)
    if cap <= 0.0:
        raise ValueError(f"cap must be > 0; got {cap}")

    active_counts = active_mask.sum(axis=1).astype(float)  # (W,)

    # Check feasibility: if active_count * cap < total, cap is infeasible
    # For infeasible rows, just return capped values (won't sum to total)
    cap_feasible = (active_counts * cap) >= total

    result = np.zeros_like(v)

    # Binary search for lambda such that sum(clip(v - lambda, 0, cap)) = total
    # Vectorized: do binary search in parallel for all rows

    # Initialize bounds
    has_active = active_counts > 0.0
    min_v = np.min(np.where(active_mask, v, np.inf), axis=1)
    max_v = np.max(np.where(active_mask, v, -np.inf), axis=1)
    lo = np.where(has_active, min_v - cap, 0.0)  # (W,)
    hi = np.where(has_active, max_v, 0.0)  # (W,)

    # 30 iterations of binary search (plenty for float64 precision)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        # Compute sum of clipped values for each row
        clipped = np.clip(v - mid[:, None], 0.0, cap)
        clipped = np.where(active_mask, clipped, 0.0)
        s = clipped.sum(axis=1)
        # Update bounds
        lo = np.where(s > total, mid, lo)
        hi = np.where(s <= total, mid, hi)

    # Final projection
    result = np.clip(v - hi[:, None], 0.0, cap)
    result = np.where(active_mask, result, 0.0)

    # For infeasible rows, relax cap to make it feasible
    infeasible = ~cap_feasible & (active_counts > 0)
    if infeasible.any():
        # Minimum cap needed: total / active_count
        min_cap = total / np.maximum(active_counts, 1.0)
        # Re-project these rows with relaxed cap
        for r in np.flatnonzero(infeasible):
            row_active = active_mask[r]
            if not row_active.any():
                continue
            cap_eff = float(min_cap[r])
            # Use scalar version for these edge cases
            idxs = np.where(row_active)[0]
            result[r, idxs] = _project_to_capped_simplex(v[r, idxs], total, cap_eff)

    return result


def _project_to_capped_simplex(v: np.ndarray, total: float, cap: float) -> np.ndarray:
    """Euclidean projection of v onto {0 <= x <= cap, sum(x) = total}."""
    v = np.asarray(v, dtype=float)
    n = int(v.size)
    if n == 0:
        return v
    total = float(total)
    cap = float(cap)
    if total <= 0.0:
        return np.zeros_like(v)
    if cap <= 0.0:
        raise ValueError(f"cap must be > 0; got {cap}")
    if float(n) * cap <= total:
        return np.full_like(v, cap)

    # Find lambda such that sum(clip(v - lambda, 0, cap)) = total.
    lo = float(np.min(v - cap))
    hi = float(np.max(v))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = float(np.clip(v - mid, 0.0, cap).sum())
        if s > total:
            lo = mid
        else:
            hi = mid
    return np.clip(v - hi, 0.0, cap)


def recenter_team_minutes_to_conditional_means(
    minutes_world: np.ndarray,  # (W, N)
    active_mask: np.ndarray,  # (W, N) bool
    *,
    target_minutes_conditional: np.ndarray,  # (N,) target E[min | active]
    total_minutes: float = 240.0,
    cap_minutes: float | None = None,
    max_iters: int = 10,
    step: float = 1.0,
    tol: float = 1e-2,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """
    Post-reconcile mean-preservation correction for minutes.

    Adjusts minutes *only* for active players in each world to better match the
    per-player conditional mean targets (E[min | active]) while preserving:
      - per-world team total = total_minutes (when any players are active)
      - non-negativity
      - optional hard cap (skipped per-row when infeasible)

    This is a lightweight iterative method:
      1) compute conditional mean error per player
      2) add a per-player bias to active worlds
      3) project each world back onto the (capped) simplex (L2-minimal per row)

    Note: Uses vectorized batch projections for performance.
    """
    m = np.asarray(minutes_world, dtype=float)
    a = np.asarray(active_mask, dtype=bool)
    if m.shape != a.shape:
        raise ValueError(f"minutes_world and active_mask must have same shape; got {m.shape} vs {a.shape}")
    if m.size == 0:
        return m.copy(), {"enabled": 1, "n_iters": 0, "max_abs_err_before": 0.0, "max_abs_err_after": 0.0}

    w, n = m.shape
    total_minutes = float(total_minutes)
    cap = float(cap_minutes) if cap_minutes is not None else None

    active_counts = a.sum(axis=0).astype(float)
    target = np.asarray(target_minutes_conditional, dtype=float).reshape(-1)
    if target.size != n:
        raise ValueError(f"target_minutes_conditional must have shape ({n},); got {target.shape}")

    # Players never active in this chunk can't be matched; treat their target as 0.
    target = target.copy()
    target[active_counts <= 0.0] = 0.0

    def _conditional_means(x: np.ndarray) -> np.ndarray:
        sums = x.sum(axis=0, dtype=float)
        return np.divide(sums, active_counts, out=np.zeros_like(sums), where=active_counts > 0.0)

    enforce_idx = active_counts > 0.0
    # Weight mean preservation to avoid over-penalizing deep-bench players with ~0 targets.
    # This keeps them from either being forced to exactly 0 (star minutes blow up) or acting as
    # totally free slack (they can soak unrealistic minutes when active).
    weights = np.clip(target / 8.0, 0.2, 1.0)
    weights = np.where(enforce_idx, weights, 0.0)

    m = np.where(a, m, 0.0)
    means0 = _conditional_means(m)
    err0 = target - means0
    max_abs0 = float(np.max(np.abs(err0[enforce_idx]))) if np.any(enforce_idx) else 0.0

    bias = np.zeros(n, dtype=float)
    n_iters_done = 0
    max_abs_bias = 10.0
    for it in range(int(max_iters)):
        n_iters_done = it + 1
        means = _conditional_means(m)
        err = target - means
        max_abs = float(np.max(np.abs(err[enforce_idx]))) if np.any(enforce_idx) else 0.0
        if max_abs <= float(tol):
            break
        bias[enforce_idx] = np.clip(
            bias[enforce_idx] + float(step) * (err[enforce_idx] * weights[enforce_idx]),
            -max_abs_bias,
            max_abs_bias,
        )

        # Vectorized projection: apply bias and project all rows at once
        v = m + bias[None, :] * a.astype(float)
        v = np.where(a, v, 0.0)

        if cap is None:
            m = _project_to_simplex_nonnegative_batch(v, a, total_minutes)
        else:
            m = _project_to_capped_simplex_batch(v, a, total_minutes, cap)

    means1 = _conditional_means(m)
    err1 = target - means1
    max_abs1 = float(np.max(np.abs(err1[enforce_idx]))) if np.any(enforce_idx) else 0.0

    return m, {
        "enabled": 1,
        "n_iters": int(n_iters_done),
        "n_rows": int(w),
        "n_players": int(n),
        "n_enforced_players": int(enforce_idx.sum()),
        "max_abs_err_before": max_abs0,
        "max_abs_err_after": max_abs1,
    }


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
