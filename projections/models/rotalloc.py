"""Rotation + minutes + team allocation baseline helpers.

This module provides a deterministic allocator that:
  - combines a rotation probability `p_rot` and conditional mean minutes `mu`
  - converts them into per-player weights
  - allocates exactly 240 minutes per (game_id, team_id)
  - applies per-player caps with water-fill redistribution
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TEAM_TOTAL_MINUTES = 240.0


def waterfill_redistribute(
    minutes: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    *,
    cap_max: float = 48.0,
    target_sum: float = DEFAULT_TEAM_TOTAL_MINUTES,
    eps: float = 1e-12,
) -> np.ndarray:
    """Clamp minutes to [0, cap_max] and redistribute leftover among unclamped players.

    Redistribution is proportional to `weights` over unclamped players, falling back to
    uniform when weights sum to ~0.

    Guarantees:
      - masked-out players have exactly 0 minutes
      - for any non-empty mask, sum(minutes[mask]) == target_sum within 1e-9
    """
    minutes64 = np.asarray(minutes, dtype=np.float64).copy()
    weights64 = np.asarray(weights, dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)

    if minutes64.shape != weights64.shape or minutes64.shape != mask_bool.shape:
        raise ValueError(
            "minutes, weights, and mask must have the same shape. "
            f"minutes={minutes64.shape} weights={weights64.shape} mask={mask_bool.shape}"
        )

    eligible_idx = np.flatnonzero(mask_bool)
    if eligible_idx.size == 0:
        return np.zeros_like(minutes64, dtype=np.float64)

    # Ensure caps are feasible for small candidate sets.
    n_eligible = int(eligible_idx.size)
    cap_eff = float(cap_max)
    min_cap = float(target_sum) / float(n_eligible)
    if cap_eff < min_cap:
        cap_eff = min_cap

    # Mask out ineligible players.
    minutes64[~mask_bool] = 0.0
    minutes64 = np.clip(minutes64, 0.0, cap_eff)

    if not np.isfinite(minutes64[mask_bool]).all():
        raise ValueError("Non-finite minutes provided to waterfill_redistribute()")

    total = float(minutes64[mask_bool].sum())
    if not math.isfinite(total):
        raise ValueError("Non-finite team total in waterfill_redistribute()")

    # If clamping removed minutes, redistribute the leftover.
    leftover = float(target_sum) - total
    tol = 1e-12
    iters = 0
    while leftover > tol:
        active = mask_bool & (minutes64 < cap_eff - tol)
        if not active.any():
            break

        active_w = weights64[active]
        denom = float(active_w.sum())
        if denom <= eps or not math.isfinite(denom):
            add = np.full(active_w.shape, leftover / float(active_w.size), dtype=np.float64)
        else:
            add = leftover * (active_w / denom)

        minutes64[active] += add
        minutes64 = np.clip(minutes64, 0.0, cap_eff)
        total = float(minutes64[mask_bool].sum())
        leftover = float(target_sum) - total

        iters += 1
        if iters > 128:
            break

    # Final exact-sum adjustment (deterministic).
    total = float(minutes64[mask_bool].sum())
    residual = float(target_sum) - total
    if abs(residual) > 1e-9:
        if residual > 0:
            slack = cap_eff - minutes64
            slack[~mask_bool] = -np.inf
            idx = int(np.nanargmax(slack))
            minutes64[idx] += residual
        else:
            candidate = np.where(mask_bool, minutes64, -np.inf)
            idx = int(np.nanargmax(candidate))
            minutes64[idx] += residual

    minutes64[~mask_bool] = 0.0
    total = float(minutes64[mask_bool].sum())
    if abs(total - float(target_sum)) > 1e-9:
        # Last-resort correction (should not trigger in normal conditions).
        idx = int(eligible_idx[0])
        minutes64[idx] += float(target_sum) - total

    minutes64[~mask_bool] = 0.0
    return minutes64


def apply_regulation_cap(
    minutes: np.ndarray,
    weights: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    cap_p50: float = 40.0,
    k_core: int = 8,
    target_sum: float = DEFAULT_TEAM_TOTAL_MINUTES,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Apply post-allocation regulation cap with core rotation redistribution.
    
    This caps minutes_p50 (mean) at cap_p50 and redistributes excess to preserve
    team total of 240, prioritizing core rotation players.
    
    Args:
        minutes: Allocated minutes array (already sums to 240 over eligible)
        weights: Proxy weights (p_rot**a * mu**mu_power) for redistribution priority
        eligible_mask: Boolean mask of eligible players
        cap_p50: Regulation cap for p50/mean (default 40.0)
        k_core: Number of top-proxy players forming core rotation (default 8)
        target_sum: Team total minutes (default 240)
        eps: Epsilon for numerical stability
        
    Returns:
        Tuple of (capped_minutes, diagnostics_dict)
        
    Algorithm:
        1. Clip all eligible players to cap_p50
        2. Compute residual = target_sum - sum(clipped)
        3. Pass 1: Redistribute residual to core rotation (top k_core by proxy)
        4. Pass 2: Redistribute remaining residual to all eligible
        5. If residual still remains, raise cap or fallback
    """
    m = np.asarray(minutes, dtype=np.float64).copy()
    w = np.asarray(weights, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return np.zeros_like(m), {"residual_pass1": 0.0, "residual_pass2": 0.0, "cap_raised": False}
    
    # Ensure cap is feasible for eligible set size
    min_cap = float(target_sum) / float(n_eligible)
    cap_eff = max(float(cap_p50), min_cap)
    cap_raised = cap_eff > float(cap_p50)
    
    # Zero out ineligible
    m[~eligible] = 0.0
    
    # Step 1: Clip to regulation cap
    m_before_clip = m.copy()
    m = np.clip(m, 0.0, cap_eff)
    
    # Step 2: Compute residual
    current_sum = float(m[eligible].sum())
    residual = float(target_sum) - current_sum
    
    # Identify core rotation (top k_core by proxy among eligible)
    ranked_proxy = np.where(eligible, w, -np.inf)
    order = np.argsort(-ranked_proxy, kind="mergesort")
    k_eff = min(k_core, n_eligible)
    core_idx = order[:k_eff]
    core_mask = np.zeros_like(eligible, dtype=bool)
    core_mask[core_idx] = True
    core_mask = core_mask & eligible
    
    residual_pass1 = residual
    residual_pass2 = 0.0
    
    # Pass 1: Redistribute to core rotation
    if residual > eps:
        iters = 0
        while residual > eps and iters < 64:
            active = core_mask & (m < cap_eff - eps)
            if not active.any():
                break
            
            active_w = w[active]
            denom = float(active_w.sum())
            if denom <= eps:
                add = np.full(active_w.shape, residual / float(active_w.size))
            else:
                add = residual * (active_w / denom)
            
            m[active] = np.minimum(m[active] + add, cap_eff)
            current_sum = float(m[eligible].sum())
            residual = float(target_sum) - current_sum
            iters += 1
        
        residual_pass1 = float(target_sum) - float(m[eligible].sum())
    
    # Pass 2: Redistribute remaining to all eligible
    residual_pass2 = float(target_sum) - float(m[eligible].sum())
    if residual_pass2 > eps:
        iters = 0
        while residual_pass2 > eps and iters < 64:
            active = eligible & (m < cap_eff - eps)
            if not active.any():
                break
            
            active_w = w[active]
            denom = float(active_w.sum())
            if denom <= eps:
                add = np.full(active_w.shape, residual_pass2 / float(active_w.size))
            else:
                add = residual_pass2 * (active_w / denom)
            
            m[active] = np.minimum(m[active] + add, cap_eff)
            current_sum = float(m[eligible].sum())
            residual_pass2 = float(target_sum) - current_sum
            iters += 1
    
    # Final exact-sum adjustment
    final_residual = float(target_sum) - float(m[eligible].sum())
    if abs(final_residual) > 1e-9:
        # Add/subtract from highest-minutes player
        candidate = np.where(eligible, m, -np.inf)
        idx = int(np.argmax(candidate))
        m[idx] += final_residual
    
    m[~eligible] = 0.0
    
    diagnostics = {
        "residual_pass1": residual_pass1,
        "residual_pass2": residual_pass2,
        "cap_raised": cap_raised,
        "cap_eff": cap_eff,
        "n_core": int(core_mask.sum()),
        "n_clipped": int((m_before_clip > cap_eff).sum()),
    }
    
    return m, diagnostics


def allocate_two_tier(
    p_rot: np.ndarray,
    mu: np.ndarray,
    mask: np.ndarray,
    *,
    a: float = 1.0,
    mu_power: float = 1.0,
    k_core: int = 8,
    core_minutes_floor: float = 200.0,
    fringe_cap_max: float = 15.0,
    cap_max: float = 48.0,
    target_sum: float = DEFAULT_TEAM_TOTAL_MINUTES,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Two-tier allocation: core rotation + fringe players.
    
    This prevents "scrub mass inflation" by:
    1. Identifying core rotation (top k_core by proxy)
    2. Guaranteeing core gets at least core_minutes_floor (e.g., 200)
    3. Capping each fringe player at fringe_cap_max (e.g., 15)
    
    Args:
        p_rot: Rotation probability for each player
        mu: Conditional minutes (mu_cond) for each player
        mask: Boolean mask of eligible candidates (not OUT, play_prob > 0)
        a: Exponent for p_rot in proxy calculation
        mu_power: Exponent for mu in proxy calculation
        k_core: Number of core rotation players (default 8)
        core_minutes_floor: Minimum total minutes for core (default 200)
        fringe_cap_max: Max minutes for any single fringe player (default 15)
        cap_max: Max minutes for any single core player (default 48)
        target_sum: Team total minutes (default 240)
        eps: Epsilon for numerical stability
        
    Returns:
        Tuple of (allocated_minutes, diagnostics_dict)
    """
    p = np.asarray(p_rot, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    
    n = len(p)
    n_eligible = int(mask_bool.sum())
    
    if n_eligible == 0:
        return np.zeros(n, dtype=np.float64), {
            "core_k": 0, "core_minutes_sum": 0.0, "fringe_minutes_sum": 0.0,
            "max_fringe_minutes": 0.0, "n_fringe": 0,
        }
    
    # Compute proxy for ranking
    p = np.clip(p, 0.0, 1.0)
    m = np.maximum(m, 0.0)
    proxy = np.power(p, float(a)) * np.power(m, float(mu_power))
    proxy = np.where(np.isfinite(proxy), proxy, 0.0)
    proxy = np.where(mask_bool, proxy, -np.inf)
    
    # Identify core vs fringe
    order = np.argsort(-proxy, kind="mergesort")
    k_eff = min(k_core, n_eligible)
    core_idx = order[:k_eff]
    core_mask = np.zeros(n, dtype=bool)
    core_mask[core_idx] = True
    core_mask = core_mask & mask_bool
    
    fringe_mask = mask_bool & ~core_mask
    n_core = int(core_mask.sum())
    n_fringe = int(fringe_mask.sum())
    
    # Compute budgets
    # Ensure core gets at least core_minutes_floor (but can get more if fringe is small)
    max_fringe_budget = float(n_fringe) * float(fringe_cap_max) if n_fringe > 0 else 0.0
    fringe_budget = min(float(target_sum) - float(core_minutes_floor), max_fringe_budget)
    fringe_budget = max(0.0, fringe_budget)
    core_budget = float(target_sum) - fringe_budget
    
    minutes = np.zeros(n, dtype=np.float64)
    
    # Allocate to core
    if n_core > 0:
        core_weights = np.where(core_mask, proxy, 0.0)
        core_weights = np.maximum(core_weights, 0.0)
        w_sum = float(core_weights.sum())
        if w_sum > eps:
            minutes[core_mask] = core_budget * (core_weights[core_mask] / w_sum)
        else:
            minutes[core_mask] = core_budget / float(n_core)
        
        # Apply waterfill capping to core
        minutes = _waterfill_tier(minutes, core_weights, core_mask, cap_max, core_budget, eps)
    
    # Allocate to fringe
    if n_fringe > 0 and fringe_budget > eps:
        fringe_weights = np.where(fringe_mask, proxy, 0.0)
        fringe_weights = np.maximum(fringe_weights, 0.0)
        w_sum = float(fringe_weights.sum())
        if w_sum > eps:
            minutes[fringe_mask] = fringe_budget * (fringe_weights[fringe_mask] / w_sum)
        else:
            minutes[fringe_mask] = fringe_budget / float(n_fringe)
        
        # Apply waterfill capping to fringe
        minutes = _waterfill_tier(minutes, fringe_weights, fringe_mask, fringe_cap_max, fringe_budget, eps)
    
    # Final exact-sum adjustment to hit 240
    total = float(minutes[mask_bool].sum())
    residual = float(target_sum) - total
    if abs(residual) > 1e-9:
        # Add/subtract from highest-minutes player
        candidate = np.where(mask_bool, minutes, -np.inf)
        idx = int(np.argmax(candidate))
        minutes[idx] += residual
    
    minutes[~mask_bool] = 0.0
    
    core_sum = float(minutes[core_mask].sum()) if n_core > 0 else 0.0
    fringe_sum = float(minutes[fringe_mask].sum()) if n_fringe > 0 else 0.0
    max_fringe = float(minutes[fringe_mask].max()) if n_fringe > 0 else 0.0
    
    diagnostics = {
        "core_k": n_core,
        "core_minutes_sum": core_sum,
        "fringe_minutes_sum": fringe_sum,
        "max_fringe_minutes": max_fringe,
        "n_fringe": n_fringe,
        "core_budget": core_budget,
        "fringe_budget": fringe_budget,
    }
    
    return minutes, diagnostics


def allocate_adaptive_depth(
    p_rot: np.ndarray,
    mu: np.ndarray,
    mask: np.ndarray,
    *,
    a: float = 1.0,
    mu_power: float = 1.0,
    bench_share_pred: float = 0.15,
    bench_share_min: float = 0.05,
    bench_share_max: float = 0.35,
    core_k_min: int | None = None,
    core_k_max: int | None = None,
    fringe_cap_max: float = 15.0,
    cap_max: float = 48.0,
    target_sum: float = DEFAULT_TEAM_TOTAL_MINUTES,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Adaptive depth allocation using cumulative proxy mass for core/fringe split.
    
    This replaces fixed k_core with a dynamic threshold based on proxy mass:
    1. Compute normalized proxy weights for all eligible players
    2. Define core as players whose cumulative proxy weight reaches (1 - bench_share_pred)
    3. Allocate (1 - bench_share_pred) * 240 to core, rest to fringe
    4. Cap fringe players at fringe_cap_max to prevent scrub inflation
    
    Args:
        p_rot: Rotation probability for each player
        mu: Conditional minutes (mu_cond) for each player
        mask: Boolean mask of eligible candidates (not OUT, play_prob > 0)
        a: Exponent for p_rot in proxy calculation
        mu_power: Exponent for mu in proxy calculation
        bench_share_pred: Predicted fraction of minutes for fringe players (0.0-1.0)
        bench_share_min: Minimum bench share (floor to avoid degenerate allocations)
        bench_share_max: Maximum bench share (cap to prevent over-spreading)
        fringe_cap_max: Max minutes for any single fringe player
        cap_max: Max minutes for any single core player
        target_sum: Team total minutes (default 240)
        eps: Epsilon for numerical stability
        
    Returns:
        Tuple of (allocated_minutes, diagnostics_dict)
        
    Key insight: Core is defined by cumulative proxy mass, not fixed player count.
    If bench_share_pred=0.15, then players with top 85% of proxy weight are "core".
    """
    p = np.asarray(p_rot, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    mask_bool = np.asarray(mask, dtype=bool)
    
    n = len(p)
    n_eligible = int(mask_bool.sum())
    
    if n_eligible == 0:
        return np.zeros(n, dtype=np.float64), {
            "core_k": 0, "core_minutes_sum": 0.0, "fringe_minutes_sum": 0.0,
            "max_fringe_minutes": 0.0, "n_fringe": 0, "bench_share_pred": bench_share_pred,
            "bench_share_actual": 0.0, "core_proxy_mass": 0.0,
        }
    
    # Clamp bench_share to valid range
    bench_share_eff = float(np.clip(bench_share_pred, bench_share_min, bench_share_max))
    core_share = 1.0 - bench_share_eff
    
    # Compute proxy weights
    p = np.clip(p, 0.0, 1.0)
    m = np.maximum(m, 0.0)
    proxy = np.power(p, float(a)) * np.power(m, float(mu_power))
    proxy = np.where(np.isfinite(proxy), proxy, 0.0)
    proxy = np.where(mask_bool, proxy, 0.0)
    
    proxy_sum = float(proxy.sum())
    if proxy_sum <= eps:
        # All weights zero - uniform fallback
        minutes = np.zeros(n, dtype=np.float64)
        minutes[mask_bool] = float(target_sum) / float(n_eligible)
        return minutes, {
            "core_k": n_eligible, "core_minutes_sum": float(target_sum),
            "fringe_minutes_sum": 0.0, "max_fringe_minutes": 0.0, "n_fringe": 0,
            "bench_share_pred": bench_share_eff, "bench_share_actual": 0.0,
            "core_proxy_mass": 1.0, "fallback": "uniform",
        }
    
    # Normalize proxy weights
    proxy_norm = proxy / proxy_sum
    
    # Sort by proxy descending and compute cumulative mass
    eligible_idx = np.flatnonzero(mask_bool)
    eligible_proxy = proxy[eligible_idx]
    order = np.argsort(-eligible_proxy, kind="mergesort")
    sorted_idx = eligible_idx[order]
    sorted_proxy_norm = proxy_norm[sorted_idx]
    cumsum = np.cumsum(sorted_proxy_norm)
    
    # Find core: players until cumulative proxy reaches core_share.
    # We want to include the player that crosses the threshold (hence +1).
    core_count = int(np.searchsorted(cumsum, core_share, side="right")) + 1
    core_count = min(core_count, n_eligible)
    core_count = max(core_count, 1)  # At least 1 player in core

    # Optional bounds on core size. This helps avoid "core too deep" outcomes
    # when proxy weights are diffuse, which can flatten 6th–8th men.
    if core_k_min is not None or core_k_max is not None:
        k_min_eff = 1
        k_max_eff = n_eligible
        if core_k_min is not None:
            k_min_eff = int(core_k_min)
        if core_k_max is not None:
            k_max_eff = int(core_k_max)
        k_min_eff = max(1, min(k_min_eff, n_eligible))
        k_max_eff = max(1, min(k_max_eff, n_eligible))
        if k_min_eff > k_max_eff:
            k_min_eff = k_max_eff
        core_count = max(k_min_eff, min(core_count, k_max_eff))
    
    core_idx = sorted_idx[:core_count]
    core_mask = np.zeros(n, dtype=bool)
    core_mask[core_idx] = True
    
    fringe_mask = mask_bool & ~core_mask
    n_core = int(core_mask.sum())
    n_fringe = int(fringe_mask.sum())
    
    # Actual core proxy mass (for diagnostics)
    core_proxy_mass = float(proxy_norm[core_mask].sum()) if n_core > 0 else 0.0
    
    # Compute budgets based on actual core/fringe split
    # Use the target bench share for budget, but respect feasibility
    max_fringe_budget = float(n_fringe) * float(fringe_cap_max) if n_fringe > 0 else 0.0
    desired_fringe_budget = bench_share_eff * float(target_sum)
    fringe_budget = min(desired_fringe_budget, max_fringe_budget)
    fringe_budget = max(0.0, fringe_budget)
    core_budget = float(target_sum) - fringe_budget
    
    minutes = np.zeros(n, dtype=np.float64)
    
    # Allocate to core (proportional to proxy)
    if n_core > 0:
        core_weights = np.where(core_mask, proxy, 0.0)
        w_sum = float(core_weights.sum())
        if w_sum > eps:
            minutes[core_mask] = core_budget * (core_weights[core_mask] / w_sum)
        else:
            minutes[core_mask] = core_budget / float(n_core)
        
        # Apply waterfill capping to core
        minutes = _waterfill_tier(minutes, core_weights, core_mask, cap_max, core_budget, eps)
    
    # Allocate to fringe (proportional to proxy, capped at fringe_cap_max)
    if n_fringe > 0 and fringe_budget > eps:
        fringe_weights = np.where(fringe_mask, proxy, 0.0)
        w_sum = float(fringe_weights.sum())
        if w_sum > eps:
            minutes[fringe_mask] = fringe_budget * (fringe_weights[fringe_mask] / w_sum)
        else:
            minutes[fringe_mask] = fringe_budget / float(n_fringe)
        
        # Apply waterfill capping to fringe
        minutes = _waterfill_tier(minutes, fringe_weights, fringe_mask, fringe_cap_max, fringe_budget, eps)
    
    # Final exact-sum adjustment
    total = float(minutes[mask_bool].sum())
    residual = float(target_sum) - total
    if abs(residual) > 1e-9:
        candidate = np.where(mask_bool, minutes, -np.inf)
        idx = int(np.argmax(candidate))
        minutes[idx] += residual
    
    minutes[~mask_bool] = 0.0
    
    core_sum = float(minutes[core_mask].sum()) if n_core > 0 else 0.0
    fringe_sum = float(minutes[fringe_mask].sum()) if n_fringe > 0 else 0.0
    max_fringe = float(minutes[fringe_mask].max()) if n_fringe > 0 else 0.0
    bench_share_actual = fringe_sum / float(target_sum) if target_sum > 0 else 0.0
    
    diagnostics = {
        "core_k": n_core,
        "core_minutes_sum": core_sum,
        "fringe_minutes_sum": fringe_sum,
        "max_fringe_minutes": max_fringe,
        "n_fringe": n_fringe,
        "core_budget": core_budget,
        "fringe_budget": fringe_budget,
        "bench_share_pred": bench_share_eff,
        "bench_share_actual": bench_share_actual,
        "core_proxy_mass": core_proxy_mass,
        "core_k_min": None if core_k_min is None else int(core_k_min),
        "core_k_max": None if core_k_max is None else int(core_k_max),
    }
    
    return minutes, diagnostics


def allocate_fringe_alpha_blend(
    p_rot: np.ndarray,
    mu: np.ndarray,
    share_weight: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    a: float = 1.5,
    mu_power: float = 1.5,
    k_core: int = 8,
    alpha_core: float = 0.8,
    alpha_fringe: float = 0.3,
    share_gamma: float = 1.0,
    cap_max: float = 48.0,
    target_sum: float = DEFAULT_TEAM_TOTAL_MINUTES,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Allocate minutes using a core/fringe blend of share and RotAlloc proxy weights.

    This is a small, deterministic "shape layer" designed to fix the common failure mode
    where RotAlloc proxy weights are diffuse and the allocator assigns ~uniform minutes
    across ranks 6–11 (bench spread too wide; 6th–8th men crushed).

    Algorithm (single team-game):
      1) Compute RotAlloc proxy weights: w_rot = (p_rot ** a) * (mu ** mu_power)
      2) Build a core set: top-k_core eligible players by w_rot (stable ties)
      3) Compute normalized weights within eligible: w_rot_norm, w_share_norm
      4) Blend weights:
           core:   w = alpha_core  * w_share_norm + (1-alpha_core)  * w_rot_norm
           fringe: w = alpha_fringe* w_share_norm + (1-alpha_fringe)* w_rot_norm
      5) Allocate exactly target_sum minutes with waterfill capping.

    Notes:
      - share_weight can be any non-negative proxy (e.g., roll_mean_5 minutes).
      - This allocator never gives minutes to ineligible players.
    """
    p = np.clip(np.asarray(p_rot, dtype=np.float64), 0.0, 1.0)
    m = np.maximum(np.asarray(mu, dtype=np.float64), 0.0)
    s = np.maximum(np.asarray(share_weight, dtype=np.float64), 0.0)
    eligible = np.asarray(eligible_mask, dtype=bool)

    if p.shape != m.shape or p.shape != s.shape or p.shape != eligible.shape:
        raise ValueError(
            "p_rot, mu, share_weight, and eligible_mask must have the same shape. "
            f"p_rot={p.shape} mu={m.shape} share_weight={s.shape} eligible_mask={eligible.shape}"
        )

    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return np.zeros_like(p, dtype=np.float64), {
            "core_k": 0,
            "n_eligible": 0,
            "fallback": "empty",
        }

    k_core_eff = int(k_core)
    if k_core_eff <= 0:
        k_core_eff = 0
    k_core_eff = min(k_core_eff, n_eligible)

    alpha_core_eff = float(np.clip(alpha_core, 0.0, 1.0))
    alpha_fringe_eff = float(np.clip(alpha_fringe, 0.0, 1.0))

    gamma_eff = float(share_gamma)
    if not math.isfinite(gamma_eff) or gamma_eff <= 0.0:
        gamma_eff = 1.0
    if abs(gamma_eff - 1.0) > 1e-9:
        s = np.power(s, gamma_eff)

    w_rot = np.power(p, float(a)) * np.power(m, float(mu_power))
    w_rot = np.where(np.isfinite(w_rot) & eligible, w_rot, 0.0)
    w_share = np.where(np.isfinite(s) & eligible, s, 0.0)

    w_rot_sum = float(w_rot.sum())
    w_share_sum = float(w_share.sum())

    # Normalize weights within eligible, with uniform fallback.
    if w_rot_sum > eps:
        w_rot_norm = w_rot / w_rot_sum
    else:
        w_rot_norm = np.zeros_like(w_rot)
        w_rot_norm[eligible] = 1.0 / float(n_eligible)

    if w_share_sum > eps:
        w_share_norm = w_share / w_share_sum
    else:
        w_share_norm = np.zeros_like(w_share)
        w_share_norm[eligible] = 1.0 / float(n_eligible)

    # Stable core selection by w_rot (not normalized).
    core_mask = np.zeros_like(eligible, dtype=bool)
    if k_core_eff > 0:
        ranked = np.where(eligible, w_rot, -np.inf)
        order = np.argsort(-ranked, kind="mergesort")
        core_idx = order[:k_core_eff]
        core_mask[core_idx] = True
        core_mask &= eligible

    # Blend weights by tier.
    w = np.zeros_like(w_rot_norm)
    if k_core_eff > 0:
        w[core_mask] = alpha_core_eff * w_share_norm[core_mask] + (1.0 - alpha_core_eff) * w_rot_norm[core_mask]
    fringe_mask = eligible & ~core_mask
    if fringe_mask.any():
        w[fringe_mask] = alpha_fringe_eff * w_share_norm[fringe_mask] + (1.0 - alpha_fringe_eff) * w_rot_norm[fringe_mask]

    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)

    minutes = waterfill_redistribute(
        np.zeros_like(w, dtype=np.float64),
        w,
        eligible,
        cap_max=cap_max,
        target_sum=target_sum,
    )

    diagnostics = {
        "core_k": int(core_mask.sum()),
        "n_eligible": n_eligible,
        "alpha_core": alpha_core_eff,
        "alpha_fringe": alpha_fringe_eff,
        "share_gamma": gamma_eff,
        "w_rot_sum": w_rot_sum,
        "w_share_sum": w_share_sum,
    }
    return minutes, diagnostics


def compute_bench_share_prior(
    team_bench_share_avg: float | None = None,
    spread: float | None = None,
    total: float | None = None,
    out_count: int = 0,
    *,
    league_bench_share: float = 0.15,
    spread_coef: float = -0.002,
    total_coef: float = 0.0003,
    out_coef: float = 0.015,
    team_prior_weight: float = 0.6,
) -> float:
    """Compute team-game depth prior (predicted bench_share).
    
    Bench share = fraction of 240 minutes going to fringe players.
    Higher bench share = more spread-out rotation, lower = more top-heavy.
    
    Model:
        bench_share = team_prior * team_weight + context_adj * (1 - team_weight)
        context_adj = league_base + spread_effect + total_effect + out_effect
        
    Args:
        team_bench_share_avg: Team's rolling average bench share (0.0-1.0)
        spread: Point spread (negative = favorite, positive = underdog)
        total: Game total (higher = faster pace, more spread)
        out_count: Number of players marked OUT for this team-game
        league_bench_share: League-wide fallback when no team prior
        spread_coef: Effect of spread on bench share (favorites play starters more)
        total_coef: Effect of total on bench share (high-scoring games spread minutes)
        out_coef: Effect of OUT players on bench share (injuries spread minutes)
        team_prior_weight: Weight for team-specific prior vs context adjustment
        
    Returns:
        Predicted bench share (0.05 - 0.35 range)
        
    Example:
        - Neutral game, no injuries: ~0.15
        - Big favorite (-10): 0.15 - 0.002*(-10) = 0.17 (more rest for starters)
        - Heavy underdog (+10): 0.15 - 0.002*(10) = 0.13 (play starters more)
        - 3 players OUT: +0.045 (remaining players spread minutes)
    """
    # Start with league-wide base
    context_adj = float(league_bench_share)
    
    # Spread effect: favorites spread minutes more (rest starters)
    # Underdogs play starters more (negative effect)
    if spread is not None and np.isfinite(spread):
        context_adj += float(spread_coef) * float(spread)
    
    # Total effect: high-scoring games slightly spread minutes
    if total is not None and np.isfinite(total):
        # Center around league average (~220-225)
        total_centered = float(total) - 222.0
        context_adj += float(total_coef) * total_centered
    
    # OUT effect: injuries force bench depth
    if out_count > 0:
        context_adj += float(out_coef) * min(int(out_count), 5)
    
    # Blend team prior with context adjustment
    if team_bench_share_avg is not None and np.isfinite(team_bench_share_avg):
        bench_share = (
            float(team_prior_weight) * float(team_bench_share_avg) +
            (1.0 - float(team_prior_weight)) * context_adj
        )
    else:
        bench_share = context_adj
    
    # Clamp to valid range
    return float(np.clip(bench_share, 0.05, 0.35))


def compute_sixth_man_crush_metric(
    minutes: np.ndarray,
    mu_scaled: np.ndarray,
    proxy: np.ndarray,
    mask: np.ndarray,
    *,
    rank_start: int = 6,
    rank_end: int = 8,
) -> float:
    """Compute the 6th-man crush metric: haircut ratio for ranks 6-8.
    
    This measures how much the 6th-8th best players (by proxy) are getting
    "crushed" relative to their expected minutes (mu_scaled).
    
    Returns:
        Average ratio of (allocated / mu_scaled) for ranks 6-8.
        Values < 0.85 indicate significant crushing of bench rotation.
        Values near 1.0 indicate healthy allocation.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    n_eligible = int(mask_bool.sum())
    
    if n_eligible < rank_start:
        return float("nan")
    
    # Rank by proxy
    eligible_idx = np.flatnonzero(mask_bool)
    eligible_proxy = proxy[eligible_idx]
    order = np.argsort(-eligible_proxy, kind="mergesort")
    sorted_idx = eligible_idx[order]
    
    # Get ranks 6-8 (0-indexed: 5-7)
    start_idx = rank_start - 1
    end_idx = min(rank_end, n_eligible)
    
    if start_idx >= end_idx:
        return float("nan")
    
    target_idx = sorted_idx[start_idx:end_idx]
    
    # Compute haircut ratio
    allocated = minutes[target_idx]
    expected = mu_scaled[target_idx]
    
    # Avoid division by zero
    valid = expected > 1.0
    if not valid.any():
        return float("nan")
    
    ratios = allocated[valid] / expected[valid]
    return float(np.mean(ratios))


def _waterfill_tier(
    minutes: np.ndarray,
    weights: np.ndarray,
    tier_mask: np.ndarray,
    cap: float,
    target_sum: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply waterfill redistribution within a tier to respect cap."""
    m = minutes.copy()
    tier_idx = np.flatnonzero(tier_mask)
    if tier_idx.size == 0:
        return m
    
    # Clip to cap
    m[tier_mask] = np.minimum(m[tier_mask], cap)
    
    # Redistribute if capping reduced sum below target
    current = float(m[tier_mask].sum())
    leftover = target_sum - current
    
    iters = 0
    while leftover > eps and iters < 64:
        active = tier_mask & (m < cap - eps)
        if not active.any():
            break
        
        active_w = weights[active]
        denom = float(active_w.sum())
        if denom <= eps:
            add = np.full(active_w.shape, leftover / float(active_w.size))
        else:
            add = leftover * (active_w / denom)
        
        m[active] = np.minimum(m[active] + add, cap)
        current = float(m[tier_mask].sum())
        leftover = target_sum - current
        iters += 1
    
    return m


def build_eligible_mask(
    p_rot: np.ndarray,
    mu: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    a: float = 1.5,
    mu_power: float = 1.0,
    p_cutoff: float | None = None,
    topk: int | None = None,
    k_min: int = 8,
    k_max: int = 11,
    use_expected_k: bool = False,
) -> np.ndarray:
    """Build a pruning mask from rotation + minutes priors.

    Pruning logic:
      - If p_cutoff is not None: exclude players with p_rot < p_cutoff.
      - If use_expected_k: k = clip(round(sum(p_rot over remaining eligible)), k_min, k_max)
        BUT: never shrink below the cutoff set size when cutoff set is already <= k_max.
        This prevents good rotation players from being excluded just because sum(p_rot) is low.
      - Else if topk is not None: keep topk by proxy.
      - If all weights zero after pruning, fall back to top-1 by proxy over mask.
    """
    p = np.asarray(p_rot, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    if p.shape != m.shape:
        raise ValueError(f"p_rot and mu must have the same shape: {p.shape} vs {m.shape}")
    if p.ndim != 1:
        raise ValueError("build_eligible_mask expects 1D arrays for a single team-game")

    if mask is None:
        mask_bool = np.ones_like(p, dtype=bool)
    else:
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != p.shape:
            raise ValueError(f"mask must match shape: mask={mask_bool.shape} p_rot={p.shape}")

    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)

    p = np.clip(p, 0.0, 1.0)
    m = np.maximum(m, 0.0)
    proxy = np.power(p, float(a)) * np.power(m, float(mu_power))
    proxy = np.where(np.isfinite(proxy), proxy, 0.0)

    eligible = mask_bool.copy()
    if p_cutoff is not None:
        eligible = eligible & (p >= float(p_cutoff))

    cutoff_set_size = int(eligible.sum())
    
    k = None
    if use_expected_k:
        expected_k = int(np.round(p[eligible].sum()))
        # Clamp expected_k to [k_min, k_max]
        k = max(int(k_min), min(int(k_max), expected_k))
        # KEY FIX: Never shrink below the cutoff set size if it's already <= k_max
        # This ensures players above p_cutoff stay eligible when the team has injuries
        if cutoff_set_size <= k_max:
            k = max(k, cutoff_set_size)
    elif topk is not None:
        if int(topk) > 0:
            k = int(topk)

    if k is not None and k < cutoff_set_size:
        # Only apply top-k filtering if we're actually shrinking
        ranked_proxy = np.where(eligible, proxy, -np.inf)
        order = np.argsort(-ranked_proxy, kind="mergesort")
        k_eff = min(int(k), int(eligible.sum()))
        top_idx = order[:k_eff]
        eligible = np.zeros_like(mask_bool, dtype=bool)
        eligible[top_idx] = True
        eligible = eligible & mask_bool

    if not np.any(eligible):
        # Fallback: top-1 by proxy over mask (ignores p_cutoff).
        ranked_proxy = np.where(mask_bool, proxy, -np.inf)
        best = int(np.argmax(ranked_proxy))
        eligible = np.zeros_like(mask_bool, dtype=bool)
        eligible[best] = True

    return eligible


def allocate_team_minutes(
    p_rot: np.ndarray,
    mu: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    a: float = 1.5,
    mu_power: float = 1.0,
    cap_max: float = 48.0,
    target_sum: float = DEFAULT_TEAM_TOTAL_MINUTES,
    eps: float = 1e-12,
    p_cutoff: float | None = None,
    topk: int | None = None,
    k_min: int = 8,
    k_max: int = 11,
    use_expected_k: bool = False,
) -> np.ndarray:
    """Allocate minutes to a team-game using rotation probabilities + conditional minutes.

    weights w_i = (p_rot_i ** a) * max(mu_i, 0)
    minutes_i = 240 * w_i / sum(w_i) over eligible players
    then apply capping + water-fill redistribution.
    """
    p = np.asarray(p_rot, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    if p.shape != m.shape:
        raise ValueError(f"p_rot and mu must have the same shape: {p.shape} vs {m.shape}")
    if p.ndim != 1:
        raise ValueError("allocate_team_minutes expects 1D arrays for a single team-game")

    if mask is None:
        mask_bool = np.ones_like(p, dtype=bool)
    else:
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != p.shape:
            raise ValueError(f"mask must match shape: mask={mask_bool.shape} p_rot={p.shape}")

    eligible = build_eligible_mask(
        p,
        m,
        mask_bool,
        a=float(a),
        mu_power=float(mu_power),
        p_cutoff=p_cutoff,
        topk=topk,
        k_min=int(k_min),
        k_max=int(k_max),
        use_expected_k=bool(use_expected_k),
    )
    if not eligible.any():
        return np.zeros_like(p, dtype=np.float64)

    p = np.clip(p, 0.0, 1.0)
    m = np.maximum(m, 0.0)
    w = np.power(p, float(a)) * np.power(m, float(mu_power))
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.where(eligible, w, 0.0)

    w_sum = float(w.sum())
    if w_sum <= eps or not math.isfinite(w_sum):
        minutes = np.zeros_like(w, dtype=np.float64)
        minutes[eligible] = float(target_sum) / float(int(eligible.sum()))
    else:
        minutes = (float(target_sum) * w / w_sum).astype(np.float64)

    minutes = waterfill_redistribute(
        minutes,
        w,
        eligible,
        cap_max=float(cap_max),
        target_sum=float(target_sum),
        eps=float(eps),
    )
    return minutes


def leak_dnp_by_teamgame(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str = "minutes_actual",
    group_cols: Iterable[str] = ("game_id", "team_id"),
) -> pd.Series:
    """Leak DNP minutes per team-game: sum(pred minutes where actual==0)."""
    group_cols = tuple(group_cols)
    dnp = df[actual_col] == 0
    totals = df.groupby(list(group_cols))[pred_col].sum()
    leak = df.loc[dnp].groupby(list(group_cols))[pred_col].sum()
    return leak.reindex(totals.index, fill_value=0.0)


def topk_overlap(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str = "minutes_actual",
    id_col: str = "player_id",
    top_k: int = 8,
    group_cols: Iterable[str] = ("game_id", "team_id"),
) -> float:
    """Average per-team-game top-k overlap (set overlap fraction)."""
    group_cols = tuple(group_cols)
    overlaps: list[float] = []
    for _, g in df.groupby(list(group_cols), sort=False):
        if g.empty:
            continue
        k = min(int(top_k), int(len(g)))
        if k <= 0:
            continue
        top_actual = set(g.nlargest(k, actual_col)[id_col].tolist())
        top_pred = set(g.nlargest(k, pred_col)[id_col].tolist())
        overlaps.append(len(top_actual & top_pred) / k)
    return float(np.mean(overlaps)) if overlaps else 0.0


__all__ = [
    "allocate_team_minutes",
    "allocate_two_tier",
    "allocate_adaptive_depth",
    "apply_regulation_cap",
    "build_eligible_mask",
    "waterfill_redistribute",
    "leak_dnp_by_teamgame",
    "topk_overlap",
    "compute_bench_share_prior",
    "compute_sixth_man_crush_metric",
]
