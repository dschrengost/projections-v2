"""Weighted, bounded team-minutes allocator for sim_v2.

Goal: per (team, world) project noisy conditional minutes ("demand") onto a feasible
allocation that sums to `target_total` (NBA team minutes = 240) while protecting
high-priority players from being squeezed.

We solve the convex QP:
  minimize    Σ_i w_i (m_i - d_i)^2
  subject to  Σ_i m_i = target_total
              0 <= m_i <= cap_i
              m_i = 0 if inactive

Closed-form (for a given Lagrange multiplier λ):
  m_i(λ) = clip(d_i - λ / (2 w_i), 0, cap_i)

We find λ via bisection ("weighted waterfilling with bounds").
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _as_bool_mask(values: Any) -> np.ndarray:
    """Best-effort conversion for bool/int/float/object masks to bool ndarray."""
    # pd.to_numeric handles bool -> {0,1} and strings -> NaN (coerced)
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return arr > 0.5


def _normalize_priority_unit_interval(priority: np.ndarray, active: np.ndarray, eps: float) -> np.ndarray:
    """Normalize priority to [0,1] within the active set; inactive values become 0."""
    z = np.zeros_like(priority, dtype=float)
    if not np.any(active):
        return z
    p_act = priority[active].astype(float, copy=False)
    p_min = float(np.min(p_act))
    p_max = float(np.max(p_act))
    denom = p_max - p_min
    if denom <= eps:
        # Constant priority within the active set => equal protection.
        z[active] = 0.0
    else:
        z[active] = (p_act - p_min) / denom
    return z


def allocate_team_minutes(
    df_team: pd.DataFrame,
    demand_col: str,
    active_col: str,
    priority_col: str,
    cap_col: str | None = None,
    target_total: float = 240.0,
    k: float = 3.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Allocate team minutes for a single team in a single world.

    Returns an ndarray aligned to df_team rows.
    """
    n = int(len(df_team))
    if n == 0:
        return np.zeros(0, dtype=float)

    target = float(target_total)
    if target <= 0.0:
        return np.zeros(n, dtype=float)

    active = _as_bool_mask(df_team[active_col])
    if not np.any(active):
        return np.zeros(n, dtype=float)

    demand = pd.to_numeric(df_team[demand_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    demand = np.clip(demand, 0.0, None)
    demand = np.where(active, demand, 0.0)

    if float(demand[active].sum()) <= eps:
        return np.zeros(n, dtype=float)

    if cap_col is None:
        cap = np.full(n, 48.0, dtype=float)
    else:
        cap = pd.to_numeric(df_team[cap_col], errors="coerce").fillna(48.0).to_numpy(dtype=float)
    cap = np.clip(cap, 0.0, None)
    cap = np.where(active, cap, 0.0)

    cap_sum = float(cap[active].sum())
    if cap_sum + eps < target:
        # Infeasible under caps (should be rare); return the max-possible minutes.
        LOGGER.warning(
            "minutes allocator infeasible under caps: cap_sum=%.3f target=%.3f n_active=%d",
            cap_sum,
            target,
            int(active.sum()),
        )
        return cap

    priority = pd.to_numeric(df_team[priority_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    z = _normalize_priority_unit_interval(priority, active=active, eps=eps)
    w = np.exp(float(k) * z)
    w = np.where(active, w, 1.0)

    # Active-only arrays for solving.
    d = demand[active]
    c = cap[active]
    ww = w[active]

    # Bracket λ such that:
    # - at λ_lo => everyone at cap  (sum = cap_sum)
    # - at λ_hi => everyone at 0    (sum = 0)
    lam_lo = float(np.min(2.0 * ww * (d - c))) - 1.0
    lam_hi = float(np.max(2.0 * ww * d)) + 1.0

    lo = lam_lo
    hi = lam_hi
    mid = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m = np.clip(d - mid / (2.0 * ww), 0.0, c)
        s = float(m.sum())
        if abs(s - target) <= 1e-6:
            break
        if s > target:
            # Need to reduce minutes -> increase λ.
            lo = mid
        else:
            # Need to increase minutes -> decrease λ.
            hi = mid

    m = np.clip(d - mid / (2.0 * ww), 0.0, c)
    out = np.zeros(n, dtype=float)
    out[active] = m
    return out


def allocate_team_minutes_matrix(
    demand_worlds: np.ndarray,  # (W, N)
    active_mask: np.ndarray,  # (W, N)
    *,
    priority: np.ndarray,  # (N,)
    cap: float | np.ndarray = 48.0,
    max_increase: np.ndarray | None = None,  # (N,) optional additional cap above anchor
    baseline: np.ndarray | None = None,  # (N,) optional anchor minutes (max(demand, baseline))
    target_total: float = 240.0,
    k: float = 3.0,
    eps: float = 1e-6,
    max_iters: int = 60,
    return_infeasible_mask: bool = False,
) -> tuple[np.ndarray, dict[str, float | int | np.ndarray]]:
    """Vectorized allocator for a single team across many worlds.

    This matches allocate_team_minutes semantics, but solves one λ per world row.
    """
    d0 = np.asarray(demand_worlds, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    if d0.shape != active.shape:
        raise ValueError(f"demand_worlds and active_mask must have same shape; got {d0.shape} vs {active.shape}")

    n_rows, n_players = d0.shape
    if n_rows == 0 or n_players == 0:
        return d0.copy(), {"n_rows": int(n_rows), "n_all_inactive": int(n_rows)}

    target = float(target_total)
    if target <= 0.0:
        return np.zeros_like(d0), {"n_rows": int(n_rows), "n_all_inactive": int((~active.any(axis=1)).sum())}

    p = np.asarray(priority, dtype=float)
    if p.shape != (n_players,):
        raise ValueError(f"priority must have shape (N,), got {p.shape}")

    if np.isscalar(cap):
        cap_vec = np.full(n_players, float(cap), dtype=float)
    else:
        cap_vec = np.asarray(cap, dtype=float)
        if cap_vec.shape != (n_players,):
            raise ValueError(f"cap must be a scalar or have shape (N,), got {cap_vec.shape}")
    cap_vec = np.clip(cap_vec, 0.0, None)

    # Apply play/inactive semantics: inactive players are fixed at 0 with cap=0.
    demand = np.clip(d0, 0.0, None) * active.astype(float)

    if max_increase is None:
        cap_eff = cap_vec[None, :] * active.astype(float)
    else:
        inc = np.asarray(max_increase, dtype=float)
        if inc.shape != (n_players,):
            raise ValueError(f"max_increase must have shape (N,), got {inc.shape}")
        inc = np.clip(inc, 0.0, None)

        if baseline is None:
            anchor = demand
        else:
            base = np.asarray(baseline, dtype=float)
            if base.shape != (n_players,):
                raise ValueError(f"baseline must have shape (N,), got {base.shape}")
            anchor = np.maximum(demand, base[None, :])

        cap_dyn = anchor + inc[None, :]
        cap_eff = np.minimum(cap_vec[None, :], cap_dyn) * active.astype(float)

    all_inactive = ~active.any(axis=1)
    n_all_inactive = int(all_inactive.sum())

    # Degenerate: if a row has no demand among active players, return zeros (matches single-world behavior).
    demand_row_sum = demand.sum(axis=1)
    zero_demand = (demand_row_sum <= eps) & (~all_inactive)

    # Cap feasibility per row.
    cap_row_sum = cap_eff.sum(axis=1)
    cap_infeasible = (cap_row_sum > 0.0) & (cap_row_sum + eps < target)
    n_cap_infeasible = int(cap_infeasible.sum())

    out = np.zeros_like(demand)
    # Infeasible rows: return max-possible under caps.
    if cap_infeasible.any():
        out[cap_infeasible] = cap_eff[cap_infeasible]

    # Solvable rows: bisection on λ.
    solvable = (~all_inactive) & (~zero_demand) & (~cap_infeasible)
    if solvable.any():
        # Per-row normalization of priority to [0,1] among active players.
        # Inactive players get z=0 (weight=1, but cap=0 anyway).
        p_row = p[None, :]  # (1, N) broadcast
        p_min = np.min(np.where(active, p_row, np.inf), axis=1)
        p_max = np.max(np.where(active, p_row, -np.inf), axis=1)
        denom = p_max - p_min
        denom = np.where(denom > eps, denom, 1.0)  # avoid divide-by-zero
        z = (p_row - p_min[:, None]) / denom[:, None]
        z = np.clip(z, 0.0, 1.0)
        z = np.where(active, z, 0.0)
        w = np.exp(float(k) * z)
        w = np.where(active, w, 1.0)

        # Bracket λ per row.
        lam_lo = np.min(2.0 * w * (demand - cap_eff), axis=1) - 1.0
        lam_hi = np.max(2.0 * w * demand, axis=1) + 1.0

        lo = lam_lo.copy()
        hi = lam_hi.copy()

        # Only update solvable rows.
        for _ in range(int(max_iters)):
            mid = 0.5 * (lo + hi)
            m = np.clip(demand - mid[:, None] / (2.0 * w), 0.0, cap_eff)
            s = m.sum(axis=1)
            too_high = s > target
            # If sum too high -> increase λ (move lo up). Else decrease λ (move hi down).
            lo = np.where(solvable & too_high, mid, lo)
            hi = np.where(solvable & (~too_high), mid, hi)

            max_abs = float(np.max(np.abs(s[solvable] - target))) if solvable.any() else 0.0
            if max_abs <= 1e-6:
                break

        mid = 0.5 * (lo + hi)
        m = np.clip(demand - mid[:, None] / (2.0 * w), 0.0, cap_eff)
        out[solvable] = m[solvable]

    # zero-demand rows -> all zeros (already).
    out[zero_demand] = 0.0
    out[all_inactive] = 0.0

    # Keep inactive exactly 0.
    out = np.where(active, out, 0.0)

    sums = out.sum(axis=1)
    residual = np.zeros(n_rows, dtype=float)
    residual[~all_inactive] = sums[~all_inactive] - target
    max_abs_residual = float(np.max(np.abs(residual[solvable]))) if solvable.any() else 0.0

    cap_bind = active & (out >= (cap_eff - 1e-9))
    n_cap_bind_rows = int((cap_bind.any(axis=1) & (~cap_infeasible) & (~all_inactive)).sum())

    return out, {
        "n_rows": int(n_rows),
        "n_all_inactive": int(n_all_inactive),
        "n_cap_infeasible_rows": int(n_cap_infeasible),
        "n_cap_bind_rows": int(n_cap_bind_rows),
        "max_abs_residual": float(max_abs_residual),
        **({"cap_infeasible_mask": cap_infeasible} if return_infeasible_mask else {}),
    }


__all__ = ["allocate_team_minutes", "allocate_team_minutes_matrix"]
