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


def build_eligible_mask(
    p_rot: np.ndarray,
    mu: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    a: float = 1.5,
    p_cutoff: float | None = None,
    topk: int | None = None,
    k_min: int = 8,
    k_max: int = 11,
    use_expected_k: bool = False,
) -> np.ndarray:
    """Build a pruning mask from rotation + minutes priors.

    Pruning logic:
      - If p_cutoff is not None: exclude players with p_rot < p_cutoff.
      - If use_expected_k: k = clip(round(sum(p_rot over remaining eligible)), k_min, k_max),
        keep top-k by proxy.
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
    proxy = np.power(p, float(a)) * m
    proxy = np.where(np.isfinite(proxy), proxy, 0.0)

    eligible = mask_bool.copy()
    if p_cutoff is not None:
        eligible = eligible & (p >= float(p_cutoff))

    k = None
    if use_expected_k:
        expected_k = int(np.round(p[eligible].sum()))
        k = max(int(k_min), min(int(k_max), expected_k))
    elif topk is not None:
        if int(topk) > 0:
            k = int(topk)

    if k is not None:
        # Rank by proxy, but only among current eligible.
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
    w = np.power(p, float(a)) * m
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
    "build_eligible_mask",
    "waterfill_redistribute",
    "leak_dnp_by_teamgame",
    "topk_overlap",
]
