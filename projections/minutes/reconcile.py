"""Deterministic team-minutes reconciliation helpers.

This module intentionally avoids optimization solvers. The goal is a small, stable
projection layer that:
- forces per-team totals to a target (240 regulation, or 240+OT in sim worlds),
- respects per-player caps when feasible (and relaxes caps only when infeasible),
- keeps OUT/DNP rows at 0,
- is deterministic (no RNG usage) and converges in a few passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

# Import canonical contract definitions
from projections.minutes.contract import (
    IN_ROTATION_THRESHOLD_MIN,
    MINUTES_CONTRACT_VERSION,
    minutes_contract_hash,
)

MinutesState = Literal["out", "dnp", "cameo", "rotation", "unknown"]


@dataclass(frozen=True, slots=True)
class TeamReconcileDiagnostics:
    target_minutes: float
    pre_sum: float
    post_sum: float
    residual: float
    passes: int
    cap_infeasible: bool
    locked_infeasible: bool
    n_players: int
    n_out_or_dnp: int
    n_cameo: int
    n_rotation: int
    n_cap_hits: int


def _coerce_float_array(series: pd.Series | np.ndarray, *, fill: float = 0.0) -> np.ndarray:
    if isinstance(series, np.ndarray):
        out = series.astype(float, copy=False)
        out = np.where(np.isfinite(out), out, float(fill))
        return out.astype(float, copy=False)
    return pd.to_numeric(series, errors="coerce").fillna(float(fill)).to_numpy(dtype=float)


def _infer_states(
    df_team: pd.DataFrame,
    *,
    minutes: np.ndarray,
    state_col: str | None,
    status_col: str | None = "status",
    play_prob_col: str | None = "play_prob",
    in_rotation_threshold_min: float = IN_ROTATION_THRESHOLD_MIN,
) -> np.ndarray:
    n = int(len(df_team))
    if n == 0:
        return np.zeros(0, dtype=object)

    if state_col and state_col in df_team.columns:
        raw = df_team[state_col].astype("string").str.strip().str.lower()
        out = raw.fillna("unknown").to_numpy(dtype=object)
        # Normalize common variants.
        mapping = {
            "inactive": "dnp",
            "did_not_play": "dnp",
            "did-not-play": "dnp",
        }
        out = np.array([mapping.get(str(x), str(x)) for x in out], dtype=object)
        return out

    status = None
    if status_col and status_col in df_team.columns:
        status = df_team[status_col].astype("string").str.strip().str.lower()
    play_prob = None
    if play_prob_col and play_prob_col in df_team.columns:
        play_prob = pd.to_numeric(df_team[play_prob_col], errors="coerce")

    out_mask = np.zeros(n, dtype=bool)
    if status is not None:
        out_mask |= status.eq("out").fillna(False).to_numpy(dtype=bool)
    if play_prob is not None:
        out_mask |= play_prob.fillna(0.0).to_numpy(dtype=float) <= 0.0

    rotation_mask = (~out_mask) & (minutes >= float(in_rotation_threshold_min))
    cameo_mask = (~out_mask) & (~rotation_mask) & (minutes > 0.0)

    states = np.full(n, "unknown", dtype=object)
    states[out_mask] = "out"
    states[cameo_mask] = "cameo"
    states[rotation_mask] = "rotation"
    return states


def _state_to_tier(state: str) -> int:
    s = str(state).strip().lower()
    if s in {"out", "dnp"}:
        return 0
    if s == "cameo":
        return 1
    if s == "rotation":
        return 2
    return 2  # Default to rotation to avoid starving unknown players.


def _distribute_delta_1d(
    values: np.ndarray,
    *,
    delta: float,
    lower: float,
    upper: np.ndarray,
    weights: np.ndarray,
    eligible: np.ndarray,
    max_passes: int,
    eps: float,
) -> tuple[np.ndarray, float, int]:
    """Distribute delta across eligible entries with per-entry bounds.

    Returns (new_values, remaining_delta, passes_used).
    """

    if values.size == 0 or abs(delta) <= eps:
        return values, float(delta), 0

    out = values.astype(float, copy=True)
    remaining = float(delta)
    passes_used = 0

    for i in range(int(max_passes)):
        if abs(remaining) <= eps:
            break
        if remaining > 0:
            headroom = (upper - out).clip(min=0.0)
            elig = eligible & (headroom > eps)
            if not np.any(elig):
                break
            w = np.where(elig, weights, 0.0)
            wsum = float(w.sum())
            if wsum <= eps:
                w = np.where(elig, headroom, 0.0)
                wsum = float(w.sum())
            if wsum <= eps:
                break
            share = w / wsum
            inc = np.minimum(remaining * share, headroom)
            out = out + inc
            remaining -= float(inc.sum())
        else:
            reducible = (out - float(lower)).clip(min=0.0)
            elig = eligible & (reducible > eps)
            if not np.any(elig):
                break
            w = np.where(elig, weights, 0.0)
            wsum = float(w.sum())
            if wsum <= eps:
                w = np.where(elig, reducible, 0.0)
                wsum = float(w.sum())
            if wsum <= eps:
                break
            share = w / wsum
            dec = np.minimum((-remaining) * share, reducible)
            out = out - dec
            remaining += float(dec.sum())
        passes_used = i + 1

    return out, remaining, passes_used


def reconcile_team_minutes(
    df_team: pd.DataFrame,
    target_minutes: float,
    *,
    minutes_col: str = "minutes_p50",
    cap_col: str | None = None,
    weight_col: str | None = None,
    state_col: str | None = None,
    locked_mask: pd.Series | np.ndarray | None = None,
    in_rotation_threshold_min: float = IN_ROTATION_THRESHOLD_MIN,
    default_cap: float = 48.0,
    max_passes: int = 5,
    eps: float = 1e-6,
) -> tuple[pd.Series, TeamReconcileDiagnostics]:
    """Reconcile a single (game_id, team_id) frame to `target_minutes`.

    Returns (minutes_final_series, diagnostics).
    """

    if df_team.empty:
        empty = pd.Series([], index=df_team.index, dtype=float)
        diag = TeamReconcileDiagnostics(
            target_minutes=float(target_minutes),
            pre_sum=0.0,
            post_sum=0.0,
            residual=0.0,
            passes=0,
            cap_infeasible=False,
            locked_infeasible=False,
            n_players=0,
            n_out_or_dnp=0,
            n_cameo=0,
            n_rotation=0,
            n_cap_hits=0,
        )
        return empty, diag

    if minutes_col not in df_team.columns:
        raise ValueError(f"Missing required minutes_col={minutes_col!r}")

    minutes = _coerce_float_array(df_team[minutes_col], fill=0.0)
    minutes = np.clip(minutes, 0.0, None)

    if cap_col and cap_col in df_team.columns:
        caps = _coerce_float_array(df_team[cap_col], fill=default_cap)
        caps = np.clip(caps, 0.0, None)
    else:
        caps = np.full(len(df_team), float(default_cap), dtype=float)

    if weight_col and weight_col in df_team.columns:
        weights = _coerce_float_array(df_team[weight_col], fill=0.0)
        weights = np.clip(weights, 0.0, None)
    else:
        weights = minutes.copy()

    if locked_mask is None:
        locked = np.zeros(len(df_team), dtype=bool)
    elif isinstance(locked_mask, pd.Series):
        locked = locked_mask.reindex(df_team.index).fillna(False).to_numpy(dtype=bool)
    else:
        locked = np.asarray(locked_mask, dtype=bool)
        if locked.shape != minutes.shape:
            raise ValueError("locked_mask must align to df_team rows")

    # Determine states/tier semantics.
    states = _infer_states(
        df_team,
        minutes=minutes,
        state_col=state_col,
        in_rotation_threshold_min=in_rotation_threshold_min,
    )
    tiers = np.array([_state_to_tier(cast(str, s)) for s in states], dtype=int)
    out_mask = tiers == 0

    # Force OUT/DNP rows to 0 and lock them (never allocate to them).
    if out_mask.any():
        minutes = minutes.copy()
        minutes[out_mask] = 0.0
        locked = locked | out_mask

    # Clip to caps before reconcile.
    minutes = np.minimum(minutes, caps)

    target = float(target_minutes)
    pre_sum = float(minutes.sum())

    # Cap infeasibility: if sum(caps for eligible) < target, relax caps proportionally.
    cap_sum = float(caps[tiers > 0].sum())
    cap_infeasible = (cap_sum > eps) and (target > cap_sum + eps)
    caps_eff = caps
    if cap_infeasible:
        scale = target / cap_sum
        caps_eff = caps * float(scale)

    # If nothing can play, we can't force 240.
    if int((tiers > 0).sum()) == 0:
        out = pd.Series(minutes, index=df_team.index, dtype=float)
        diag = TeamReconcileDiagnostics(
            target_minutes=target,
            pre_sum=pre_sum,
            post_sum=float(out.sum()),
            residual=target - float(out.sum()),
            passes=0,
            cap_infeasible=cap_infeasible,
            locked_infeasible=False,
            n_players=int(len(df_team)),
            n_out_or_dnp=int(out_mask.sum()),
            n_cameo=int((tiers == 1).sum()),
            n_rotation=int((tiers == 2).sum()),
            n_cap_hits=int((np.abs(out.to_numpy(dtype=float) - caps_eff) <= 1e-9).sum()),
        )
        return out, diag

    # Default: keep locked players fixed, but relax the lock when infeasible.
    locked_sum = float(minutes[locked].sum())
    locked_infeasible = locked_sum > target + eps

    values = minutes.copy()

    def _step(eligible: np.ndarray, remaining: float) -> tuple[float, int]:
        nonlocal values
        if abs(remaining) <= eps:
            return 0.0, 0
        adj_mask = (~locked) & eligible
        if locked_infeasible:
            # If locked minutes alone exceed the target, allow locked rows (except OUT)
            # to participate in reduction so we still hit the team contract.
            adj_mask = eligible & (tiers > 0)
        values, remaining2, passes_used = _distribute_delta_1d(
            values,
            delta=remaining,
            lower=0.0,
            upper=caps_eff,
            weights=weights,
            eligible=adj_mask,
            max_passes=max_passes,
            eps=eps,
        )
        return remaining2, passes_used

    remaining = target - float(values.sum())
    passes_used_total = 0

    if remaining > eps:
        # Add minutes: rotation first, then cameo, then any eligible.
        for tier_val in (2, 1):
            remaining, passes_used = _step(tiers == tier_val, remaining)
            passes_used_total = max(passes_used_total, passes_used)
            if remaining <= eps:
                break
        if remaining > eps:
            remaining, passes_used = _step(tiers > 0, remaining)
            passes_used_total = max(passes_used_total, passes_used)
    elif remaining < -eps:
        # Remove minutes: cameo first, then rotation, then any eligible.
        for tier_val in (1, 2):
            remaining, passes_used = _step(tiers == tier_val, remaining)
            passes_used_total = max(passes_used_total, passes_used)
            if remaining >= -eps:
                break
        if remaining < -eps:
            remaining, passes_used = _step(tiers > 0, remaining)
            passes_used_total = max(passes_used_total, passes_used)

    # Final deterministic residual push (avoid drift from floating point).
    residual = target - float(values.sum())
    if abs(residual) > eps:
        if residual > 0:
            headroom = (caps_eff - values).clip(min=0.0)
            eligible = (tiers > 0) & (headroom > eps)
            if eligible.any():
                idx = int(np.argmax(np.where(eligible, headroom, -np.inf)))
                values[idx] = min(values[idx] + residual, float(caps_eff[idx]))
        else:
            reducible = values.clip(min=0.0)
            eligible = (tiers > 0) & (reducible > eps)
            if eligible.any():
                idx = int(np.argmax(np.where(eligible, reducible, -np.inf)))
                values[idx] = max(values[idx] + residual, 0.0)

    post_sum = float(values.sum())
    residual_final = target - post_sum
    n_cap_hits = int((np.abs(values - caps_eff) <= 1e-9).sum())

    diag = TeamReconcileDiagnostics(
        target_minutes=target,
        pre_sum=pre_sum,
        post_sum=post_sum,
        residual=residual_final,
        passes=int(passes_used_total),
        cap_infeasible=cap_infeasible,
        locked_infeasible=locked_infeasible,
        n_players=int(len(df_team)),
        n_out_or_dnp=int(out_mask.sum()),
        n_cameo=int((tiers == 1).sum()),
        n_rotation=int((tiers == 2).sum()),
        n_cap_hits=n_cap_hits,
    )

    return pd.Series(values, index=df_team.index, dtype=float), diag


def reconcile_team_minutes_matrix(
    minutes_worlds: np.ndarray,
    active_mask: np.ndarray,
    *,
    target_minutes: float = 240.0,
    cap_minutes: float = 48.0,
    weights: np.ndarray | None = None,
    tiers: np.ndarray | None = None,
    in_rotation_threshold_min: float = IN_ROTATION_THRESHOLD_MIN,
    max_passes: int = 5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Reconcile (W, N) minutes to `target_minutes` per row, active-only.

    Caps are enforced when feasible. If infeasible (active_count * cap < target),
    caps are relaxed proportionally so the row can still sum to target.
    """

    m0 = np.asarray(minutes_worlds, dtype=float)
    active = np.asarray(active_mask, dtype=bool)
    if m0.shape != active.shape:
        raise ValueError(f"minutes_worlds and active_mask must have same shape; got {m0.shape} vs {active.shape}")

    n_rows, n_players = m0.shape
    if n_rows == 0 or n_players == 0:
        return m0.copy(), {"n_rows": int(n_rows), "n_all_inactive": int(n_rows)}

    base_weights = None
    if weights is not None:
        base_weights = np.asarray(weights, dtype=float)
        if base_weights.shape != (n_players,):
            raise ValueError(f"weights must have shape (N,), got {base_weights.shape}")
        base_weights = np.clip(base_weights, 0.0, None)

    # Default tiers derived from weights/minutes (rotation vs cameo). OUT/DNP are represented by active_mask=False.
    if tiers is not None:
        tier_arr = np.asarray(tiers, dtype=int)
        if tier_arr.shape != (n_players,):
            raise ValueError(f"tiers must have shape (N,), got {tier_arr.shape}")
        tier_rot = tier_arr == 2
        tier_cameo = tier_arr == 1
    else:
        ref = base_weights if base_weights is not None else np.maximum(m0.mean(axis=0), 0.0)
        tier_rot = ref >= float(in_rotation_threshold_min)
        tier_cameo = (ref > 0.0) & (~tier_rot)

    cap = float(cap_minutes)
    if cap <= 0.0:
        raise ValueError("cap_minutes must be > 0")

    # Active-only minutes, clipped non-negative.
    out = np.maximum(m0, 0.0) * active.astype(float)

    all_inactive = ~active.any(axis=1)
    n_all_inactive = int(all_inactive.sum())

    # Per-row cap feasibility; relax caps when infeasible.
    active_counts = active.sum(axis=1).astype(float)
    cap_sum = active_counts * cap
    cap_infeasible = (cap_sum > 0.0) & (float(target_minutes) > (cap_sum + eps))
    n_cap_infeasible = int(cap_infeasible.sum())
    cap_scale = np.ones(n_rows, dtype=float)
    cap_scale[cap_infeasible] = float(target_minutes) / cap_sum[cap_infeasible]
    cap_eff = cap * cap_scale[:, None]  # (W, 1)

    # Initial scale to hit target (before cap).
    row_sum = out.sum(axis=1)
    scale = np.ones(n_rows, dtype=float)
    nonzero = row_sum > eps
    scale[nonzero] = float(target_minutes) / row_sum[nonzero]
    out = out * scale[:, None]

    # Apply caps (relaxed when infeasible).
    out = np.minimum(out, cap_eff)

    def _distribute_add(delta: np.ndarray, eligible_players: np.ndarray) -> np.ndarray:
        nonlocal out
        if not np.any(delta > eps):
            return delta
        headroom = (cap_eff - out).clip(min=0.0)
        elig = (delta[:, None] > eps) & active & eligible_players[None, :] & (headroom > eps)
        if not np.any(elig):
            return delta
        if base_weights is None:
            w = np.where(elig, out, 0.0)
        else:
            w = np.where(elig, base_weights[None, :], 0.0)
        wsum = w.sum(axis=1, keepdims=True)
        # Fallback to headroom when weights vanish.
        w = np.where(wsum > eps, w, np.where(elig, headroom, 0.0))
        wsum = w.sum(axis=1, keepdims=True)
        share = np.divide(w, wsum, out=np.zeros_like(w), where=wsum > eps)
        inc = share * delta[:, None]
        inc = np.minimum(inc, headroom)
        out = out + inc
        delta = delta - inc.sum(axis=1)
        return delta

    def _distribute_remove(delta: np.ndarray, eligible_players: np.ndarray) -> np.ndarray:
        nonlocal out
        if not np.any(delta < -eps):
            return delta
        reducible = out.clip(min=0.0)
        elig = (delta[:, None] < -eps) & active & eligible_players[None, :] & (reducible > eps)
        if not np.any(elig):
            return delta
        if base_weights is None:
            w = np.where(elig, out, 0.0)
        else:
            w = np.where(elig, base_weights[None, :], 0.0)
        wsum = w.sum(axis=1, keepdims=True)
        w = np.where(wsum > eps, w, np.where(elig, reducible, 0.0))
        wsum = w.sum(axis=1, keepdims=True)
        share = np.divide(w, wsum, out=np.zeros_like(w), where=wsum > eps)
        dec = share * (-delta[:, None])
        dec = np.minimum(dec, reducible)
        out = out - dec
        delta = delta + dec.sum(axis=1)
        return delta

    for _ in range(int(max_passes)):
        if all_inactive.all():
            break
        row_sum = out.sum(axis=1)
        delta = float(target_minutes) - row_sum
        if float(np.max(np.abs(delta[~all_inactive])) if (~all_inactive).any() else 0.0) <= eps:
            break

        # Priority semantics (cameo vs rotation).
        delta = _distribute_add(delta, tier_rot)
        delta = _distribute_add(delta, tier_cameo)
        delta = _distribute_add(delta, np.ones(n_players, dtype=bool))

        delta = _distribute_remove(delta, tier_cameo)
        delta = _distribute_remove(delta, tier_rot)
        delta = _distribute_remove(delta, np.ones(n_players, dtype=bool))

    # Final residual push for rows still off target (deterministic).
    row_sum = out.sum(axis=1)
    delta = float(target_minutes) - row_sum
    needs = (~all_inactive) & (np.abs(delta) > eps)
    if needs.any():
        for r in np.flatnonzero(needs):
            if delta[r] > 0:
                headroom = (cap_eff[r, 0] - out[r]).clip(min=0.0)
                elig = active[r] & (headroom > eps)
                if not np.any(elig):
                    continue
                idx = int(np.argmax(np.where(elig, headroom, -np.inf)))
                out[r, idx] = min(out[r, idx] + delta[r], cap_eff[r, 0])
            else:
                reducible = out[r].clip(min=0.0)
                elig = active[r] & (reducible > eps)
                if not np.any(elig):
                    continue
                idx = int(np.argmax(np.where(elig, reducible, -np.inf)))
                out[r, idx] = max(out[r, idx] + delta[r], 0.0)

    # Keep inactive exactly 0.
    out = np.where(active, out, 0.0)

    final_sum = out.sum(axis=1)
    max_abs_residual = float(np.max(np.abs(final_sum[~all_inactive] - float(target_minutes)))) if (~all_inactive).any() else 0.0

    cap_bind = active & (out >= (cap_eff - 1e-9))
    n_cap_bind_rows = int((cap_bind.any(axis=1) & (~cap_infeasible) & (~all_inactive)).sum())

    return out, {
        "n_rows": int(n_rows),
        "n_all_inactive": int(n_all_inactive),
        "n_cap_infeasible_rows": int(n_cap_infeasible),
        "n_cap_bind_rows": int(n_cap_bind_rows),
        "max_abs_residual": max_abs_residual,
    }
