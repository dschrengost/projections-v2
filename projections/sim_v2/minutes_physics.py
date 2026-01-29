from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from projections.sim_v2.config import (
    MinutesAbsorptionCapsConfig,
    MinutesAvailabilityPolicyConfig,
    MinutesFeasibilityConfig,
)


@dataclass(frozen=True)
class AvailabilityPolicyDiagnostics:
    enabled: bool
    n_players: int
    n_rotation_locks: int
    n_floored: int
    max_floor_delta: float


def compute_rotation_lock_mask(
    *,
    baseline_minutes: np.ndarray,  # (P,)
    is_starter: np.ndarray | None,  # (P,) bool
    group_map: dict[Any, np.ndarray],
    top_k: int,
    minutes_threshold: float,
) -> np.ndarray:
    """Heuristic rotation-lock classifier (per-slate, per-player).

    A player is a rotation lock if:
      - is_starter=True (if provided), OR
      - in top-K by baseline minutes within their team, OR
      - baseline minutes >= minutes_threshold.
    """
    mins = np.asarray(baseline_minutes, dtype=float)
    n_players = int(mins.shape[0])
    lock = np.zeros(n_players, dtype=bool)

    if is_starter is not None:
        starter = np.asarray(is_starter, dtype=bool)
        if starter.shape != (n_players,):
            raise ValueError(f"is_starter must have shape (P,), got {starter.shape}")
        lock |= starter

    threshold = float(minutes_threshold)
    if threshold > 0.0:
        lock |= mins >= threshold

    k = int(max(0, top_k))
    if k <= 0 or not group_map:
        return lock

    # Deterministic iteration order.
    for _team_key in sorted(group_map.keys()):
        idxs = np.asarray(group_map[_team_key], dtype=int)
        if idxs.size == 0:
            continue
        team_mins = mins[idxs]
        order = np.argsort(-team_mins, kind="mergesort")
        take = min(k, int(order.size))
        if take <= 0:
            continue
        lock[idxs[order[:take]]] = True

    return lock


def apply_minutes_availability_policy(
    *,
    play_prob_raw: np.ndarray,  # (P,)
    baseline_minutes: np.ndarray,  # (P,)
    is_starter: np.ndarray | None,  # (P,) bool
    status_bucket: np.ndarray | None,  # (P,) str
    group_map: dict[Any, np.ndarray],
    cfg: MinutesAvailabilityPolicyConfig,
) -> tuple[np.ndarray, np.ndarray, AvailabilityPolicyDiagnostics]:
    """Return (rotation_lock_mask, play_prob_eff, diagnostics)."""
    p_raw = np.asarray(play_prob_raw, dtype=float)
    if p_raw.ndim != 1:
        raise ValueError("play_prob_raw must be a 1D array")
    p_raw = np.clip(p_raw, 0.0, 1.0)

    if not cfg.enabled:
        return (
            np.zeros_like(p_raw, dtype=bool),
            p_raw,
            AvailabilityPolicyDiagnostics(
                enabled=False,
                n_players=int(p_raw.size),
                n_rotation_locks=0,
                n_floored=0,
                max_floor_delta=0.0,
            ),
        )

    lock = compute_rotation_lock_mask(
        baseline_minutes=baseline_minutes,
        is_starter=is_starter,
        group_map=group_map,
        top_k=cfg.rotation_lock_top_k,
        minutes_threshold=cfg.rotation_lock_minutes_threshold,
    )

    eligible = lock.copy()
    if status_bucket is not None and status_bucket.size:
        sb = np.asarray(status_bucket, dtype=str)
        if sb.shape != p_raw.shape:
            raise ValueError(f"status_bucket must have shape {p_raw.shape}, got {sb.shape}")
        exclude = {str(x).strip().lower() for x in cfg.exclude_status_buckets}
        if exclude:
            eligible &= ~np.isin(np.char.lower(sb), np.array(sorted(exclude), dtype=str))

    p_floor = float(cfg.play_prob_floor)
    p_floor = float(np.clip(p_floor, 0.0, 1.0))
    p_eff = p_raw.copy()
    before = p_eff[eligible].copy()
    p_eff[eligible] = np.maximum(p_eff[eligible], p_floor)

    max_delta = float(np.max(p_eff[eligible] - before)) if eligible.any() else 0.0

    return (
        lock,
        p_eff,
        AvailabilityPolicyDiagnostics(
            enabled=True,
            n_players=int(p_raw.size),
            n_rotation_locks=int(lock.sum()),
            n_floored=int((p_eff > p_raw).sum()),
            max_floor_delta=max_delta,
        ),
    )


def compute_max_increase_by_depth(
    *,
    baseline_minutes: np.ndarray,  # (P,)
    is_starter: np.ndarray | None,  # (P,) bool
    group_map: dict[Any, np.ndarray],
    cfg: MinutesAbsorptionCapsConfig,
) -> np.ndarray:
    """Return per-player max increase minutes (delta_i_max)."""
    mins = np.asarray(baseline_minutes, dtype=float)
    n_players = int(mins.shape[0])
    delta = np.full(n_players, float(cfg.deep_delta_max), dtype=float)

    starter = None
    if is_starter is not None:
        starter = np.asarray(is_starter, dtype=bool)
        if starter.shape != (n_players,):
            raise ValueError(f"is_starter must have shape (P,), got {starter.shape}")

    if not group_map:
        return np.clip(delta, 0.0, None)

    core_k = int(max(0, cfg.core_rank_max))
    rot_k = int(max(core_k, cfg.rotation_rank_max))
    bench_k = int(max(rot_k, cfg.bench_rank_max))
    fringe_k = int(max(bench_k, cfg.fringe_rank_max))

    for _team_key in sorted(group_map.keys()):
        idxs = np.asarray(group_map[_team_key], dtype=int)
        if idxs.size == 0:
            continue
        team_mins = mins[idxs]
        order = np.argsort(-team_mins, kind="mergesort")
        # 1-indexed ranks.
        ranks = np.empty(order.size, dtype=int)
        ranks[order] = np.arange(1, order.size + 1)

        d_team = np.full(order.size, float(cfg.deep_delta_max), dtype=float)
        d_team[ranks <= fringe_k] = float(cfg.fringe_delta_max)
        d_team[ranks <= bench_k] = float(cfg.bench_delta_max)
        d_team[ranks <= rot_k] = float(cfg.rotation_delta_max)
        d_team[ranks <= core_k] = float(cfg.core_delta_max)

        if starter is not None:
            d_team = np.where(starter[idxs], np.maximum(d_team, float(cfg.core_delta_max)), d_team)

        delta[idxs] = d_team

    return np.clip(delta, 0.0, None)


@dataclass(frozen=True)
class FeasibilityGateDiagnostics:
    enabled: bool
    n_team_worlds: int
    n_infeasible_pre_resample: int
    n_resampled_team_worlds: int
    resample_attempts_total: int
    n_promoted_team_worlds: int
    promoted_players_total: int


def apply_team_feasibility_gate(
    active_mask: np.ndarray,  # (W, P)
    *,
    play_prob: np.ndarray,  # (P,)
    baseline_minutes: np.ndarray,  # (P,)
    cap_upper: np.ndarray,  # (P,)
    group_map: dict[Any, np.ndarray],
    cfg: MinutesFeasibilityConfig,
    rng: np.random.Generator,
    eligible_mask: np.ndarray | None = None,  # (P,) optional hard eligibility mask
    rotation_lock_mask: np.ndarray | None = None,  # (P,)
    target_total: float = 240.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, FeasibilityGateDiagnostics]:
    """Resample team availability draws until per-world constraints are feasible.

    This mutates and returns `active_mask` so callers can avoid extra allocations.
    """
    if not cfg.enabled:
        return active_mask, FeasibilityGateDiagnostics(
            enabled=False,
            n_team_worlds=0,
            n_infeasible_pre_resample=0,
            n_resampled_team_worlds=0,
            resample_attempts_total=0,
            n_promoted_team_worlds=0,
            promoted_players_total=0,
        )

    active = np.asarray(active_mask, dtype=bool)
    p = np.asarray(play_prob, dtype=float)
    base = np.asarray(baseline_minutes, dtype=float)
    cap_u = np.asarray(cap_upper, dtype=float)

    if active.ndim != 2:
        raise ValueError("active_mask must be 2D (W, P)")
    n_worlds, n_players = active.shape
    if p.shape != (n_players,):
        raise ValueError(f"play_prob must have shape (P,), got {p.shape}")
    if base.shape != (n_players,):
        raise ValueError(f"baseline_minutes must have shape (P,), got {base.shape}")
    if cap_u.shape != (n_players,):
        raise ValueError(f"cap_upper must have shape (P,), got {cap_u.shape}")

    eligible = None
    if eligible_mask is not None:
        eligible_arr = np.asarray(eligible_mask, dtype=bool)
        if eligible_arr.shape != (n_players,):
            raise ValueError(f"eligible_mask must have shape (P,), got {eligible_arr.shape}")
        eligible = eligible_arr

    lock = None
    if rotation_lock_mask is not None and cfg.min_rotation_locks_active is not None:
        lock_arr = np.asarray(rotation_lock_mask, dtype=bool)
        if lock_arr.shape != (n_players,):
            raise ValueError(f"rotation_lock_mask must have shape (P,), got {lock_arr.shape}")
        lock = lock_arr

    min_active = int(max(0, cfg.min_active_players))
    min_sum_demand = float(cfg.min_sum_demand)
    max_attempts = int(max(0, cfg.max_resample_attempts))
    min_locks = cfg.min_rotation_locks_active
    if min_locks is not None:
        min_locks = int(max(0, min_locks))

    n_team_worlds = 0
    n_infeasible_pre = 0
    n_resampled_rows = 0
    resample_attempts_total = 0
    n_promoted_rows = 0
    promoted_players_total = 0

    if not group_map or n_worlds == 0:
        return active, FeasibilityGateDiagnostics(
            enabled=True,
            n_team_worlds=0,
            n_infeasible_pre_resample=0,
            n_resampled_team_worlds=0,
            resample_attempts_total=0,
            n_promoted_team_worlds=0,
            promoted_players_total=0,
        )

    for _team_key in sorted(group_map.keys()):
        idxs = np.asarray(group_map[_team_key], dtype=int)
        if idxs.size == 0:
            continue
        if min_active and idxs.size < min_active:
            # Cannot satisfy constraints with the provided roster slice.
            continue

        team_active = active[:, idxs].copy()
        p_team = np.clip(p[idxs], 0.0, 1.0)
        base_team = np.clip(base[idxs], 0.0, None)
        cap_team = np.clip(cap_u[idxs], 0.0, None)
        lock_team = lock[idxs] if lock is not None else None
        eligible_team = eligible[idxs] if eligible is not None else np.ones(idxs.size, dtype=bool)
        if min_active and int(eligible_team.sum()) < min_active:
            # The eligible slice is too small to satisfy constraints.
            continue

        # Deterministic promotion order: highest play_prob, then baseline minutes, then index.
        # np.lexsort sorts ascending on the last key first; using (-p, -minutes, idx)
        # yields highest play_prob first with deterministic tie-breaking.
        order = np.lexsort((np.arange(idxs.size), -base_team, -p_team))

        def _infeasible(mask: np.ndarray) -> np.ndarray:
            # mask: (W, N) bool
            n_act = mask.sum(axis=1).astype(float)
            sum_dem = (mask.astype(float) * base_team[None, :]).sum(axis=1)
            sum_cap = (mask.astype(float) * cap_team[None, :]).sum(axis=1)
            bad = np.zeros(n_act.shape[0], dtype=bool)
            if min_active > 0:
                bad |= n_act < float(min_active)
            if min_sum_demand > 0.0:
                bad |= sum_dem < float(min_sum_demand)
            bad |= sum_cap + float(eps) < float(target_total)
            if lock_team is not None and min_locks is not None and min_locks > 0:
                bad |= (mask & lock_team[None, :]).sum(axis=1) < int(min_locks)
            return bad

        infeasible = _infeasible(team_active)
        n_team_worlds += int(n_worlds)
        n_infeasible_pre += int(infeasible.sum())

        attempts = np.zeros(n_worlds, dtype=int)
        resampled = np.zeros(n_worlds, dtype=bool)

        for _ in range(max_attempts):
            bad_rows = np.flatnonzero(infeasible)
            if bad_rows.size == 0:
                break
            u = rng.random(size=(bad_rows.size, idxs.size))
            team_active[bad_rows] = (u < p_team[None, :]) & eligible_team[None, :]
            attempts[bad_rows] += 1
            resampled[bad_rows] = True
            infeasible = _infeasible(team_active)

        if resampled.any():
            n_resampled_rows += int(resampled.sum())
            resample_attempts_total += int(attempts.sum())

        # Fallback: promote additional actives deterministically until constraints satisfied.
        bad_rows = np.flatnonzero(infeasible)
        if bad_rows.size:
            for w in bad_rows:
                row = team_active[w].copy()
                # Track counts so we can early-exit promotion.
                n_act = int(row.sum())
                sum_dem = float((row.astype(float) * base_team).sum())
                sum_cap = float((row.astype(float) * cap_team).sum())
                n_lock = int((row & lock_team).sum()) if lock_team is not None else 0

                promoted = 0
                for j in order:
                    if not eligible_team[j]:
                        continue
                    if row[j]:
                        continue
                    row[j] = True
                    promoted += 1
                    promoted_players_total += 1
                    n_act += 1
                    sum_dem += float(base_team[j])
                    sum_cap += float(cap_team[j])
                    if lock_team is not None and lock_team[j]:
                        n_lock += 1

                    if min_active > 0 and n_act < min_active:
                        continue
                    if min_sum_demand > 0.0 and sum_dem < min_sum_demand:
                        continue
                    if sum_cap + float(eps) < float(target_total):
                        continue
                    if lock_team is not None and min_locks is not None and min_locks > 0 and n_lock < min_locks:
                        continue
                    break

                if promoted:
                    n_promoted_rows += 1
                team_active[w] = row

        active[:, idxs] = team_active

    return active, FeasibilityGateDiagnostics(
        enabled=True,
        n_team_worlds=int(n_team_worlds),
        n_infeasible_pre_resample=int(n_infeasible_pre),
        n_resampled_team_worlds=int(n_resampled_rows),
        resample_attempts_total=int(resample_attempts_total),
        n_promoted_team_worlds=int(n_promoted_rows),
        promoted_players_total=int(promoted_players_total),
    )


__all__ = [
    "AvailabilityPolicyDiagnostics",
    "FeasibilityGateDiagnostics",
    "apply_minutes_availability_policy",
    "apply_team_feasibility_gate",
    "compute_max_increase_by_depth",
    "compute_rotation_lock_mask",
]
