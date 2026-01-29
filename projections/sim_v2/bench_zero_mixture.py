from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BenchZeroMixtureStats:
    enabled: bool
    n_player_worlds_dropped: int
    n_player_worlds_restored_for_feasibility: int
    min_active_needed: int


def apply_bench_zero_mixture(
    minutes_worlds: np.ndarray,  # (W, P)
    active_mask: np.ndarray,  # (W, P) bool
    *,
    group_map: dict[Any, list[int]],
    minutes_target: np.ndarray,  # (P,)
    minutes_threshold: float,
    p_zero_base: float,
    p_zero_slope: float,
    cap_minutes: float,
    total_minutes: float = 240.0,
    rng: np.random.Generator | None = None,
    min_active_needed_override: int | None = None,
) -> BenchZeroMixtureStats:
    """
    Two-part minutes model for low-minute players:

    For players with minutes_target < minutes_threshold, set minutes=0 with probability p_zero,
    otherwise leave minutes unchanged. Dropped players are treated as inactive (active_mask=False).

    A feasibility guard ensures each team keeps enough active players per world to support the
    minutes cap (i.e., active_count >= ceil(total_minutes / cap_minutes)).
    """
    if rng is None:
        rng = np.random.default_rng()

    minutes = np.asarray(minutes_worlds)
    active = np.asarray(active_mask, dtype=bool)
    target = np.asarray(minutes_target, dtype=float)

    if minutes.shape != active.shape:
        raise ValueError(f"minutes_worlds and active_mask must have same shape; got {minutes.shape} vs {active.shape}")
    if target.shape[0] != minutes.shape[1]:
        raise ValueError("minutes_target must have length equal to n_players (axis=1 of minutes_worlds)")

    if minutes_threshold <= 0.0 or p_zero_base < 0.0 or cap_minutes <= 0.0:
        return BenchZeroMixtureStats(
            enabled=False,
            n_player_worlds_dropped=0,
            n_player_worlds_restored_for_feasibility=0,
            min_active_needed=int(np.ceil(float(total_minutes) / float(cap_minutes))),
        )

    min_active_needed = int(np.ceil(float(total_minutes) / float(cap_minutes)))
    if min_active_needed_override is not None:
        min_active_needed = max(min_active_needed, int(min_active_needed_override))

    in_bucket = target < float(minutes_threshold)
    if not np.any(in_bucket):
        return BenchZeroMixtureStats(
            enabled=False,
            n_player_worlds_dropped=0,
            n_player_worlds_restored_for_feasibility=0,
            min_active_needed=min_active_needed,
        )

    x = np.clip((float(minutes_threshold) - target) / float(minutes_threshold), 0.0, 1.0)
    p_zero = np.clip(float(p_zero_base) + float(p_zero_slope) * x, 0.0, 0.95)
    p_zero = np.where(in_bucket, p_zero, 0.0)

    drop = (rng.random(size=minutes.shape) < p_zero[None, :]) & active

    restored = 0
    if group_map and min_active_needed > 0:
        for _, idxs_list in group_map.items():
            idxs = np.asarray(idxs_list, dtype=int)
            if idxs.size == 0:
                continue
            team_drop = drop[:, idxs]  # (W, N)
            if not team_drop.any():
                continue
            team_active = active[:, idxs]
            after_active = team_active & ~team_drop
            after_count = after_active.sum(axis=1)
            team_count = team_active.sum(axis=1)
            need = np.where((team_count > 0) & (after_count < min_active_needed))[0]
            if need.size == 0:
                continue

            priorities = target[idxs]
            for w in need:
                k = int(min_active_needed - int(after_count[w]))
                if k <= 0:
                    continue
                dropped_rel = np.where(team_drop[w])[0]
                if dropped_rel.size == 0:
                    continue
                # Restore the "most likely to play" among dropped candidates.
                # Use minutes_target as the primary signal; higher target restored first.
                order = np.argsort(priorities[dropped_rel])[::-1]
                restore_rel = dropped_rel[order[:k]]
                team_drop[w, restore_rel] = False
                restored += int(restore_rel.size)

            drop[:, idxs] = team_drop

    minutes[drop] = 0.0
    active[drop] = False

    return BenchZeroMixtureStats(
        enabled=True,
        n_player_worlds_dropped=int(drop.sum()),
        n_player_worlds_restored_for_feasibility=int(restored),
        min_active_needed=min_active_needed,
    )


__all__ = ["apply_bench_zero_mixture", "BenchZeroMixtureStats"]
