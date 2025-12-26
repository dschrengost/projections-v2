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


def sample_minutes_noise_per_world(
    *,
    minutes_reconciled: np.ndarray,  # (P,) reconciled minutes per player
    minutes_p10: np.ndarray,  # (P,) lower quantile
    minutes_p90: np.ndarray,  # (P,) upper quantile
    is_starter: np.ndarray,  # (P,) bool
    team_indices: np.ndarray,  # (P,) int, team code per player
    n_worlds: int,
    sigma_starter: float = 2.0,
    sigma_bench: float = 3.0,
    min_minutes_for_noise: float = 8.0,
    cap_abs: float = 6.0,
    use_student_t: bool = False,
    t_df: float = 8.0,
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
    is_starter_arr = np.asarray(is_starter, dtype=bool)
    sigma_per_player = np.where(is_starter_arr, sigma_starter, sigma_bench)

    # Mask: only apply noise to players with enough minutes
    noise_mask = minutes_reconciled >= min_minutes_for_noise

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
        m_noisy, team_indices, lo, hi, noise_mask, n_worlds, n_teams
    )

    return m_final, stats


def _project_team_240_fast(
    m_noisy: np.ndarray,  # (W, P)
    team_indices: np.ndarray,  # (P,)
    lo: np.ndarray,  # (P,)
    hi: np.ndarray,  # (P,)
    adjustable_mask: np.ndarray,  # (P,) bool - which players can be adjusted
    n_worlds: int,
    n_teams: int,
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
        adjustable_team = adjustable_mask[team_players]

        for w in range(n_worlds):
            total_team_worlds += 1
            m = out[w, team_players].copy()
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
                        max_idx = int(np.argmax(m))
                        m[max_idx] = max(m[max_idx] + delta, 0.0)
                        teams_residual_push += 1
                        max_residual = max(max_residual, abs(delta))

            # Final correction to max-minute player if still off
            final_sum = float(m.sum())
            final_delta = 240.0 - final_sum
            if abs(final_delta) > 1e-6:
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
    "apply_pre_sim_qp_reconcile",
    "sample_minutes_noise_per_world",
]
