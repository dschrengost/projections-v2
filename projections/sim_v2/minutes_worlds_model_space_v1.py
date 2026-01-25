"""Model-space minutes worlds sampling (PR5 backend).

This module provides the new transformer-based minutes worlds sampler that uses
the rotation_set model's auxiliary outputs (gate logits, share logits, router probs)
to sample minutes worlds directly in model space.

Key design principles:
- Pure function with no side effects (deterministic given rng)
- Uses only the passed-in RNG (never touches global RNG)
- Mutually exclusive with other minutes backends (structured noise, game scripts, fallback)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projections.minutes import PLAY_THRESHOLD_MINUTES, ROTATION_THRESHOLD_MINUTES


@dataclass(frozen=True)
class MinutesWorldsConfig:
    """Configuration for model-space minutes worlds sampling."""

    # Temperature scaling for gate logits (1.0 = no scaling)
    gate_temperature: float = 1.0
    # Whether to use bench-zero mixture (adds mass at zero for low-minute players)
    use_bench_zero_mixture: bool = True
    # Threshold below which players are candidates for bench-zero mixture
    bench_zero_minutes_threshold: float = 8.0
    # Base probability of zero minutes for bench players
    bench_zero_p_base: float = 0.25
    # Slope for zero probability as minutes decrease
    bench_zero_p_slope: float = 0.5


@dataclass
class MinutesWorldsResult:
    """Result from model-space minutes worlds sampling."""

    # Minutes worlds array: (n_worlds, n_players)
    minutes_worlds: np.ndarray
    # Active mask: (n_worlds, n_players) - True if player is active in that world
    active_mask: np.ndarray
    # Diagnostics dict with sampling statistics
    diagnostics: dict


def sample_minutes_worlds_model_space_v1(
    *,
    minutes_mean: np.ndarray,
    gate_logit: np.ndarray | None,
    gate_prob: np.ndarray | None,
    share_logit: np.ndarray,
    play_prob: np.ndarray,
    team_indices: np.ndarray,
    n_worlds: int,
    rng: np.random.Generator,
    config: MinutesWorldsConfig | None = None,
    router_pi: np.ndarray | None = None,
) -> MinutesWorldsResult:
    """Sample minutes worlds using transformer model outputs.

    This function samples minutes for each player in each world using the
    rotation_set model's auxiliary outputs. It enforces team-240 constraints
    and handles active/inactive status via play_prob.

    Args:
        minutes_mean: (P,) baseline minutes allocation from model
        gate_logit: (P,) gate logits (log-odds of being in rotation), or None if no gate head
        gate_prob: (P,) gate probabilities (sigmoid of gate_logit), or None
        share_logit: (P,) share logits for within-rotation allocation
        play_prob: (P,) probability each player appears in the game at all
        team_indices: (P,) integer team index for each player
        n_worlds: number of worlds to sample
        rng: numpy random generator (must be per-chunk, not global)
        config: sampling configuration (uses defaults if None)
        router_pi: (G, E) optional router probabilities per group (for MoE models)

    Returns:
        MinutesWorldsResult with minutes_worlds, active_mask, and diagnostics

    Raises:
        ValueError: if play_prob contains NaN or is missing
    """
    cfg = config or MinutesWorldsConfig()
    n_players = len(minutes_mean)

    # Validate inputs
    if np.isnan(play_prob).any():
        raise ValueError("play_prob contains NaN values - PR5 backend requires valid play_prob")
    if len(play_prob) != n_players:
        raise ValueError(f"play_prob length {len(play_prob)} != n_players {n_players}")

    # 1. Sample active mask from play_prob (Bernoulli per player per world)
    u_active = rng.random(size=(n_worlds, n_players))
    active_mask = u_active < play_prob[None, :]

    # 2. Apply gate probability to determine rotation membership
    # If gate_prob is provided, use it; otherwise assume all active players are in rotation
    if gate_prob is not None:
        # Apply temperature scaling to gate logit if configured
        if gate_logit is not None and cfg.gate_temperature != 1.0:
            scaled_logit = gate_logit / cfg.gate_temperature
            gate_prob_scaled = 1.0 / (1.0 + np.exp(-scaled_logit))
        else:
            gate_prob_scaled = gate_prob

        # Sample rotation membership: active AND passes gate
        u_gate = rng.random(size=(n_worlds, n_players))
        in_rotation_mask = active_mask & (u_gate < gate_prob_scaled[None, :])
    else:
        # No gate head - all active players are in rotation
        in_rotation_mask = active_mask.copy()

    # 3. Initialize minutes worlds with baseline allocation
    minutes_worlds = np.broadcast_to(minutes_mean[None, :], (n_worlds, n_players)).copy()

    # 4. Apply bench-zero mixture for low-minute players
    if cfg.use_bench_zero_mixture:
        # Players below threshold get additional zero probability
        low_minutes_mask = minutes_mean < cfg.bench_zero_minutes_threshold
        # p_zero increases as minutes decrease: p_zero = base + slope * (1 - minutes/threshold)
        minutes_ratio = np.clip(minutes_mean / cfg.bench_zero_minutes_threshold, 0.0, 1.0)
        p_zero = cfg.bench_zero_p_base + cfg.bench_zero_p_slope * (1.0 - minutes_ratio)
        p_zero = np.clip(p_zero, 0.0, 0.95)  # Cap at 95% to avoid pathological cases
        p_zero = np.where(low_minutes_mask, p_zero, 0.0)

        # Sample zero events
        u_zero = rng.random(size=(n_worlds, n_players))
        bench_zero_mask = u_zero < p_zero[None, :]

        # Apply bench-zero: player gets zero minutes if bench_zero fires
        # This is combined with active status: inactive OR bench_zero => 0 minutes
        in_rotation_mask = in_rotation_mask & ~bench_zero_mask

    # 5. Zero out minutes for players not in rotation
    minutes_worlds = np.where(in_rotation_mask, minutes_worlds, 0.0)

    # 6. Enforce team-240 constraint per team per world
    n_teams = int(team_indices.max()) + 1 if team_indices.size else 0
    if n_teams > 0:
        minutes_worlds = _enforce_team_240_simple(
            minutes_worlds=minutes_worlds,
            team_indices=team_indices,
            in_rotation_mask=in_rotation_mask,
        )

    # 7. Compute diagnostics
    active_count = active_mask.sum()
    rotation_count = in_rotation_mask.sum()
    zero_minutes_count = (minutes_worlds == 0.0).sum()

    diagnostics = {
        "n_worlds": n_worlds,
        "n_players": n_players,
        "active_rate": float(active_count / (n_worlds * n_players)) if n_players > 0 else 0.0,
        "rotation_rate": float(rotation_count / (n_worlds * n_players)) if n_players > 0 else 0.0,
        "zero_minutes_rate": float(zero_minutes_count / (n_worlds * n_players)) if n_players > 0 else 0.0,
        "gate_temperature": cfg.gate_temperature,
        "bench_zero_mixture": cfg.use_bench_zero_mixture,
    }

    return MinutesWorldsResult(
        minutes_worlds=minutes_worlds,
        active_mask=active_mask,
        diagnostics=diagnostics,
    )


def _enforce_team_240_simple(
    minutes_worlds: np.ndarray,
    team_indices: np.ndarray,
    in_rotation_mask: np.ndarray,
) -> np.ndarray:
    """Enforce 240 minutes per team per world via proportional scaling.

    Simple implementation that scales rotation players proportionally to hit 240.
    """
    n_worlds, n_players = minutes_worlds.shape
    n_teams = int(team_indices.max()) + 1 if team_indices.size else 0

    out = minutes_worlds.copy()
    team_one_hot = np.eye(n_teams, dtype=float)[team_indices]  # (P, T)

    # Sum of minutes per team per world
    team_sums = out @ team_one_hot  # (W, T)

    # Scale factor per team per world
    scale = np.ones_like(team_sums)
    nonzero = team_sums > 1e-6
    scale[nonzero] = 240.0 / team_sums[nonzero]

    # Clamp scale to avoid extreme values
    scale = np.clip(scale, 0.5, 2.0)

    # Apply scale per player based on their team
    scale_per_player = scale[:, team_indices]  # (W, P)
    out = out * scale_per_player

    # Ensure non-rotation players stay at 0
    out = np.where(in_rotation_mask, out, 0.0)

    return out


def compute_minutes_quantiles(
    minutes_worlds: np.ndarray,
    active_mask: np.ndarray,
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unconditional and conditional minutes quantiles from worlds.

    Args:
        minutes_worlds: (W, P) minutes per world per player
        active_mask: (W, P) boolean mask of active worlds
        quantiles: tuple of quantile values to compute (default: 10th, 50th, 90th)

    Returns:
        uncond_quantiles: (Q, P) unconditional quantiles (DNP => 0 minutes)
        cond_quantiles: (Q, P) conditional quantiles (given player plays)

    Contract:
        - Unconditional includes all worlds, with DNP worlds contributing 0 minutes
        - Conditional only considers worlds where player is active (active_mask=True)
    """
    n_worlds, n_players = minutes_worlds.shape
    n_quantiles = len(quantiles)
    q_arr = np.array(quantiles) * 100  # Convert to percentiles for np.percentile

    # Unconditional: compute over all worlds (DNP => 0 is already in minutes_worlds)
    uncond_quantiles = np.percentile(minutes_worlds, q_arr, axis=0).astype(np.float32)

    # Conditional: compute only over active worlds
    cond_quantiles = np.zeros((n_quantiles, n_players), dtype=np.float32)
    for p in range(n_players):
        active_worlds = active_mask[:, p]
        if active_worlds.any():
            player_minutes = minutes_worlds[active_worlds, p]
            cond_quantiles[:, p] = np.percentile(player_minutes, q_arr)
        else:
            # No active worlds - use 0 (or could use uncond, but 0 is clearer)
            cond_quantiles[:, p] = 0.0

    return uncond_quantiles, cond_quantiles


__all__ = [
    "MinutesWorldsConfig",
    "MinutesWorldsResult",
    "sample_minutes_worlds_model_space_v1",
    "compute_minutes_quantiles",
    "PLAY_THRESHOLD_MINUTES",
    "ROTATION_THRESHOLD_MINUTES",
]
