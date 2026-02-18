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
from projections.sim_v2.minutes_allocator import allocate_team_minutes_matrix


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
    # Whether to use share_logits for per-world demand sampling.
    use_share_logits: bool = True
    # Temperature applied to share logits before softmax.
    share_temperature: float = 1.0
    # Gaussian noise std added to share logits per world.
    share_noise_std: float = 0.0
    # Blend weight between share-driven demand and baseline minutes demand.
    # 0.0 => baseline-only, 1.0 => share-only.
    share_blend_weight: float = 0.5


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
    active_mask: np.ndarray | None = None,
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
        active_mask: optional precomputed availability mask (W, P). When provided,
            sampler uses this mask directly instead of resampling from play_prob.
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

    # 1. Resolve active mask.
    if active_mask is not None:
        active_mask_resolved = np.asarray(active_mask, dtype=bool)
        if active_mask_resolved.shape != (n_worlds, n_players):
            raise ValueError(
                f"active_mask shape {active_mask_resolved.shape} != {(n_worlds, n_players)}"
            )
        active_mask_resolved = active_mask_resolved.copy()
        active_mask_source = "provided"
    else:
        # Sample active mask from play_prob (Bernoulli per player per world)
        u_active = rng.random(size=(n_worlds, n_players))
        active_mask_resolved = u_active < play_prob[None, :]
        active_mask_source = "sampled_from_play_prob"

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
        in_rotation_mask = active_mask_resolved & (u_gate < gate_prob_scaled[None, :])
    else:
        # No gate head - all active players are in rotation
        in_rotation_mask = active_mask_resolved.copy()

    # 3. Build per-world demand minutes.
    minutes_mean_safe = np.maximum(np.asarray(minutes_mean, dtype=float), 0.0)
    minutes_worlds = np.broadcast_to(minutes_mean_safe[None, :], (n_worlds, n_players)).copy()

    use_share_logits = (
        bool(cfg.use_share_logits)
        and share_logit is not None
        and len(share_logit) == n_players
        and n_players > 0
    )
    if use_share_logits:
        share_demand_worlds = _build_share_driven_demand_worlds(
            share_logit=np.asarray(share_logit, dtype=float),
            in_rotation_mask=in_rotation_mask,
            team_indices=team_indices,
            n_worlds=n_worlds,
            rng=rng,
            share_temperature=float(cfg.share_temperature),
            share_noise_std=float(cfg.share_noise_std),
        )
        blend = float(np.clip(cfg.share_blend_weight, 0.0, 1.0))
        minutes_worlds = (1.0 - blend) * minutes_worlds + blend * share_demand_worlds

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
    active_count = active_mask_resolved.sum()
    rotation_count = in_rotation_mask.sum()
    zero_minutes_count = (minutes_worlds == 0.0).sum()

    diagnostics = {
        "n_worlds": n_worlds,
        "n_players": n_players,
        "active_rate": float(active_count / (n_worlds * n_players)) if n_players > 0 else 0.0,
        "rotation_rate": float(rotation_count / (n_worlds * n_players)) if n_players > 0 else 0.0,
        "zero_minutes_rate": float(zero_minutes_count / (n_worlds * n_players)) if n_players > 0 else 0.0,
        "active_mask_source": active_mask_source,
        "gate_temperature": cfg.gate_temperature,
        "bench_zero_mixture": cfg.use_bench_zero_mixture,
        "share_logits_used": bool(use_share_logits),
        "share_temperature": float(cfg.share_temperature),
        "share_noise_std": float(cfg.share_noise_std),
        "share_blend_weight": float(np.clip(cfg.share_blend_weight, 0.0, 1.0)),
    }

    return MinutesWorldsResult(
        minutes_worlds=minutes_worlds,
        active_mask=active_mask_resolved,
        diagnostics=diagnostics,
    )


def _enforce_team_240_simple(
    minutes_worlds: np.ndarray,
    team_indices: np.ndarray,
    in_rotation_mask: np.ndarray,
) -> np.ndarray:
    """Enforce 240 minutes per team per world via weighted bounded projection.

    Uses `minutes_mean`-like priorities implicitly (demand itself) to avoid smearing
    top-minute players when scaling up/down to team total.
    """
    n_worlds, _n_players = minutes_worlds.shape
    n_teams = int(team_indices.max()) + 1 if team_indices.size else 0

    out = minutes_worlds.copy()
    for t in range(n_teams):
        idxs = np.flatnonzero(team_indices == t)
        if idxs.size == 0:
            continue
        demand = out[:, idxs]
        active = in_rotation_mask[:, idxs]
        # Priority: higher demand minutes => more protected.
        priority = np.maximum(demand.mean(axis=0), 0.0)
        allocated, _stats = allocate_team_minutes_matrix(
            demand,
            active,
            priority=priority,
            cap=48.0,
            target_total=240.0,
            k=3.0,
            eps=1e-6,
        )
        out[:, idxs] = allocated

    # Ensure non-rotation players stay at 0 exactly.
    out = np.where(in_rotation_mask, out, 0.0)
    return out


def _build_share_driven_demand_worlds(
    *,
    share_logit: np.ndarray,
    in_rotation_mask: np.ndarray,
    team_indices: np.ndarray,
    n_worlds: int,
    rng: np.random.Generator,
    share_temperature: float,
    share_noise_std: float,
) -> np.ndarray:
    """Build per-world demand from share logits under team-240 constraint."""
    n_players = len(share_logit)
    out = np.zeros((n_worlds, n_players), dtype=float)
    if n_players == 0 or team_indices.size == 0:
        return out

    n_teams = int(team_indices.max()) + 1
    temp = max(float(share_temperature), 1e-3)
    noise_std = max(float(share_noise_std), 0.0)

    base = np.nan_to_num(share_logit.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    for team_id in range(n_teams):
        idx = np.flatnonzero(team_indices == team_id)
        if idx.size == 0:
            continue

        team_logits = base[idx]
        team_logits = team_logits - float(np.mean(team_logits))
        if noise_std > 0.0:
            noise = rng.normal(0.0, noise_std, size=(n_worlds, idx.size))
            logits_world = (team_logits[None, :] + noise) / temp
        else:
            logits_world = np.broadcast_to(team_logits[None, :] / temp, (n_worlds, idx.size)).copy()

        active_team = in_rotation_mask[:, idx]
        shares = _masked_softmax_rows(logits_world, active_team)
        out[:, idx] = shares * 240.0

    return out


def _masked_softmax_rows(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-wise masked softmax returning zeros for rows with no active slots."""
    if logits.shape != mask.shape:
        raise ValueError(f"logits/mask shape mismatch: {logits.shape} vs {mask.shape}")
    if logits.size == 0:
        return np.zeros_like(logits, dtype=float)

    masked_logits = np.where(mask, logits, -1e9)
    row_max = masked_logits.max(axis=1, keepdims=True)
    stable = masked_logits - row_max
    exp_vals = np.exp(stable)
    exp_vals = np.where(mask, exp_vals, 0.0)
    denom = exp_vals.sum(axis=1, keepdims=True)
    return np.divide(exp_vals, denom, out=np.zeros_like(exp_vals), where=denom > 0.0)


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
