from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class GameFactorStats:
    enabled: bool
    n_games: int
    n_worlds: int
    mode: str
    beta_basis: str


def apply_game_factor(
    fpts_world: np.ndarray,  # (W, P)
    active_mask: np.ndarray,  # (W, P) bool
    *,
    game_ids: np.ndarray,  # (P,)
    beta_basis: np.ndarray,  # (P,) non-negative basis to form per-game shares
    sigma: float,
    mode: Literal["additive", "multiplicative"] = "additive",
    rng: np.random.Generator | None = None,
    game_shocks: np.ndarray | None = None,  # (W, G) optional override (tests)
) -> GameFactorStats:
    """
    Apply a single latent factor per (game, world) to induce cross-team correlation.

    - additive: fpts += shock_game * beta_share
      where beta_share sums to 1 within each game (both teams).
    - multiplicative: fpts *= (1 + shock_game * beta_share)
      (expects sigma to be small, e.g. 0.02..0.10)

    Inactive player-worlds remain hard-zero.
    """
    if rng is None:
        rng = np.random.default_rng()
    fpts = np.asarray(fpts_world)
    if not np.issubdtype(fpts.dtype, np.floating):
        raise TypeError(f"fpts_world must be a floating dtype; got {fpts.dtype}")
    active = np.asarray(active_mask, dtype=bool)
    gids = np.asarray(game_ids)
    basis = np.asarray(beta_basis, dtype=float)

    if fpts.shape != active.shape:
        raise ValueError(f"fpts_world and active_mask must have same shape; got {fpts.shape} vs {active.shape}")
    if gids.shape[0] != fpts.shape[1] or basis.shape[0] != fpts.shape[1]:
        raise ValueError("game_ids and beta_basis must have length equal to n_players (axis=1 of fpts_world)")

    unique_games = np.unique(gids)
    n_games = int(unique_games.size)
    n_worlds = int(fpts.shape[0])
    if n_games == 0 or n_worlds == 0:
        return GameFactorStats(enabled=False, n_games=n_games, n_worlds=n_worlds, mode=mode, beta_basis="custom")

    if game_shocks is None:
        shocks = rng.normal(loc=0.0, scale=float(sigma), size=(n_worlds, n_games))
    else:
        shocks = np.asarray(game_shocks, dtype=float)
        if shocks.shape != (n_worlds, n_games):
            raise ValueError(f"game_shocks must have shape ({n_worlds}, {n_games}); got {shocks.shape}")

    for gi, gid in enumerate(unique_games):
        idxs = np.where(gids == gid)[0]
        if idxs.size == 0:
            continue

        active_g = active[:, idxs]  # (W, N)
        active_count = active_g.sum(axis=1)  # (W,)
        if not active_count.any():
            continue

        basis_g = np.clip(basis[idxs], 0.0, None)[None, :]  # (1, N)
        b_active = basis_g * active_g.astype(float)  # (W, N)
        denom = b_active.sum(axis=1)  # (W,)

        beta = np.zeros_like(b_active)
        has_denom = denom > 0.0
        if has_denom.any():
            beta[has_denom] = b_active[has_denom] / denom[has_denom][:, None]

        # If all basis is 0 but there are active players, use uniform shares among actives.
        fallback = ~has_denom & (active_count > 0)
        if fallback.any():
            beta[fallback] = active_g[fallback].astype(float) / active_count[fallback][:, None]

        shock = shocks[:, gi][:, None]  # (W, 1)
        adj = shock * beta  # (W, N)

        if mode == "additive":
            updated = fpts[:, idxs] + adj
        elif mode == "multiplicative":
            updated = fpts[:, idxs] * (1.0 + adj)
        else:
            raise ValueError(f"Unsupported game_factor.mode={mode!r}")

        fpts[:, idxs] = np.where(active_g, updated, 0.0)

    return GameFactorStats(
        enabled=True,
        n_games=n_games,
        n_worlds=n_worlds,
        mode=str(mode),
        beta_basis="custom",
    )


__all__ = ["apply_game_factor", "GameFactorStats"]
