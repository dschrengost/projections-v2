"""Unit tests for the rotalloc fringe-alpha blend allocator."""

from __future__ import annotations

import numpy as np

from projections.models.rotalloc import allocate_fringe_alpha_blend


def _synthetic_team(*, n_players: int = 12, n_eligible: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build simple synthetic inputs with a clear proxy ordering."""
    if n_eligible > n_players:
        raise ValueError("n_eligible must be <= n_players")

    # Higher p_rot/mu for earlier players.
    p_rot = np.clip(0.95 - 0.06 * np.arange(n_players), 0.0, 1.0)
    mu = np.clip(28.0 - 1.5 * np.arange(n_players), 0.0, None)

    # Share proxy favors the middle bench (to simulate a heavy 6th man signal).
    share = np.zeros(n_players, dtype=float)
    share[:n_players] = np.linspace(0.5, 1.5, n_players)

    eligible = np.zeros(n_players, dtype=bool)
    eligible[:n_eligible] = True
    return p_rot, mu, share, eligible


def test_fringe_alpha_blend_respects_mask_and_sums_240() -> None:
    p_rot, mu, share, eligible = _synthetic_team(n_players=12, n_eligible=10)
    minutes, diag = allocate_fringe_alpha_blend(
        p_rot,
        mu,
        share,
        eligible,
        k_core=8,
        alpha_core=0.8,
        alpha_fringe=0.3,
        cap_max=48.0,
    )

    assert abs(float(minutes[eligible].sum()) - 240.0) < 1e-6
    assert (minutes[~eligible] == 0.0).all()
    assert (minutes[eligible] >= 0.0).all()
    assert float(minutes.max()) <= 48.0 + 1e-6
    assert diag["n_eligible"] == 10
    assert diag["core_k"] == 8


def test_fringe_alpha_blend_core_k_caps_at_eligible() -> None:
    p_rot, mu, share, eligible = _synthetic_team(n_players=10, n_eligible=6)
    minutes, diag = allocate_fringe_alpha_blend(
        p_rot,
        mu,
        share,
        eligible,
        k_core=20,
        alpha_core=0.8,
        alpha_fringe=0.3,
    )
    assert abs(float(minutes[eligible].sum()) - 240.0) < 1e-6
    assert diag["core_k"] == 6


def test_fringe_alpha_blend_uniform_fallback_when_all_weights_zero() -> None:
    n_players = 10
    eligible = np.ones(n_players, dtype=bool)
    p_rot = np.zeros(n_players, dtype=float)
    mu = np.zeros(n_players, dtype=float)
    share = np.zeros(n_players, dtype=float)

    minutes, diag = allocate_fringe_alpha_blend(
        p_rot,
        mu,
        share,
        eligible,
        k_core=8,
        alpha_core=0.8,
        alpha_fringe=0.3,
        cap_max=48.0,
    )
    assert abs(float(minutes.sum()) - 240.0) < 1e-6
    assert np.allclose(minutes, 240.0 / n_players)
    assert diag["w_rot_sum"] == 0.0
    assert diag["w_share_sum"] == 0.0


def test_fringe_alpha_blend_clamps_alpha_and_gamma() -> None:
    p_rot, mu, share, eligible = _synthetic_team(n_players=12, n_eligible=10)
    minutes, diag = allocate_fringe_alpha_blend(
        p_rot,
        mu,
        share,
        eligible,
        k_core=8,
        alpha_core=2.0,  # clamp to 1.0
        alpha_fringe=-1.0,  # clamp to 0.0
        share_gamma=-3.0,  # fallback to 1.0
        cap_max=48.0,
    )
    assert abs(float(minutes[eligible].sum()) - 240.0) < 1e-6
    assert diag["alpha_core"] == 1.0
    assert diag["alpha_fringe"] == 0.0
    assert diag["share_gamma"] == 1.0

