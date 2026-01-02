"""Monte Carlo minutes quantiles from rotshare point predictions.

This is a minimal, shippable step toward the minutes_v3 direction:
derive `minutes_p10/p50/p90` by simulating team-joint worlds that sum to 240,
then taking empirical percentiles.

We use the existing rotshare outputs as the *mean* allocation:
  - `normalized_share` (sums to 1 per team-game)
  - `play_prob` (used as an inclusion probability per world)

Sampling model (per team-game):
  1) Sample active players via Bernoulli(play_prob), with a small safety floor.
  2) Conditional on active set, sample a Dirichlet over shares with
     alpha_i = concentration * base_share_i (implemented via Gamma draws).
  3) Scale shares to 240 minutes (optional cap + renormalize).

This intentionally avoids hard "top-N" caps; rotation depth emerges from play_prob.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TEAM_TOTAL_MINUTES = 240.0
PLAYER_MINUTES_CAP = 48.0


@dataclass(frozen=True)
class RotshareMonteCarloConfig:
    n_worlds: int = 25_000
    concentration: float = 60.0
    seed: int = 42
    min_active_players: int = 5
    cap_minutes: float = PLAYER_MINUTES_CAP
    max_cap_redistribution_passes: int = 5
    # Center used for the exported "p50" column:
    # - "mean": sums to 240 by construction (E[sum]=240)
    # - "p50": per-player median (does not generally sum to 240)
    center: str = "mean"


def _stable_team_seed(seed: int, game_id: int, team_id: int) -> int:
    # Deterministic mixing; avoid Python hash randomization.
    mixed = (int(game_id) * 1_000_003 + int(team_id) * 97_531 + int(seed) * 1_013) % (2**32)
    return int(mixed)


def _ensure_min_active(
    active: np.ndarray,
    play_prob: np.ndarray,
    *,
    min_active: int,
) -> np.ndarray:
    if int(active.sum()) >= int(min_active):
        return active
    order = np.argsort(-play_prob, kind="mergesort")
    forced = order[: min(int(min_active), len(order))]
    out = active.copy()
    out[forced] = True
    return out


def _sample_dirichlet_gamma(
    rng: np.random.Generator,
    alpha: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    alpha = np.where(np.isfinite(alpha) & (alpha > 0.0), alpha, 0.0)
    if float(alpha.sum()) <= 0.0:
        # Fallback: uniform over non-zero entries (or all if none).
        out = np.ones_like(alpha, dtype=float)
        total = float(out.sum())
        return out / (total if total > 0 else 1.0)
    draws = rng.gamma(shape=np.maximum(alpha, epsilon), scale=1.0)
    total = float(draws.sum())
    if total <= 0.0 or not np.isfinite(total):
        out = np.ones_like(alpha, dtype=float)
        return out / float(out.sum())
    return draws / total


def _cap_and_renormalize_minutes(
    minutes: np.ndarray,
    *,
    cap: float,
    target_total: float,
    max_passes: int,
) -> np.ndarray:
    if cap <= 0.0:
        return minutes
    mins = np.asarray(minutes, dtype=float)
    mins = np.maximum(mins, 0.0)
    mins = np.minimum(mins, cap)
    for _ in range(int(max_passes)):
        total = float(mins.sum())
        gap = float(target_total - total)
        if abs(gap) <= 1e-6:
            break
        if gap > 0:
            room = cap - mins
            eligible = room > 1e-9
            if not bool(np.any(eligible)):
                break
            room_total = float(room[eligible].sum())
            if room_total <= 0.0:
                break
            mins[eligible] = mins[eligible] + gap * (room[eligible] / room_total)
            mins = np.minimum(mins, cap)
        else:
            eligible = mins > 1e-9
            if not bool(np.any(eligible)):
                break
            m_total = float(mins[eligible].sum())
            if m_total <= 0.0:
                break
            mins[eligible] = np.maximum(0.0, mins[eligible] + gap * (mins[eligible] / m_total))
    # Final correction to keep exact team total (push to max-minute player).
    final_gap = float(target_total - float(mins.sum()))
    if abs(final_gap) > 1e-6 and mins.size:
        idx = int(np.argmax(mins))
        mins[idx] = np.clip(mins[idx] + final_gap, 0.0, cap)
    return mins


def _cap_and_renormalize_minutes_worlds(
    minutes_worlds: np.ndarray,
    *,
    cap: float,
    target_total: float,
    max_passes: int,
) -> np.ndarray:
    if cap <= 0.0:
        return minutes_worlds
    mins = np.asarray(minutes_worlds, dtype=float)
    mins = np.maximum(mins, 0.0)
    mins = np.minimum(mins, cap)
    for _ in range(int(max_passes)):
        totals = mins.sum(axis=1)
        gap = target_total - totals
        if float(np.max(np.abs(gap))) <= 1e-6:
            break

        # Add minutes to players with room.
        pos = gap > 1e-6
        if bool(np.any(pos)):
            pos_idx = np.flatnonzero(pos)
            room = np.maximum(cap - mins[pos_idx], 0.0)
            room_total = room.sum(axis=1)
            eligible = room_total > 1e-9
            if bool(np.any(eligible)):
                sel = pos_idx[eligible]
                add = (gap[sel] / room_total[eligible])[:, None] * room[eligible]
                mins[sel] = np.minimum(mins[sel] + add, cap)

        # Remove minutes proportionally from players with minutes.
        neg = gap < -1e-6
        if bool(np.any(neg)):
            neg_idx = np.flatnonzero(neg)
            m = mins[neg_idx]
            m_total = m.sum(axis=1)
            eligible = m_total > 1e-9
            if bool(np.any(eligible)):
                sel = neg_idx[eligible]
                sub = ((-gap[sel]) / m_total[eligible])[:, None] * m[eligible]
                mins[sel] = np.maximum(0.0, mins[sel] - sub)

    # Final correction: push residual to max-minute player per world.
    totals = mins.sum(axis=1)
    gap = target_total - totals
    if float(np.max(np.abs(gap))) > 1e-6 and mins.shape[1] > 0:
        idx = np.argmax(mins, axis=1)
        mins[np.arange(mins.shape[0]), idx] = np.clip(
            mins[np.arange(mins.shape[0]), idx] + gap,
            0.0,
            cap,
        )
    return mins


def sample_team_minutes_worlds(
    *,
    base_share: np.ndarray,
    play_prob: np.ndarray,
    is_out: np.ndarray | None,
    config: RotshareMonteCarloConfig,
    game_id: int,
    team_id: int,
) -> np.ndarray:
    """Sample (W, P) minutes for one team-game."""
    n_worlds = int(config.n_worlds)
    if n_worlds <= 0:
        raise ValueError("n_worlds must be > 0")
    base_share = np.asarray(base_share, dtype=float)
    play_prob = np.asarray(play_prob, dtype=float)
    if base_share.shape != play_prob.shape:
        raise ValueError("base_share and play_prob must have the same shape")

    out_mask = np.zeros_like(play_prob, dtype=bool)
    if is_out is not None:
        out_mask = np.asarray(is_out, dtype=bool)
        if out_mask.shape != play_prob.shape:
            raise ValueError("is_out must align with base_share shape")

    base_share = np.where(out_mask, 0.0, base_share)
    play_prob = np.where(out_mask, 0.0, np.clip(play_prob, 0.0, 1.0))
    base_share = np.where(np.isfinite(base_share) & (base_share > 0.0), base_share, 0.0)
    base_total = float(base_share.sum())
    if base_total <= 0.0:
        # If weights are all zero, fall back to play_prob (or uniform).
        base_share = np.where(play_prob > 0.0, play_prob, 0.0)
        base_total = float(base_share.sum())
        if base_total <= 0.0:
            base_share = np.ones_like(base_share, dtype=float)
            base_total = float(base_share.sum())
    base_share = base_share / base_total

    concentration = float(config.concentration)
    if not np.isfinite(concentration) or concentration <= 0.0:
        raise ValueError("concentration must be finite and > 0")

    rng = np.random.default_rng(_stable_team_seed(config.seed, game_id, team_id))

    n_players = len(base_share)
    alpha_base = np.maximum(base_share * concentration, 1e-6).astype(float)

    active = rng.random(size=(n_worlds, n_players)) < play_prob[None, :]
    if int(config.min_active_players) > 0 and n_players:
        forced = np.argsort(-play_prob, kind="mergesort")[: min(int(config.min_active_players), n_players)]
        need = active.sum(axis=1) < int(config.min_active_players)
        if bool(np.any(need)):
            active[np.ix_(need, forced)] = True
    if out_mask.any():
        active &= ~out_mask[None, :]

    # Ensure at least one active player (if any non-out players exist).
    if n_players:
        non_out = ~out_mask
        if bool(np.any(non_out)):
            top_idx = int(np.argmax(np.where(non_out, play_prob, -1.0)))
            none_active = active.sum(axis=1) == 0
            if bool(np.any(none_active)):
                active[none_active, top_idx] = True

    alpha = np.where(active, alpha_base[None, :], 1e-6)
    draws = rng.gamma(shape=alpha, scale=1.0)
    draws = draws * active.astype(float)
    totals = draws.sum(axis=1)
    bad = totals <= 0.0
    if bool(np.any(bad)):
        fallback = active[bad].astype(float)
        denom = fallback.sum(axis=1)
        denom = np.where(denom > 0.0, denom, 1.0)
        fallback = fallback / denom[:, None]
        shares = np.zeros_like(draws, dtype=float)
        shares[~bad] = draws[~bad] / totals[~bad, None]
        shares[bad] = fallback
    else:
        shares = draws / totals[:, None]

    minutes_worlds = shares * TEAM_TOTAL_MINUTES
    if config.cap_minutes is not None:
        minutes_worlds = _cap_and_renormalize_minutes_worlds(
            minutes_worlds,
            cap=float(config.cap_minutes),
            target_total=TEAM_TOTAL_MINUTES,
            max_passes=int(config.max_cap_redistribution_passes),
        )

    return minutes_worlds


def compute_minutes_quantiles_from_worlds(
    minutes_worlds: np.ndarray,
    *,
    qs: tuple[float, float, float] = (0.1, 0.5, 0.9),
    center: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if minutes_worlds.ndim != 2:
        raise ValueError("minutes_worlds must be a 2D array (W, P)")
    q10, q50, q90 = qs
    p10 = np.quantile(minutes_worlds, q10, axis=0).astype(float)
    p90 = np.quantile(minutes_worlds, q90, axis=0).astype(float)
    if center == "p50":
        mid = np.quantile(minutes_worlds, q50, axis=0).astype(float)
    elif center == "mean":
        mid = np.mean(minutes_worlds, axis=0).astype(float)
    else:
        raise ValueError("center must be 'mean' or 'p50'")
    return p10, mid, p90


def add_rotshare_mc_quantiles(
    df: pd.DataFrame,
    *,
    game_col: str = "game_id",
    team_col: str = "team_id",
    base_share_col: str = "normalized_share",
    play_prob_col: str = "play_prob",
    is_out_col: str | None = "is_out",
    config: RotshareMonteCarloConfig,
    out_prefix: str = "minutes",
) -> pd.DataFrame:
    """Attach minutes quantiles computed from team-joint Monte Carlo sampling."""
    if df.empty:
        return df.copy()

    required = {game_col, team_col, base_share_col, play_prob_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    out = df.copy()
    out[f"{out_prefix}_p10"] = 0.0
    out[f"{out_prefix}_p50"] = 0.0
    out[f"{out_prefix}_p90"] = 0.0

    for (game_id, team_id), g in out.groupby([game_col, team_col], sort=False):
        base_share = pd.to_numeric(g[base_share_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        play_prob = pd.to_numeric(g[play_prob_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        is_out = None
        if is_out_col is not None and is_out_col in g.columns:
            is_out = pd.to_numeric(g[is_out_col], errors="coerce").fillna(0).astype(int).to_numpy(dtype=bool)
        worlds = sample_team_minutes_worlds(
            base_share=base_share,
            play_prob=play_prob,
            is_out=is_out,
            config=config,
            game_id=int(game_id),
            team_id=int(team_id),
        )
        p10, p50, p90 = compute_minutes_quantiles_from_worlds(worlds, center=str(config.center))
        out.loc[g.index, f"{out_prefix}_p10"] = p10
        out.loc[g.index, f"{out_prefix}_p50"] = p50
        out.loc[g.index, f"{out_prefix}_p90"] = p90

    # Ensure monotonic quantiles and bounds.
    out[f"{out_prefix}_p10"] = np.maximum(pd.to_numeric(out[f"{out_prefix}_p10"], errors="coerce").fillna(0.0), 0.0)
    out[f"{out_prefix}_p50"] = np.maximum(pd.to_numeric(out[f"{out_prefix}_p50"], errors="coerce").fillna(0.0), 0.0)
    out[f"{out_prefix}_p90"] = np.maximum(pd.to_numeric(out[f"{out_prefix}_p90"], errors="coerce").fillna(0.0), 0.0)
    out[f"{out_prefix}_p10"] = np.minimum(out[f"{out_prefix}_p10"], out[f"{out_prefix}_p50"])
    out[f"{out_prefix}_p90"] = np.maximum(out[f"{out_prefix}_p90"], out[f"{out_prefix}_p50"])
    out[f"{out_prefix}_p10"] = np.minimum(out[f"{out_prefix}_p10"], float(config.cap_minutes))
    out[f"{out_prefix}_p50"] = np.minimum(out[f"{out_prefix}_p50"], float(config.cap_minutes))
    out[f"{out_prefix}_p90"] = np.minimum(out[f"{out_prefix}_p90"], float(config.cap_minutes))

    return out


__all__ = [
    "RotshareMonteCarloConfig",
    "add_rotshare_mc_quantiles",
    "compute_minutes_quantiles_from_worlds",
    "sample_team_minutes_worlds",
]
