from __future__ import annotations

import numpy as np


def compute_played_mask(
    *,
    minutes_worlds: np.ndarray | None,
    play_prob_eff: np.ndarray | None,
    use_play_prob_masking: bool,
    play_threshold_minutes: float,
) -> np.ndarray:
    """Return a (W, P) boolean mask for "played" worlds.

    Definitions:
    - When `use_play_prob_masking=True` (sim_v3/prod), worlds are unconditional and
      DNP => 0 minutes, so "played" is defined as minutes >= play_threshold_minutes.
    - When `use_play_prob_masking=False` (baseline), worlds are generated conditional-on-playing
      except for players with play_prob_eff==0; so conditional moments should count all worlds
      for play_prob_eff>0 players.
    """
    if minutes_worlds is None:
        raise ValueError("minutes_worlds is required to compute played mask")
    mins = np.asarray(minutes_worlds, dtype=float)
    if mins.ndim != 2:
        raise ValueError(f"minutes_worlds must be 2D (W, P), got {mins.shape}")

    if use_play_prob_masking:
        threshold = float(play_threshold_minutes)
        return mins >= threshold

    if play_prob_eff is None:
        raise ValueError("play_prob_eff is required when use_play_prob_masking=False")
    p = np.asarray(play_prob_eff, dtype=float)
    if p.ndim != 1 or p.shape[0] != mins.shape[1]:
        raise ValueError(f"play_prob_eff must have shape (P,), got {p.shape} for minutes_worlds {mins.shape}")
    return np.broadcast_to(p > 0.0, mins.shape)


__all__ = ["compute_played_mask"]

