from __future__ import annotations

import numpy as np


def _interp_conditional_quantile_q10_q50_q90(
    tau_pos: np.ndarray,
    *,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    play_threshold: float,
) -> np.ndarray:
    """Piecewise-linear inverse CDF interpolation for the conditional minutes distribution.

    We only have (q10, q50, q90). For adjusted taus that fall between these anchors we
    interpolate linearly. For tau_pos < 0.10 we interpolate between (0.0, play_threshold)
    and (0.10, q10) (this is the critical path for avoiding 'floor inflation' from clamping).
    For tau_pos > 0.90 we extrapolate using the slope from the (0.50, 0.90) segment.
    """

    tau_pos = np.asarray(tau_pos, dtype=np.float64)
    q10 = np.asarray(q10, dtype=np.float64)
    q50 = np.asarray(q50, dtype=np.float64)
    q90 = np.asarray(q90, dtype=np.float64)

    out = np.empty_like(tau_pos, dtype=np.float64)

    # [0.00, 0.10]
    m0 = tau_pos <= 0.10
    if np.any(m0):
        # Avoid division by zero if someone configures tau=0.0 anchors in the future.
        denom = 0.10
        out[m0] = float(play_threshold) + (q10[m0] - float(play_threshold)) * (tau_pos[m0] / denom)

    # (0.10, 0.50]
    m1 = (tau_pos > 0.10) & (tau_pos <= 0.50)
    if np.any(m1):
        out[m1] = q10[m1] + (q50[m1] - q10[m1]) * ((tau_pos[m1] - 0.10) / 0.40)

    # (0.50, 0.90]
    m2 = (tau_pos > 0.50) & (tau_pos <= 0.90)
    if np.any(m2):
        out[m2] = q50[m2] + (q90[m2] - q50[m2]) * ((tau_pos[m2] - 0.50) / 0.40)

    # (0.90, 1.00] extrapolate with the last slope
    m3 = tau_pos > 0.90
    if np.any(m3):
        # Continue with the slope implied by q50->q90 across 0.50..0.90.
        slope = (q90[m3] - q50[m3]) / 0.40
        out[m3] = q90[m3] + slope * (tau_pos[m3] - 0.90)

    return out


def mixture_quantile_q10_q50_q90(
    *,
    p_play: np.ndarray,
    q10_cond: np.ndarray,
    q50_cond: np.ndarray,
    q90_cond: np.ndarray,
    tau: float,
    play_threshold: float,
) -> np.ndarray:
    """Unconditional quantile for a point-mass-at-zero hurdle model.

    Mixture CDF:
        F(y) = (1-p) * 1[y >= 0] + p * F_pos(y)

    Unconditional quantile:
      - If tau <= (1-p): q_tau = 0
      - Else: q_tau = q_pos((tau - (1-p)) / p)
    """

    p = np.asarray(p_play, dtype=np.float64)
    q10 = np.asarray(q10_cond, dtype=np.float64)
    q50 = np.asarray(q50_cond, dtype=np.float64)
    q90 = np.asarray(q90_cond, dtype=np.float64)
    tau_f = float(tau)

    out = np.zeros_like(p, dtype=np.float64)
    # tau <= 1-p -> 0 (including equality). Use a small epsilon to avoid float
    # representation edge-cases like 0.2 > (1-0.8)=0.199999999999...
    eps = 1e-12
    positive_mask = (p > 0.0) & ((tau_f - (1.0 - p)) > eps)
    if not np.any(positive_mask):
        return out

    p_sel = p[positive_mask]
    tau_pos = (tau_f - (1.0 - p_sel)) / p_sel  # in (0, 1]
    out[positive_mask] = _interp_conditional_quantile_q10_q50_q90(
        tau_pos,
        q10=q10[positive_mask],
        q50=q50[positive_mask],
        q90=q90[positive_mask],
        play_threshold=play_threshold,
    )
    return out


def mixture_quantiles_q10_q50_q90(
    *,
    p_play: np.ndarray,
    q10_cond: np.ndarray,
    q50_cond: np.ndarray,
    q90_cond: np.ndarray,
    taus: list[float],
    play_threshold: float,
) -> dict[float, np.ndarray]:
    return {
        float(tau): mixture_quantile_q10_q50_q90(
            p_play=p_play,
            q10_cond=q10_cond,
            q50_cond=q50_cond,
            q90_cond=q90_cond,
            tau=float(tau),
            play_threshold=play_threshold,
        )
        for tau in taus
    }


__all__ = [
    "mixture_quantile_q10_q50_q90",
    "mixture_quantiles_q10_q50_q90",
]
