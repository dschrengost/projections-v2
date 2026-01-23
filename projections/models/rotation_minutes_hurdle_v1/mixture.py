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

    NOTE: This is the v1.0 interpolation with only 3 quantiles. Use
    _interp_conditional_quantile for v1.1 with 7 quantiles.

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


def _interp_conditional_quantile(
    tau_pos: np.ndarray,
    *,
    q05: np.ndarray,
    q10: np.ndarray,
    q25: np.ndarray,
    q50: np.ndarray,
    q75: np.ndarray,
    q90: np.ndarray,
    q95: np.ndarray,
    play_threshold: float,
) -> np.ndarray:
    """Piecewise-linear inverse CDF interpolation for the conditional minutes distribution.

    v1.1: Uses all 7 quantiles for better tail fidelity and reduced extrapolation.

    Segments:
        [0.00, 0.05]: interpolate from (0, play_threshold) to q05
        (0.05, 0.10]: interpolate q05 -> q10
        (0.10, 0.25]: interpolate q10 -> q25
        (0.25, 0.50]: interpolate q25 -> q50
        (0.50, 0.75]: interpolate q50 -> q75
        (0.75, 0.90]: interpolate q75 -> q90
        (0.90, 0.95]: interpolate q90 -> q95
        (0.95, 1.00]: extrapolate from q95 using q90->q95 slope
    """
    tau_pos = np.asarray(tau_pos, dtype=np.float64)
    q05 = np.asarray(q05, dtype=np.float64)
    q10 = np.asarray(q10, dtype=np.float64)
    q25 = np.asarray(q25, dtype=np.float64)
    q50 = np.asarray(q50, dtype=np.float64)
    q75 = np.asarray(q75, dtype=np.float64)
    q90 = np.asarray(q90, dtype=np.float64)
    q95 = np.asarray(q95, dtype=np.float64)

    out = np.empty_like(tau_pos, dtype=np.float64)
    T = float(play_threshold)

    # [0.00, 0.05]: interpolate from (0, T) to (0.05, q05)
    m = tau_pos <= 0.05
    if np.any(m):
        out[m] = T + (q05[m] - T) * (tau_pos[m] / 0.05)

    # (0.05, 0.10]: interpolate q05 -> q10
    m = (tau_pos > 0.05) & (tau_pos <= 0.10)
    if np.any(m):
        out[m] = q05[m] + (q10[m] - q05[m]) * ((tau_pos[m] - 0.05) / 0.05)

    # (0.10, 0.25]: interpolate q10 -> q25
    m = (tau_pos > 0.10) & (tau_pos <= 0.25)
    if np.any(m):
        out[m] = q10[m] + (q25[m] - q10[m]) * ((tau_pos[m] - 0.10) / 0.15)

    # (0.25, 0.50]: interpolate q25 -> q50
    m = (tau_pos > 0.25) & (tau_pos <= 0.50)
    if np.any(m):
        out[m] = q25[m] + (q50[m] - q25[m]) * ((tau_pos[m] - 0.25) / 0.25)

    # (0.50, 0.75]: interpolate q50 -> q75
    m = (tau_pos > 0.50) & (tau_pos <= 0.75)
    if np.any(m):
        out[m] = q50[m] + (q75[m] - q50[m]) * ((tau_pos[m] - 0.50) / 0.25)

    # (0.75, 0.90]: interpolate q75 -> q90
    m = (tau_pos > 0.75) & (tau_pos <= 0.90)
    if np.any(m):
        out[m] = q75[m] + (q90[m] - q75[m]) * ((tau_pos[m] - 0.75) / 0.15)

    # (0.90, 0.95]: interpolate q90 -> q95
    m = (tau_pos > 0.90) & (tau_pos <= 0.95)
    if np.any(m):
        out[m] = q90[m] + (q95[m] - q90[m]) * ((tau_pos[m] - 0.90) / 0.05)

    # (0.95, 1.00]: extrapolate using q90->q95 slope
    m = tau_pos > 0.95
    if np.any(m):
        slope = (q95[m] - q90[m]) / 0.05
        out[m] = q95[m] + slope * (tau_pos[m] - 0.95)

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


def mixture_quantile(
    *,
    p_play: np.ndarray,
    q05_cond: np.ndarray,
    q10_cond: np.ndarray,
    q25_cond: np.ndarray,
    q50_cond: np.ndarray,
    q75_cond: np.ndarray,
    q90_cond: np.ndarray,
    q95_cond: np.ndarray,
    tau: float,
    play_threshold: float,
) -> np.ndarray:
    """Unconditional quantile for a point-mass-at-zero hurdle model (v1.1).

    v1.1: Uses all 7 conditional quantiles for improved tail fidelity.

    Mixture CDF:
        F(y) = (1-p) * 1[y >= 0] + p * F_pos(y)

    Unconditional quantile:
      - If tau <= (1-p): q_tau = 0
      - Else: q_tau = q_pos((tau - (1-p)) / p)
    """
    p = np.asarray(p_play, dtype=np.float64)
    q05 = np.asarray(q05_cond, dtype=np.float64)
    q10 = np.asarray(q10_cond, dtype=np.float64)
    q25 = np.asarray(q25_cond, dtype=np.float64)
    q50 = np.asarray(q50_cond, dtype=np.float64)
    q75 = np.asarray(q75_cond, dtype=np.float64)
    q90 = np.asarray(q90_cond, dtype=np.float64)
    q95 = np.asarray(q95_cond, dtype=np.float64)
    tau_f = float(tau)

    out = np.zeros_like(p, dtype=np.float64)
    eps = 1e-12
    positive_mask = (p > 0.0) & ((tau_f - (1.0 - p)) > eps)
    if not np.any(positive_mask):
        return out

    p_sel = p[positive_mask]
    tau_pos = (tau_f - (1.0 - p_sel)) / p_sel  # in (0, 1]
    out[positive_mask] = _interp_conditional_quantile(
        tau_pos,
        q05=q05[positive_mask],
        q10=q10[positive_mask],
        q25=q25[positive_mask],
        q50=q50[positive_mask],
        q75=q75[positive_mask],
        q90=q90[positive_mask],
        q95=q95[positive_mask],
        play_threshold=play_threshold,
    )
    return out


def mixture_quantiles(
    *,
    p_play: np.ndarray,
    q05_cond: np.ndarray,
    q10_cond: np.ndarray,
    q25_cond: np.ndarray,
    q50_cond: np.ndarray,
    q75_cond: np.ndarray,
    q90_cond: np.ndarray,
    q95_cond: np.ndarray,
    taus: list[float],
    play_threshold: float,
) -> dict[float, np.ndarray]:
    """Compute multiple unconditional quantiles at once (v1.1)."""
    return {
        float(tau): mixture_quantile(
            p_play=p_play,
            q05_cond=q05_cond,
            q10_cond=q10_cond,
            q25_cond=q25_cond,
            q50_cond=q50_cond,
            q75_cond=q75_cond,
            q90_cond=q90_cond,
            q95_cond=q95_cond,
            tau=float(tau),
            play_threshold=play_threshold,
        )
        for tau in taus
    }


__all__ = [
    "mixture_quantile",
    "mixture_quantile_q10_q50_q90",
    "mixture_quantiles",
    "mixture_quantiles_q10_q50_q90",
]
