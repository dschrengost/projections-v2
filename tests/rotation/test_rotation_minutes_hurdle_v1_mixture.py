from __future__ import annotations

import numpy as np
import pytest

from projections.models.rotation_minutes_hurdle_v1.mixture import mixture_quantile_q10_q50_q90


def test_mixture_quantile_tau_le_one_minus_p_is_zero():
    # tau <= (1-p) -> 0 (including equality)
    p = np.array([0.8, 0.8], dtype=float)
    q10 = np.array([10.0, 10.0], dtype=float)
    q50 = np.array([20.0, 20.0], dtype=float)
    q90 = np.array([30.0, 30.0], dtype=float)

    out_eq = mixture_quantile_q10_q50_q90(
        p_play=p,
        q10_cond=q10,
        q50_cond=q50,
        q90_cond=q90,
        tau=0.20,  # 1-p = 0.20
        play_threshold=1.0,
    )
    assert np.all(out_eq == 0.0)

    out_lt = mixture_quantile_q10_q50_q90(
        p_play=p,
        q10_cond=q10,
        q50_cond=q50,
        q90_cond=q90,
        tau=0.10,  # < 1-p
        play_threshold=1.0,
    )
    assert np.all(out_lt == 0.0)


def test_mixture_quantile_maps_tau_above_mass_without_clamping():
    # Choose p just above 0.10 so that tau=0.90 maps to tau_pos < 0.10.
    p = np.array([0.11], dtype=float)
    q10 = np.array([15.0], dtype=float)
    q50 = np.array([25.0], dtype=float)
    q90 = np.array([35.0], dtype=float)

    tau = 0.90
    tau_pos = (tau - (1.0 - p[0])) / p[0]  # ~= 0.0909
    assert 0.0 < tau_pos < 0.10

    out = mixture_quantile_q10_q50_q90(
        p_play=p,
        q10_cond=q10,
        q50_cond=q50,
        q90_cond=q90,
        tau=tau,
        play_threshold=1.0,
    )[0]

    # Expected: interpolate between (0.0, play_threshold) and (0.10, q10) (no clamping to q10).
    expected = 1.0 + (15.0 - 1.0) * (tau_pos / 0.10)
    assert out == pytest.approx(expected, abs=1e-6)
    assert out < q10[0]


def test_mixture_quantile_mid_segment_interpolation():
    p = np.array([0.8], dtype=float)
    q10 = np.array([10.0], dtype=float)
    q50 = np.array([20.0], dtype=float)
    q90 = np.array([30.0], dtype=float)

    tau = 0.50  # > 1-p (=0.2), so maps into conditional
    tau_pos = (tau - (1.0 - p[0])) / p[0]  # 0.375
    assert 0.10 < tau_pos < 0.50

    out = mixture_quantile_q10_q50_q90(
        p_play=p,
        q10_cond=q10,
        q50_cond=q50,
        q90_cond=q90,
        tau=tau,
        play_threshold=1.0,
    )[0]

    expected = 10.0 + (20.0 - 10.0) * ((tau_pos - 0.10) / 0.40)
    assert out == pytest.approx(expected, abs=1e-6)
