from __future__ import annotations

import numpy as np
import pytest

from projections.models.rotation_minutes_hurdle_v1.mixture import (
    mixture_quantile,
    mixture_quantile_q10_q50_q90,
)


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


# --- v1.1 tests for 7-quantile mixture function ---


def test_mixture_quantile_v11_tau_le_one_minus_p_is_zero():
    """v1.1: tau <= (1-p) -> 0 (including equality)."""
    p = np.array([0.8, 0.8], dtype=float)
    q05 = np.array([5.0, 5.0], dtype=float)
    q10 = np.array([10.0, 10.0], dtype=float)
    q25 = np.array([15.0, 15.0], dtype=float)
    q50 = np.array([20.0, 20.0], dtype=float)
    q75 = np.array([25.0, 25.0], dtype=float)
    q90 = np.array([30.0, 30.0], dtype=float)
    q95 = np.array([35.0, 35.0], dtype=float)

    out_eq = mixture_quantile(
        p_play=p,
        q05_cond=q05,
        q10_cond=q10,
        q25_cond=q25,
        q50_cond=q50,
        q75_cond=q75,
        q90_cond=q90,
        q95_cond=q95,
        tau=0.20,  # 1-p = 0.20
        play_threshold=5.0,
    )
    assert np.all(out_eq == 0.0)


def test_mixture_quantile_v11_maps_tau_above_mass_without_clamping():
    """v1.1: tau_pos < 0.05 should interpolate from (0, T) to q05 (no clamping)."""
    p = np.array([0.051], dtype=float)  # p chosen so tau_pos falls in (0, 0.05)
    q05 = np.array([8.0], dtype=float)
    q10 = np.array([12.0], dtype=float)
    q25 = np.array([18.0], dtype=float)
    q50 = np.array([25.0], dtype=float)
    q75 = np.array([30.0], dtype=float)
    q90 = np.array([35.0], dtype=float)
    q95 = np.array([38.0], dtype=float)

    tau = 0.95
    tau_pos = (tau - (1.0 - p[0])) / p[0]  # ~= 0.0167
    assert 0.0 < tau_pos < 0.05

    out = mixture_quantile(
        p_play=p,
        q05_cond=q05,
        q10_cond=q10,
        q25_cond=q25,
        q50_cond=q50,
        q75_cond=q75,
        q90_cond=q90,
        q95_cond=q95,
        tau=tau,
        play_threshold=5.0,
    )[0]

    # Expected: interpolate between (0.0, play_threshold=5) and (0.05, q05=8)
    expected = 5.0 + (8.0 - 5.0) * (tau_pos / 0.05)
    assert out == pytest.approx(expected, abs=1e-6)
    assert out < q05[0]  # Should be below q05 since tau_pos < 0.05


def test_mixture_quantile_v11_mid_segment_interpolation():
    """v1.1: Test interpolation in a middle segment (q25 to q50)."""
    p = np.array([0.8], dtype=float)
    q05 = np.array([5.0], dtype=float)
    q10 = np.array([10.0], dtype=float)
    q25 = np.array([15.0], dtype=float)
    q50 = np.array([20.0], dtype=float)
    q75 = np.array([25.0], dtype=float)
    q90 = np.array([30.0], dtype=float)
    q95 = np.array([35.0], dtype=float)

    # Choose tau such that tau_pos falls in (0.25, 0.50]
    tau = 0.50  # > 1-p (=0.2), so maps into conditional
    tau_pos = (tau - (1.0 - p[0])) / p[0]  # 0.375
    assert 0.25 < tau_pos < 0.50

    out = mixture_quantile(
        p_play=p,
        q05_cond=q05,
        q10_cond=q10,
        q25_cond=q25,
        q50_cond=q50,
        q75_cond=q75,
        q90_cond=q90,
        q95_cond=q95,
        tau=tau,
        play_threshold=5.0,
    )[0]

    # Expected: interpolate in q25->q50 segment
    expected = 15.0 + (20.0 - 15.0) * ((tau_pos - 0.25) / 0.25)
    assert out == pytest.approx(expected, abs=1e-6)


def test_mixture_quantile_v11_extrapolation_above_q95():
    """v1.1: tau_pos > 0.95 should extrapolate using q90->q95 slope."""
    p = np.array([1.0], dtype=float)  # p=1 so tau_pos=tau
    q05 = np.array([5.0], dtype=float)
    q10 = np.array([10.0], dtype=float)
    q25 = np.array([15.0], dtype=float)
    q50 = np.array([20.0], dtype=float)
    q75 = np.array([25.0], dtype=float)
    q90 = np.array([30.0], dtype=float)
    q95 = np.array([35.0], dtype=float)

    tau = 0.98  # > 0.95, should extrapolate
    tau_pos = tau  # since p=1

    out = mixture_quantile(
        p_play=p,
        q05_cond=q05,
        q10_cond=q10,
        q25_cond=q25,
        q50_cond=q50,
        q75_cond=q75,
        q90_cond=q90,
        q95_cond=q95,
        tau=tau,
        play_threshold=5.0,
    )[0]

    # Expected: extrapolate using slope from q90->q95
    slope = (35.0 - 30.0) / 0.05  # = 100
    expected = 35.0 + slope * (tau_pos - 0.95)
    assert out == pytest.approx(expected, abs=1e-6)
    assert out > q95[0]  # Should be above q95 since we're extrapolating
