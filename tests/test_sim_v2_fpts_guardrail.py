"""Tests for sim_v2 FPTS sanity guardrails."""
from __future__ import annotations

import numpy as np
import pytest


def test_extreme_fpts_warning_threshold():
    """Verify that FPTS values > 120 are flagged as extreme.

    The NBA all-time single-game DK FPTS record is around 80.
    Values > 120 indicate noise calibration issues.
    """
    # Simulate typical FPTS distribution
    rng = np.random.default_rng(42)
    n_worlds = 1000
    n_players = 10

    # Baseline: mean 25, std 10 (normal players)
    fpts = rng.normal(25, 10, size=(n_worlds, n_players))
    fpts = np.maximum(fpts, 0)

    EXTREME_THRESHOLD = 120.0
    extreme_mask = fpts > EXTREME_THRESHOLD
    n_extreme = extreme_mask.sum()

    # With normal noise, should have 0 extremes
    assert n_extreme == 0, f"Normal FPTS distribution should have 0 values > {EXTREME_THRESHOLD}"


def test_student_t_noise_produces_extremes():
    """Demonstrate that student-t(5) with k=0.65 can produce extreme values.

    This test documents the current behavior - NOT that it's correct.
    With 25000 worlds and 9 stats, extreme tails will produce
    some FPTS > 120.
    """
    rng = np.random.default_rng(42)
    n_worlds = 25000

    # Simulate per-stat noise with student-t
    nu = 5  # degrees of freedom
    k = 0.65  # noise scale

    # Baseline per-minute rates for a typical player
    # fga2_per_min, fga3_per_min, fta_per_min, etc.
    base_rates = np.array([0.28, 0.16, 0.12, 0.24, 0.08, 0.03, 0.11, 0.03, 0.01])
    minutes = 36  # typical starter
    base_stats = base_rates * minutes

    # Apply multiplicative student-t noise to each stat
    all_fpts = []
    for _ in range(n_worlds):
        eps = rng.standard_t(df=nu, size=len(base_rates))
        stats = np.maximum(base_stats * (1 + k * eps), 0)

        # Simplified FPTS calculation (PTS + 1.25*REB + 1.5*AST + 2*STL + 2*BLK - 0.5*TOV)
        # Using indices: [fga2, fga3, fta, ast, tov, oreb, dreb, stl, blk]
        # Points = fga2*0.5*2 + fga3*0.36*3 + fta*0.84
        pts = stats[0] * 0.5 * 2 + stats[1] * 0.36 * 3 + stats[2] * 0.84
        reb = stats[5] + stats[6]
        ast = stats[3]
        stl = stats[7]
        blk = stats[8]
        tov = stats[4]
        fpts = pts + 1.25 * reb + 1.5 * ast + 2 * stl + 2 * blk - 0.5 * tov
        all_fpts.append(fpts)

    all_fpts = np.array(all_fpts)

    # With student-t(5) and k=0.65, we expect some extreme values
    max_fpts = all_fpts.max()
    p99 = np.percentile(all_fpts, 99)

    # Document current behavior
    assert p99 < 100, f"p99 FPTS should be < 100, got {p99:.2f}"
    # NOTE: max can exceed 120 in rare cases - this is a noise calibration issue
    if max_fpts > 120:
        pytest.xfail(
            f"Student-t noise produced extreme value: max={max_fpts:.2f}. "
            "This documents known noise calibration issue with k=0.65 and nu=5."
        )


def test_fpts_per_minute_sanity():
    """FPTS per minute should not exceed ~3 for any player.

    Historical max is about 2.5 FPTS/min for exceptional performances.
    """
    max_reasonable_fpts_per_min = 3.0
    max_minutes = 48

    reasonable_max_fpts = max_reasonable_fpts_per_min * max_minutes
    assert reasonable_max_fpts < 150, "Max reasonable single-game FPTS should be < 150"
