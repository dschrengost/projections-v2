from __future__ import annotations

import numpy as np
import pytest

from projections.sim_v2.worlds_summary import compute_played_mask


def test_compute_played_mask_masking_enabled_uses_minutes_threshold() -> None:
    mins = np.array(
        [
            [0.0, 0.5, 1.0, 10.0],
            [2.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    mask = compute_played_mask(
        minutes_worlds=mins,
        play_prob_eff=None,
        use_play_prob_masking=True,
        play_threshold_minutes=1.0,
    )
    expected = mins >= 1.0
    np.testing.assert_array_equal(mask, expected)


def test_compute_played_mask_masking_disabled_counts_all_worlds_for_nonzero_p() -> None:
    mins = np.zeros((3, 4), dtype=float)
    p_eff = np.array([1.0, 0.2, 0.0, 0.9], dtype=float)
    mask = compute_played_mask(
        minutes_worlds=mins,
        play_prob_eff=p_eff,
        use_play_prob_masking=False,
        play_threshold_minutes=1.0,
    )
    expected = np.broadcast_to(p_eff > 0.0, mins.shape)
    np.testing.assert_array_equal(mask, expected)


def test_compute_played_mask_requires_play_prob_eff_when_masking_disabled() -> None:
    mins = np.zeros((2, 2), dtype=float)
    with pytest.raises(ValueError, match="play_prob_eff is required"):
        compute_played_mask(
            minutes_worlds=mins,
            play_prob_eff=None,
            use_play_prob_masking=False,
            play_threshold_minutes=1.0,
        )

