from __future__ import annotations

import numpy as np

from projections.sim_v2.minutes_noise import enforce_team_240_minutes


def test_enforce_team_240_rotation_cap_preserves_starters() -> None:
    # Team with too many players projected for meaningful minutes.
    minutes = np.array(
        [
            35.17143,
            33.836637,
            31.256229,
            30.042844,
            29.155958,
            21.659549,
            20.996871,
            16.850449,
            16.283114,
            15.117910,
            15.024008,
            14.364298,
            13.321511,
            12.941548,
        ],
        dtype=float,
    )
    minutes_world = minutes[None, :]
    team_indices = np.zeros(minutes.size, dtype=int)
    starter_mask = np.zeros(minutes.size, dtype=bool)
    starter_mask[:5] = True

    out = enforce_team_240_minutes(
        minutes_world=minutes_world,
        team_indices=team_indices,
        rotation_mask=minutes >= 12.0,
        bench_mask=(minutes > 0.0) & (minutes < 12.0),
        baseline_minutes=minutes,
        clamp_scale=(0.7, 1.3),
        starter_mask=starter_mask,
        max_rotation_size=10,
    )[0]

    # Rotation capped: no more than 10 players receive minutes.
    assert int((out > 0.0).sum()) <= 10

    # Starters should not be nerfed just because the team is oversubscribed.
    np.testing.assert_allclose(out[:5], minutes[:5], rtol=0.0, atol=1e-6)

    # Bench minutes should shrink to make room for starters (when oversubscribed).
    assert out[5] < minutes[5]

    # The dropped tail should be zeroed out.
    assert (out[10:] == 0.0).any()


def test_enforce_team_240_promotes_minutes_when_starter_inactive() -> None:
    minutes = np.array(
        [
            35.0,
            34.0,
            31.0,
            30.0,
            29.0,
            22.0,
            21.0,
            17.0,
            16.0,
            15.0,
            15.0,
            14.0,
            13.0,
            13.0,
        ],
        dtype=float,
    )
    minutes_world = minutes[None, :]
    active_mask = np.ones_like(minutes_world, dtype=bool)
    active_mask[0, 0] = False  # top starter out

    team_indices = np.zeros(minutes.size, dtype=int)
    starter_mask = np.zeros(minutes.size, dtype=bool)
    starter_mask[:5] = True

    out = enforce_team_240_minutes(
        minutes_world=minutes_world,
        team_indices=team_indices,
        rotation_mask=minutes >= 12.0,
        bench_mask=(minutes > 0.0) & (minutes < 12.0),
        baseline_minutes=minutes,
        clamp_scale=(0.7, 1.3),
        active_mask=active_mask,
        starter_mask=starter_mask,
        max_rotation_size=10,
    )[0]

    assert out[0] == 0.0
    assert int((out > 0.0).sum()) <= 10
    # Remaining starters should absorb some of the missing minutes (scale up).
    assert out[1] > minutes[1]


def test_enforce_team_240_rotation_cap_uses_baseline_minutes_for_selection() -> None:
    # If baseline_minutes is provided, bench rotation selection should follow
    # baseline ordering (not per-world sampled minutes).
    minutes = np.array(
        [
            # Starters (kept)
            35.0,
            34.0,
            33.0,
            32.0,
            31.0,
            # Bench (6 candidates; cap allows 5)
            20.0,
            19.0,
            18.0,
            17.0,
            16.0,
            30.0,  # sampled spike, but baseline says this is a fringe player
        ],
        dtype=float,
    )
    baseline_minutes = np.array(
        [
            35.0,
            34.0,
            33.0,
            32.0,
            31.0,
            20.0,
            19.0,
            18.0,
            17.0,
            16.0,
            1.0,  # fringe bench by baseline
        ],
        dtype=float,
    )
    minutes_world = minutes[None, :]
    team_indices = np.zeros(minutes.size, dtype=int)
    starter_mask = np.zeros(minutes.size, dtype=bool)
    starter_mask[:5] = True

    out = enforce_team_240_minutes(
        minutes_world=minutes_world,
        team_indices=team_indices,
        rotation_mask=baseline_minutes >= 12.0,
        bench_mask=(baseline_minutes > 0.0) & (baseline_minutes < 12.0),
        baseline_minutes=baseline_minutes,
        clamp_scale=(0.7, 1.3),
        starter_mask=starter_mask,
        max_rotation_size=10,
    )[0]

    # The fringe player should be dropped, even if their sampled minutes are high.
    assert out[-1] == 0.0
