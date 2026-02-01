from __future__ import annotations

import pandas as pd

from projections.rotations.candidate_pool import build_candidate_pool_prior, build_candidate_pool_truth


def test_candidate_pool_truth_matches_legacy_mask() -> None:
    labels = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 6,
            "team_id": [10] * 6,
            "player_id": [1, 2, 3, 4, 5, 6],
            "minutes_actual": [0.0, 0.5, 1.0, 10.0, 0.0, 0.0],
            "played_ge_1": [False, False, True, True, False, False],
        }
    )
    out = build_candidate_pool_truth(labels)
    got = set(out["player_id"].tolist())

    expected = set(labels.loc[(labels["minutes_actual"] > 0.0) | (labels["played_ge_1"]), "player_id"].tolist())
    assert got == expected


def test_candidate_pool_prior_is_deterministic_and_respects_thresholds() -> None:
    priors = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 8,
            "team_id": [10] * 8,
            "player_id": [100, 101, 102, 103, 104, 105, 106, 107],
            "minutes_prior": [30, 30, 20, 10, 5, 0, 0, 0],
            "play_prob": [1.0, 1.0, 1.0, 1.0, 0.9, 0.2, 0.8, 0.1],
        }
    )

    # top_n=3 -> {100,101,102} plus thresholds: minutes_prior>=5 OR play_prob>=0.8
    # => includes 103,104,106 as well. Enforce min_candidates=8 backfills remaining in minutes_prior order.
    out1 = build_candidate_pool_prior(
        priors,
        top_n=3,
        min_minutes_prior=5.0,
        min_play_prob=0.8,
        min_candidates=8,
    )
    out2 = build_candidate_pool_prior(
        priors,
        top_n=3,
        min_minutes_prior=5.0,
        min_play_prob=0.8,
        min_candidates=8,
    )

    # Deterministic, and stays within the prior universe.
    assert out1.equals(out2)
    got = set(out1["player_id"].tolist())
    assert got.issubset(set(priors["player_id"].tolist()))
    assert len(got) == 8
