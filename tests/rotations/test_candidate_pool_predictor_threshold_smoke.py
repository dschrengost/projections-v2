from __future__ import annotations

import pandas as pd

from projections.rotations.candidate_pool import (
    build_candidate_pool_predictor_threshold,
    build_candidate_pool_prior_topn_team_game,
)


def test_candidate_pool_predictor_threshold_respects_required_thresholds_and_max_size() -> None:
    game_id = "0000000001"
    team_id = 10
    player_ids = list(range(100, 115))  # 15 players

    # Priors: descending minutes_prior with deterministic tie-breaks.
    priors = pd.DataFrame(
        {
            "game_id": [game_id] * len(player_ids),
            "team_id": [team_id] * len(player_ids),
            "player_id": player_ids,
            "minutes_prior": list(range(30, 15, -1)),  # 30..16
        }
    )

    starters = [100, 101, 102, 103, 104]

    # Predictor probabilities: enough threshold candidates to force truncation beyond required set.
    probs = pd.DataFrame(
        {
            "game_id": [game_id] * len(player_ids),
            "team_id": [team_id] * len(player_ids),
            "player_id": player_ids,
            "p_ge15_pred": [
                0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55,  # 100..107 (required by top_n)
                0.40, 0.38, 0.36, 0.10, 0.34, 0.00, 0.00,  # 108..114
            ],
            "p_ge5_pred": [
                0.95, 0.9, 0.9, 0.88, 0.85, 0.8, 0.75, 0.7,  # 100..107
                0.30, 0.20, 0.10, 0.60, 0.90, 0.40, 0.10,  # 108..114
            ],
        }
    )

    out1 = build_candidate_pool_predictor_threshold(
        priors,
        probs,
        starters=starters,
        pool_max_size=11,
        t_ge15=0.35,
        t_ge5=0.35,
        always_include_starters=True,
        always_include_top_n=8,
    )
    out2 = build_candidate_pool_predictor_threshold(
        priors,
        probs,
        starters=starters,
        pool_max_size=11,
        t_ge15=0.35,
        t_ge5=0.35,
        always_include_starters=True,
        always_include_top_n=8,
    )

    assert out1.equals(out2)

    got = set(out1["player_id"].tolist())
    assert len(got) <= 11

    # Required: starters + top 8 by minutes_prior.
    top8 = set(player_ids[:8])
    assert set(starters).issubset(got)
    assert top8.issubset(got)

    # Thresholding adds 108/109/110, but truncation keeps only the top-ranked three.
    assert {108, 109, 110}.issubset(got)
    assert 112 not in got  # would qualify via p_ge5, but is truncated out by rank and pool_max_size
    assert 111 not in got  # would qualify via p_ge5, but is truncated out by rank and pool_max_size
    assert 114 not in got  # never qualifies


def test_candidate_pool_predictor_threshold_fail_open_is_prior_topn_by_minutes() -> None:
    game_id = "0000000001"
    team_id = 10
    player_ids = list(range(100, 115))

    priors = pd.DataFrame(
        {
            "game_id": [game_id] * len(player_ids),
            "team_id": [team_id] * len(player_ids),
            "player_id": player_ids,
            "minutes_prior": list(range(30, 15, -1)),  # 30..16
        }
    )

    out = build_candidate_pool_prior_topn_team_game(priors, top_n=11)
    got = out["player_id"].tolist()

    expected = player_ids[:11]  # top 11 by minutes_prior desc
    assert got == expected

