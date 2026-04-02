from __future__ import annotations

import pytest
import pandas as pd

from projections.cli.score_minutes_v1 import _build_replay_eval_packet


def test_build_replay_eval_packet_includes_promotion_and_propless_slices() -> None:
    df = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "game_date": ["2026-01-01"] * 4,
            "team_id": [10, 10, 20, 20],
            "minutes": [0.0, 22.0, 18.0, 35.0],
            "minutes_p50": [5.0, 20.0, 14.0, 30.0],
            "minutes_p90": [12.0, 28.0, 25.0, 40.0],
            "play_prob": [0.20, 0.80, 0.40, 0.90],
            "promotion_signal_score": [0.20, 0.95, 0.80, 0.10],
            "an_has_any_props": [0.0, 0.0, 1.0, 0.0],
            "prior_play_prob": [0.20, 0.10, 0.60, 0.40],
            "promotion_signal_flag": [0, 1, 1, 0],
            "promotion_signal": [0, 1, 0, 1],
            "promotion_signal_sparse_prior": [1, 1, 0, 0],
            "promotion_signal_propless": [1, 1, 0, 1],
        }
    )

    packet = _build_replay_eval_packet(df)
    slices = packet["slices"]

    assert packet["actual_minutes_col"] == "minutes"
    assert packet["p95_source"] == "derived_from_p50_p90"
    assert int(slices["broad_packet"]["rows"]) == 4
    assert int(slices["target_sparse_propless"]["rows"]) == 3
    assert int(slices["promotion_signal_flag"]["rows"]) == 2
    assert int(slices["promotion_signal"]["rows"]) == 2
    assert int(slices["propless"]["rows"]) == 3
    assert int(slices["actual_minutes_ge_20"]["rows"]) == 2

    broad_play_prob = slices["broad_packet"]["play_prob_calibration"]
    assert broad_play_prob is not None
    assert int(broad_play_prob["rows"]) == 4
    assert broad_play_prob["actual_play_rate"] == pytest.approx(0.75)

    tails = slices["broad_packet"]["tail_hit_rates"]
    assert tails["p90_hit_rate"] == pytest.approx(1.0)
    assert tails["p95_hit_rate"] == pytest.approx(1.0)

    team_hit = slices["broad_packet"]["team_hit_at_k"]
    assert team_hit is not None
    assert team_hit["team_dates"] == 2
    assert team_hit["any_hit_rate"] == pytest.approx(1.0)
    assert team_hit["top1_hit_rate"] == pytest.approx(0.5)
    assert team_hit["top2_hit_rate"] == pytest.approx(1.0)


def test_build_replay_eval_packet_handles_missing_actual_minutes() -> None:
    df = pd.DataFrame(
        {
            "player_id": [1, 2],
            "minutes_p50": [12.0, 16.0],
            "minutes_p90": [20.0, 26.0],
            "play_prob": [0.30, 0.70],
            "promotion_signal_flag": [0, 1],
            "promotion_signal_propless": [1, 0],
        }
    )

    packet = _build_replay_eval_packet(df)
    broad = packet["slices"]["broad_packet"]

    assert packet["actual_minutes_col"] is None
    assert broad["actual_rows"] == 0
    assert broad["play_prob_calibration"] is None
    assert broad["team_hit_at_k"] is None
    assert broad["tail_hit_rates"]["p90_hit_rate"] is None
    assert packet["slices"]["actual_minutes_ge_20"]["rows"] == 0


def test_build_replay_eval_packet_uses_play_prob_sparse_fallback() -> None:
    df = pd.DataFrame(
        {
            "player_id": [1, 2],
            "minutes": [6.0, 20.0],
            "minutes_p50": [8.0, 18.0],
            "minutes_p90": [18.0, 28.0],
            "play_prob": [0.20, 0.90],
            "prior_play_prob": [0.97, 0.97],
            "an_has_any_props": [0.0, 0.0],
        }
    )

    packet = _build_replay_eval_packet(df)
    slices = packet["slices"]

    assert int(slices["sparse_prior"]["rows"]) == 1
    assert int(slices["target_sparse_propless"]["rows"]) == 1
