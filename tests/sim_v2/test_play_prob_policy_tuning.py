from __future__ import annotations

import json

import numpy as np
import pandas as pd

from projections.sim_v2.config import PlayProbPolicyConfig
from projections.sim_v2.play_prob_policy_tuning import (
    PolicyObjectiveWeights,
    build_policy_grid_overrides,
    compute_policy_metrics,
    evaluate_policy_candidate,
)


def test_compute_policy_metrics_smoke() -> None:
    df = pd.DataFrame(
        {
            "plays_target": [1, 0, 1, 0],
            "play_prob_raw": [0.80, 0.80, 0.60, 0.20],
            "play_prob_eff": [0.90, 0.40, 0.70, 0.20],
            "is_starter": [1, 1, 0, 0],
        }
    )
    out = compute_policy_metrics(df, bins=4)

    assert out["rows"] == 4.0
    assert out["plays_rate"] == 0.5
    assert 0.0 <= out["brier_raw"] <= 1.0
    assert 0.0 <= out["brier_eff"] <= 1.0
    assert np.isfinite(out["logloss_raw"])
    assert np.isfinite(out["logloss_eff"])
    assert out["false_active_p90_eff"] == 0.0
    assert out["starter_mean_p_eff"] == np.mean([0.9, 0.4])


def test_build_policy_grid_overrides_dedupes_and_includes_baseline() -> None:
    base = PlayProbPolicyConfig()
    candidates = build_policy_grid_overrides(
        base,
        grids={
            "core_floor": [0.90, 0.90, 0.92],
            "core_lock_topk": [base.core_lock_topk],
        },
        include_baseline=True,
    )

    assert len(candidates) == 2
    assert candidates[0]["core_floor"] == base.core_floor
    assert candidates[0]["core_lock_topk"] == base.core_lock_topk
    assert {float(c["core_floor"]) for c in candidates} == {0.9, 0.92}


def test_evaluate_policy_candidate_returns_objective_and_reasons() -> None:
    cfg = PlayProbPolicyConfig(
        enabled=True,
        mode="guarded_v2",
        starter_floor=0.995,
        core_floor=0.9,
        core_lock_min_cond_p50=20.0,
        core_lock_topk=2,
        max_floor_delta=0.30,
        min_raw_play_prob_for_floor=0.2,
        min_rotation_prob_for_floor=0.5,
        depth_block_roles=("limited", "not_listed"),
        depth_block_min_ahead_global=7,
        dnp_block_streak_threshold=3.0,
        dnp_block_rate_threshold=0.5,
        dnp_block_inactive_streak_threshold=2.0,
    )
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "player_id": [100, 101, 102],
            "play_prob": [0.70, 0.50, 0.40],
            "rotation_prob": [0.95, 0.80, 0.20],
            "minutes_p50": [32.0, 26.0, 10.0],
            "is_starter": [1, 0, 0],
            "status": ["Ava", "Ava", "Ava"],
            "dc_role": ["starter", "rotation", "limited"],
            "dc_ahead_global": [0, 2, 9],
            "consecutive_active_dnp": [0, 0, 5],
            "active_but_dnp_rate_last10": [0.0, 0.1, 0.8],
            "inactive_streak_len": [0, 0, 0],
            "plays_target": [1, 1, 0],
        }
    )

    out = evaluate_policy_candidate(
        df,
        cfg,
        bins=5,
        weights=PolicyObjectiveWeights(),
    )

    assert np.isfinite(out["objective"])
    assert out["rows"] == 3.0
    assert out["n_changed"] >= 1.0
    parsed_reasons = json.loads(out["reason_counts_json"])
    assert isinstance(parsed_reasons, dict)
