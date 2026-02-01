from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from projections.rotations.rotation_gate import GateConfig, apply_rotation_gate


def test_rotation_gate_tiers_and_caps_smoke() -> None:
    cfg = replace(
        GateConfig(),
        enabled=True,
        protect_top_n=False,
        core_ge15_min=0.40,
        bench_ge5_min=0.30,
        bench_minutes_prior_min=10.0,
        bench_minutes_cap=14.0,
        fringe_minutes_cap=6.0,
        fringe_play_prob_cap=0.70,
    )

    priors = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 5,
            "team_id": [10] * 5,
            "player_id": [1, 2, 3, 4, 5],
            "minutes_prior": [32.0, 22.0, 16.0, 9.0, 3.5],
            "minutes_p10": [28.0, 18.0, 12.0, 6.0, 0.0],
            "minutes_p90": [36.0, 26.0, 20.0, 12.0, 6.0],
            "play_prob": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    preds = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 5,
            "team_id": [10] * 5,
            "player_id": [1, 2, 3, 4, 5],
            "p_ge5_pred": [0.99, 0.20, 0.70, 0.20, 0.01],
            "p_ge15_pred": [0.95, 0.55, 0.10, 0.05, 0.00],
        }
    )

    out = apply_rotation_gate(priors, preds, starters_set={1}, cfg=cfg, seed=0)
    assert set(["p_ge5_pred", "p_ge15_pred", "gate_tier", "gate_reason", "minutes_prior_adj", "play_prob_adj"]).issubset(
        set(out.columns)
    )

    by_pid = {int(r.player_id): r for r in out.itertuples(index=False)}

    assert by_pid[1].gate_tier == "starter"
    assert float(by_pid[1].minutes_prior_adj) == 32.0

    assert by_pid[2].gate_tier == "core"  # p_ge15 protects
    assert float(by_pid[2].minutes_prior_adj) == 22.0

    assert by_pid[3].gate_tier == "bench"
    assert float(by_pid[3].minutes_prior_adj) == 14.0  # capped

    assert by_pid[4].gate_tier == "fringe"
    assert float(by_pid[4].minutes_prior_adj) == 6.0  # capped
    assert float(by_pid[4].play_prob_adj) == 0.70

    # Gate is non-structural: never excludes players.
    assert by_pid[5].gate_tier == "fringe"
    assert bool(by_pid[5].gate_excluded) is False
    assert float(by_pid[5].minutes_prior_adj) == 3.5
    assert float(by_pid[5].play_prob_adj) == 0.70


def test_rotation_gate_preserves_player_ids_and_never_excludes() -> None:
    cfg = replace(GateConfig(), enabled=True, protect_top_n=False)

    priors = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 4,
            "team_id": [10] * 4,
            "player_id": [1, 2, 3, 4],
            "minutes_prior": [6.0, 6.0, 3.5, 3.2],
            "play_prob": [1.0, 1.0, 1.0, 1.0],
        }
    )
    preds = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 4,
            "team_id": [10] * 4,
            "player_id": [1, 2, 3, 4],
            # Low-probability fringe players should still be present (gate is non-structural).
            "p_ge5_pred": [0.9, 0.9, 0.04, 0.01],
            "p_ge15_pred": [0.9, 0.9, 0.0, 0.0],
        }
    )

    out = apply_rotation_gate(priors, preds, starters_set=set(), cfg=cfg, seed=123)
    assert set(out["player_id"].astype(int).tolist()) == {1, 2, 3, 4}
    assert bool(out["gate_excluded"].fillna(False).any()) is False


def test_rotation_gate_missing_preds_are_noop() -> None:
    cfg = replace(
        GateConfig(),
        enabled=True,
        protect_top_n=False,
        protect_starters=False,
        core_ge15_min=0.99,
        core_minutes_prior_min=999.0,
        bench_ge5_min=0.99,
        bench_minutes_prior_min=999.0,
        bench_minutes_cap=14.0,
        fringe_minutes_cap=6.0,
        fringe_play_prob_cap=0.70,
    )

    priors = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 3,
            "team_id": [10] * 3,
            "player_id": [1, 2, 3],
            "minutes_prior": [30.0, 12.0, 12.0],
            "minutes_p10": [26.0, 10.0, 10.0],
            "minutes_p90": [34.0, 14.0, 14.0],
            "play_prob": [1.0, 1.0, 0.9],
        }
    )
    preds = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 3,
            "team_id": [10] * 3,
            "player_id": [1, 2, 3],
            "p_ge5_pred": [0.99, 0.10, np.nan],
            "p_ge15_pred": [0.99, 0.10, np.nan],
        }
    )

    out = apply_rotation_gate(priors, preds, starters_set=set(), cfg=cfg, seed=0)
    by_pid = {int(r.player_id): r for r in out.itertuples(index=False)}

    # Player 2 has predictions: should be fringe-capped under this config.
    assert by_pid[2].gate_tier == "fringe"
    assert float(by_pid[2].minutes_prior_adj) == 6.0
    assert float(by_pid[2].play_prob_adj) == 0.70

    # Player 3 is missing predictions: strict no-op (no cap, no exclusion).
    assert by_pid[3].gate_tier == "unknown"
    assert by_pid[3].gate_reason == "missing_pred"
    assert bool(by_pid[3].gate_missing_pred) is True
    assert bool(by_pid[3].gate_excluded) is False
    assert bool(np.isfinite(float(by_pid[3].gate_minutes_cap))) is False
    assert bool(np.isfinite(float(by_pid[3].gate_play_prob_cap))) is False
    assert float(by_pid[3].minutes_prior_adj) == 12.0
    assert float(by_pid[3].minutes_p10_adj) == 10.0
    assert float(by_pid[3].minutes_p90_adj) == 14.0
    assert float(by_pid[3].play_prob_adj) == 0.9
