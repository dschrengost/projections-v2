import pandas as pd
import pytest

from projections.sim_v2.config import PlayProbPolicyConfig
from projections.sim_v2.play_prob_policy import apply_play_prob_policy_with_diagnostics


def test_rotation_lock_heuristic_topk_threshold_and_starter() -> None:
    cfg = PlayProbPolicyConfig(
        enabled=True,
        rotation_lock_min_cond_p50=18.0,
        rotation_lock_topk=2,
        rotation_lock_floor=0.995,
        probable_floor=0.90,
    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1, 1],
            "team_id": [10, 10, 10, 20, 20, 20],
            "player_id": [101, 102, 103, 201, 202, 203],
            # Team 10: top2 should be 101,102; threshold should also lock 103.
            # Team 20: equal minutes -> stable top2 should pick first two rows for that team.
            "minutes_p50_cond": [30.0, 22.0, 18.0, 10.0, 10.0, 1.0],
            "starter_flag": [0, 1, 0, 0, 0, 0],
            "play_prob": [0.5] * 6,
            "status_bucket": ["healthy"] * 6,
        }
    )

    out, _diag = apply_play_prob_policy_with_diagnostics(df, cfg)
    locks = out["rotation_lock"].tolist()

    # Team 10: player 101 (topK), 102 (starter + topK), 103 (threshold).
    assert locks[0] is True
    assert locks[1] is True
    assert locks[2] is True

    # Team 20: top2 stable selection among equal minutes picks first two (201,202).
    assert locks[3] is True
    assert locks[4] is True
    assert locks[5] is False


def test_play_prob_policy_rules_and_reasons() -> None:
    cfg = PlayProbPolicyConfig(
        enabled=True,
        rotation_lock_min_cond_p50=18.0,
        rotation_lock_topk=2,
        rotation_lock_floor=0.995,
        probable_floor=0.90,
    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1],
            "team_id": [10, 10, 10, 10, 10],
            "player_id": [101, 102, 103, 104, 105],
            "minutes_p50_cond": [32.0, 28.0, 12.0, 28.0, 6.0],
            "starter_flag": [1, 0, 0, 0, 0],
            "play_prob": [0.8, 0.2, 0.5, 0.6, 0.3],
            "status_bucket": ["out", "healthy", "probable", "questionable", "healthy"],
        }
    )

    out, _diag = apply_play_prob_policy_with_diagnostics(df, cfg)

    # OUT always forces 0.
    assert float(out.loc[0, "play_prob_eff"]) == 0.0
    assert out.loc[0, "play_prob_policy_reason"] == "out_like"

    # Healthy rotation lock floors to ~1.0.
    assert bool(out.loc[1, "rotation_lock"]) is True
    assert float(out.loc[1, "play_prob_eff"]) == cfg.rotation_lock_floor
    assert out.loc[1, "play_prob_policy_reason"] == "rotation_lock_floor"

    # Probable floors to probable_floor even if not a rotation lock.
    assert float(out.loc[2, "play_prob_eff"]) == cfg.probable_floor
    assert out.loc[2, "play_prob_policy_reason"] == "probable_floor"

    # Questionable should not be floored just because they are a rotation lock.
    assert bool(out.loc[3, "rotation_lock"]) is True
    assert float(out.loc[3, "play_prob_eff"]) == float(out.loc[3, "play_prob_raw"])
    assert out.loc[3, "play_prob_policy_reason"] == "raw"

    # Fringe/healthy remains unchanged.
    assert bool(out.loc[4, "rotation_lock"]) is False
    assert float(out.loc[4, "play_prob_eff"]) == float(out.loc[4, "play_prob_raw"])
    assert out.loc[4, "play_prob_policy_reason"] == "raw"


def test_play_prob_policy_guarded_v2_blocks_depth_and_dnp_risk() -> None:
    cfg = PlayProbPolicyConfig(
        enabled=True,
        mode="guarded_v2",
        rotation_lock_min_cond_p50=8.0,
        rotation_lock_topk=3,
        starter_floor=0.995,
        core_floor=0.92,
        core_lock_min_cond_p50=24.0,
        core_lock_topk=2,
        max_floor_delta=0.25,
        min_raw_play_prob_for_floor=0.35,
        min_rotation_prob_for_floor=0.65,
        dnp_block_streak_threshold=3.0,
        dnp_block_rate_threshold=0.5,
        dnp_block_inactive_streak_threshold=2.0,
    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "player_id": [101, 102, 103],
            "minutes_p50_cond": [34.0, 28.0, 26.0],
            "starter_flag": [1, 0, 0],
            "rotation_prob": [0.95, 0.90, 0.90],
            "play_prob": [0.40, 0.60, 0.60],
            "status_bucket": ["healthy", "healthy", "healthy"],
            "dc_role": ["starter", "rotation", "limited"],
            "dc_ahead_global": [0, 1, 9],
            "consecutive_active_dnp": [0, 0, 7],
            "active_but_dnp_rate_last10": [0.0, 0.0, 0.7],
            "inactive_streak_len": [0, 0, 0],
        }
    )

    out, _diag = apply_play_prob_policy_with_diagnostics(df, cfg)

    # Starter receives bounded floor uplift (0.40 -> min(0.995, 0.40+0.25)=0.65).
    assert float(out.loc[0, "play_prob_eff"]) == pytest.approx(0.65, abs=1e-9)
    assert out.loc[0, "play_prob_policy_reason"] == "starter_floor_guarded_v2"

    # Core non-starter receives bounded core floor uplift (0.60 -> min(0.92, 0.60+0.25)=0.85).
    assert float(out.loc[1, "play_prob_eff"]) == pytest.approx(0.85, abs=1e-9)
    assert out.loc[1, "play_prob_policy_reason"] == "core_floor_guarded_v2"

    # Depth + DNP risk blocks flooring.
    assert float(out.loc[2, "play_prob_eff"]) == float(out.loc[2, "play_prob_raw"])
    assert out.loc[2, "play_prob_policy_reason"] == "raw_blocked_depth_dnp"


def test_play_prob_policy_guarded_v2_requires_raw_and_rotation_thresholds() -> None:
    cfg = PlayProbPolicyConfig(
        enabled=True,
        mode="guarded_v2",
        rotation_lock_min_cond_p50=8.0,
        rotation_lock_topk=3,
        starter_floor=0.995,
        core_floor=0.92,
        core_lock_min_cond_p50=20.0,
        core_lock_topk=3,
        max_floor_delta=0.25,
        min_raw_play_prob_for_floor=0.35,
        min_rotation_prob_for_floor=0.65,
    )

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "player_id": [201, 202, 203],
            "minutes_p50_cond": [30.0, 30.0, 30.0],
            "starter_flag": [0, 0, 1],
            "rotation_prob": [0.90, 0.40, 0.20],
            "play_prob": [0.20, 0.60, 0.20],
            "status_bucket": ["healthy", "healthy", "healthy"],
            "dc_role": ["rotation", "rotation", "starter"],
            "dc_ahead_global": [0, 0, 0],
            "consecutive_active_dnp": [0, 0, 0],
            "active_but_dnp_rate_last10": [0.0, 0.0, 0.0],
            "inactive_streak_len": [0, 0, 0],
        }
    )

    out, _diag = apply_play_prob_policy_with_diagnostics(df, cfg)

    # Raw play_prob too low for flooring.
    assert float(out.loc[0, "play_prob_eff"]) == float(out.loc[0, "play_prob_raw"])
    assert out.loc[0, "play_prob_policy_reason"] == "raw"

    # Rotation probability too low for non-starter core floor.
    assert float(out.loc[1, "play_prob_eff"]) == float(out.loc[1, "play_prob_raw"])
    assert out.loc[1, "play_prob_policy_reason"] == "raw"

    # Starter with low raw is still blocked by min_raw threshold.
    assert float(out.loc[2, "play_prob_eff"]) == float(out.loc[2, "play_prob_raw"])
    assert out.loc[2, "play_prob_policy_reason"] == "raw"
