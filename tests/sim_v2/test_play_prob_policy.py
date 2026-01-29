import pandas as pd

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
