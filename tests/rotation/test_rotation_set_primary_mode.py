import pandas as pd

from projections.cli.score_minutes_rotation_set_v1 import (
    _derive_gate_probs,
    _derive_out_mask,
    _scale_minutes_to_team_target,
)
from projections.minutes import PLAY_THRESHOLD_MINUTES, ROTATION_THRESHOLD_MINUTES


def test_primary_mode_out_zero_and_scale_to_240() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2],
            "team_id": [10, 10, 10, 20, 20],
            "player_id": [101, 102, 103, 201, 202],
            "status": ["OUT", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE"],
            "is_out": [1, 0, 0, 0, 0],
        }
    )
    minutes = pd.Series([30.0, 20.0, 10.0, 100.0, 20.0], index=df.index)
    out_mask = _derive_out_mask(df)
    minutes = minutes.where(~out_mask, 0.0)

    scaled, summary = _scale_minutes_to_team_target(df, minutes, team_target=240.0)
    df["scaled"] = scaled

    team_sums = df.groupby(["game_id", "team_id"])["scaled"].sum()
    assert (team_sums.round(6) == 240.0).all()
    assert float(df.loc[0, "scaled"]) == 0.0
    assert summary["zero_sum_teams"] == 0


def test_primary_mode_gate_probs_use_thresholds() -> None:
    minutes = pd.Series(
        [
            0.0,
            PLAY_THRESHOLD_MINUTES - 0.01,
            PLAY_THRESHOLD_MINUTES,
            ROTATION_THRESHOLD_MINUTES - 0.01,
            ROTATION_THRESHOLD_MINUTES,
        ]
    )
    out_mask = pd.Series([False, False, False, False, True])
    play_prob, rotation_prob = _derive_gate_probs(minutes, out_mask=out_mask)

    assert play_prob.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    assert rotation_prob.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
