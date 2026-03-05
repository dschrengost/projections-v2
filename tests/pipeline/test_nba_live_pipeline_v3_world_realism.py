from __future__ import annotations

import pandas as pd

from prefect_flows.live_nba_pipeline_v3 import (
    _apply_low_minutes_tail_damping_to_worlds,
    _recompute_dk_fpts,
    _resample_extreme_game_worlds,
)


def test_low_minutes_tail_damping_reduces_short_minute_spikes() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 1],
            "game_id": [11, 11],
            "team_id": [101, 101],
            "player_id": [1001, 1001],
            "minutes": [8.0, 30.0],
            "pts": [40.0, 20.0],
            "reb": [12.0, 6.0],
            "ast": [6.0, 4.0],
            "stl": [1.0, 1.0],
            "blk": [1.0, 0.0],
            "tov": [2.0, 2.0],
            "oreb": [4.0, 2.0],
            "dreb": [8.0, 4.0],
        }
    )
    worlds["dk_fpts"] = _recompute_dk_fpts(worlds)

    out, report = _apply_low_minutes_tail_damping_to_worlds(
        worlds,
        minutes_threshold=12.0,
        min_scale=0.55,
    )

    assert report["applied"] is True
    assert report["affected_rows"] == 1
    assert out.loc[0, "pts"] < worlds.loc[0, "pts"]
    assert out.loc[1, "pts"] == worlds.loc[1, "pts"]
    assert out.loc[0, "reb"] == out.loc[0, "oreb"] + out.loc[0, "dreb"]
    assert out.loc[1, "reb"] == out.loc[1, "oreb"] + out.loc[1, "dreb"]
    pd.testing.assert_series_equal(
        out["dk_fpts"].round(6),
        _recompute_dk_fpts(out).round(6),
        check_names=False,
    )


def test_resample_extreme_game_worlds_replaces_bad_pair() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 0, 1, 1],
            "game_id": [22, 22, 22, 22],
            "team_id": [201, 202, 201, 202],
            "player_id": [2001, 2002, 2001, 2002],
            "minutes": [10.0, 30.0, 10.0, 30.0],
            "pts": [28.0, 18.0, 8.0, 18.0],
            "reb": [5.0, 6.0, 2.0, 6.0],
            "ast": [4.0, 5.0, 1.0, 5.0],
            "stl": [1.0, 1.0, 0.0, 1.0],
            "blk": [0.0, 0.0, 0.0, 0.0],
            "tov": [1.0, 2.0, 1.0, 2.0],
            "dk_fpts": [40.0, 35.0, 12.0, 35.0],
        }
    )

    out, report = _resample_extreme_game_worlds(
        worlds,
        random_seed=7,
        max_passes=1,
        game_pts_max=500.0,
        game_pts_min=0.0,
    )

    assert report["applied"] is True
    assert report["total_replaced_pairs"] == 1
    # world_idx=0 was bad and should now match world_idx=1 values (except world_idx)
    out_w0 = out.loc[out["world_idx"] == 0, ["team_id", "player_id", "minutes", "pts", "dk_fpts"]].sort_values(
        ["team_id", "player_id"]
    )
    out_w1 = out.loc[out["world_idx"] == 1, ["team_id", "player_id", "minutes", "pts", "dk_fpts"]].sort_values(
        ["team_id", "player_id"]
    )
    pd.testing.assert_frame_equal(out_w0.reset_index(drop=True), out_w1.reset_index(drop=True))
