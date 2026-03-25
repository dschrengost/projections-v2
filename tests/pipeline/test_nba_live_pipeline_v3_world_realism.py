from __future__ import annotations

import pandas as pd

from prefect_flows.live_nba_pipeline_v3 import (
    _apply_low_minutes_tail_damping_to_worlds,
    _apply_mid_minutes_tail_calibration_to_worlds,
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


def test_resample_extreme_game_worlds_replaces_multiple_bad_pairs_from_same_donor() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 0, 1, 1, 2, 2],
            "game_id": [33, 33, 33, 33, 33, 33],
            "team_id": [301, 302, 301, 302, 301, 302],
            "player_id": [3001, 3002, 3001, 3002, 3001, 3002],
            "minutes": [9.0, 30.0, 11.0, 29.0, 28.0, 30.0],
            "pts": [31.0, 18.0, 29.0, 16.0, 12.0, 14.0],
            "reb": [4.0, 6.0, 3.0, 5.0, 5.0, 7.0],
            "ast": [3.0, 4.0, 2.0, 4.0, 5.0, 6.0],
            "stl": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "blk": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "tov": [1.0, 2.0, 1.0, 2.0, 2.0, 2.0],
            "dk_fpts": [38.0, 24.0, 36.0, 22.0, 18.0, 24.0],
        }
    )

    out, report = _resample_extreme_game_worlds(
        worlds,
        random_seed=11,
        max_passes=1,
        game_pts_max=500.0,
        game_pts_min=0.0,
    )

    assert report["applied"] is True
    assert report["total_replaced_pairs"] == 2
    donor = out.loc[out["world_idx"] == 2, ["team_id", "player_id", "minutes", "pts", "dk_fpts"]].sort_values(
        ["team_id", "player_id"]
    )
    for world_idx in (0, 1):
        replaced = out.loc[
            out["world_idx"] == world_idx,
            ["team_id", "player_id", "minutes", "pts", "dk_fpts"],
        ].sort_values(["team_id", "player_id"])
        pd.testing.assert_frame_equal(replaced.reset_index(drop=True), donor.reset_index(drop=True))


def test_mid_minutes_tail_calibration_lifts_positive_residuals_in_bucket() -> None:
    worlds = pd.DataFrame(
        {
            "world_idx": [0, 1],
            "game_id": [44, 44],
            "team_id": [401, 401],
            "player_id": [4001, 4001],
            "minutes": [16.0, 30.0],
            "pts": [30.0, 10.0],
            "reb": [8.0, 4.0],
            "ast": [6.0, 2.0],
            "stl": [1.0, 0.2],
            "blk": [1.0, 0.1],
            "tov": [2.0, 2.0],
            "oreb": [2.0, 1.0],
            "dreb": [6.0, 3.0],
        }
    )
    worlds["dk_fpts"] = _recompute_dk_fpts(worlds)

    out, report = _apply_mid_minutes_tail_calibration_to_worlds(
        worlds,
        enabled=True,
        min_minutes=12.0,
        max_minutes=20.0,
        tail_boost=0.20,
    )

    assert report["applied"] is True
    assert report["affected_rows"] == 1
    assert out.loc[0, "pts"] > worlds.loc[0, "pts"]
    assert out.loc[1, "pts"] == worlds.loc[1, "pts"]
    assert out.loc[0, "reb"] == out.loc[0, "oreb"] + out.loc[0, "dreb"]
    pd.testing.assert_series_equal(
        out["dk_fpts"].round(6),
        _recompute_dk_fpts(out).round(6),
        check_names=False,
    )
