from __future__ import annotations

import numpy as np
import pandas as pd

from projections.models.rotation_minutes_hurdle_v1.live_features import prepare_live_features_for_rmh


def test_prepare_live_features_for_rmh_adds_derived_features_and_normalizes_injury() -> None:
    df = pd.DataFrame(
        [
            # Team 100 (1 out, 1 UNK/no injury row)
            {
                "game_id": 1,
                "team_id": 100,
                "player_id": 1,
                "pos_bucket": "BIG",
                "is_out": 0,
                "minutes_from_stints_prior_20": 30.0,
                "starter_flag": 1,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 100,
                "player_id": 2,
                "pos_bucket": "G",
                "is_out": 0,
                "minutes_from_stints_prior_20": 28.0,
                "starter_flag": 1,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 100,
                "player_id": 3,
                "pos_bucket": "W",
                "is_out": 0,
                "minutes_from_stints_prior_20": 26.0,
                "starter_flag": 1,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 100,
                "player_id": 4,
                "pos_bucket": "W",
                "is_out": 0,
                "minutes_from_stints_prior_20": 24.0,
                "starter_flag": 1,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 100,
                "player_id": 5,
                "pos_bucket": "G",
                "is_out": 0,
                "minutes_from_stints_prior_20": 22.0,
                "starter_flag": 1,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 100,
                "player_id": 6,
                "pos_bucket": "BIG",
                "is_out": 1,
                "minutes_from_stints_prior_20": 20.0,
                "starter_flag": 0,
                "injury_row_present": True,
                "status": "OUT",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.0,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 100,
                "player_id": 7,
                "pos_bucket": "UNK",
                "is_out": 0,
                "minutes_from_stints_prior_20": 10.0,
                "starter_flag": 0,
                "injury_row_present": False,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            # Team 200 (bench pool == 0 after scaling)
            {
                "game_id": 1,
                "team_id": 200,
                "player_id": 8,
                "pos_bucket": "BIG",
                "is_out": 0,
                "minutes_from_stints_prior_20": 20.0,
                "starter_flag": 1,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 200,
                "player_id": 9,
                "pos_bucket": "G",
                "is_out": 0,
                "minutes_from_stints_prior_20": 18.0,
                "starter_flag": 1,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 200,
                "player_id": 10,
                "pos_bucket": "W",
                "is_out": 0,
                "minutes_from_stints_prior_20": 16.0,
                "starter_flag": 1,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 200,
                "player_id": 11,
                "pos_bucket": "W",
                "is_out": 0,
                "minutes_from_stints_prior_20": 14.0,
                "starter_flag": 1,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 200,
                "player_id": 12,
                "pos_bucket": "G",
                "is_out": 0,
                "minutes_from_stints_prior_20": 12.0,
                "starter_flag": 1,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 200,
                "player_id": 13,
                "pos_bucket": "BIG",
                "is_out": 0,
                "minutes_from_stints_prior_20": 0.0,
                "starter_flag": 0,
                "injury_row_present": True,
                "status": "AVAIL",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.97,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
            {
                "game_id": 1,
                "team_id": 200,
                "player_id": 14,
                "pos_bucket": "W",
                "is_out": 1,
                "minutes_from_stints_prior_20": 0.0,
                "starter_flag": 0,
                "injury_row_present": True,
                "status": "OUT",
                "injury_as_of_ts": "2026-01-24T10:00:00Z",
                "injury_snapshot_missing": 0,
                "prior_play_prob": 0.0,
                "vac_min_szn": 0.0,
                "vac_min_guard_szn": 0.0,
                "vac_min_wing_szn": 0.0,
                "vac_min_big_szn": 0.0,
            },
        ]
    )

    out = prepare_live_features_for_rmh(df)

    expected_new_cols = {
        "started_proxy",
        "has_injury_row",
        "team_n_players",
        "team_n_not_out",
        "available_B_not_out",
        "available_G_not_out",
        "available_W_not_out",
        "depth_same_pos_not_out",
        "vacated_minutes_prior_20_total",
        "vacated_minutes_prior_20_same_pos",
        "team_prior_minutes_20_not_out",
        "prior_minutes_share_20",
        "rotation_team_missing",
        "rotation_missing",
        "rotation_player_row_missing_raw",
        "rotation_player_filled_zero",
        "depth_6",
        "depth_10",
        "depth_14",
        "effective_n",
        "starter_pool_minutes",
        "bench_pool_minutes",
        "bench_conc_top1",
        "bench_conc_top2",
        "vac_missing",
    }
    assert expected_new_cols.issubset(out.columns)

    # Injury semantics normalization (injury_row_present == False).
    row_unk = out.loc[out["player_id"] == 7].iloc[0]
    assert row_unk["status"] == "Ava"
    assert pd.isna(row_unk["injury_as_of_ts"])
    assert int(row_unk["injury_snapshot_missing"]) == 1
    assert pd.isna(row_unk["prior_play_prob"])
    assert int(row_unk["has_injury_row"]) == 0

    # Team 100 derived counts and vacancy math.
    team_100 = out.loc[out["team_id"] == 100].copy()
    assert int(team_100["team_n_players"].iloc[0]) == 7
    assert int(team_100["team_n_not_out"].iloc[0]) == 6
    assert int(team_100["available_B_not_out"].iloc[0]) == 1
    assert int(team_100["available_G_not_out"].iloc[0]) == 2
    assert int(team_100["available_W_not_out"].iloc[0]) == 2

    p1 = team_100.loc[team_100["player_id"] == 1].iloc[0]
    p6 = team_100.loc[team_100["player_id"] == 6].iloc[0]
    p7 = team_100.loc[team_100["player_id"] == 7].iloc[0]
    assert int(p1["depth_same_pos_not_out"]) == 0  # only other BIGs
    assert int(p6["depth_same_pos_not_out"]) == 1  # BIG out -> don't subtract self
    assert int(p7["depth_same_pos_not_out"]) == 0  # UNK always 0

    assert np.isclose(float(p1["vacated_minutes_prior_20_total"]), 20.0)
    assert np.isclose(float(p1["vacated_minutes_prior_20_same_pos"]), 20.0)  # BIG
    assert np.isclose(float(p7["vacated_minutes_prior_20_same_pos"]), 20.0)  # UNK uses total
    assert np.isclose(float(p1["team_prior_minutes_20_not_out"]), 140.0)
    assert np.isclose(float(p1["prior_minutes_share_20"]), 30.0 / 140.0)
    assert np.isclose(float(p6["prior_minutes_share_20"]), 0.0)  # out rows forced to 0

    # Rotation flags approximate OUT as missing rotation row.
    assert int(p1["rotation_player_row_missing_raw"]) == 0
    assert int(p6["rotation_player_row_missing_raw"]) == 1

    # Team-shape features for team 100 use a 240-scaled minutes proxy.
    assert np.isclose(float(p1["depth_6"]), 6.0)
    assert np.isclose(float(p1["depth_10"]), 6.0)
    assert np.isclose(float(p1["depth_14"]), 6.0)
    assert np.isclose(float(p1["bench_conc_top1"]), 1.0)
    assert np.isclose(float(p1["bench_conc_top2"]), 1.0)

    # Team 200 bench pool is 0 -> bench concentration should be 0 (no divide-by-zero).
    team_200 = out.loc[out["team_id"] == 200].copy()
    p8 = team_200.loc[team_200["player_id"] == 8].iloc[0]
    assert np.isclose(float(p8["bench_pool_minutes"]), 0.0)
    assert np.isclose(float(p8["bench_conc_top1"]), 0.0)
    assert np.isclose(float(p8["bench_conc_top2"]), 0.0)

