"""Tests for rotation_set_minutes derived features computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.rotation.rotation_set_minutes_features_v1 import (
    ROTATION_SET_DERIVED_FEATURES,
    add_rotation_set_derived_features,
)


def test_add_rotation_set_derived_features_computes_all_15_features() -> None:
    """Test that all 15 derived features are computed correctly."""
    df = pd.DataFrame(
        [
            # Team 100: 7 players, 1 out (BIG with 20 prior minutes)
            {"game_id": 1, "team_id": 100, "player_id": 1, "pos_bucket": "BIG", "is_out": 0, "minutes_from_stints_prior_20": 30.0},
            {"game_id": 1, "team_id": 100, "player_id": 2, "pos_bucket": "G", "is_out": 0, "minutes_from_stints_prior_20": 28.0},
            {"game_id": 1, "team_id": 100, "player_id": 3, "pos_bucket": "W", "is_out": 0, "minutes_from_stints_prior_20": 26.0},
            {"game_id": 1, "team_id": 100, "player_id": 4, "pos_bucket": "W", "is_out": 0, "minutes_from_stints_prior_20": 24.0},
            {"game_id": 1, "team_id": 100, "player_id": 5, "pos_bucket": "G", "is_out": 0, "minutes_from_stints_prior_20": 22.0},
            {"game_id": 1, "team_id": 100, "player_id": 6, "pos_bucket": "BIG", "is_out": 1, "minutes_from_stints_prior_20": 20.0},  # OUT
            {"game_id": 1, "team_id": 100, "player_id": 7, "pos_bucket": "UNK", "is_out": 0, "minutes_from_stints_prior_20": 10.0},
            # Team 200: 6 players, 1 out (W with 15 prior minutes)
            {"game_id": 1, "team_id": 200, "player_id": 8, "pos_bucket": "BIG", "is_out": 0, "minutes_from_stints_prior_20": 32.0},
            {"game_id": 1, "team_id": 200, "player_id": 9, "pos_bucket": "G", "is_out": 0, "minutes_from_stints_prior_20": 30.0},
            {"game_id": 1, "team_id": 200, "player_id": 10, "pos_bucket": "W", "is_out": 0, "minutes_from_stints_prior_20": 28.0},
            {"game_id": 1, "team_id": 200, "player_id": 11, "pos_bucket": "W", "is_out": 1, "minutes_from_stints_prior_20": 15.0},  # OUT
            {"game_id": 1, "team_id": 200, "player_id": 12, "pos_bucket": "G", "is_out": 0, "minutes_from_stints_prior_20": 25.0},
            {"game_id": 1, "team_id": 200, "player_id": 13, "pos_bucket": "BIG", "is_out": 0, "minutes_from_stints_prior_20": 18.0},
        ]
    )

    result = add_rotation_set_derived_features(df)

    # All 11 features should be present
    for col in ROTATION_SET_DERIVED_FEATURES:
        assert col in result.columns, f"Missing feature: {col}"

    # Team 100 checks
    team_100 = result[result["team_id"] == 100]
    assert int(team_100["team_n_players"].iloc[0]) == 7
    assert int(team_100["team_n_not_out"].iloc[0]) == 6  # 7 - 1 out
    assert int(team_100["available_G_not_out"].iloc[0]) == 2  # players 2, 5
    assert int(team_100["available_W_not_out"].iloc[0]) == 2  # players 3, 4
    assert int(team_100["available_B_not_out"].iloc[0]) == 1  # player 1 (6 is out)

    # Player 1 (BIG, not out): depth_same_pos_not_out = available_B_not_out - 1 = 0
    p1 = result[result["player_id"] == 1].iloc[0]
    assert int(p1["depth_same_pos_not_out"]) == 0

    # Player 6 (BIG, out): depth_same_pos_not_out = available_B_not_out (don't subtract self)
    p6 = result[result["player_id"] == 6].iloc[0]
    assert int(p6["depth_same_pos_not_out"]) == 1

    # Player 7 (UNK): depth_same_pos_not_out = 0 always
    p7 = result[result["player_id"] == 7].iloc[0]
    assert int(p7["depth_same_pos_not_out"]) == 0

    # Vacancy features for team 100: player 6 is out with 20 prior minutes
    assert np.isclose(float(p1["vacated_minutes_prior_20_total"]), 20.0)
    assert np.isclose(float(p1["vacated_minutes_prior_20_same_pos"]), 20.0)  # BIG out
    assert np.isclose(float(p7["vacated_minutes_prior_20_same_pos"]), 20.0)  # UNK uses total

    # Player 2 (G): vacated_minutes_prior_20_same_pos = 0 (no G out)
    p2 = result[result["player_id"] == 2].iloc[0]
    assert np.isclose(float(p2["vacated_minutes_prior_20_same_pos"]), 0.0)

    # Team prior minutes for not-out players: 30+28+26+24+22+10 = 140
    assert np.isclose(float(p1["team_prior_minutes_20_not_out"]), 140.0)

    # prior_minutes_share_20 for player 1: 30/140
    assert np.isclose(float(p1["prior_minutes_share_20"]), 30.0 / 140.0)

    # Player 6 (out): prior_minutes_share_20 = 0
    assert np.isclose(float(p6["prior_minutes_share_20"]), 0.0)

    # Team 200 checks
    team_200 = result[result["team_id"] == 200]
    assert int(team_200["team_n_players"].iloc[0]) == 6
    assert int(team_200["team_n_not_out"].iloc[0]) == 5  # 6 - 1 out

    # Vacancy for team 200: player 11 (W) is out with 15 prior minutes
    p10 = result[result["player_id"] == 10].iloc[0]
    assert np.isclose(float(p10["vacated_minutes_prior_20_total"]), 15.0)
    assert np.isclose(float(p10["vacated_minutes_prior_20_same_pos"]), 15.0)  # W out

    # Player 8 (BIG): vacated_minutes_prior_20_same_pos = 0 (no BIG out)
    p8 = result[result["player_id"] == 8].iloc[0]
    assert np.isclose(float(p8["vacated_minutes_prior_20_same_pos"]), 0.0)


def test_add_rotation_set_derived_features_with_vac_columns() -> None:
    """Test vac_missing flag when vacancy columns are present."""
    df = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 100, "player_id": 1, "pos_bucket": "G", "is_out": 0,
             "minutes_from_stints_prior_20": 20.0, "vac_min_szn": 5.0},
            {"game_id": 1, "team_id": 100, "player_id": 2, "pos_bucket": "W", "is_out": 0,
             "minutes_from_stints_prior_20": 18.0, "vac_min_szn": np.nan},
        ]
    )

    result = add_rotation_set_derived_features(df)

    # Player 1 has vac_min_szn not null -> vac_missing = 0
    assert int(result[result["player_id"] == 1]["vac_missing"].iloc[0]) == 0

    # Player 2 has vac_min_szn null -> vac_missing = 1
    assert int(result[result["player_id"] == 2]["vac_missing"].iloc[0]) == 1


def test_add_rotation_set_derived_features_without_vac_columns() -> None:
    """Test vac_missing = 1 when no vacancy columns are present."""
    df = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 100, "player_id": 1, "pos_bucket": "G", "is_out": 0,
             "minutes_from_stints_prior_20": 20.0},
        ]
    )

    result = add_rotation_set_derived_features(df)

    # No vac columns -> vac_missing = 1
    assert int(result["vac_missing"].iloc[0]) == 1


def test_add_rotation_set_derived_features_handles_status_column() -> None:
    """Test that 'status' column is used when is_out is not present."""
    df = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 100, "player_id": 1, "pos_bucket": "G", "status": "AVAIL",
             "minutes_from_stints_prior_20": 20.0},
            {"game_id": 1, "team_id": 100, "player_id": 2, "pos_bucket": "W", "status": "OUT",
             "minutes_from_stints_prior_20": 15.0},
        ]
    )

    result = add_rotation_set_derived_features(df)

    assert int(result["team_n_not_out"].iloc[0]) == 1  # Only player 1 is not out
    assert np.isclose(float(result["vacated_minutes_prior_20_total"].iloc[0]), 15.0)


def test_add_rotation_set_derived_features_handles_position_normalization() -> None:
    """Test position normalization from raw position strings."""
    df = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 100, "player_id": 1, "dk_pos": "PG", "is_out": 0,
             "minutes_from_stints_prior_20": 20.0},
            {"game_id": 1, "team_id": 100, "player_id": 2, "dk_pos": "SG", "is_out": 0,
             "minutes_from_stints_prior_20": 18.0},
            {"game_id": 1, "team_id": 100, "player_id": 3, "dk_pos": "SF", "is_out": 0,
             "minutes_from_stints_prior_20": 16.0},
            {"game_id": 1, "team_id": 100, "player_id": 4, "dk_pos": "PF", "is_out": 0,
             "minutes_from_stints_prior_20": 14.0},
            {"game_id": 1, "team_id": 100, "player_id": 5, "dk_pos": "C", "is_out": 0,
             "minutes_from_stints_prior_20": 12.0},
        ]
    )

    result = add_rotation_set_derived_features(df)

    # PG, SG -> G (2 players)
    assert int(result["available_G_not_out"].iloc[0]) == 2
    # SF, PF -> W (2 players)
    assert int(result["available_W_not_out"].iloc[0]) == 2
    # C -> B (1 player)
    assert int(result["available_B_not_out"].iloc[0]) == 1


def test_add_rotation_set_derived_features_selective_columns() -> None:
    """Test that only requested features are computed when feature_columns is provided."""
    df = pd.DataFrame(
        [
            {"game_id": 1, "team_id": 100, "player_id": 1, "pos_bucket": "G", "is_out": 0,
             "minutes_from_stints_prior_20": 20.0},
        ]
    )

    # Only request team_n_players and team_n_not_out
    result = add_rotation_set_derived_features(
        df,
        feature_columns=["team_n_players", "team_n_not_out", "some_other_col"],
    )

    assert "team_n_players" in result.columns
    assert "team_n_not_out" in result.columns
    # Other derived features should not be computed
    assert "available_G_not_out" not in result.columns
    assert "vacated_minutes_prior_20_total" not in result.columns


def test_add_rotation_set_derived_features_missing_keys_raises() -> None:
    """Test that missing key columns raises ValueError."""
    df = pd.DataFrame(
        [
            {"team_id": 100, "player_id": 1, "pos_bucket": "G", "is_out": 0,
             "minutes_from_stints_prior_20": 20.0},
        ]
    )

    with pytest.raises(ValueError, match="Missing required key columns"):
        add_rotation_set_derived_features(df)


def test_role_change_features_computed_correctly() -> None:
    """Test role-change features are correct differences of short vs long window priors."""
    df = pd.DataFrame(
        [
            # Player 1: bench player recently promoted to starter (next-man-up)
            {
                "game_id": 1, "team_id": 100, "player_id": 1, "pos_bucket": "B", "is_out": 0,
                "minutes_from_stints_prior_20": 8.0,
                "started_proxy_rate_prior_5": 0.4,
                "started_proxy_rate_prior_10": 0.1,
                "started_proxy_rate_prior_20": 0.05,
                "minutes_from_stints_prior_5": 22.0,
                "minutes_from_stints_prior_10": 12.0,
            },
            # Player 2: stable starter (no role change)
            {
                "game_id": 1, "team_id": 100, "player_id": 2, "pos_bucket": "G", "is_out": 0,
                "minutes_from_stints_prior_20": 32.0,
                "started_proxy_rate_prior_5": 1.0,
                "started_proxy_rate_prior_10": 1.0,
                "started_proxy_rate_prior_20": 0.95,
                "minutes_from_stints_prior_5": 33.0,
                "minutes_from_stints_prior_10": 32.5,
            },
            # Player 3: recently demoted (negative role change)
            {
                "game_id": 1, "team_id": 100, "player_id": 3, "pos_bucket": "W", "is_out": 0,
                "minutes_from_stints_prior_20": 28.0,
                "started_proxy_rate_prior_5": 0.0,
                "started_proxy_rate_prior_10": 0.4,
                "started_proxy_rate_prior_20": 0.7,
                "minutes_from_stints_prior_5": 10.0,
                "minutes_from_stints_prior_10": 18.0,
            },
        ]
    )

    result = add_rotation_set_derived_features(df)

    # Player 1: bench → starter (positive divergence)
    p1 = result[result["player_id"] == 1].iloc[0]
    assert np.isclose(float(p1["role_change_starter_5v20"]), 0.4 - 0.05)  # 0.35
    assert np.isclose(float(p1["role_change_minutes_5v20"]), 22.0 - 8.0)  # 14.0
    assert np.isclose(float(p1["role_change_starter_5v10"]), 0.4 - 0.1)   # 0.3
    assert np.isclose(float(p1["role_change_minutes_5v10"]), 22.0 - 12.0) # 10.0

    # Player 2: stable starter (near-zero divergence)
    p2 = result[result["player_id"] == 2].iloc[0]
    assert np.isclose(float(p2["role_change_starter_5v20"]), 1.0 - 0.95)  # 0.05
    assert np.isclose(float(p2["role_change_minutes_5v20"]), 33.0 - 32.0) # 1.0
    assert np.isclose(float(p2["role_change_starter_5v10"]), 1.0 - 1.0)   # 0.0
    assert np.isclose(float(p2["role_change_minutes_5v10"]), 33.0 - 32.5) # 0.5

    # Player 3: recently demoted (negative divergence)
    p3 = result[result["player_id"] == 3].iloc[0]
    assert np.isclose(float(p3["role_change_starter_5v20"]), 0.0 - 0.7)   # -0.7
    assert np.isclose(float(p3["role_change_minutes_5v20"]), 10.0 - 28.0) # -18.0
    assert np.isclose(float(p3["role_change_starter_5v10"]), 0.0 - 0.4)   # -0.4
    assert np.isclose(float(p3["role_change_minutes_5v10"]), 10.0 - 18.0) # -8.0


def test_role_change_features_zero_when_priors_missing() -> None:
    """Test role-change features gracefully fall back to 0 when prior columns are absent."""
    df = pd.DataFrame(
        [
            {
                "game_id": 1, "team_id": 100, "player_id": 1, "pos_bucket": "G", "is_out": 0,
                "minutes_from_stints_prior_20": 20.0,
                # No started_proxy_rate_prior_* or minutes_from_stints_prior_5/10
            },
        ]
    )

    result = add_rotation_set_derived_features(df)

    # Starter features: all source columns missing → 0 - 0 = 0
    assert np.isclose(float(result["role_change_starter_5v20"].iloc[0]), 0.0)
    assert np.isclose(float(result["role_change_starter_5v10"].iloc[0]), 0.0)

    # Minutes features: _prior_5 missing (→ 0) but _prior_20 present (20.0) → 0 - 20 = -20
    assert np.isclose(float(result["role_change_minutes_5v20"].iloc[0]), -20.0)
    # _prior_10 also missing → 0 - 0 = 0
    assert np.isclose(float(result["role_change_minutes_5v10"].iloc[0]), 0.0)


def test_context_priors_match_bucket_and_backoff_to_global() -> None:
    df = pd.DataFrame(
        [
            {
                "game_id": 1,
                "team_id": 100,
                "player_id": 1,
                "depth_same_pos_not_out": 0,
                "minutes_from_stints_prior_20": 20.0,
                "started_proxy_rate_prior_20": 0.2,
                "minutes_from_stints_ctx_same_pos_thin_prior_20": 32.0,
                "minutes_from_stints_ctx_same_pos_normal_prior_20": 18.0,
                "started_proxy_rate_ctx_same_pos_thin_prior_20": 1.0,
                "started_proxy_rate_ctx_same_pos_normal_prior_20": 0.4,
                "ctx_same_pos_thin_prior_n_games_20": 3,
                "ctx_same_pos_normal_prior_n_games_20": 5,
            },
            {
                "game_id": 1,
                "team_id": 100,
                "player_id": 2,
                "depth_same_pos_not_out": 2,
                "minutes_from_stints_prior_20": 14.0,
                "started_proxy_rate_prior_20": 0.1,
                "minutes_from_stints_ctx_same_pos_normal_prior_20": 25.0,
                "started_proxy_rate_ctx_same_pos_normal_prior_20": 0.8,
                "ctx_same_pos_normal_prior_n_games_20": 1,
            },
            {
                "game_id": 1,
                "team_id": 100,
                "player_id": 3,
                "depth_same_pos_not_out": 5,
                "minutes_from_stints_prior_20": 10.0,
                "started_proxy_rate_prior_20": 0.0,
            },
        ]
    )

    result = add_rotation_set_derived_features(
        df,
        feature_columns=[
            "ctx_minutes_from_stints_prior_20",
            "ctx_started_proxy_rate_prior_20",
            "ctx_prior_n_games_20",
            "ctx_prior_backoff_used_20",
        ],
    )

    p1 = result.loc[result["player_id"] == 1].iloc[0]
    assert np.isclose(float(p1["ctx_minutes_from_stints_prior_20"]), 32.0)
    assert np.isclose(float(p1["ctx_started_proxy_rate_prior_20"]), 1.0)
    assert int(p1["ctx_prior_n_games_20"]) == 3
    assert int(p1["ctx_prior_backoff_used_20"]) == 0

    p2 = result.loc[result["player_id"] == 2].iloc[0]
    assert np.isclose(float(p2["ctx_minutes_from_stints_prior_20"]), 14.0)
    assert np.isclose(float(p2["ctx_started_proxy_rate_prior_20"]), 0.1)
    assert int(p2["ctx_prior_n_games_20"]) == 1
    assert int(p2["ctx_prior_backoff_used_20"]) == 1

    p3 = result.loc[result["player_id"] == 3].iloc[0]
    assert np.isclose(float(p3["ctx_minutes_from_stints_prior_20"]), 10.0)
    assert np.isclose(float(p3["ctx_started_proxy_rate_prior_20"]), 0.0)
    assert int(p3["ctx_prior_n_games_20"]) == 0
    assert int(p3["ctx_prior_backoff_used_20"]) == 1
