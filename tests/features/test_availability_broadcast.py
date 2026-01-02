"""Tests for availability features - injury_as_of_ts broadcasting."""

from __future__ import annotations

import pandas as pd
import pytest

from projections.features.availability import attach_availability_features
from projections.minutes_v1.constants import AvailabilityStatus


class TestInjuryAsOfTsBroadcast:
    """Regression tests for injury_as_of_ts per-game broadcasting."""

    def test_injury_as_of_ts_broadcast_to_all_rows(self):
        """injury_as_of_ts should be set for ALL rows in a game, not just matched players."""
        # Base features with 3 players in game 1001
        base_df = pd.DataFrame({
            "game_id": [1001, 1001, 1001],
            "player_id": [101, 102, 103],
            "player_name": ["Alice", "Bob", "Charlie"],
            "tip_ts": pd.Timestamp("2025-01-02 19:00:00", tz="UTC"),
        })

        # Injury snapshot only has 1 player (Bob=102 is OUT)
        injuries_snapshot = pd.DataFrame({
            "game_id": [1001],
            "player_id": [102],
            "status": ["OUT"],
            "restriction_flag": [False],
            "ramp_flag": [False],
            "games_since_return": [0],
            "days_since_return": [0],
            "as_of_ts": pd.Timestamp("2025-01-02 17:30:00", tz="UTC"),
        })

        result = attach_availability_features(base_df, injuries_snapshot)

        # All 3 rows should have the same injury_as_of_ts
        assert result["injury_as_of_ts"].notna().all(), \
            f"injury_as_of_ts should be set for ALL rows, got: {result['injury_as_of_ts'].tolist()}"
        
        # Should have exactly 1 unique value per game
        unique_ts = result["injury_as_of_ts"].dropna().unique()
        assert len(unique_ts) == 1, f"Expected 1 unique injury_as_of_ts, got {len(unique_ts)}"
        
        # injury_snapshot_missing should be 0 for all rows since snapshot exists for game
        assert (result["injury_snapshot_missing"] == 0).all(), \
            f"injury_snapshot_missing should be 0 when snapshot exists, got: {result['injury_snapshot_missing'].tolist()}"

    def test_injury_row_present_indicator(self):
        """injury_row_present should be True only for players with injury records."""
        base_df = pd.DataFrame({
            "game_id": [1001, 1001, 1001],
            "player_id": [101, 102, 103],
            "player_name": ["Alice", "Bob", "Charlie"],
            "tip_ts": pd.Timestamp("2025-01-02 19:00:00", tz="UTC"),
        })

        injuries_snapshot = pd.DataFrame({
            "game_id": [1001],
            "player_id": [102],  # Only Bob has injury record
            "status": ["OUT"],
            "restriction_flag": [False],
            "ramp_flag": [False],
            "games_since_return": [0],
            "days_since_return": [0],
            "as_of_ts": pd.Timestamp("2025-01-02 17:30:00", tz="UTC"),
        })

        result = attach_availability_features(base_df, injuries_snapshot)

        # Only player 102 (Bob) should have injury_row_present=True
        bob_row = result[result["player_id"] == 102].iloc[0]
        alice_row = result[result["player_id"] == 101].iloc[0]
        charlie_row = result[result["player_id"] == 103].iloc[0]

        assert bob_row["injury_row_present"] is True or bob_row["injury_row_present"] == True
        assert alice_row["injury_row_present"] is False or alice_row["injury_row_present"] == False
        assert charlie_row["injury_row_present"] is False or charlie_row["injury_row_present"] == False

    def test_is_out_only_for_matched_player(self):
        """is_out should only be 1 for the player with OUT status, not broadcast."""
        base_df = pd.DataFrame({
            "game_id": [1001, 1001, 1001],
            "player_id": [101, 102, 103],
            "player_name": ["Alice", "Bob", "Charlie"],
            "tip_ts": pd.Timestamp("2025-01-02 19:00:00", tz="UTC"),
        })

        injuries_snapshot = pd.DataFrame({
            "game_id": [1001],
            "player_id": [102],
            "status": ["OUT"],
            "restriction_flag": [False],
            "ramp_flag": [False],
            "games_since_return": [0],
            "days_since_return": [0],
            "as_of_ts": pd.Timestamp("2025-01-02 17:30:00", tz="UTC"),
        })

        result = attach_availability_features(base_df, injuries_snapshot)

        # Only Bob (102) should be is_out=1
        assert result[result["player_id"] == 102]["is_out"].iloc[0] == 1
        assert result[result["player_id"] == 101]["is_out"].iloc[0] == 0
        assert result[result["player_id"] == 103]["is_out"].iloc[0] == 0

    def test_empty_injuries_sets_snapshot_missing(self):
        """When no injuries exist, injury_snapshot_missing should be 1."""
        base_df = pd.DataFrame({
            "game_id": [1001, 1001],
            "player_id": [101, 102],
            "player_name": ["Alice", "Bob"],
        })

        result = attach_availability_features(base_df, injuries_snapshot=None)

        assert (result["injury_snapshot_missing"] == 1).all()
        assert result["injury_as_of_ts"].isna().all()
        assert (result["injury_row_present"] == False).all()

    def test_multi_game_broadcast(self):
        """Each game should get its own injury_as_of_ts from its snapshot."""
        base_df = pd.DataFrame({
            "game_id": [1001, 1001, 1002, 1002],
            "player_id": [101, 102, 201, 202],
            "player_name": ["Alice", "Bob", "Carol", "Dave"],
            "tip_ts": [
                pd.Timestamp("2025-01-02 19:00:00", tz="UTC"),
                pd.Timestamp("2025-01-02 19:00:00", tz="UTC"),
                pd.Timestamp("2025-01-02 21:00:00", tz="UTC"),
                pd.Timestamp("2025-01-02 21:00:00", tz="UTC"),
            ],
        })

        injuries_snapshot = pd.DataFrame({
            "game_id": [1001, 1002],
            "player_id": [102, 201],
            "status": ["OUT", "QUESTIONABLE"],
            "restriction_flag": [False, False],
            "ramp_flag": [False, False],
            "games_since_return": [0, 0],
            "days_since_return": [0, 0],
            "as_of_ts": [
                pd.Timestamp("2025-01-02 17:30:00", tz="UTC"),
                pd.Timestamp("2025-01-02 19:30:00", tz="UTC"),
            ],
        })

        result = attach_availability_features(base_df, injuries_snapshot)

        # All 4 rows should have non-null injury_as_of_ts
        assert result["injury_as_of_ts"].notna().all()

        # Game 1001 rows should have same ts
        game1_ts = result[result["game_id"] == 1001]["injury_as_of_ts"].unique()
        assert len(game1_ts) == 1

        # Game 1002 rows should have same ts (but different from game 1001)
        game2_ts = result[result["game_id"] == 1002]["injury_as_of_ts"].unique()
        assert len(game2_ts) == 1

        # The timestamps should be different between games
        assert game1_ts[0] != game2_ts[0]
