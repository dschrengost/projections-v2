"""Tests for DNP history features.

These tests validate:
1. Point-in-time correctness (no future data leakage)
2. Correct computation of each feature
3. Per-(player_id, team_id) history reset on team change
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projections.features.dnp_history import (
    DNP_HISTORY_FEATURE_COLUMNS,
    DNPHistoryConfig,
    compute_dnp_history_features,
    compute_dnp_history_features_for_live,
    derive_roster_active_pre_tip,
)


def _make_synthetic_df() -> pd.DataFrame:
    """Create a synthetic dataset for testing.

    Timeline:
    - Player 1 on Team A: Games 1-5
      - Game 1: Active, played 20 mins
      - Game 2: Active, DNP (0 mins)
      - Game 3: OUT
      - Game 4: Active, DNP (0 mins)
      - Game 5: Active, played 15 mins (current game)

    - Player 1 on Team B (traded): Games 6-8
      - Game 6: Active, played 25 mins
      - Game 7: Active, DNP
      - Game 8: Active, played 10 mins (current game)

    - Player 2 on Team A: Games 1-5
      - Game 1: Active, played 30 mins
      - Game 2: Active, played 28 mins
      - Game 3: Active, played 25 mins
      - Game 4: Active, played 22 mins
      - Game 5: Active, played 20 mins (current game)
    """
    data = [
        # Player 1, Team A
        {"player_id": 1, "team_id": 100, "game_date": "2024-01-01", "is_out": 0, "minutes": 20.0},
        {"player_id": 1, "team_id": 100, "game_date": "2024-01-02", "is_out": 0, "minutes": 0.0},
        {"player_id": 1, "team_id": 100, "game_date": "2024-01-03", "is_out": 1, "minutes": 0.0},
        {"player_id": 1, "team_id": 100, "game_date": "2024-01-04", "is_out": 0, "minutes": 0.0},
        {"player_id": 1, "team_id": 100, "game_date": "2024-01-05", "is_out": 0, "minutes": 15.0},
        # Player 1, Team B (traded)
        {"player_id": 1, "team_id": 200, "game_date": "2024-01-10", "is_out": 0, "minutes": 25.0},
        {"player_id": 1, "team_id": 200, "game_date": "2024-01-11", "is_out": 0, "minutes": 0.0},
        {"player_id": 1, "team_id": 200, "game_date": "2024-01-12", "is_out": 0, "minutes": 10.0},
        # Player 2, Team A
        {"player_id": 2, "team_id": 100, "game_date": "2024-01-01", "is_out": 0, "minutes": 30.0},
        {"player_id": 2, "team_id": 100, "game_date": "2024-01-02", "is_out": 0, "minutes": 28.0},
        {"player_id": 2, "team_id": 100, "game_date": "2024-01-03", "is_out": 0, "minutes": 25.0},
        {"player_id": 2, "team_id": 100, "game_date": "2024-01-04", "is_out": 0, "minutes": 22.0},
        {"player_id": 2, "team_id": 100, "game_date": "2024-01-05", "is_out": 0, "minutes": 20.0},
    ]
    df = pd.DataFrame(data)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["tip_ts"] = df["game_date"] + pd.Timedelta(hours=19)  # 7pm tip
    return df


class TestDeriveRosterActivePretip:
    """Tests for derive_roster_active_pre_tip function."""

    def test_out_players_are_inactive(self):
        df = pd.DataFrame({"is_out": [0, 1, 0, 1]})
        result = derive_roster_active_pre_tip(df)
        assert list(result) == [1, 0, 1, 0]

    def test_missing_is_out_defaults_to_active(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = derive_roster_active_pre_tip(df)
        assert list(result) == [1, 1, 1]

    def test_nan_is_out_treated_as_active(self):
        df = pd.DataFrame({"is_out": [0, np.nan, 1, None]})
        result = derive_roster_active_pre_tip(df)
        assert list(result) == [1, 1, 0, 1]


class TestComputeDNPHistoryFeatures:
    """Tests for compute_dnp_history_features function."""

    def test_output_columns_present(self):
        df = _make_synthetic_df()
        result = compute_dnp_history_features(df, validate_pit=False)
        for col in DNP_HISTORY_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_first_game_has_no_history(self):
        """First game for a player-team should have no prior history."""
        df = _make_synthetic_df()
        result = compute_dnp_history_features(df, validate_pit=False)

        # Player 1, Team A, Game 1 (2024-01-01)
        first_game = result[
            (result["player_id"] == 1)
            & (result["team_id"] == 100)
            & (result["game_date"] == "2024-01-01")
        ].iloc[0]

        assert first_game["games_since_last_roster_active"] == 99  # Capped
        assert first_game["never_roster_active_before"] == 1
        assert first_game["consecutive_active_dnp"] == 0
        assert first_game["inactive_streak_len"] == 0

    def test_point_in_time_correctness(self):
        """Features should only use prior games, not current or future."""
        df = _make_synthetic_df()
        result = compute_dnp_history_features(df, validate_pit=False)

        # Player 1, Team A, Game 2 (2024-01-02)
        # Prior: Game 1 where they played (active, not DNP)
        game2 = result[
            (result["player_id"] == 1)
            & (result["team_id"] == 100)
            & (result["game_date"] == "2024-01-02")
        ].iloc[0]

        assert game2["games_since_last_roster_active"] == 1  # 1 game since last active
        assert game2["never_roster_active_before"] == 0
        assert game2["consecutive_active_dnp"] == 0  # Prior game was not DNP

    def test_consecutive_active_dnp_accumulates(self):
        """DNP streak should accumulate across consecutive active-but-DNP games."""
        df = _make_synthetic_df()
        result = compute_dnp_history_features(df, validate_pit=False)

        # Player 1, Team A, Game 5 (2024-01-05)
        # Prior: G1(played), G2(active-DNP), G3(OUT), G4(active-DNP)
        # Consecutive active DNP before G5 = 1 (only G4, since G3 was OUT)
        game5 = result[
            (result["player_id"] == 1)
            & (result["team_id"] == 100)
            & (result["game_date"] == "2024-01-05")
        ].iloc[0]

        assert game5["consecutive_active_dnp"] == 1  # Only G4

    def test_team_change_resets_history(self):
        """History should reset when player changes teams."""
        df = _make_synthetic_df()
        result = compute_dnp_history_features(df, validate_pit=False)

        # Player 1, Team B, Game 6 (2024-01-10) - first game on new team
        new_team_first = result[
            (result["player_id"] == 1)
            & (result["team_id"] == 200)
            & (result["game_date"] == "2024-01-10")
        ].iloc[0]

        assert new_team_first["games_since_last_roster_active"] == 99  # No prior on this team
        assert new_team_first["never_roster_active_before"] == 1

    def test_inactive_streak_counts_out_games(self):
        """inactive_streak_len should count consecutive OUT games."""
        df = _make_synthetic_df()
        result = compute_dnp_history_features(df, validate_pit=False)

        # Player 1, Team A, Game 4 (2024-01-04)
        # Prior: G1(active), G2(active), G3(OUT)
        # Inactive streak = 1 (only G3)
        game4 = result[
            (result["player_id"] == 1)
            & (result["team_id"] == 100)
            & (result["game_date"] == "2024-01-04")
        ].iloc[0]

        assert game4["inactive_streak_len"] == 1

    def test_dnp_rate_shrinkage_with_few_games(self):
        """DNP rate should be shrunk toward prior with few observations."""
        config = DNPHistoryConfig(alpha=1.0, beta=1.0, min_opportunities=3)
        df = _make_synthetic_df()
        result = compute_dnp_history_features(df, config=config, validate_pit=False)

        # Player 1, Team A, Game 3 (2024-01-03)
        # Prior active games: G1(played), G2(DNP) -> 2 active games, 1 DNP
        # Shrunk: (1 + 1) / (2 + 1 + 1) = 0.5
        game3 = result[
            (result["player_id"] == 1)
            & (result["team_id"] == 100)
            & (result["game_date"] == "2024-01-03")
        ].iloc[0]

        # With 2 active games (< min_opportunities=3), shrinkage is applied
        expected_rate = (1 + 1.0) / (2 + 1.0 + 1.0)  # = 0.5
        assert abs(game3["active_but_dnp_rate_last10"] - expected_rate) < 0.001

    def test_consistent_player_never_dnp(self):
        """Player 2 who always plays should have 0 DNP rate."""
        df = _make_synthetic_df()
        result = compute_dnp_history_features(df, validate_pit=False)

        # Player 2, Team A, Game 5 (2024-01-05)
        # Prior: G1-G4 all played (no DNPs)
        game5_p2 = result[
            (result["player_id"] == 2)
            & (result["team_id"] == 100)
            & (result["game_date"] == "2024-01-05")
        ].iloc[0]

        assert game5_p2["consecutive_active_dnp"] == 0
        # With 4 prior active games and 0 DNPs, raw rate = 0
        # Since 4 >= min_opportunities (3), no shrinkage
        assert game5_p2["active_but_dnp_rate_last10"] == 0.0


class TestPointInTimeViolationDetection:
    """Tests that would FAIL if future data was used."""

    def test_future_games_cannot_affect_features(self):
        """If we accidentally used future games, this test would fail.

        We create a scenario where using future data would give different results.
        """
        # Create a dataset where the pattern changes after a certain date
        data = [
            # Player always plays in first 3 games
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-01", "is_out": 0, "minutes": 20.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-02", "is_out": 0, "minutes": 20.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-03", "is_out": 0, "minutes": 20.0},
            # Then DNPs in games 4-5
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-04", "is_out": 0, "minutes": 0.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-05", "is_out": 0, "minutes": 0.0},
        ]
        df = pd.DataFrame(data)
        df["game_date"] = pd.to_datetime(df["game_date"])

        result = compute_dnp_history_features(df, validate_pit=False)

        # At Game 3 (2024-01-03), there should be NO DNPs in history
        # If future data was used, we'd see the DNPs from games 4-5
        game3 = result[result["game_date"] == "2024-01-03"].iloc[0]
        assert game3["consecutive_active_dnp"] == 0
        # With 2 prior active games (< min_opportunities=3), shrinkage is applied
        # (0 DNPs + 1) / (2 games + 1 + 1) = 0.25
        assert abs(game3["active_but_dnp_rate_last10"] - 0.25) < 0.001

        # At Game 4 (2024-01-04), there should still be NO DNPs in prior history
        game4 = result[result["game_date"] == "2024-01-04"].iloc[0]
        assert game4["consecutive_active_dnp"] == 0

        # At Game 5 (2024-01-05), there should be 1 prior DNP (Game 4)
        game5 = result[result["game_date"] == "2024-01-05"].iloc[0]
        assert game5["consecutive_active_dnp"] == 1


class TestLiveInference:
    """Tests for live inference variant."""

    def test_live_inference_basic(self):
        """Test live inference with separate current/historical DataFrames."""
        # Historical data
        historical = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-01", "is_out": 0, "minutes": 20.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-02", "is_out": 0, "minutes": 0.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-03", "is_out": 0, "minutes": 15.0},
        ])
        historical["game_date"] = pd.to_datetime(historical["game_date"])

        # Current game (no minutes yet)
        current = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-04", "is_out": 0},
        ])
        current["game_date"] = pd.to_datetime(current["game_date"])

        result = compute_dnp_history_features_for_live(
            current, historical
        )

        assert len(result) == 1
        row = result.iloc[0]

        # Prior history: G1(played), G2(DNP), G3(played)
        # Last active was G3, so games_since = 1
        assert row["games_since_last_roster_active"] == 1
        assert row["never_roster_active_before"] == 0
        # Consecutive DNP = 0 (G3 was not DNP)
        assert row["consecutive_active_dnp"] == 0
        # DNP rate: 1 DNP out of 3 active games
        # With min_opportunities=3, this is >= threshold, so raw rate
        assert abs(row["active_but_dnp_rate_last10"] - 1/3) < 0.001

    def test_live_inference_new_player(self):
        """Test live inference for player with no history."""
        historical = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-01", "is_out": 0, "minutes": 20.0},
        ])
        historical["game_date"] = pd.to_datetime(historical["game_date"])

        # New player not in historical data
        current = pd.DataFrame([
            {"player_id": 999, "team_id": 100, "game_date": "2024-01-02", "is_out": 0},
        ])
        current["game_date"] = pd.to_datetime(current["game_date"])

        result = compute_dnp_history_features_for_live(current, historical)

        row = result.iloc[0]
        assert row["games_since_last_roster_active"] == 99
        assert row["never_roster_active_before"] == 1
        assert row["consecutive_active_dnp"] == 0
        # Prior mean with alpha=1, beta=1
        assert abs(row["active_but_dnp_rate_last10"] - 0.5) < 0.001


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["player_id", "team_id", "game_date", "is_out", "minutes"])
        df["game_date"] = pd.to_datetime(df["game_date"])
        result = compute_dnp_history_features(df, validate_pit=False)
        assert len(result) == 0
        for col in DNP_HISTORY_FEATURE_COLUMNS:
            assert col in result.columns

    def test_single_row(self):
        df = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-01", "is_out": 0, "minutes": 20.0},
        ])
        df["game_date"] = pd.to_datetime(df["game_date"])
        result = compute_dnp_history_features(df, validate_pit=False)
        assert len(result) == 1
        # First game should have default values
        row = result.iloc[0]
        assert row["games_since_last_roster_active"] == 99
        assert row["never_roster_active_before"] == 1

    def test_all_out_player(self):
        """Player who is always OUT should have no roster_active history."""
        df = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-01", "is_out": 1, "minutes": 0.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-02", "is_out": 1, "minutes": 0.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-03", "is_out": 1, "minutes": 0.0},
        ])
        df["game_date"] = pd.to_datetime(df["game_date"])
        result = compute_dnp_history_features(df, validate_pit=False)

        for _, row in result.iterrows():
            assert row["roster_active_pre_tip"] == 0
            assert row["never_roster_active_before"] == 1
            assert row["consecutive_active_dnp"] == 0

        # Last row should have inactive_streak = 2 (prior 2 OUT games)
        last_row = result.iloc[-1]
        assert last_row["inactive_streak_len"] == 2

    def test_missing_minutes_column(self):
        """Should handle missing minutes column gracefully."""
        df = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-01", "is_out": 0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-02", "is_out": 0},
        ])
        df["game_date"] = pd.to_datetime(df["game_date"])
        # Should not raise, but DNP features will be based on NaN minutes
        result = compute_dnp_history_features(df, validate_pit=False, minutes_col="minutes")
        assert len(result) == 2


class TestInjurySnapshotMissing:
    """Tests for injury_snapshot_missing handling.

    These tests validate that players with missing injury snapshots and zero
    minutes are treated as inactive (not as active-DNP), preventing corrupted
    DNP history for players returning from injury.
    """

    def test_snapshot_missing_with_zero_minutes_treated_as_inactive(self):
        """When snapshot_missing=1 AND minutes=0, treat as inactive (not active-DNP)."""
        df = pd.DataFrame({
            "is_out": [0, 0, 0],
            "injury_snapshot_missing": [0, 1, 0],
            "minutes": [20.0, 0.0, 15.0],
        })
        result = derive_roster_active_pre_tip(
            df,
            is_out_col="is_out",
            injury_snapshot_missing_col="injury_snapshot_missing",
            minutes_col="minutes",
        )
        # Row 0: is_out=0, snapshot_ok => active
        # Row 1: is_out=0, snapshot_missing=1, minutes=0 => inactive
        # Row 2: is_out=0, snapshot_ok => active
        assert list(result) == [1, 0, 1]

    def test_snapshot_missing_with_positive_minutes_stays_active(self):
        """When snapshot_missing=1 BUT minutes>0, player was clearly active."""
        df = pd.DataFrame({
            "is_out": [0, 0],
            "injury_snapshot_missing": [1, 1],
            "minutes": [25.0, 10.0],
        })
        result = derive_roster_active_pre_tip(
            df,
            is_out_col="is_out",
            injury_snapshot_missing_col="injury_snapshot_missing",
            minutes_col="minutes",
        )
        # Both played minutes, so both are active despite missing snapshot
        assert list(result) == [1, 1]

    def test_snapshot_missing_without_minutes_col_treats_as_inactive(self):
        """When minutes_col not provided, snapshot_missing=1 => inactive."""
        df = pd.DataFrame({
            "is_out": [0, 0, 0],
            "injury_snapshot_missing": [0, 1, 1],
        })
        result = derive_roster_active_pre_tip(
            df,
            is_out_col="is_out",
            injury_snapshot_missing_col="injury_snapshot_missing",
            minutes_col=None,
        )
        # Row 0: snapshot_ok => active
        # Row 1, 2: snapshot_missing, no minutes to check => inactive
        assert list(result) == [1, 0, 0]

    def test_snapshot_missing_with_active_flag_recovers_active(self):
        """Explicit active_flag should recover active status for missing snapshots."""
        df = pd.DataFrame({
            "is_out": [0, 0, 0],
            "injury_snapshot_missing": [1, 1, 1],
            "minutes": [0.0, 0.0, 0.0],
            "active_flag": [True, False, True],
        })
        result = derive_roster_active_pre_tip(
            df,
            is_out_col="is_out",
            injury_snapshot_missing_col="injury_snapshot_missing",
            minutes_col="minutes",
        )
        assert list(result) == [1, 0, 1]

    def test_snapshot_missing_with_minutes_uses_no_row_ava_signal(self):
        """Historical rows with no injury row + Ava should be treated as active."""
        df = pd.DataFrame({
            "is_out": [0, 0],
            "injury_snapshot_missing": [1, 1],
            "injury_row_present": [False, True],
            "status": ["Ava", "Ava"],
            "minutes": [0.0, 0.0],
        })
        result = derive_roster_active_pre_tip(
            df,
            is_out_col="is_out",
            injury_snapshot_missing_col="injury_snapshot_missing",
            minutes_col="minutes",
        )
        # Row 0: no row + Ava => active
        # Row 1: row present => conservative inactive
        assert list(result) == [1, 0]

    def test_snapshot_missing_without_minutes_uses_no_row_ava_signal(self):
        """No injury row + Ava should be treated as active for current-game rows."""
        df = pd.DataFrame({
            "is_out": [0, 0, 0],
            "injury_snapshot_missing": [1, 1, 1],
            "injury_row_present": [False, True, False],
            "status": ["Ava", "Ava", "UNK"],
        })
        result = derive_roster_active_pre_tip(
            df,
            is_out_col="is_out",
            injury_snapshot_missing_col="injury_snapshot_missing",
            minutes_col=None,
        )
        # Row 0: no row + Ava => active
        # Row 1: row present => still conservative inactive
        # Row 2: no row but not Ava => conservative inactive
        assert list(result) == [1, 0, 0]

    def test_dnp_history_with_snapshot_missing_returns_from_injury(self):
        """Simulate player returning from injury with missing historical snapshots.

        This is the Hartenstein case: player was injured for several games,
        injury snapshots are missing (showing is_out=0), but they had 0 minutes.
        Without the fix, this would create a long consecutive_active_dnp streak.
        With the fix, those games are treated as inactive.
        """
        # Historical games: player "injured" with missing snapshots
        historical = pd.DataFrame([
            # Game 1: Played before injury
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-01",
             "is_out": 0, "injury_snapshot_missing": 0, "minutes": 30.0},
            # Games 2-5: Injured but snapshot missing (would wrongly show is_out=0)
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-02",
             "is_out": 0, "injury_snapshot_missing": 1, "minutes": 0.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-03",
             "is_out": 0, "injury_snapshot_missing": 1, "minutes": 0.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-04",
             "is_out": 0, "injury_snapshot_missing": 1, "minutes": 0.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-05",
             "is_out": 0, "injury_snapshot_missing": 1, "minutes": 0.0},
        ])
        historical["game_date"] = pd.to_datetime(historical["game_date"])

        # Current game: player returns (cleared, no longer on injury report)
        current = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-06",
             "is_out": 0, "injury_snapshot_missing": 1},
        ])
        current["game_date"] = pd.to_datetime(current["game_date"])

        result = compute_dnp_history_features_for_live(
            current,
            historical,
            injury_snapshot_missing_col="injury_snapshot_missing",
        )

        row = result.iloc[0]
        # With the fix: Games 2-5 are treated as inactive (not active-DNP)
        # So consecutive_active_dnp should be 0 (last active game was G1, played)
        assert row["consecutive_active_dnp"] == 0, (
            f"Expected 0 consecutive_active_dnp, got {row['consecutive_active_dnp']}. "
            "Games with missing snapshots and 0 minutes should be treated as inactive."
        )
        # inactive_streak_len should be 4 (games 2-5 were inactive)
        assert row["inactive_streak_len"] == 4
        # games_since_last_roster_active should be 5 (G1 was active, then 4 inactive + current)
        assert row["games_since_last_roster_active"] == 5

    def test_dnp_history_without_snapshot_missing_column(self):
        """Should fall back to old behavior when column doesn't exist."""
        historical = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-01",
             "is_out": 0, "minutes": 30.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-02",
             "is_out": 0, "minutes": 0.0},  # This would be active-DNP
        ])
        historical["game_date"] = pd.to_datetime(historical["game_date"])

        current = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-03", "is_out": 0},
        ])
        current["game_date"] = pd.to_datetime(current["game_date"])

        result = compute_dnp_history_features_for_live(
            current,
            historical,
            injury_snapshot_missing_col="injury_snapshot_missing",  # Not in data
        )

        row = result.iloc[0]
        # Without snapshot_missing column, G2 is treated as active-DNP
        assert row["consecutive_active_dnp"] == 1

    def test_live_history_with_active_flag_missing_snapshot_counts_active_dnp(self):
        """Historical active_flag should preserve active-DNP signal under missing snapshots."""
        historical = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-01", "is_out": 0, "injury_snapshot_missing": 1, "active_flag": True, "minutes": 0.0},
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-02", "is_out": 0, "injury_snapshot_missing": 1, "active_flag": True, "minutes": 0.0},
        ])
        historical["game_date"] = pd.to_datetime(historical["game_date"])

        current = pd.DataFrame([
            {"player_id": 1, "team_id": 100, "game_date": "2024-01-03", "is_out": 0, "injury_snapshot_missing": 1, "active_flag": True},
        ])
        current["game_date"] = pd.to_datetime(current["game_date"])

        result = compute_dnp_history_features_for_live(
            current,
            historical,
            injury_snapshot_missing_col="injury_snapshot_missing",
        )

        row = result.iloc[0]
        assert row["consecutive_active_dnp"] == 2
