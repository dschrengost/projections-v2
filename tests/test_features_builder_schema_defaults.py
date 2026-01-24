"""Test that SharedFeaturesBuilder fills missing live-critical columns.

These tests verify that the feature builder's _ensure_schema_columns method
correctly fills missing columns with defaults to prevent scoring failures
when upstream data sources (odds, injuries, roster) are unavailable.

Related: REQUIRED_MINUTES_FEATURES in build_minutes_live.py
"""

from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from projections.builders.features_builder import SharedFeaturesBuilder, FeatureBuildConfig
from datetime import date
from pathlib import Path


class TestEnsureSchemaColumns:
    """Test the _ensure_schema_columns method fills missing columns."""

    @pytest.fixture
    def builder(self, tmp_path: Path) -> SharedFeaturesBuilder:
        """Create a SharedFeaturesBuilder with minimal config."""
        config = FeatureBuildConfig(
            data_root=tmp_path,
            season=2025,
            target_day=date(2025, 12, 15),
            as_of_ts=pd.Timestamp("2025-12-15T18:30:00Z"),
        )
        return SharedFeaturesBuilder(config)

    def test_fills_missing_starter_flag_no_signal(self, builder: SharedFeaturesBuilder) -> None:
        """Test that missing starter_flag defaults to 0 when no starter signal exists."""
        df = pd.DataFrame({
            "game_id": [22500100],
            "player_id": [101],
            "team_id": [1610612751],
        })

        result = builder._ensure_schema_columns(df)

        assert "starter_flag" in result.columns
        assert result["starter_flag"].iloc[0] == 0
        assert bool(result["starter_flag_missing"].iloc[0]) is True

    def test_derives_starter_flag_from_is_projected_starter(self, builder: SharedFeaturesBuilder) -> None:
        """Test that starter_flag is derived from is_projected_starter when present."""
        df = pd.DataFrame({
            "game_id": [22500100, 22500100],
            "player_id": [101, 102],
            "team_id": [1610612751, 1610612751],
            "is_projected_starter": [True, False],
        })

        result = builder._ensure_schema_columns(df)

        assert "starter_flag" in result.columns
        assert result["starter_flag"].iloc[0] == 1
        assert result["starter_flag"].iloc[1] == 0
        # starter_flag_missing should be False since we had a signal
        assert bool(result["starter_flag_missing"].iloc[0]) is False
        assert bool(result["starter_flag_missing"].iloc[1]) is False

    def test_derives_starter_flag_from_is_confirmed_starter(self, builder: SharedFeaturesBuilder) -> None:
        """Test that starter_flag is derived from is_confirmed_starter when present."""
        df = pd.DataFrame({
            "game_id": [22500100, 22500100],
            "player_id": [101, 102],
            "team_id": [1610612751, 1610612751],
            "is_confirmed_starter": [True, False],
        })

        result = builder._ensure_schema_columns(df)

        assert "starter_flag" in result.columns
        assert result["starter_flag"].iloc[0] == 1
        assert result["starter_flag"].iloc[1] == 0
        assert bool(result["starter_flag_missing"].iloc[0]) is False

    def test_derives_starter_flag_from_is_starter(self, builder: SharedFeaturesBuilder) -> None:
        """Test that starter_flag is derived from is_starter when present."""
        df = pd.DataFrame({
            "game_id": [22500100, 22500100],
            "player_id": [101, 102],
            "team_id": [1610612751, 1610612751],
            "is_starter": [1, 0],
        })

        result = builder._ensure_schema_columns(df)

        assert "starter_flag" in result.columns
        assert result["starter_flag"].iloc[0] == 1
        assert result["starter_flag"].iloc[1] == 0
        assert bool(result["starter_flag_missing"].iloc[0]) is False

    def test_starter_signal_priority_confirmed_over_projected(self, builder: SharedFeaturesBuilder) -> None:
        """Test that is_confirmed_starter takes priority over is_projected_starter."""
        df = pd.DataFrame({
            "game_id": [22500100],
            "player_id": [101],
            "team_id": [1610612751],
            "is_confirmed_starter": [True],
            "is_projected_starter": [False],
        })

        result = builder._ensure_schema_columns(df)

        # is_confirmed_starter should win
        assert result["starter_flag"].iloc[0] == 1
        assert bool(result["starter_flag_missing"].iloc[0]) is False

    def test_leaves_team_context_columns_as_nan(self, builder: SharedFeaturesBuilder) -> None:
        """Test that missing team context columns are left as NaN (not filled with 0.0)."""
        df = pd.DataFrame({
            "game_id": [22500100],
            "player_id": [101],
        })

        result = builder._ensure_schema_columns(df)

        team_ctx_cols = ["team_pace_szn", "team_off_rtg_szn", "team_def_rtg_szn"]
        for col in team_ctx_cols:
            assert col in result.columns, f"Missing column: {col}"
            assert pd.isna(result[col].iloc[0]), f"Expected NaN for {col}, got {result[col].iloc[0]}"
        assert bool(result["team_ctx_missing"].iloc[0]) is True

    def test_leaves_opp_context_columns_as_nan(self, builder: SharedFeaturesBuilder) -> None:
        """Test that missing opponent context columns are left as NaN (not filled with 0.0)."""
        df = pd.DataFrame({
            "game_id": [22500100],
            "player_id": [101],
        })

        result = builder._ensure_schema_columns(df)

        opp_ctx_cols = ["opp_pace_szn", "opp_def_rtg_szn"]
        for col in opp_ctx_cols:
            assert col in result.columns, f"Missing column: {col}"
            assert pd.isna(result[col].iloc[0]), f"Expected NaN for {col}, got {result[col].iloc[0]}"
        assert bool(result["opp_ctx_missing"].iloc[0]) is True

    def test_fills_missing_vacancy_columns_with_zero(self, builder: SharedFeaturesBuilder) -> None:
        """Test that missing vacancy columns are filled with 0.0."""
        df = pd.DataFrame({
            "game_id": [22500100],
            "player_id": [101],
        })

        result = builder._ensure_schema_columns(df)

        vac_cols = ["vac_min_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn"]
        for col in vac_cols:
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].iloc[0] == 0.0, f"Wrong default for {col}"
        assert bool(result["vac_ctx_missing"].iloc[0]) is True

    def test_all_ten_required_columns_present_after_fill(self, builder: SharedFeaturesBuilder) -> None:
        """Test that all 10 required live columns exist after processing."""
        # Simulate a DataFrame with NONE of the required columns
        df = pd.DataFrame({
            "game_id": [22500100, 22500101],
            "player_id": [101, 102],
            "team_id": [1610612751, 1610612744],
        })

        result = builder._ensure_schema_columns(df)

        required_columns = [
            "starter_flag",
            "team_pace_szn",
            "team_off_rtg_szn",
            "team_def_rtg_szn",
            "opp_pace_szn",
            "opp_def_rtg_szn",
            "vac_min_szn",
            "vac_min_guard_szn",
            "vac_min_wing_szn",
            "vac_min_big_szn",
        ]
        for col in required_columns:
            assert col in result.columns, f"Missing required column: {col}"

    def test_preserves_existing_columns(self, builder: SharedFeaturesBuilder) -> None:
        """Test that existing columns are not overwritten."""
        df = pd.DataFrame({
            "game_id": [22500100],
            "player_id": [101],
            "starter_flag": [1],
            "team_pace_szn": [102.5],
            "opp_def_rtg_szn": [108.3],
        })

        result = builder._ensure_schema_columns(df)

        # Existing values should be preserved
        assert result["starter_flag"].iloc[0] == 1
        assert result["team_pace_szn"].iloc[0] == 102.5
        assert result["opp_def_rtg_szn"].iloc[0] == 108.3
        # Missing flags should be False for preserved columns
        assert bool(result["starter_flag_missing"].iloc[0]) is False

    def test_missing_flags_are_false_when_columns_present(self, builder: SharedFeaturesBuilder) -> None:
        """Test that missing flags are False when all columns are already present."""
        df = pd.DataFrame({
            "game_id": [22500100],
            "player_id": [101],
            "team_id": [1610612751],
            # All required columns present
            "starter_flag": [1],
            "team_pace_szn": [100.0],
            "team_off_rtg_szn": [110.0],
            "team_def_rtg_szn": [105.0],
            "opp_pace_szn": [98.0],
            "opp_def_rtg_szn": [107.0],
            "vac_min_szn": [10.0],
            "vac_min_guard_szn": [5.0],
            "vac_min_wing_szn": [3.0],
            "vac_min_big_szn": [2.0],
        })

        result = builder._ensure_schema_columns(df)

        # All missing flags should be False
        assert bool(result["starter_flag_missing"].iloc[0]) is False
        assert bool(result["team_ctx_missing"].iloc[0]) is False
        assert bool(result["opp_ctx_missing"].iloc[0]) is False
        assert bool(result["vac_ctx_missing"].iloc[0]) is False

    def test_dtypes_are_correct_with_signals(self, builder: SharedFeaturesBuilder) -> None:
        """Test that filled columns have correct dtypes when starter signal available."""
        df = pd.DataFrame({
            "game_id": [22500100],
            "player_id": [101],
            "is_projected_starter": [True],  # Provide starter signal
            "team_pace_szn": [100.0],  # Provide ctx to avoid NaN dtype issues
            "team_off_rtg_szn": [110.0],
            "team_def_rtg_szn": [105.0],
            "opp_pace_szn": [98.0],
            "opp_def_rtg_szn": [107.0],
        })

        result = builder._ensure_schema_columns(df)

        # starter_flag should be int-like
        assert result["starter_flag"].dtype in (int, "int64", "int32", np.int64, np.int32)

        # Float columns should be float (only vac columns are auto-filled as 0.0)
        vac_cols = ["vac_min_szn", "vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn"]
        for col in vac_cols:
            assert result[col].dtype == float, f"Expected float for {col}, got {result[col].dtype}"

        # Boolean flags
        flag_cols = ["starter_flag_missing", "team_ctx_missing", "opp_ctx_missing", "vac_ctx_missing"]
        for col in flag_cols:
            assert result[col].dtype == bool, f"Expected bool for {col}, got {result[col].dtype}"

    def test_handles_nan_in_starter_signals(self, builder: SharedFeaturesBuilder) -> None:
        """Test that NaN values in starter signals are handled correctly (filled as 0)."""
        df = pd.DataFrame({
            "game_id": [22500100, 22500100],
            "player_id": [101, 102],
            "team_id": [1610612751, 1610612751],
            "is_projected_starter": [True, pd.NA],
        })

        result = builder._ensure_schema_columns(df)

        assert result["starter_flag"].iloc[0] == 1
        assert result["starter_flag"].iloc[1] == 0  # NaN becomes 0
        assert bool(result["starter_flag_missing"].iloc[0]) is False

    def test_derives_starter_flag_when_5_per_team_game(self, builder: SharedFeaturesBuilder) -> None:
        """Test that starter_flag is populated from is_projected_starter with 5 per team-game."""
        # Simulate 2 teams with 10 players each, 5 starters per team
        team1_id, team2_id = 1610612751, 1610612744
        game_id = 22500100

        df = pd.DataFrame({
            "game_id": [game_id] * 20,
            "player_id": list(range(101, 121)),
            "team_id": [team1_id] * 10 + [team2_id] * 10,
            # 5 starters per team
            "is_projected_starter": [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5,
        })

        result = builder._ensure_schema_columns(df)

        # Should have exactly 10 starters (5 per team)
        assert result["starter_flag"].sum() == 10
        # Per-team starter counts
        team1_starters = result[result["team_id"] == team1_id]["starter_flag"].sum()
        team2_starters = result[result["team_id"] == team2_id]["starter_flag"].sum()
        assert team1_starters == 5
        assert team2_starters == 5
        assert bool(result["starter_flag_missing"].iloc[0]) is False

    def test_corrects_all_zeros_starter_flag(self, builder: SharedFeaturesBuilder) -> None:
        """Test that starter_flag is corrected when all zeros but is_projected_starter has positives."""
        df = pd.DataFrame({
            "game_id": [22500100] * 10,
            "player_id": list(range(101, 111)),
            "team_id": [1610612751] * 10,
            "starter_flag": [0] * 10,  # All zeros - clearly wrong
            "is_projected_starter": [True] * 5 + [False] * 5,  # Has 5 starters
        })

        result = builder._ensure_schema_columns(df)

        # starter_flag should now match is_projected_starter
        assert result["starter_flag"].sum() == 5
        assert list(result["starter_flag"].iloc[:5]) == [1, 1, 1, 1, 1]
        assert list(result["starter_flag"].iloc[5:]) == [0, 0, 0, 0, 0]
        assert bool(result["starter_flag_missing"].iloc[0]) is False

    def test_preserves_nonzero_starter_flag(self, builder: SharedFeaturesBuilder) -> None:
        """Test that starter_flag with non-zero mass is preserved (not overridden)."""
        df = pd.DataFrame({
            "game_id": [22500100] * 10,
            "player_id": list(range(101, 111)),
            "team_id": [1610612751] * 10,
            "starter_flag": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 3 starters
            "is_projected_starter": [True] * 5 + [False] * 5,  # Different: 5 starters
        })

        result = builder._ensure_schema_columns(df)

        # Original starter_flag should be preserved since it has non-zero mass
        assert result["starter_flag"].sum() == 3
        assert list(result["starter_flag"].iloc[:3]) == [1, 1, 1]
        assert bool(result["starter_flag_missing"].iloc[0]) is False

    def test_or_combines_multiple_starter_signals(self, builder: SharedFeaturesBuilder) -> None:
        """Test that multiple starter signals are OR'd together."""
        df = pd.DataFrame({
            "game_id": [22500100] * 4,
            "player_id": [101, 102, 103, 104],
            "team_id": [1610612751] * 4,
            # Different signals are true for different players
            "is_confirmed_starter": [True, False, False, False],
            "is_projected_starter": [False, True, False, False],
            "is_starter": [0, 0, 1, 0],
        })

        result = builder._ensure_schema_columns(df)

        # OR of all signals: 3 players should be starters
        assert result["starter_flag"].sum() == 3
        assert result["starter_flag"].iloc[0] == 1  # from is_confirmed_starter
        assert result["starter_flag"].iloc[1] == 1  # from is_projected_starter
        assert result["starter_flag"].iloc[2] == 1  # from is_starter
        assert result["starter_flag"].iloc[3] == 0  # none
        assert bool(result["starter_flag_missing"].iloc[0]) is False

    def test_fills_lineup_semantics_from_starters_and_out(self, builder: SharedFeaturesBuilder) -> None:
        """Fill lineup_role/status/roster_status when missing.

        Training expects:
        - lineup_role in {bench, confirmed_starter, projected_starter, out}
        - lineup_status: Expected only for projected_starter, otherwise Confirmed
        - lineup_roster_status: Inactive only for out, otherwise Active
        """
        df = pd.DataFrame(
            {
                "game_id": [22500100] * 4,
                "team_id": [1610612751] * 4,
                "player_id": [101, 102, 103, 104],
                "active_flag": [True, True, True, False],
                "is_out": [0, 0, 0, 0],
                "is_confirmed_starter": [True, False, False, False],
                "is_projected_starter": [False, True, False, False],
                # Leave lineup_* missing on purpose.
                "lineup_role": [pd.NA, pd.NA, pd.NA, pd.NA],
                "lineup_status": [pd.NA, pd.NA, pd.NA, pd.NA],
                "lineup_roster_status": [pd.NA, pd.NA, pd.NA, pd.NA],
            }
        )

        result = builder._ensure_schema_columns(df)

        by_pid = result.set_index("player_id")

        assert str(by_pid.loc[101, "lineup_role"]) == "confirmed_starter"
        assert str(by_pid.loc[101, "lineup_status"]) == "Confirmed"
        assert str(by_pid.loc[101, "lineup_roster_status"]) == "Active"

        assert str(by_pid.loc[102, "lineup_role"]) == "projected_starter"
        assert str(by_pid.loc[102, "lineup_status"]) == "Expected"
        assert str(by_pid.loc[102, "lineup_roster_status"]) == "Active"

        assert str(by_pid.loc[103, "lineup_role"]) == "bench"
        assert str(by_pid.loc[103, "lineup_status"]) == "Confirmed"
        assert str(by_pid.loc[103, "lineup_roster_status"]) == "Active"

        # inactive roster -> out role
        assert str(by_pid.loc[104, "lineup_role"]) == "out"
        assert str(by_pid.loc[104, "lineup_status"]) == "Confirmed"
        assert str(by_pid.loc[104, "lineup_roster_status"]) == "Inactive"
