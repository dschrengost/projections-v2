"""Unit tests for scale_shares_to_240 function.

Tests cover:
1. Sum-to-240 invariant
2. Cap redistribution
3. Empty-sum fallback behavior
"""

import numpy as np
import pandas as pd
import pytest

from projections.eval.minutes_alloc_abtest import (
    ScaleSharesDiagnostics,
    scale_shares_to_240,
)


def _make_team_df(
    shares: list[float],
    game_id: int = 1,
    team_id: int = 100,
    statuses: list[str] | None = None,
    play_probs: list[float] | None = None,
) -> pd.DataFrame:
    """Helper to create a single-team dataframe."""
    n = len(shares)
    df = pd.DataFrame({
        "game_id": [game_id] * n,
        "team_id": [team_id] * n,
        "player_id": list(range(1, n + 1)),
        "share": shares,
    })
    if statuses:
        df["status"] = statuses
    if play_probs:
        df["play_prob"] = play_probs
    return df


class TestScaleSharesToSum240:
    """Test that minutes sum to exactly 240 per team."""

    def test_basic_sum_to_240(self):
        """Basic case: all eligible players, shares sum to 240."""
        df = _make_team_df([0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.05, 0.02])
        out, diag = scale_shares_to_240(df, shares_col="share")

        team_sum = out["team_minutes_sum"].iloc[0]
        assert abs(team_sum - 240.0) < 1e-6, f"Team sum should be 240, got {team_sum}"
        assert diag.fallback_count == 0

    def test_sum_240_with_varying_shares(self):
        """Shares that don't sum to 1 should still produce 240 minutes."""
        df = _make_team_df([0.5, 0.3, 0.2, 0.1, 0.05])  # Sum = 1.15
        out, diag = scale_shares_to_240(df, shares_col="share")

        team_sum = out["team_minutes_sum"].iloc[0]
        assert abs(team_sum - 240.0) < 1e-6

    def test_multiple_teams_sum_240(self):
        """Each team should sum to 240 independently."""
        # Note: Each team needs at least 5 players to achieve 240 with cap_max=48
        df1 = _make_team_df([0.3, 0.25, 0.2, 0.15, 0.10], game_id=1, team_id=100)
        df2 = _make_team_df([0.4, 0.3, 0.15, 0.10, 0.05], game_id=1, team_id=200)  # 5 players
        df3 = _make_team_df([0.35, 0.30, 0.20, 0.10, 0.05], game_id=2, team_id=300)  # 5 players
        df = pd.concat([df1, df2, df3], ignore_index=True)

        out, diag = scale_shares_to_240(df, shares_col="share")

        for (gid, tid), g in out.groupby(["game_id", "team_id"]):
            team_sum = g["team_minutes_sum"].iloc[0]
            assert abs(team_sum - 240.0) < 1e-6, \
                f"Team {gid}/{tid} sum should be 240, got {team_sum}"


class TestCapRedistribution:
    """Test cap application and redistribution."""

    def test_cap_applied(self):
        """High share player should be capped at cap_max."""
        # One player with 80% share would get 192 min without cap
        df = _make_team_df([0.80, 0.10, 0.05, 0.03, 0.02])
        out, diag = scale_shares_to_240(df, shares_col="share", cap_max=48.0)

        max_minutes = out["minutes_mean_A"].max()
        assert max_minutes <= 48.0 + 1e-6, f"Max minutes should be ≤48, got {max_minutes}"

    def test_cap_redistribution_preserves_sum(self):
        """After capping, sum should still be 240."""
        df = _make_team_df([0.80, 0.10, 0.05, 0.03, 0.02])
        out, diag = scale_shares_to_240(df, shares_col="share", cap_max=48.0)

        team_sum = out["team_minutes_sum"].iloc[0]
        # With redistribution, sum should still be 240
        assert abs(team_sum - 240.0) < 1e-6, f"Sum after cap should be 240, got {team_sum}"

    def test_no_redistribution_option(self):
        """Without redistribution, sum may be less than 240 after cap."""
        df = _make_team_df([0.80, 0.10, 0.05, 0.03, 0.02])
        out, diag = scale_shares_to_240(
            df, shares_col="share", cap_max=48.0, redistribute_after_cap=False
        )

        max_minutes = out["minutes_mean_A"].max()
        assert max_minutes <= 48.0 + 1e-6

        # Without redistribution, team sum will be less than 240
        team_sum = out["minutes_mean_A"].sum()
        assert team_sum < 240.0  # 48 + 24 + 12 + ... = less than 240

    def test_multiple_capped_players(self):
        """Multiple players hitting cap should redistribute correctly."""
        # 8 players so we have room for redistribution with cap=42
        # With 8 players at cap=42, max possible = 336. Sum=240 is feasible.
        df = _make_team_df([0.35, 0.30, 0.15, 0.08, 0.05, 0.03, 0.02, 0.02])
        out, diag = scale_shares_to_240(df, shares_col="share", cap_max=42.0)

        max_minutes = out["minutes_mean_A"].max()
        assert max_minutes <= 42.0 + 1e-6

        team_sum = out["team_minutes_sum"].iloc[0]
        assert abs(team_sum - 240.0) < 1e-6


class TestFallbackBehavior:
    """Test fallback when share sum is zero."""

    def test_fallback_all_zero_shares(self):
        """Zero shares for all eligible should trigger fallback."""
        # Need 5+ players to achieve 240 with cap=48
        df = _make_team_df([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        out, diag = scale_shares_to_240(df, shares_col="share")

        assert diag.fallback_count == 1
        team_sum = out["team_minutes_sum"].iloc[0]
        assert abs(team_sum - 240.0) < 1e-6  # Still should sum to 240

    def test_fallback_uniform_distribution(self):
        """Zero shares should result in uniform distribution."""
        # 6 eligible players, each gets 40 minutes (cap=48 allows this)
        df = _make_team_df([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        out, diag = scale_shares_to_240(df, shares_col="share")

        # 6 eligible players, each should get 40 minutes
        expected_per_player = 240.0 / 6
        for m in out["minutes_mean_A"]:
            assert abs(m - expected_per_player) < 1e-6

    def test_fallback_with_out_players(self):
        """Fallback with some OUT players."""
        # 6 total, 5 active -> each active gets 48 min (at cap), sum = 240
        df = _make_team_df(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            statuses=["ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "OUT"],
        )
        out, diag = scale_shares_to_240(df, shares_col="share")

        # 5 eligible players, each gets 48 minutes (uniform fallback)
        eligible = out[out["eligible_flag"] == 1]
        assert len(eligible) == 5
        expected_per_player = 240.0 / 5  # = 48
        for m in eligible["minutes_mean_A"]:
            assert abs(m - expected_per_player) < 1e-6


class TestEligibilityGating:
    """Test eligibility based on status and play_prob."""

    def test_out_players_excluded(self):
        """OUT players should get 0 minutes."""
        # 7 players, 5 active -> enough for 240
        df = _make_team_df(
            [0.30, 0.25, 0.20, 0.15, 0.05, 0.03, 0.02],
            statuses=["ACTIVE", "OUT", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "OUT"],
        )
        out, diag = scale_shares_to_240(df, shares_col="share")

        # Check OUT players (indices 1 and 6)
        out_mask = df["status"] == "OUT"
        for _, row in out[out_mask].iterrows():
            assert row["minutes_mean_A"] == 0.0, "OUT players should have 0 minutes"

        # Check team sum still 240
        team_sum = out["team_minutes_sum"].iloc[0]
        assert abs(team_sum - 240.0) < 1e-6

    def test_zero_play_prob_excluded(self):
        """Players with play_prob=0 should get 0 minutes."""
        df = _make_team_df(
            [0.30, 0.25, 0.20, 0.15, 0.10],
            play_probs=[1.0, 0.0, 1.0, 0.5, 0.0],
        )
        out, diag = scale_shares_to_240(df, shares_col="share")

        # Check zero play_prob players
        zero_prob = df["play_prob"] == 0.0
        for _, row in out[zero_prob].iterrows():
            assert row["minutes_mean_A"] == 0.0

    def test_case_insensitive_status(self):
        """Status check should be case-insensitive."""
        df = _make_team_df(
            [0.30, 0.25, 0.20, 0.15, 0.10],
            statuses=["active", "out", "Out", "OUT", "Active"],
        )
        out, diag = scale_shares_to_240(df, shares_col="share")

        # Players 2, 3, 4 should be out (various casings of "out")
        out_indices = [1, 2, 3]  # 0-indexed
        for i in out_indices:
            assert out.iloc[i]["minutes_mean_A"] == 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_player_team(self):
        """Single player gets capped at 48 minutes (can't reach 240 alone)."""
        # Physical constraint: 1 player * 48 cap = 48 max, not 240
        df = _make_team_df([0.5])
        out, diag = scale_shares_to_240(df, shares_col="share")

        # With cap=48, single player can only get 48 max
        assert out["minutes_mean_A"].iloc[0] == 48.0
        # Team sum will be < 240 (physically impossible with 1 player and cap)
        assert out["team_minutes_sum"].iloc[0] == 48.0

    def test_missing_shares_column_raises(self):
        """Missing shares column should raise error."""
        df = _make_team_df([0.3, 0.2, 0.1])
        df = df.drop(columns=["share"])

        with pytest.raises(ValueError, match="Shares column.*not found"):
            scale_shares_to_240(df, shares_col="share")

    def test_negative_shares_clipped(self):
        """Negative shares should be clipped to 0."""
        # 6 eligible players (first has negative share, clipped to 0)
        df = _make_team_df([-0.1, 0.3, 0.2, 0.15, 0.10, 0.08])
        out, diag = scale_shares_to_240(df, shares_col="share")

        # First player should have 0 minutes (negative share clipped)
        assert out.iloc[0]["minutes_mean_A"] == 0.0
        # Team sum should still be 240 (5 positive-share players)
        team_sum = out["team_minutes_sum"].iloc[0]
        assert abs(team_sum - 240.0) < 1e-6

    def test_nan_shares_treated_as_zero(self):
        """NaN shares should be treated as 0."""
        df = _make_team_df([float("nan"), 0.3, 0.2, 0.1])
        out, diag = scale_shares_to_240(df, shares_col="share")

        # First player (NaN share) should have 0 minutes
        assert out.iloc[0]["minutes_mean_A"] == 0.0


class TestDiagnostics:
    """Test diagnostic output."""

    def test_diagnostics_populated(self):
        """Diagnostics should be populated with correct counts."""
        df1 = _make_team_df([0.3, 0.2, 0.1], game_id=1, team_id=100)
        df2 = _make_team_df([0.4, 0.3], game_id=1, team_id=200)
        df = pd.concat([df1, df2], ignore_index=True)

        out, diag = scale_shares_to_240(df, shares_col="share")

        assert diag.n_players == 5
        assert diag.n_teams == 2
        assert diag.n_eligible == 5
        assert isinstance(diag.team_sum_dev_max, float)

    def test_cap_count_tracked(self):
        """Cap application should be tracked in diagnostics."""
        df = _make_team_df([0.80, 0.10, 0.05, 0.05])
        out, diag = scale_shares_to_240(df, shares_col="share", cap_max=48.0)

        assert diag.cap_applied_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
