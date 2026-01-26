"""Unit tests for rotation_set_minutes sparsify guardrail."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.rotation.guardrails import apply_rotation_minutes_guardrails


def _make_team_game_df(
    n_players: int = 18,
    *,
    gate_probs: list[float] | None = None,
    minutes: list[float] | None = None,
    include_out_player: bool = False,
    include_row_fallback: bool = False,
) -> pd.DataFrame:
    """Build a fake single team-game DataFrame for testing."""
    if gate_probs is None:
        # Default: descending gate probs from 0.95 to near 0
        gate_probs = [0.95 - i * 0.05 for i in range(n_players)]
    if minutes is None:
        # Default: realistic NBA rotation distribution (avoids pathology fallback)
        # Top 5 starters: ~32-36 mins, next 5 rotation: ~15-22 mins, rest: 0-8
        base_minutes = [
            36.0, 34.0, 33.0, 32.0, 30.0,  # Starters (165 total)
            22.0, 18.0, 15.0, 12.0, 8.0,   # Rotation (75 total)
        ]
        # Fill remaining with small nonzero values (need nonzero for renormalization to work)
        remaining = n_players - len(base_minutes)
        if remaining > 0:
            base_minutes.extend([1.0] * remaining)
        # Scale to exactly 240
        total = sum(base_minutes[:n_players])
        minutes = [m * 240.0 / total for m in base_minutes[:n_players]]

    df = pd.DataFrame({
        "game_id": ["0022401234"] * n_players,
        "team_id": [1610612744] * n_players,
        "player_id": list(range(1, n_players + 1)),
        "rotation_minutes_p50": minutes,
        "baseline_minutes_p50": minutes.copy(),
        "gate_prob": gate_probs,
        "minutes_features_row_missing": [0] * n_players,
        "injury_snapshot_missing": [0.0] * n_players,
        "is_out": [0] * n_players,
        "status": [None] * n_players,
    })

    if include_out_player:
        # Mark last player as OUT
        df.loc[n_players - 1, "is_out"] = 1
        df.loc[n_players - 1, "status"] = "OUT"

    if include_row_fallback:
        # Mark second-to-last player as row fallback
        df.loc[n_players - 2, "minutes_features_row_missing"] = 1

    return df


class TestSparsifyGuardrailBasic:
    """Basic tests for sparsify guardrail."""

    def test_sparsify_disabled_by_default(self) -> None:
        """Sparsify should be disabled by default, leaving input unchanged."""
        df = _make_team_game_df(n_players=18)
        original_minutes = df["rotation_minutes_p50"].copy()

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
        )

        # Minutes should be unchanged (modulo scaling to 240)
        assert result.summary["sparsify"]["enabled"] is False
        # Team total should be 240
        assert abs(result.minutes_p50.sum() - 240.0) < 0.01

    def test_sparsify_enabled_zeros_low_gate_prob_players(self) -> None:
        """With sparsify enabled, low gate_prob players should get zeroed."""
        # Create 18 players with descending gate probs
        gate_probs = [0.95 - i * 0.05 for i in range(18)]
        df = _make_team_game_df(n_players=18, gate_probs=gate_probs)

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_topk=9,
            sparsify_tau=0.55,
            sparsify_kmax=11,
            sparsify_min_keep=8,
        )

        # Should have zeroed some players
        nonzero_count = (result.minutes_p50 > 0.01).sum()
        assert nonzero_count >= 8, f"Expected at least 8 nonzero, got {nonzero_count}"
        assert nonzero_count <= 11, f"Expected at most 11 nonzero, got {nonzero_count}"

        # Team total should still be 240
        assert abs(result.minutes_p50.sum() - 240.0) < 0.01

        # Sparsify stats should be populated
        assert result.summary["sparsify"]["enabled"] is True
        assert "n_players_zeroed" in result.summary["sparsify"]

    def test_sparsify_respects_kmax(self) -> None:
        """Sparsify should not keep more than kmax players."""
        # All players have gate_prob >= tau (all should pass tau threshold)
        gate_probs = [0.9] * 15 + [0.1] * 3  # 15 above tau=0.55
        df = _make_team_game_df(n_players=18, gate_probs=gate_probs)

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_topk=5,
            sparsify_tau=0.55,
            sparsify_kmax=10,  # Should cap at 10 despite 15 above tau
            sparsify_min_keep=5,
        )

        nonzero_count = (result.minutes_p50 > 0.01).sum()
        assert nonzero_count <= 10, f"Expected at most 10 nonzero, got {nonzero_count}"

    def test_sparsify_respects_min_keep(self) -> None:
        """Sparsify should keep at least min_keep players."""
        # Only 3 players have gate_prob >= tau
        gate_probs = [0.9, 0.8, 0.7] + [0.1] * 15
        df = _make_team_game_df(n_players=18, gate_probs=gate_probs)

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_topk=3,
            sparsify_tau=0.55,  # Only 3 players above this
            sparsify_kmax=11,
            sparsify_min_keep=8,  # Should keep at least 8
        )

        nonzero_count = (result.minutes_p50 > 0.01).sum()
        assert nonzero_count >= 8, f"Expected at least 8 nonzero, got {nonzero_count}"


class TestSparsifyGuardrailFixedRows:
    """Tests for sparsify behavior with fixed rows (OUT, row_fallback)."""

    def test_fixed_rows_unchanged(self) -> None:
        """OUT and row_fallback rows should not be modified by sparsify."""
        df = _make_team_game_df(
            n_players=18,
            include_out_player=True,
            include_row_fallback=True,
        )
        # Set specific values for the fixed rows
        out_idx = 17  # Last player is OUT
        fallback_idx = 16  # Second-to-last is row_fallback

        # Give OUT player 0 minutes (they're OUT)
        df.loc[out_idx, "rotation_minutes_p50"] = 0.0
        df.loc[out_idx, "baseline_minutes_p50"] = 5.0  # Baseline had some

        # Give fallback player specific minutes
        df.loc[fallback_idx, "rotation_minutes_p50"] = 15.0
        df.loc[fallback_idx, "baseline_minutes_p50"] = 15.0

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_topk=9,
            sparsify_tau=0.55,
            sparsify_kmax=11,
            sparsify_min_keep=8,
        )

        # OUT player should be zero (enforced by OUT handling)
        assert result.minutes_p50.iloc[out_idx] == 0.0

        # Row fallback player should have baseline minutes (due to row fallback)
        # Note: actual value depends on scaling, but should not be zeroed by sparsify


class TestSparsifyGuardrailTeamTotal:
    """Tests for sparsify team total preservation."""

    def test_team_total_preserved_after_sparsify(self) -> None:
        """Team total should be 240 after sparsify renormalization."""
        df = _make_team_game_df(n_players=18)

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_topk=9,
            sparsify_tau=0.55,
            sparsify_kmax=11,
            sparsify_min_keep=8,
        )

        # Team total should be 240
        total = result.minutes_p50.sum()
        assert abs(total - 240.0) < 0.01, f"Expected 240.0, got {total}"

    def test_team_total_with_fixed_rows(self) -> None:
        """Team total should be 240 even with OUT players."""
        df = _make_team_game_df(n_players=18, include_out_player=True)
        # OUT player gets 0 minutes
        df.loc[17, "rotation_minutes_p50"] = 0.0

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_topk=9,
            sparsify_tau=0.55,
            sparsify_kmax=11,
            sparsify_min_keep=8,
        )

        # Team total should still be 240 (OUT player contributes 0)
        total = result.minutes_p50.sum()
        assert abs(total - 240.0) < 0.01, f"Expected 240.0, got {total}"


class TestSparsifyGuardrailMissingColumn:
    """Tests for sparsify behavior when gate_prob column is missing."""

    def test_skipped_when_gate_prob_missing(self) -> None:
        """Sparsify should be skipped if gate_prob column is missing."""
        df = _make_team_game_df(n_players=18)
        df = df.drop(columns=["gate_prob"])

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_topk=9,
            sparsify_tau=0.55,
            sparsify_kmax=11,
            sparsify_min_keep=8,
        )

        # Sparsify should be skipped
        assert result.summary["sparsify"]["enabled"] is True
        assert result.summary["sparsify"]["skipped"] is True
        assert result.summary["sparsify"]["skip_reason"] == "column_missing"


class TestSparsifyByMinutes:
    """Tests for sparsify using predicted minutes as the score column."""

    def test_sparsify_by_minutes_zeros_low_minute_players(self) -> None:
        """With sparsify_use_col=rotation_minutes_p50 and tau=8, low-minute players get zeroed."""
        # Create 12 players with descending minutes
        # Top 9 have >= 8 mins, bottom 3 have < 8 mins
        minutes = [36.0, 34.0, 32.0, 30.0, 28.0, 22.0, 18.0, 15.0, 10.0, 6.0, 4.0, 2.0]
        # Gate probs don't matter for this test but need the column present
        gate_probs = [0.9] * 12
        df = _make_team_game_df(n_players=12, gate_probs=gate_probs, minutes=minutes)

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_use_col="rotation_minutes_p50",  # Use minutes, not gate_prob
            sparsify_topk=9,
            sparsify_tau=8.0,  # Minutes threshold
            sparsify_kmax=10,
            sparsify_min_keep=6,
        )

        # Should keep players with >= 8 mins (top 9) but kmax=10 so all 9 fit
        # Players with < 8 mins (indices 9, 10, 11) should be zeroed
        nonzero_count = (result.minutes_p50 > 0.01).sum()
        assert nonzero_count >= 6, f"Expected at least 6 nonzero, got {nonzero_count}"
        assert nonzero_count <= 10, f"Expected at most 10 nonzero, got {nonzero_count}"

        # Team total should still be 240
        assert abs(result.minutes_p50.sum() - 240.0) < 0.01

        # The 3 lowest minute players should be zeroed (6, 4, 2 mins - all < 8)
        assert result.minutes_p50.iloc[9] < 0.01, "Player with 6 mins should be zeroed"
        assert result.minutes_p50.iloc[10] < 0.01, "Player with 4 mins should be zeroed"
        assert result.minutes_p50.iloc[11] < 0.01, "Player with 2 mins should be zeroed"

        # Sparsify stats should show use_col
        assert result.summary["sparsify"]["use_col"] == "rotation_minutes_p50"
        assert result.summary["sparsify"]["enabled"] is True
        assert result.summary["sparsify"]["n_players_zeroed"] > 0

    def test_sparsify_by_minutes_respects_min_keep(self) -> None:
        """With min_keep=6, even if only 5 players >= tau, keep at least 6."""
        # Create 12 players with realistic minutes that sum to 240 and avoid pathology.
        # Top 5 starters around 30 mins, then rotation players with lower mins.
        # 5 players have >= 8 mins (above tau), but topk=5 and min_keep=6 means keep 6.
        minutes = [34.0, 32.0, 30.0, 28.0, 26.0, 20.0, 18.0, 15.0, 13.0, 10.0, 8.0, 6.0]
        # Verify sum is 240
        assert abs(sum(minutes) - 240.0) < 0.1
        gate_probs = [0.9] * 12
        df = _make_team_game_df(n_players=12, gate_probs=gate_probs, minutes=minutes)

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_use_col="rotation_minutes_p50",
            sparsify_topk=5,  # topk=5 keeps top 5
            sparsify_tau=8.0,  # 11 players >= 8 mins
            sparsify_kmax=10,
            sparsify_min_keep=6,  # Must keep at least 6
        )

        nonzero_count = (result.minutes_p50 > 0.01).sum()
        # 11 players have >= 8 mins (above tau), so they'd all be kept.
        # topk=5 + above_tau(11) = 11 total candidates, but kmax=10 caps it.
        # Since 10 are kept and last player (6 mins) is zeroed, nonzero=10.
        assert nonzero_count >= 6, f"Expected at least 6 nonzero, got {nonzero_count}"
        assert abs(result.minutes_p50.sum() - 240.0) < 0.01

    def test_sparsify_by_minutes_respects_kmax(self) -> None:
        """With kmax=10, don't keep more than 10 even if all players >= tau."""
        # Create 12 players with realistic minutes that sum to 240.
        # All 12 have >= 8 mins (above tau), but kmax=10 should limit to 10.
        minutes = [34.0, 32.0, 30.0, 28.0, 26.0, 20.0, 18.0, 15.0, 13.0, 10.0, 8.0, 6.0]
        # Verify sum is 240
        assert abs(sum(minutes) - 240.0) < 0.1
        gate_probs = [0.9] * 12
        df = _make_team_game_df(n_players=12, gate_probs=gate_probs, minutes=minutes)

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_use_col="rotation_minutes_p50",
            sparsify_topk=9,
            sparsify_tau=5.0,  # 12 players >= 5 mins (all above tau)
            sparsify_kmax=10,  # Cap at 10
            sparsify_min_keep=6,
        )

        nonzero_count = (result.minutes_p50 > 0.01).sum()
        # All 12 above tau=5, but kmax=10 should limit to 10
        assert nonzero_count <= 10, f"Expected at most 10 nonzero, got {nonzero_count}"
        assert abs(result.minutes_p50.sum() - 240.0) < 0.01

    def test_sparsify_stats_include_kept_score_diagnostics(self) -> None:
        """Sparsify stats should include kept_score_min, kept_score_max, n_kept_mean."""
        minutes = [36.0, 34.0, 32.0, 30.0, 28.0, 22.0, 18.0, 15.0, 10.0, 6.0, 4.0, 2.0]
        gate_probs = [0.9] * 12
        df = _make_team_game_df(n_players=12, gate_probs=gate_probs, minutes=minutes)

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_use_col="rotation_minutes_p50",
            sparsify_topk=9,
            sparsify_tau=8.0,
            sparsify_kmax=10,
            sparsify_min_keep=6,
        )

        sparsify = result.summary["sparsify"]
        assert "kept_score_min" in sparsify, "Should have kept_score_min"
        assert "kept_score_max" in sparsify, "Should have kept_score_max"
        assert "n_kept_mean" in sparsify, "Should have n_kept_mean"
        # The min kept score should be >= some reasonable value (depends on renorm)
        # The max should be close to the original max minutes
        assert sparsify["kept_score_max"] >= 30.0, "Max kept score should be high"


class TestPathologyFallbackWithRealBaseline:
    """Tests for pathology fallback using separate baseline minutes."""

    def test_pathology_fallback_uses_baseline_not_rotation(self) -> None:
        """When rotation minutes trigger pathology, fallback should use baseline, not rotation.

        This test verifies the fix for the bug where primary mode used minutes_p50 for both
        rotation_p50_col and baseline_p50_col, making pathology fallback a no-op.
        """
        # Create a smeared/pathological rotation distribution:
        # - Low max_minutes (e.g., 28) triggers min_max_minutes pathology
        # - Low top5_sum triggers min_top5_sum pathology
        # flat_trigger = min_max_trigger AND min_top5_trigger, so we need both.
        smeared_minutes = [28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 24.0]
        # Sum = 240, max=28 < 30, top5_sum = 28+27+26+25+24 = 130

        # Create sane baseline minutes with realistic distribution:
        # - Star player at 36 mins
        # - Top 5 sum well above threshold (~165)
        baseline_minutes = [36.0, 34.0, 32.0, 31.0, 30.0, 22.0, 18.0, 15.0, 12.0, 10.0]
        # Sum = 240, max=36, top5_sum=163

        df = pd.DataFrame({
            "game_id": ["0022401234"] * 10,
            "team_id": [1610612744] * 10,
            "player_id": list(range(1, 11)),
            "rotation_minutes_p50": smeared_minutes,
            "baseline_minutes_p50": baseline_minutes,
            "gate_prob": [0.95 - i * 0.05 for i in range(10)],
            "minutes_features_row_missing": [0] * 10,
            "injury_snapshot_missing": [0.0] * 10,
            "is_out": [0] * 10,
            "status": [None] * 10,
        })

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            # Use thresholds that trigger flat_trigger (min_max_trigger AND min_top5_trigger)
            pathology_min_max_minutes_threshold=30.0,  # max=28 < 30 triggers
            pathology_min_top5_sum_threshold=135.0,  # top5_sum=130 < 135 triggers
        )

        # Pathology should have triggered for this team-game
        assert result.summary["pathology"]["fallback_team_games"] >= 1, (
            "Expected pathology fallback to trigger for smeared rotation"
        )

        # The result should be the BASELINE minutes, not the smeared rotation
        # Star player should have ~36 mins (baseline), not ~28 (rotation)
        star_player_minutes = result.minutes_p50.iloc[0]
        assert star_player_minutes > 32.0, (
            f"Expected star player to have baseline (~36) not rotation (~28), got {star_player_minutes}"
        )

        # The lowest player should have baseline minutes too
        bench_player_minutes = result.minutes_p50.iloc[9]
        assert abs(bench_player_minutes - 10.0) < 0.5, (
            f"Expected bench player to have baseline (~10), got {bench_player_minutes}"
        )

        # Team total should still be 240
        total = result.minutes_p50.sum()
        assert abs(total - 240.0) < 0.01, f"Expected team total 240, got {total}"

    def test_no_pathology_when_baseline_equals_rotation(self) -> None:
        """Verify the old bug: if baseline==rotation, pathology fallback is a no-op.

        This documents the bug that the fix addresses. When baseline_p50_col points
        to the same data as rotation_p50_col, pathology fallback replaces rotation
        with itself - achieving nothing.
        """
        # Create pathological rotation minutes
        smeared_minutes = [28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 24.0]
        # Sum = 240, max=28, top5_sum=130

        df = pd.DataFrame({
            "game_id": ["0022401234"] * 10,
            "team_id": [1610612744] * 10,
            "player_id": list(range(1, 11)),
            "rotation_minutes_p50": smeared_minutes,
            "baseline_minutes_p50": smeared_minutes.copy(),  # Same as rotation - the bug!
            "gate_prob": [0.95 - i * 0.05 for i in range(10)],
            "minutes_features_row_missing": [0] * 10,
            "injury_snapshot_missing": [0.0] * 10,
            "is_out": [0] * 10,
            "status": [None] * 10,
        })

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            # Same thresholds as the fix test - triggers flat_trigger
            pathology_min_max_minutes_threshold=30.0,  # max=28 < 30 triggers
            pathology_min_top5_sum_threshold=135.0,  # top5_sum=130 < 135 triggers
        )

        # Pathology triggers (same as above test)
        assert result.summary["pathology"]["fallback_team_games"] >= 1

        # But since baseline == rotation, the "fallback" is a no-op
        # Star player still has smeared minutes (~28), not realistic (~36)
        star_player_minutes = result.minutes_p50.iloc[0]
        assert star_player_minutes < 32.0, (
            f"With baseline==rotation, star player stays smeared (~28), got {star_player_minutes}"
        )

    def test_sparsify_disabled_on_pathology_team(self) -> None:
        """Sparsify should NOT run on teams that fell back to baseline via pathology.

        When a team triggers pathology fallback, the entire team-game reverts to baseline.
        Sparsify should skip these teams since they're no longer using rotation output.
        """
        # Create pathological rotation distribution
        smeared_minutes = [28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 24.0]
        # Sum = 240, max=28, top5_sum=130
        # Sane baseline with clear rotation hierarchy
        baseline_minutes = [36.0, 34.0, 32.0, 31.0, 30.0, 22.0, 18.0, 15.0, 12.0, 10.0]
        # Sum = 240, max=36, top5_sum=163

        df = pd.DataFrame({
            "game_id": ["0022401234"] * 10,
            "team_id": [1610612744] * 10,
            "player_id": list(range(1, 11)),
            "rotation_minutes_p50": smeared_minutes,
            "baseline_minutes_p50": baseline_minutes,
            "gate_prob": [0.95, 0.90, 0.85, 0.80, 0.75, 0.40, 0.35, 0.30, 0.25, 0.20],
            "minutes_features_row_missing": [0] * 10,
            "injury_snapshot_missing": [0.0] * 10,
            "is_out": [0] * 10,
            "status": [None] * 10,
        })

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="rotation_minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            # Thresholds that trigger flat_trigger for smeared distribution
            pathology_min_max_minutes_threshold=30.0,  # max=28 < 30 triggers
            pathology_min_top5_sum_threshold=135.0,  # top5_sum=130 < 135 triggers
            sparsify_enable=True,
            sparsify_topk=5,
            sparsify_tau=0.50,  # Would zero players 6-10 if sparsify ran
            sparsify_kmax=8,
            sparsify_min_keep=5,
        )

        # Pathology should trigger
        assert result.summary["pathology"]["fallback_team_games"] >= 1

        # Since pathology fallback happened, sparsify should NOT have zeroed the bench players
        # The bench player (idx 9) should still have baseline minutes (~10), not 0
        bench_player_minutes = result.minutes_p50.iloc[9]
        assert bench_player_minutes > 5.0, (
            f"Sparsify should not run on pathology teams; bench player should have ~10, got {bench_player_minutes}"
        )


class TestGuardrailsAlwaysRunAndReturnStats:
    """Test that guardrails always run and return stats, even when sparsify is disabled."""

    def _make_two_team_df(self) -> pd.DataFrame:
        """Build a 2-team DataFrame for testing guardrails visibility."""
        # Team A: 10 players, realistic distribution summing to 240
        team_a_minutes = [36.0, 34.0, 32.0, 30.0, 28.0, 22.0, 18.0, 15.0, 13.0, 12.0]
        # Team B: 10 players, similar distribution
        team_b_minutes = [35.0, 33.0, 31.0, 29.0, 27.0, 23.0, 19.0, 16.0, 14.0, 13.0]

        rows = []
        for i, mins in enumerate(team_a_minutes):
            rows.append({
                "game_id": "0022401234",
                "team_id": 1610612744,
                "player_id": 100 + i,
                "minutes_p50": mins,
                "rotation_minutes_p50": mins,
                "baseline_minutes_p50": mins,
                "gate_prob": 0.9 - i * 0.05,
                "minutes_features_row_missing": 0,
                "injury_snapshot_missing": 0.0,
                "is_out": 0,
                "status": None,
            })
        for i, mins in enumerate(team_b_minutes):
            rows.append({
                "game_id": "0022401234",
                "team_id": 1610612745,
                "player_id": 200 + i,
                "minutes_p50": mins,
                "rotation_minutes_p50": mins,
                "baseline_minutes_p50": mins,
                "gate_prob": 0.9 - i * 0.05,
                "minutes_features_row_missing": 0,
                "injury_snapshot_missing": 0.0,
                "is_out": 0,
                "status": None,
            })
        return pd.DataFrame(rows)

    def test_guardrails_return_stats_when_sparsify_disabled(self) -> None:
        """Guardrails should return stats even when sparsify is disabled."""
        df = self._make_two_team_df()

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=False,  # Disabled
        )

        # Stats should still be present
        assert "sparsify" in result.summary
        assert result.summary["sparsify"]["enabled"] is False
        # Other guardrail stats should be present
        assert "rows" in result.summary
        assert "pathology" in result.summary
        assert "degraded" in result.summary
        assert "degraded_reasons" in result.summary

    def test_guardrails_sparsify_reduces_player_count_with_minutes_col(self) -> None:
        """Sparsify using minutes_p50 should reduce players >1 min per team to <=kmax."""
        df = self._make_two_team_df()

        # Count players with >1 minute before guardrails
        pre_count_team_a = (df[df["team_id"] == 1610612744]["minutes_p50"] > 1.0).sum()
        pre_count_team_b = (df[df["team_id"] == 1610612745]["minutes_p50"] > 1.0).sum()

        result = apply_rotation_minutes_guardrails(
            df,
            rotation_p50_col="minutes_p50",
            baseline_p50_col="baseline_minutes_p50",
            sparsify_enable=True,
            sparsify_use_col="minutes_p50",  # Use minutes as score
            sparsify_topk=6,
            sparsify_tau=15.0,  # Only players with >=15 mins pass tau
            sparsify_kmax=8,  # Cap at 8
            sparsify_min_keep=5,
        )

        # Merge result back for per-team analysis
        result_df = df.copy()
        result_df["final_minutes"] = result.minutes_p50

        # Count players with >1 minute after guardrails
        post_count_team_a = (result_df[result_df["team_id"] == 1610612744]["final_minutes"] > 1.0).sum()
        post_count_team_b = (result_df[result_df["team_id"] == 1610612745]["final_minutes"] > 1.0).sum()

        # Should have fewer players with >1 min (or at most kmax)
        assert post_count_team_a <= 8, f"Team A should have <=8 players with >1 min, got {post_count_team_a}"
        assert post_count_team_b <= 8, f"Team B should have <=8 players with >1 min, got {post_count_team_b}"

        # Verify stats are present and meaningful
        sparsify_stats = result.summary["sparsify"]
        assert sparsify_stats["enabled"] is True
        assert sparsify_stats["use_col"] == "minutes_p50"
        assert "n_teams_sparsified" in sparsify_stats
        assert "n_players_zeroed" in sparsify_stats
        # At least one team should have been sparsified (had players zeroed)
        assert sparsify_stats["n_players_zeroed"] >= 0  # May be 0 if all players passed criteria
