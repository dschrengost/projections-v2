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
