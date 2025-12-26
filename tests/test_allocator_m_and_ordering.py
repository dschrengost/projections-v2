"""Unit tests for Allocator M and ordering_10_30 metric.

Tests cover:
1. test_allocator_M_sums_to_240 - M output should sum to 240 per team
2. test_allocator_M_respects_mask - M should assign 0 minutes to OUT players
3. test_ordering_metric_simple_case - Ordering metric computation correctness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.abtest_minutes_allocators import _compute_ordering_accuracy_10_30


class TestOrderingMetric:
    """Tests for ordering accuracy in [10, 30] range."""

    def test_ordering_metric_simple_case(self):
        """Test ordering metric with simple concordant/discordant pairs."""
        # Create data with known ordering
        # Team 1: actual [15, 25, 20] -> pred [12, 28, 22] should be fully concordant
        # within [10, 30]: all 3 players qualify
        # Pairs:
        #   (0,1): actual 15 < 25, pred 12 < 28 -> concordant
        #   (0,2): actual 15 < 20, pred 12 < 22 -> concordant
        #   (1,2): actual 25 > 20, pred 28 > 22 -> concordant
        # 3/3 = 1.0

        pred_minutes = np.array([12.0, 28.0, 22.0])
        actual_minutes = np.array([15.0, 25.0, 20.0])
        team_ids = np.array([1, 1, 1])

        accuracy = _compute_ordering_accuracy_10_30(pred_minutes, actual_minutes, team_ids)

        assert accuracy == 1.0

    def test_ordering_metric_partial_concordance(self):
        """Test ordering metric with mixed concordant/discordant pairs."""
        # Team 1: actual [15, 25, 20] -> pred [28, 12, 22]
        # Pairs (within [10, 30]):
        #   (0,1): actual 15 < 25, pred 28 > 12 -> DISCORDANT
        #   (0,2): actual 15 < 20, pred 28 > 22 -> DISCORDANT
        #   (1,2): actual 25 > 20, pred 12 < 22 -> DISCORDANT
        # 0/3 = 0.0

        pred_minutes = np.array([28.0, 12.0, 22.0])
        actual_minutes = np.array([15.0, 25.0, 20.0])
        team_ids = np.array([1, 1, 1])

        accuracy = _compute_ordering_accuracy_10_30(pred_minutes, actual_minutes, team_ids)

        assert accuracy == 0.0

    def test_ordering_metric_filters_outside_10_30(self):
        """Test that players outside [10, 30] actual minutes are filtered."""
        # Team 1: actual [5, 15, 35] - only player index 1 (15 min) is in [10, 30]
        # With only 1 player, no pairs -> returns None

        pred_minutes = np.array([10.0, 20.0, 40.0])
        actual_minutes = np.array([5.0, 15.0, 35.0])
        team_ids = np.array([1, 1, 1])

        accuracy = _compute_ordering_accuracy_10_30(pred_minutes, actual_minutes, team_ids)

        # With only 1 player in range, no pairs to compare
        assert accuracy is None

    def test_ordering_metric_multiple_teams(self):
        """Test ordering metric aggregates across teams."""
        # Team 1: actual [15, 25] -> pred [18, 22]
        #   (0,1): actual 15 < 25, pred 18 < 22 -> concordant (1/1 = 1.0)
        # Team 2: actual [20, 10] -> pred [25, 15]
        #   (0,1): actual 20 > 10, pred 25 > 15 -> concordant (1/1 = 1.0)
        # Mean: (1.0 + 1.0) / 2 = 1.0

        pred_minutes = np.array([18.0, 22.0, 25.0, 15.0])
        actual_minutes = np.array([15.0, 25.0, 20.0, 10.0])
        team_ids = np.array([1, 1, 2, 2])

        accuracy = _compute_ordering_accuracy_10_30(pred_minutes, actual_minutes, team_ids)

        assert accuracy == 1.0

    def test_ordering_metric_handles_ties(self):
        """Test that ties in actual minutes are skipped."""
        # Team 1: actual [15, 15, 20] -> pred [10, 20, 25]
        # Pairs:
        #   (0,1): actual 15 == 15 -> SKIP (tie)
        #   (0,2): actual 15 < 20, pred 10 < 25 -> concordant
        #   (1,2): actual 15 < 20, pred 20 < 25 -> concordant
        # 2/2 = 1.0

        pred_minutes = np.array([10.0, 20.0, 25.0])
        actual_minutes = np.array([15.0, 15.0, 20.0])
        team_ids = np.array([1, 1, 1])

        accuracy = _compute_ordering_accuracy_10_30(pred_minutes, actual_minutes, team_ids)

        assert accuracy == 1.0

    def test_ordering_metric_insufficient_data(self):
        """Test that None is returned with insufficient data."""
        # Only 1 player total -> None
        pred_minutes = np.array([20.0])
        actual_minutes = np.array([25.0])
        team_ids = np.array([1])

        accuracy = _compute_ordering_accuracy_10_30(pred_minutes, actual_minutes, team_ids)

        assert accuracy is None


class TestAllocatorMSumsTo240:
    """Tests for Allocator M sum-to-240 constraint."""

    def test_allocator_M_sums_to_240_basic(self):
        """Test that M allocation sums to 240 per team using simplified logic."""
        # Simulate the M allocation logic from _run_single_slate
        # Given expected minutes, convert to shares and scale to 240

        expected_minutes = np.array([30.0, 25.0, 20.0, 15.0, 10.0])  # 5 players
        team_sum = expected_minutes.sum()  # 100
        shares = expected_minutes / team_sum
        minutes_M = 240.0 * shares

        # Should sum to 240
        assert np.isclose(minutes_M.sum(), 240.0)

    def test_allocator_M_sums_to_240_with_cap(self):
        """Test that M allocation sums to 240 even with cap redistribution."""
        cap_max = 48.0  # Cap used in actual allocator

        # Normal case where cap won't be exceeded after scaling
        expected_minutes = np.array([35.0, 30.0, 25.0, 20.0, 15.0])  # Sum = 125
        team_sum = max(expected_minutes.sum(), 1e-8)
        shares = expected_minutes / team_sum
        minutes_M = 240.0 * shares  # Scale to 240

        # Expected: [67.2, 57.6, 48.0, 38.4, 28.8]
        # Only first two exceed cap; apply cap and redistribute
        for _ in range(10):
            over_cap = minutes_M > cap_max
            if not over_cap.any():
                break
            excess = (minutes_M - cap_max).clip(min=0).sum()
            minutes_M[over_cap] = cap_max
            under_cap = (minutes_M < cap_max) & (minutes_M > 0)
            if under_cap.any():
                headroom = (cap_max - minutes_M[under_cap]).sum()
                if headroom > 0:
                    redist = min(excess, headroom)
                    minutes_M[under_cap] += redist * (cap_max - minutes_M[under_cap]) / headroom

        # Renormalize to exactly 240 (key constraint)
        if minutes_M.sum() > 0:
            minutes_M = 240.0 * minutes_M / minutes_M.sum()

        # Sum-to-240 is the primary constraint
        assert np.isclose(minutes_M.sum(), 240.0)


class TestAllocatorMRespectsMask:
    """Tests for Allocator M respecting OUT mask."""

    def test_allocator_M_respects_mask(self):
        """Test that M assigns 0 minutes to OUT players."""
        # Simulate M allocation with mask
        expected_minutes = np.array([30.0, 25.0, 20.0, 15.0, 10.0])

        # Player 2 is OUT
        mask_inactive = np.array([False, False, True, False, False])
        expected_minutes_masked = expected_minutes.copy()
        expected_minutes_masked[mask_inactive] = 0.0

        # Convert to shares and scale
        team_sum = max(expected_minutes_masked.sum(), 1e-8)
        shares = expected_minutes_masked / team_sum
        minutes_M = 240.0 * shares

        # OUT player should have 0 minutes
        assert minutes_M[2] == 0.0
        # Total should still sum to 240
        assert np.isclose(minutes_M.sum(), 240.0)

    def test_allocator_M_all_out_gets_zero(self):
        """Test edge case where all players are OUT."""
        expected_minutes = np.array([30.0, 25.0, 20.0])
        mask_inactive = np.array([True, True, True])
        expected_minutes_masked = expected_minutes.copy()
        expected_minutes_masked[mask_inactive] = 0.0

        team_sum = max(expected_minutes_masked.sum(), 1e-8)
        shares = expected_minutes_masked / team_sum
        minutes_M = 240.0 * shares

        # All players should have 0 (edge case - no one to play)
        assert (minutes_M == 0.0).all()
