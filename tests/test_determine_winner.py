"""Unit tests for determine_winner function in minutes_alloc_abtest_aggregate."""

import math

import pytest

from projections.eval.minutes_alloc_abtest_aggregate import determine_winner


class TestDetermineWinner:
    """Tests for the determine_winner function."""

    def test_lower_is_better_clear_winner(self):
        """A is clearly the best with lowest value."""
        result = determine_winner(
            {"A": 5.0, "B": 10.0, "C": 8.0},
            lower_is_better=True,
        )
        assert result == "A"

    def test_lower_is_better_c_wins(self):
        """C has the lowest value when lower is better."""
        result = determine_winner(
            {"A": 10.0, "B": 8.0, "C": 5.0},
            lower_is_better=True,
        )
        assert result == "C"

    def test_lower_is_better_b_wins(self):
        """B has the lowest value."""
        result = determine_winner(
            {"A": 10.0, "B": 3.0, "C": 8.0},
            lower_is_better=True,
        )
        assert result == "B"

    def test_higher_is_better_clear_winner(self):
        """A is clearly the best with highest value."""
        result = determine_winner(
            {"A": 200.0, "B": 150.0, "C": 180.0},
            lower_is_better=False,
        )
        assert result == "A"

    def test_higher_is_better_c_wins(self):
        """C has the highest value when higher is better."""
        result = determine_winner(
            {"A": 150.0, "B": 160.0, "C": 200.0},
            lower_is_better=False,
        )
        assert result == "C"

    def test_higher_is_better_b_wins(self):
        """B has the highest value."""
        result = determine_winner(
            {"A": 150.0, "B": 200.0, "C": 180.0},
            lower_is_better=False,
        )
        assert result == "B"

    def test_tie_within_tolerance(self):
        """Two values within tolerance should result in tie."""
        result = determine_winner(
            {"A": 5.0, "B": 5.0 + 1e-9, "C": 10.0},
            lower_is_better=True,
            tolerance=1e-6,
        )
        assert result == "–"

    def test_three_way_tie(self):
        """All three values equal should result in tie."""
        result = determine_winner(
            {"A": 5.0, "B": 5.0, "C": 5.0},
            lower_is_better=True,
        )
        assert result == "–"

    def test_all_nan(self):
        """All NaN values should return tie marker."""
        result = determine_winner(
            {"A": float("nan"), "B": float("nan"), "C": float("nan")},
            lower_is_better=True,
        )
        assert result == "–"

    def test_some_nan(self):
        """NaN values should be ignored, winner from valid values."""
        result = determine_winner(
            {"A": float("nan"), "B": 5.0, "C": 10.0},
            lower_is_better=True,
        )
        assert result == "B"

    def test_none_values(self):
        """None values should be ignored."""
        result = determine_winner(
            {"A": None, "B": 5.0, "C": 10.0},
            lower_is_better=True,
        )
        assert result == "B"

    def test_single_valid_value(self):
        """Single valid value should be the winner."""
        result = determine_winner(
            {"A": float("nan"), "B": 5.0, "C": None},
            lower_is_better=True,
        )
        assert result == "B"

    def test_empty_dict(self):
        """Empty dict should return tie marker."""
        result = determine_winner({}, lower_is_better=True)
        assert result == "–"

    def test_real_mae_scenario_c_wins(self):
        """Realistic MAE scenario where C should win (lowest MAE)."""
        # This is the bug scenario: A=6.5, B=7.0, C=6.2
        # C should win because 6.2 < 6.5 < 7.0
        result = determine_winner(
            {"A": 6.5, "B": 7.0, "C": 6.2},
            lower_is_better=True,
        )
        assert result == "C"

    def test_real_top5_scenario_c_wins(self):
        """Realistic top5 sum scenario where C should win (highest sum)."""
        # A=165, B=155, C=170 -> C wins (higher is better)
        result = determine_winner(
            {"A": 165.0, "B": 155.0, "C": 170.0},
            lower_is_better=False,
        )
        assert result == "C"

    def test_tolerance_edge_case(self):
        """Values just outside tolerance should have a winner."""
        result = determine_winner(
            {"A": 5.0, "B": 5.0 + 1e-4, "C": 10.0},
            lower_is_better=True,
            tolerance=1e-6,
        )
        # A is clearly lower than B now
        assert result == "A"

    def test_negative_values(self):
        """Should handle negative values correctly."""
        result = determine_winner(
            {"A": -10.0, "B": -5.0, "C": -8.0},
            lower_is_better=True,  # More negative = lower = better
        )
        assert result == "A"  # -10 is the lowest

    def test_negative_values_higher_is_better(self):
        """Should handle negative values correctly when higher is better."""
        result = determine_winner(
            {"A": -10.0, "B": -5.0, "C": -8.0},
            lower_is_better=False,  # Less negative = higher = better
        )
        assert result == "B"  # -5 is the highest
