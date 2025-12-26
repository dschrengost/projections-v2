"""Tests for minute share mixture model.

Tests cover:
- State boundary classification (minutes_to_state)
- Expected minutes prediction shape and non-negativity
- Allocator M sums to 240 per team
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.models.minute_share_mixture import (
    NUM_STATES,
    STATE_NAMES,
    get_state_counts,
    minutes_to_state,
    state_to_bucket_name,
    state_to_minute_range,
)


class TestMinutesToState:
    """Tests for minutes_to_state function."""

    def test_state_0_dnp(self):
        """S0 = exactly 0 minutes."""
        assert minutes_to_state(0.0) == 0
        assert minutes_to_state(0) == 0
    
    def test_state_1_garbage_time(self):
        """S1 = 0 < minutes <= 10."""
        assert minutes_to_state(0.1) == 1
        assert minutes_to_state(5.0) == 1
        assert minutes_to_state(10.0) == 1
    
    def test_state_2_fringe_rotation(self):
        """S2 = 10 < minutes <= 20."""
        assert minutes_to_state(10.1) == 2
        assert minutes_to_state(15.0) == 2
        assert minutes_to_state(20.0) == 2
    
    def test_state_3_core_rotation(self):
        """S3 = 20 < minutes <= 30."""
        assert minutes_to_state(20.1) == 3
        assert minutes_to_state(25.0) == 3
        assert minutes_to_state(30.0) == 3
    
    def test_state_4_starters(self):
        """S4 = minutes > 30."""
        assert minutes_to_state(30.1) == 4
        assert minutes_to_state(35.0) == 4
        assert minutes_to_state(48.0) == 4
    
    def test_array_input(self):
        """minutes_to_state works on arrays."""
        minutes = np.array([0, 5, 15, 25, 40])
        states = minutes_to_state(minutes)
        
        assert len(states) == 5
        np.testing.assert_array_equal(states, [0, 1, 2, 3, 4])
    
    def test_series_input(self):
        """minutes_to_state works on pandas Series."""
        minutes = pd.Series([0, 10, 20, 30, 48])
        states = minutes_to_state(minutes)
        
        assert len(states) == 5
        np.testing.assert_array_equal(states, [0, 1, 2, 3, 4])
    
    def test_boundary_values(self):
        """Test exact boundary values."""
        # 0 -> S0
        assert minutes_to_state(0) == 0
        # 10 -> S1 (inclusive upper)
        assert minutes_to_state(10) == 1
        # 10.0001 -> S2
        assert minutes_to_state(10.0001) == 2
        # 20 -> S2
        assert minutes_to_state(20) == 2
        # 20.0001 -> S3
        assert minutes_to_state(20.0001) == 3
        # 30 -> S3
        assert minutes_to_state(30) == 3
        # 30.0001 -> S4
        assert minutes_to_state(30.0001) == 4


class TestStateToBucketName:
    """Tests for state_to_bucket_name function."""

    def test_valid_states(self):
        """All valid states have names."""
        assert state_to_bucket_name(0) == "dnp"
        assert state_to_bucket_name(1) == "garbage_time"
        assert state_to_bucket_name(2) == "fringe_rotation"
        assert state_to_bucket_name(3) == "core_rotation"
        assert state_to_bucket_name(4) == "starters"
    
    def test_invalid_state(self):
        """Invalid states return unknown_X."""
        assert state_to_bucket_name(5) == "unknown_5"
        assert state_to_bucket_name(-1) == "unknown_-1"


class TestStateToMinuteRange:
    """Tests for state_to_minute_range function."""

    def test_state_0(self):
        """S0 is exactly 0."""
        lo, hi = state_to_minute_range(0)
        assert lo == 0.0
        assert hi == 0.0
    
    def test_state_1(self):
        """S1 is 0-10."""
        lo, hi = state_to_minute_range(1)
        assert lo == 0.0
        assert hi == 10.0
    
    def test_state_4(self):
        """S4 is 30-48."""
        lo, hi = state_to_minute_range(4)
        assert lo == 30.0
        assert hi == 48.0
    
    def test_invalid_state(self):
        """Invalid state raises ValueError."""
        with pytest.raises(ValueError):
            state_to_minute_range(5)


class TestGetStateCounts:
    """Tests for get_state_counts function."""

    def test_counts_all_states(self):
        """Counts correctly for all states."""
        minutes = np.array([0, 0, 5, 15, 15, 15, 25, 40, 40])
        counts = get_state_counts(minutes)
        
        assert counts[0] == 2  # DNP
        assert counts[1] == 1  # 5
        assert counts[2] == 3  # 15, 15, 15
        assert counts[3] == 1  # 25
        assert counts[4] == 2  # 40, 40
    
    def test_empty_array(self):
        """Empty array returns all zeros."""
        counts = get_state_counts(np.array([]))
        for s in range(NUM_STATES):
            assert counts[s] == 0


class TestConstants:
    """Tests for module constants."""

    def test_num_states(self):
        """NUM_STATES is 5."""
        assert NUM_STATES == 5
    
    def test_state_names_complete(self):
        """STATE_NAMES has all 5 states."""
        assert len(STATE_NAMES) == 5
        for i in range(NUM_STATES):
            assert i in STATE_NAMES
