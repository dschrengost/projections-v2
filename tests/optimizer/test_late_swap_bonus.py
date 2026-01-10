"""Unit tests for late swap bonus functionality."""

import numpy as np
import pandas as pd
import pytest

from projections.optimizer.objective.late_swap_bonus import (
    LateSwapBonusConfig,
    build_late_swap_bonus,
)


class TestLateSwapBonusConfig:
    """Tests for LateSwapBonusConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        cfg = LateSwapBonusConfig()
        assert cfg.bonus_per_hour == 0.2
        assert cfg.max_bonus == 1.5
        assert cfg.enabled is False

    def test_validate_positive(self):
        """Test validation accepts valid values."""
        cfg = LateSwapBonusConfig(bonus_per_hour=0.5, max_bonus=2.0, enabled=True)
        cfg.validate()  # Should not raise

    def test_validate_negative_bonus_per_hour(self):
        """Test validation rejects negative bonus_per_hour."""
        cfg = LateSwapBonusConfig(bonus_per_hour=-0.1)
        with pytest.raises(ValueError, match="bonus_per_hour must be >= 0"):
            cfg.validate()

    def test_validate_negative_max_bonus(self):
        """Test validation rejects negative max_bonus."""
        cfg = LateSwapBonusConfig(max_bonus=-1.0)
        with pytest.raises(ValueError, match="max_bonus must be >= 0"):
            cfg.validate()


class TestBuildLateSwapBonus:
    """Tests for build_late_swap_bonus function."""

    def test_disabled_returns_zeros(self):
        """When disabled, should return all zeros."""
        df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3"],
            "game_start_utc": [
                "2026-01-10T00:00:00Z",
                "2026-01-10T01:00:00Z",
                "2026-01-10T02:00:00Z",
            ],
        })
        cfg = LateSwapBonusConfig(enabled=False)
        bonus = build_late_swap_bonus(df, cfg)

        assert len(bonus) == 3
        assert np.all(bonus == 0.0)

    def test_basic_bonus_calculation(self):
        """Test basic bonus calculation based on hours after earliest."""
        df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3"],
            "game_start_utc": [
                "2026-01-10T00:00:00Z",  # earliest
                "2026-01-10T01:00:00Z",  # +1 hour
                "2026-01-10T03:00:00Z",  # +3 hours
            ],
        })
        cfg = LateSwapBonusConfig(enabled=True, bonus_per_hour=0.2, max_bonus=1.5)
        bonus = build_late_swap_bonus(df, cfg)

        assert len(bonus) == 3
        assert bonus[0] == pytest.approx(0.0)  # earliest game
        assert bonus[1] == pytest.approx(0.2)  # +1 hour * 0.2
        assert bonus[2] == pytest.approx(0.6)  # +3 hours * 0.2

    def test_max_bonus_cap(self):
        """Test that max_bonus caps the bonus."""
        df = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "game_start_utc": [
                "2026-01-10T00:00:00Z",  # earliest
                "2026-01-10T10:00:00Z",  # +10 hours
            ],
        })
        cfg = LateSwapBonusConfig(enabled=True, bonus_per_hour=0.5, max_bonus=2.0)
        bonus = build_late_swap_bonus(df, cfg)

        # 10 hours * 0.5 = 5.0, but capped at 2.0
        assert bonus[1] == pytest.approx(2.0)

    def test_missing_game_start_column(self):
        """When game_start_utc column is missing, return zeros."""
        df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3"],
        })
        cfg = LateSwapBonusConfig(enabled=True, bonus_per_hour=0.2)
        bonus = build_late_swap_bonus(df, cfg)

        assert len(bonus) == 3
        assert np.all(bonus == 0.0)

    def test_missing_game_times_handled(self):
        """Players with missing game times should get zero bonus."""
        df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3"],
            "game_start_utc": [
                "2026-01-10T00:00:00Z",
                None,  # missing
                "2026-01-10T02:00:00Z",
            ],
        })
        cfg = LateSwapBonusConfig(enabled=True, bonus_per_hour=0.2, max_bonus=1.5)
        bonus = build_late_swap_bonus(df, cfg)

        assert len(bonus) == 3
        assert bonus[0] == pytest.approx(0.0)  # earliest
        assert bonus[1] == pytest.approx(0.0)  # missing -> 0
        assert bonus[2] == pytest.approx(0.4)  # +2 hours * 0.2

    def test_all_same_game_time(self):
        """When all players have same game time, all get zero bonus."""
        df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3"],
            "game_start_utc": [
                "2026-01-10T00:00:00Z",
                "2026-01-10T00:00:00Z",
                "2026-01-10T00:00:00Z",
            ],
        })
        cfg = LateSwapBonusConfig(enabled=True, bonus_per_hour=0.2)
        bonus = build_late_swap_bonus(df, cfg)

        assert np.all(bonus == 0.0)

    def test_all_missing_game_times(self):
        """When all game times are missing, return zeros."""
        df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3"],
            "game_start_utc": [None, None, None],
        })
        cfg = LateSwapBonusConfig(enabled=True, bonus_per_hour=0.2)
        bonus = build_late_swap_bonus(df, cfg)

        assert len(bonus) == 3
        assert np.all(bonus == 0.0)

    def test_empty_dataframe(self):
        """Empty dataframe should return empty array."""
        df = pd.DataFrame({"player_id": [], "game_start_utc": []})
        cfg = LateSwapBonusConfig(enabled=True, bonus_per_hour=0.2)
        bonus = build_late_swap_bonus(df, cfg)

        assert len(bonus) == 0

    def test_partial_hours(self):
        """Test bonus calculation with partial hours."""
        df = pd.DataFrame({
            "player_id": ["p1", "p2"],
            "game_start_utc": [
                "2026-01-10T00:00:00Z",
                "2026-01-10T00:30:00Z",  # +30 minutes = +0.5 hours
            ],
        })
        cfg = LateSwapBonusConfig(enabled=True, bonus_per_hour=0.2)
        bonus = build_late_swap_bonus(df, cfg)

        assert bonus[0] == pytest.approx(0.0)
        assert bonus[1] == pytest.approx(0.1)  # 0.5 hours * 0.2
