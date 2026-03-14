"""Tests for minutes feature builder verification."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.cli.build_minutes_live import (  # noqa: E402
    REQUIRED_MINUTES_FEATURES,
    _compute_live_role_history_frame,
    _compute_live_team_dispersion_prior_frame,
    _compute_live_rest_frame,
    _compute_live_trend_frame,
    _compute_recent_start_pct_frame,
    _compute_vacancy_features,
    _select_recent_start_source,
    _verify_required_features,
)


class TestRequiredMinutesFeatures:
    """Tests for REQUIRED_MINUTES_FEATURES constant."""

    def test_contains_team_context(self):
        """Team context columns are required."""
        assert "team_pace_szn" in REQUIRED_MINUTES_FEATURES
        assert "team_off_rtg_szn" in REQUIRED_MINUTES_FEATURES
        assert "team_def_rtg_szn" in REQUIRED_MINUTES_FEATURES

    def test_contains_opp_context(self):
        """Opponent context columns are required."""
        assert "opp_pace_szn" in REQUIRED_MINUTES_FEATURES
        assert "opp_def_rtg_szn" in REQUIRED_MINUTES_FEATURES

    def test_contains_vacancy(self):
        """Vacancy feature columns are required."""
        assert "vac_min_szn" in REQUIRED_MINUTES_FEATURES
        assert "vac_min_guard_szn" in REQUIRED_MINUTES_FEATURES
        assert "vac_min_wing_szn" in REQUIRED_MINUTES_FEATURES
        assert "vac_min_big_szn" in REQUIRED_MINUTES_FEATURES

    def test_contains_trends(self):
        """Trend feature columns are required."""
        assert "roll_mean_5" in REQUIRED_MINUTES_FEATURES
        assert "roll_mean_10" in REQUIRED_MINUTES_FEATURES
        assert "min_last3" in REQUIRED_MINUTES_FEATURES
        assert "min_last5" in REQUIRED_MINUTES_FEATURES


class TestVerifyRequiredFeatures:
    """Tests for _verify_required_features function."""

    def test_all_required_present(self):
        """No warning when all required features present."""
        df = pd.DataFrame({col: [1.0] for col in REQUIRED_MINUTES_FEATURES})
        warnings = []
        
        _verify_required_features(df, "test_run", warnings)
        
        # No warnings should be added when all features present
        assert len([w for w in warnings if "Missing" in w]) == 0

    def test_missing_features_logged(self):
        """Missing features are logged as warning."""
        # Only include half the required features
        subset = list(REQUIRED_MINUTES_FEATURES)[:7]
        df = pd.DataFrame({col: [1.0] for col in subset})
        warnings = []
        
        _verify_required_features(df, "test_run", warnings)
        
        # Warning should be added
        assert len(warnings) == 1
        assert "Missing" in warnings[0]
        # All missing columns should be mentioned
        missing = REQUIRED_MINUTES_FEATURES - set(subset)
        for col in missing:
            assert col in warnings[0]


class TestNumpyLiveHistoryHelpers:
    def test_trend_and_rest_recomputes_use_history_only(self):
        history = pd.DataFrame(
            {
                "player_id": [1, 1, 1, 1, 2, 2],
                "game_date": [
                    "2026-01-01",
                    "2026-01-03",
                    "2026-01-05",
                    "2026-01-07",
                    "2026-01-04",
                    "2026-01-08",
                ],
                "minutes": [20.0, 0.0, 30.0, 40.0, 10.0, 12.0],
            }
        )

        target_day = pd.Timestamp("2026-01-10")
        trend = _compute_live_trend_frame(history, target_day).set_index("player_id")
        rest = _compute_live_rest_frame(history, target_day).set_index("player_id")

        assert trend.loc[1, "min_last1"] == pytest.approx(40.0)
        assert trend.loc[1, "min_last3"] == pytest.approx(30.0)
        assert trend.loc[1, "min_last5"] == pytest.approx(30.0)
        assert trend.loc[1, "sum_min_7d"] == pytest.approx(70.0)
        assert trend.loc[1, "roll_iqr_5"] == pytest.approx(10.0)
        assert rest.loc[1, "days_since_last_recomp"] == pytest.approx(3.0)
        assert rest.loc[1, "is_b2b_recomp"] == pytest.approx(0.0)
        assert rest.loc[2, "days_since_last_recomp"] == pytest.approx(2.0)

    def test_recent_start_recompute_uses_last_ten_rows_per_player(self):
        history = pd.DataFrame(
            {
                "player_id": [1] * 12 + [2] * 4,
                "game_date": pd.date_range("2026-01-01", periods=12).tolist()
                + pd.date_range("2026-01-02", periods=4).tolist(),
                "starter_flag": [1] * 16,
                "starter_flag_label": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            }
        )

        starter_col = _select_recent_start_source(history)
        recency = _compute_recent_start_pct_frame(history, pd.Timestamp("2026-01-20"), starter_col).set_index("player_id")

        assert starter_col == "starter_flag_label"
        assert recency.loc[1, "recent_start_pct_10_recomp"] == pytest.approx(1.0)
        assert recency.loc[2, "recent_start_pct_10_recomp"] == pytest.approx(0.5)

    def test_role_history_recompute_uses_history_and_live_starter_flag(self):
        history = pd.DataFrame(
            {
                "player_id": [1, 1, 1, 2, 2],
                "team_id": [10, 10, 10, 20, 20],
                "season": ["2025-26"] * 5,
                "game_date": [
                    "2026-01-01",
                    "2026-01-03",
                    "2026-01-05",
                    "2026-01-02",
                    "2026-01-06",
                ],
                "minutes": [20.0, 30.0, 10.0, 16.0, 18.0],
                "starter_flag_label": [1, 0, 1, 0, 0],
            }
        )
        live = pd.DataFrame(
            {
                "player_id": [1, 2],
                "team_id": [10, 20],
                "season": ["2025-26", "2025-26"],
                "starter_flag": [1, 1],
            }
        )

        role = _compute_live_role_history_frame(
            history,
            live,
            pd.Timestamp("2026-01-10"),
            "starter_flag_label",
        ).set_index("player_id")

        assert role.loc[1, "starter_prev_game_asof"] == pytest.approx(1.0)
        assert role.loc[1, "rotation_minutes_std_5g"] == pytest.approx(float(pd.Series([20.0, 30.0, 10.0]).std(ddof=0)))
        assert role.loc[1, "role_change_rate_10g"] == pytest.approx(0.75)
        assert role.loc[1, "season_phase"] > 0.0
        assert role.loc[2, "starter_prev_game_asof"] == pytest.approx(0.0)

    def test_team_dispersion_prior_uses_last_prior_team_game(self):
        history = pd.DataFrame(
            {
                "team_id": [10, 10, 10, 10, 20, 20],
                "game_id": [100, 100, 101, 101, 200, 200],
                "game_date": [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-04",
                    "2026-01-04",
                ],
                "minutes": [20.0, 28.0, 30.0, 18.0, 16.0, 22.0],
            }
        )

        result = _compute_live_team_dispersion_prior_frame(history, pd.Timestamp("2026-01-10")).set_index("team_id")

        assert result.loc[10, "team_minutes_dispersion_prior"] == pytest.approx(6.0)
        assert result.loc[20, "team_minutes_dispersion_prior"] == pytest.approx(3.0)

    def test_vacancy_features_sum_minutes_by_game_and_team(self):
        injuries = pd.DataFrame(
            {
                "game_id": [10, 10, 10, 11],
                "player_id": [1, 1, 2, 3],
                "status": ["Q", "OUT", "OUT", "OUT"],
                "as_of_ts": [
                    "2026-01-09T18:00:00Z",
                    "2026-01-09T19:00:00Z",
                    "2026-01-09T19:30:00Z",
                    "2026-01-09T20:00:00Z",
                ],
            }
        )
        roster = pd.DataFrame(
            {
                "game_id": [10, 10, 11],
                "player_id": [1, 2, 3],
                "team_id": [100, 100, 200],
                "listed_pos": ["PG", "SF", "C"],
            }
        )
        labels = pd.DataFrame(
            {
                "player_id": [1, 1, 2, 3],
                "game_date": ["2026-01-01", "2026-01-05", "2026-01-05", "2026-01-06"],
                "minutes": [10.0, 20.0, 15.0, 18.0],
            }
        )

        result = _compute_vacancy_features(
            injuries_snapshot=injuries,
            roster_nightly=roster,
            labels_source=labels,
            target_day=pd.Timestamp("2026-01-10"),
            warnings=[],
        ).sort_values(["game_id", "team_id"]).reset_index(drop=True)

        assert list(result["game_id"]) == [10, 11]
        assert list(result["team_id"]) == [100, 200]
        assert result.loc[0, "vac_min_szn"] == pytest.approx(45.0)
        assert result.loc[0, "vac_min_guard_szn"] == pytest.approx(30.0)
        assert result.loc[0, "vac_min_wing_szn"] == pytest.approx(15.0)
        assert result.loc[1, "vac_min_big_szn"] == pytest.approx(18.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
