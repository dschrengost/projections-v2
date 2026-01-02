"""Unit tests for walk-forward month window generation."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from projections.cli.walk_forward_eval_minutes_injury_regime import (
    _iter_feature_month_windows,
    _iter_month_windows,
    aggregate_walk_forward_metrics,
    extract_injury_regime_report_metrics,
)


def test_iter_month_windows_clips_partial_months():
    windows = list(_iter_month_windows(date(2023, 10, 15), date(2023, 12, 2)))
    assert windows == [
        (date(2023, 10, 15), date(2023, 10, 31)),
        (date(2023, 11, 1), date(2023, 11, 30)),
        (date(2023, 12, 1), date(2023, 12, 2)),
    ]


def test_iter_month_windows_single_month():
    windows = list(_iter_month_windows(date(2025, 11, 1), date(2025, 11, 30)))
    assert windows == [(date(2025, 11, 1), date(2025, 11, 30))]


def test_iter_feature_month_windows_skips_off_season_months():
    features = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                [
                    "2024-04-01",
                    "2024-04-15",
                    "2024-10-02",
                    "2024-11-03",
                ]
            )
        }
    )
    windows = _iter_feature_month_windows(features, start=date(2024, 4, 1), end=date(2024, 11, 30))
    assert windows == [
        (date(2024, 4, 1), date(2024, 4, 30)),
        (date(2024, 10, 1), date(2024, 10, 31)),
        (date(2024, 11, 1), date(2024, 11, 30)),
    ]


def test_extract_injury_regime_report_metrics_includes_top9_and_tail(tmp_path):
    report = {
        "models": {
            "current": {
                "injury_regime": {
                    "n_player_rows": 10,
                    "n_team_games": 2,
                    "player_mae": 3.0,
                    "bench_core_mae": 5.0,
                    "top7_sum_mae": 11.0,
                    "top9_player_mae": 4.0,
                    "top9_sum_mae": 15.0,
                    "top9_sum_mae_team240": 13.0,
                    "top9_sum_bias_team240": -9.0,
                    "tail_minutes_mae_team240": 13.0,
                    "tail_minutes_bias_team240": 9.0,
                }
            },
            "candidate": {
                "injury_regime": {
                    "n_player_rows": 10,
                    "n_team_games": 2,
                    "player_mae": 2.5,
                    "bench_core_mae": 4.5,
                    "top7_sum_mae": 11.2,
                    "top9_player_mae": 3.5,
                    "top9_sum_mae": 11.0,
                    "top9_sum_mae_team240": 11.0,
                    "top9_sum_bias_team240": -8.0,
                    "tail_minutes_mae_team240": 11.0,
                    "tail_minutes_bias_team240": 8.0,
                }
            },
        }
    }
    import json

    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    extracted = extract_injury_regime_report_metrics(path)
    assert extracted is not None
    injury = extracted["injury_regime"]
    assert injury["top9_sum_mae_team240_current"] == pytest.approx(13.0)
    assert injury["top9_sum_mae_team240_candidate"] == pytest.approx(11.0)
    assert injury["tail_minutes_mae_team240_current"] == pytest.approx(13.0)
    assert injury["tail_minutes_mae_team240_candidate"] == pytest.approx(11.0)


def test_aggregate_walk_forward_metrics_produces_deltas():
    extracted_months = [
        {
            "month": "2025-11",
            "injury_regime": {
                "n_player_rows": 100,
                "n_team_games": 10,
                "player_mae_current": 3.0,
                "player_mae_candidate": 2.0,
                "bench_core_mae_current": 5.0,
                "bench_core_mae_candidate": 4.0,
                "top7_sum_mae_current": 11.0,
                "top7_sum_mae_candidate": 12.0,
                "top9_player_mae_current": 5.0,
                "top9_player_mae_candidate": 4.0,
                "top9_sum_mae_current": 15.0,
                "top9_sum_mae_candidate": 11.0,
                "top9_sum_mae_team240_current": 13.0,
                "top9_sum_mae_team240_candidate": 11.0,
                "top9_sum_bias_team240_current": -9.0,
                "top9_sum_bias_team240_candidate": -8.0,
                "tail_minutes_mae_team240_current": 13.0,
                "tail_minutes_mae_team240_candidate": 11.0,
                "tail_minutes_bias_team240_current": 9.0,
                "tail_minutes_bias_team240_candidate": 8.0,
            },
        }
    ]
    agg = aggregate_walk_forward_metrics(extracted_months)
    injury = agg["injury_regime"]
    assert injury["top9_sum_mae_team240_delta"] == pytest.approx(-2.0)
    assert injury["tail_minutes_mae_team240_delta"] == pytest.approx(-2.0)
