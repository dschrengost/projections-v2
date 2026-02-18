"""Unit tests for live injury-regime minutes evaluator helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd

from projections.cli.eval_minutes_live_injury_regime import (
    _active_only_slice,
    _catastrophic_minutes_metrics,
    _coerce_utc_ts_series,
    _load_candidate_minutes,
)


def test_load_candidate_minutes_empty_returns_schema(tmp_path):
    required = pd.DataFrame(
        {
            "game_date": [date(2026, 1, 2)],
            "run_id": ["20260102T235959Z"],
        }
    )

    preds, coverage = _load_candidate_minutes(
        root=tmp_path,
        required=required,
        minutes_col="effective_minutes",
        require_full_coverage=False,
    )

    assert preds.empty
    assert list(preds.columns) == ["run_id", "game_id", "team_id", "player_id", "_pred_minutes"]
    assert coverage.required_pairs == 1
    assert coverage.found_pairs == 0
    assert len(coverage.missing_pairs) == 1
    assert coverage.missing_pairs[0]["reason"] == "missing_minutes_parquet"


def test_active_only_catastrophic_excludes_out_rows():
    df = pd.DataFrame(
        {
            "game_date": [pd.Timestamp("2026-01-02")] * 4,
            "run_id": ["20260102T235959Z"] * 4,
            "game_id": [1, 1, 1, 1],
            "team_id": [10, 10, 10, 10],
            "team_tricode": ["AAA"] * 4,
            "player_id": [101, 102, 103, 104],
            "player_name": ["A", "B", "C", "D"],
            "status": ["OUT", "Ava", "Ava", "Ava"],
            "starter_flag_label": [0, 0, 0, 0],
            "minutes": [0.0, 0.0, 20.0, 8.0],
            "_pred_minutes": [20.0, 20.0, 0.0, 10.0],
        }
    )

    all_cat = _catastrophic_minutes_metrics(
        df,
        pred_col="_pred_minutes",
        actual_col="minutes",
        ghost_pred_min=15.0,
        ghost_actual_max=0.0,
        missed_actual_min=15.0,
        missed_pred_max=5.0,
        top_n=5,
    )
    active_cat = _catastrophic_minutes_metrics(
        _active_only_slice(df),
        pred_col="_pred_minutes",
        actual_col="minutes",
        ghost_pred_min=15.0,
        ghost_actual_max=0.0,
        missed_actual_min=15.0,
        missed_pred_max=5.0,
        top_n=5,
    )

    assert all_cat["ghost_dnp"]["n_ghost"] == 2
    assert active_cat["ghost_dnp"]["n_ghost"] == 1
    assert all_cat["missed_run"]["n_missed"] == 1
    assert active_cat["missed_run"]["n_missed"] == 1


def test_coerce_utc_ts_series_handles_missing_column():
    df = pd.DataFrame({"x": [1, 2]})
    series = _coerce_utc_ts_series(df, "missing")
    assert series.isna().all()


def test_minutes_to_tip_uses_existing_tip_or_schedule_tip():
    snapshot = pd.DataFrame(
        {
            "run_as_of_ts": [
                "2026-01-02T23:55:00Z",
                "2026-01-03T01:55:00Z",
            ],
            "tip_ts": [
                "2026-01-03T00:00:00Z",
                None,
            ],
            "tip_ts_schedule": [
                "2026-01-03T00:01:00Z",
                "2026-01-03T02:00:00Z",
            ],
        }
    )
    run_ts = _coerce_utc_ts_series(snapshot, "run_as_of_ts")
    tip_ts = _coerce_utc_ts_series(snapshot, "tip_ts").fillna(
        _coerce_utc_ts_series(snapshot, "tip_ts_schedule")
    )
    minutes_to_tip = (tip_ts - run_ts).dt.total_seconds() / 60.0

    assert minutes_to_tip.iloc[0] == 5.0
    assert minutes_to_tip.iloc[1] == 5.0
