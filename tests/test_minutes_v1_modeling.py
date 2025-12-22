"""Tests for Minutes V1 modeling and reconciliation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.minutes_v1.modeling import (
    ConformalIntervalCalibrator,
    predict_minutes,
    train_minutes_quickstart_models,
)
from projections.minutes_v1.reconcile import ReconcileConfig, reconcile_minutes_p50_all, reconcile_team_minutes_p50


def _synthetic_training_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = 160
    return pd.DataFrame(
        {
            "game_id": rng.integers(1, 20, size=rows),
            "player_id": rng.integers(1000, 2000, size=rows),
            "team_id": rng.integers(1, 10, size=rows),
            "minutes": rng.uniform(8, 38, size=rows),
            "prior_play_prob": rng.uniform(0.4, 1.0, size=rows),
            "recent_start_pct_10": rng.uniform(0.0, 1.0, size=rows),
            "min_last3": rng.uniform(5, 35, size=rows),
            "home_flag": rng.integers(0, 2, size=rows),
            "is_b2b": rng.integers(0, 2, size=rows),
            "sum_min_7d": rng.uniform(40, 180, size=rows),
            "spread_home": rng.uniform(-12, 12, size=rows),
            "total": rng.uniform(210, 240, size=rows),
        }
    )


def test_quickstart_model_training_and_prediction():
    training_df = _synthetic_training_frame()
    artifacts = train_minutes_quickstart_models(training_df)
    assert "mae" in artifacts.baseline.metrics
    scored = predict_minutes(artifacts, training_df.iloc[:10])
    assert {"baseline_pred", "p10", "p50", "p90", "p10_calibrated", "p90_calibrated"}.issubset(scored.columns)
    assert np.all(scored["p10_calibrated"] <= scored["p90_calibrated"])


def test_reconcile_team_minutes_obeys_caps_and_total():
    team_df = pd.DataFrame(
        {
            "game_id": [1] * 8,
            "team_id": [100] * 8,
            "player_id": list(range(8)),
            "minutes_p50": [40, 38, 36, 34, 28, 24, 22, 18],
            "minutes_p10": [30, 28, 26, 24, 18, 14, 12, 8],
            "minutes_p90": [44, 42, 40, 38, 34, 30, 28, 24],
            "is_projected_starter": [True, True, True, True, False, False, False, False],
            "play_prob": [1.0] * 8,
            "rotation_prob": [1.0] * 8,
        }
    )
    cfg = ReconcileConfig()
    reconciled = reconcile_team_minutes_p50(team_df, cfg)
    assert pytest.approx(reconciled.sum(), rel=1e-4) == cfg.team_minutes.target
    assert (reconciled >= 0.0).all()
    assert reconciled.max() <= cfg.bounds.hard_cap + 1e-6


def test_reconcile_minutes_appends_column():
    df = pd.DataFrame(
        {
            "game_id": [1] * 8 + [2] * 8,
            "team_id": [10] * 8 + [20] * 8,
            "player_id": list(range(16)),
            "minutes_p50": [32, 31, 30, 28, 26, 24, 22, 20, 30, 29, 28, 26, 25, 24, 22, 20],
            "minutes_p10": [20.0] * 16,
            "minutes_p90": [40.0] * 16,
            "is_projected_starter": [True, True, True, True, False, False, False, False] * 2,
            "play_prob": [1.0] * 16,
            "rotation_prob": [1.0] * 16,
        }
    )
    reconciled = reconcile_minutes_p50_all(df, ReconcileConfig())
    assert "minutes_p50_raw" in reconciled.columns
    totals = reconciled.groupby(["game_id", "team_id"])["minutes_p50"].sum()
    assert np.allclose(totals.to_numpy(), 240.0, rtol=1e-4)


def test_reconcile_minutes_preserves_quantiles():
    df = pd.DataFrame(
        {
            "game_id": [1] * 4,
            "team_id": [30] * 4,
            "player_id": list(range(4)),
            "minutes_p50": [32, 28, 24, 22],
            "minutes_p10": [20, 18, 16, 15],
            "minutes_p90": [40, 36, 33, 30],
            "is_projected_starter": [True, True, False, False],
            "play_prob": [1.0] * 4,
            "rotation_prob": [1.0] * 4,
        }
    )
    pre = df[["minutes_p10", "minutes_p90"]].copy()
    reconciled = reconcile_minutes_p50_all(df, ReconcileConfig())
    pd.testing.assert_series_equal(pre["minutes_p10"], reconciled["minutes_p10"], check_dtype=False)
    pd.testing.assert_series_equal(pre["minutes_p90"], reconciled["minutes_p90"], check_dtype=False)


def test_conformal_calibrator_exports_offsets_after_fit():
    calibrator = ConformalIntervalCalibrator(alpha_low=0.2, alpha_high=0.2)
    with pytest.raises(ValueError):
        calibrator.export_offsets()
    with pytest.raises(ValueError):
        calibrator.calibrate(np.array([0.0]), np.array([1.0]))

    y_true = np.array([10.0, 12.0, 14.0])
    lower = np.array([8.0, 10.0, 12.0])
    upper = np.array([16.0, 18.0, 20.0])
    calibrator.fit(y_true, lower, upper)
    offsets = calibrator.export_offsets()
    assert pytest.approx(offsets["alpha_low"]) == 0.2
    assert "low_adjustment" in offsets and "high_adjustment" in offsets


def test_conformal_calibrator_export_supports_legacy_pickles():
    calibrator = ConformalIntervalCalibrator(alpha_low=0.1, alpha_high=0.1)
    calibrator._low_adjustment = 0.75
    calibrator._high_adjustment = 1.25
    if "_fitted" in calibrator.__dict__:
        del calibrator.__dict__["_fitted"]

    offsets = calibrator.export_offsets()
    assert offsets["low_adjustment"] == pytest.approx(0.75)
    assert offsets["high_adjustment"] == pytest.approx(1.25)
