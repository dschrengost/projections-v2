"""Tests for recency-weighted minutes retrain dataset builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.minutes_v1.retrain_dataset import RetrainWindows, _prepare_joined_frame, recency_weight_from_age_days


def _base_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [101, 102, 103],
            "player_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "game_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "feature_as_of_ts": [
                "2026-01-01T18:00:00Z",
                "2026-01-02T18:00:00Z",
                "2026-01-03T18:00:00Z",
            ],
            "tip_ts": [
                "2026-01-01T19:00:00Z",
                "2026-01-02T19:00:00Z",
                "2026-01-03T19:00:00Z",
            ],
            "some_feature": [0.2, 0.3, 0.4],
        }
    )


def _base_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [101, 102, 103],
            "player_id": [1, 1, 1],
            "team_id": [10, 10, 10],
            "game_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "minutes": [5.0, 12.0, 0.0],
        }
    )


def _windows() -> RetrainWindows:
    return RetrainWindows(
        train_start=pd.Timestamp("2026-01-01"),
        train_end=pd.Timestamp("2026-01-02"),
        cal_start=pd.Timestamp("2026-01-03"),
        cal_end=pd.Timestamp("2026-01-03"),
    )


def test_recency_weight_from_age_days_half_life_sanity() -> None:
    ages = np.array([0, 35, 70], dtype=float)
    weights = recency_weight_from_age_days(ages, half_life_days=35.0)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.5, rel=1e-6)
    assert weights[2] == pytest.approx(0.25, rel=1e-6)


def test_prepare_joined_frame_assigns_splits_and_train_weights() -> None:
    joined, summary = _prepare_joined_frame(
        features_df=_base_features(),
        labels_df=_base_labels(),
        windows=_windows(),
        half_life_days=35.0,
    )

    split_counts = summary["split_counts"]
    assert split_counts["train"] == 2
    assert split_counts["cal"] == 1

    train_rows = joined[joined["split"] == "train"].sort_values("game_date")
    assert train_rows.iloc[-1]["weight_recency"] == pytest.approx(1.0)
    assert float(train_rows.iloc[0]["weight_recency"]) < 1.0
    assert np.all(train_rows["weight_recency"].to_numpy(dtype=float) > 0.0)


def test_prepare_joined_frame_raises_on_leakage() -> None:
    features = _base_features()
    features.loc[0, "feature_as_of_ts"] = "2026-01-01T20:00:00Z"

    with pytest.raises(RuntimeError, match="Leakage violation"):
        _prepare_joined_frame(
            features_df=features,
            labels_df=_base_labels(),
            windows=_windows(),
            half_life_days=35.0,
        )
