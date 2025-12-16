"""Tests for Minutes V1 dataset helpers."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from projections.minutes_v1.datasets import TimeSplit, apply_time_split


def test_time_split_defaults_validation_start_day_after_train_end():
    split = TimeSplit.from_args(
        train_start=datetime(2024, 12, 1),
        train_end=datetime(2024, 12, 20),
        val_end=datetime(2024, 12, 31),
    )
    assert split.train_start == pd.Timestamp("2024-12-01")
    assert split.val_start == pd.Timestamp("2024-12-21")
    assert split.val_end == pd.Timestamp("2024-12-31")


def test_apply_time_split_filters_rows_by_game_date():
    df = pd.DataFrame(
        {
            "game_date": pd.date_range("2024-12-01", periods=6, freq="D"),
            "minutes": range(6),
        }
    )
    split = TimeSplit.from_args(
        train_start=datetime(2024, 12, 1),
        train_end=datetime(2024, 12, 3),
        val_end=datetime(2024, 12, 6),
        val_start=datetime(2024, 12, 4),
    )
    train_df, val_df = apply_time_split(df, split)
    assert len(train_df) == 3
    assert len(val_df) == 3
    assert train_df["minutes"].tolist() == [0, 1, 2]
    assert val_df["minutes"].tolist() == [3, 4, 5]


def test_apply_time_split_raises_when_window_empty():
    df = pd.DataFrame({"game_date": pd.date_range("2024-12-01", periods=2, freq="D")})
    split = TimeSplit.from_args(
        train_start=datetime(2024, 12, 1),
        train_end=datetime(2024, 12, 1),
        val_end=datetime(2024, 12, 3),
        val_start=datetime(2024, 12, 2),
    )
    with pytest.raises(ValueError):
        apply_time_split(df.iloc[:0], split)

