from __future__ import annotations

import pandas as pd
import pytest

from prefect_flows.minutes_retrain import (
    _resolve_requested_windows,
    _resolve_training_window,
)


def test_resolve_training_window_clamps_to_labels_min() -> None:
    window = _resolve_training_window(
        labels_max_date=pd.Timestamp("2026-02-05"),
        labels_min_date=pd.Timestamp("2025-10-21"),
        train_window_days=120,
        cal_window_days=14,
    )
    assert window["cal_end_date"] == "2026-02-05"
    assert window["cal_start_date"] == "2026-01-23"
    assert window["train_end_date"] == "2026-01-22"
    assert window["train_start_date"] == "2025-10-21"


def test_resolve_requested_windows_requires_all_explicit_dates() -> None:
    with pytest.raises(ValueError, match="Provide either all explicit train/cal dates or none"):
        _resolve_requested_windows(
            labels_max_date=pd.Timestamp("2026-02-05"),
            labels_min_date=pd.Timestamp("2025-10-21"),
            train_start_date="2025-11-01",
            train_end_date=None,
            cal_start_date="2026-02-01",
            cal_end_date="2026-02-05",
            train_window_days=120,
            cal_window_days=14,
        )


def test_resolve_requested_windows_uses_explicit_dates_when_provided() -> None:
    window = _resolve_requested_windows(
        labels_max_date=pd.Timestamp("2026-02-05"),
        labels_min_date=pd.Timestamp("2025-10-21"),
        train_start_date="2025-11-01T00:00:00Z",
        train_end_date="2026-01-31",
        cal_start_date="2026-02-01",
        cal_end_date="2026-02-05",
        train_window_days=120,
        cal_window_days=14,
    )
    assert window == {
        "train_start_date": "2025-11-01",
        "train_end_date": "2026-01-31",
        "cal_start_date": "2026-02-01",
        "cal_end_date": "2026-02-05",
    }


def test_resolve_training_window_raises_when_no_train_data_after_clamp() -> None:
    with pytest.raises(RuntimeError, match="invalid rolling window after clamp"):
        _resolve_training_window(
            labels_max_date=pd.Timestamp("2026-02-05"),
            labels_min_date=pd.Timestamp("2026-02-01"),
            train_window_days=120,
            cal_window_days=14,
        )
