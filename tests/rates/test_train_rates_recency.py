from __future__ import annotations

import pandas as pd
import pytest

from scripts.rates.train_rates_v1 import _compute_recency_sample_weights


def test_compute_recency_sample_weights_half_life_sanity() -> None:
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                [
                    "2026-01-10",  # age 0
                    "2025-10-27",  # age 75
                    "2025-08-13",  # age 150
                ]
            )
        }
    )
    weights, stats = _compute_recency_sample_weights(
        df,
        train_end_ts=pd.Timestamp("2026-01-10"),
        half_life_days=75.0,
        min_weight=0.0,
    )
    assert float(weights[0]) == pytest.approx(1.0, rel=1e-6)
    assert float(weights[1]) == pytest.approx(0.5, rel=1e-6)
    assert float(weights[2]) == pytest.approx(0.25, rel=1e-6)
    assert stats["max"] == pytest.approx(1.0, rel=1e-6)
