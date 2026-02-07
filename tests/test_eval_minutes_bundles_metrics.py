from __future__ import annotations

import pytest
import pandas as pd

from projections.cli.eval_minutes_bundles import compute_head_to_head_metrics


def test_compute_head_to_head_metrics_sanity_and_expected_values() -> None:
    df = pd.DataFrame(
        {
            "actual_minutes": [0.0, 5.0, 0.0, 12.0],
            "plays_target": [0, 1, 0, 1],
            "play_prob": [0.9, 0.1, 0.2, 0.8],
            "pred_p10_minutes": [0.0, 2.0, 0.0, 8.0],
            "pred_p50_minutes": [11.0, 6.0, 4.0, 10.0],
            "pred_p90_minutes": [12.0, 12.0, 8.0, 20.0],
        }
    )

    metrics = compute_head_to_head_metrics(df)

    assert 0.0 <= metrics["brier_play_prob"] <= 1.0
    assert 0.0 <= metrics["false_active_rate_p_ge_0_5"] <= 1.0
    assert 0.0 <= metrics["false_inactive_rate_p_le_0_2"] <= 1.0
    assert 0.0 <= metrics["bench_smear_proxy"] <= 1.0

    assert metrics["rows"] == 4
    assert metrics["positive_rows"] == 2
    assert metrics["false_active_rate_p_ge_0_5"] == pytest.approx(0.25)
    assert metrics["false_inactive_rate_p_le_0_2"] == pytest.approx(0.25)
    assert metrics["mae_p50_conditional"] == pytest.approx(1.5)
    assert metrics["bench_smear_proxy"] == pytest.approx(0.25)
    assert metrics["p10_coverage_leq"] == pytest.approx(0.5)
    assert metrics["p90_coverage_leq"] == pytest.approx(1.0)
