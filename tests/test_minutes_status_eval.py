"""Unit tests for status-stratified minutes evaluation helpers."""

from __future__ import annotations

import pandas as pd

from scripts.minutes_debug import status_eval_minutes as status_eval


def test_compute_status_metrics_groups_and_keys() -> None:
    df = pd.DataFrame(
        {
            "minutes": [30.0, 15.0, 0.0, 20.0],
            "minutes_p10": [25.0, 10.0, 0.0, 18.0],
            "minutes_p50": [32.0, 12.0, 4.0, 18.0],
            "minutes_p90": [38.0, 20.0, 8.0, 24.0],
            "status": ["Q", "Q", "OUT", "AVAIL"],
            "starter_flag": [1, 0, 1, 0],
        }
    )

    metrics = status_eval.compute_status_metrics(df, status_col="status", target_col="minutes")

    assert "status=Q,starter=1" in metrics
    assert metrics["status=Q,starter=1"]["rows"] == 1
    assert metrics["status=Q,starter=1"]["mae_q50"] == 2.0
    assert metrics["status=Q,starter=1"]["coverage_q10_q90"] == 1.0
    assert metrics["status=Q,starter=1"]["cond_coverage_q10_q90"] == 1.0

    assert "status=Q,starter=0" in metrics
    assert metrics["status=Q,starter=0"]["rows"] == 1
    assert metrics["status=Q,starter=0"]["mae_q50"] == 3.0

    # Ensure bucket metrics are present for each group.
    for group in metrics.values():
        assert any(key.startswith("mae_") for key in group if key != "mae_q50")
