from __future__ import annotations

from scripts.rotation.monitor_gtv2_calibration_nightly import (
    _compute_alerts,
    _parse_run_timestamp,
)


def test_parse_run_timestamp_supports_suffix_embedded_run_ids() -> None:
    ts = _parse_run_timestamp("gpucheck_postpatch_20260310T045222Z")
    assert ts is not None
    assert str(ts) == "2026-03-10 04:52:22+00:00"


def test_compute_alerts_flags_huge_rows_and_bucket_drift() -> None:
    today = {
        "date": "2026-03-25",
        "huge_pred_rows": 1,
        "propless_n": 250,
        "propless_over_p95": 0.12,
        "minutes_12_20_n": 220,
        "minutes_12_20_over_p95": 0.08,
    }
    history = [
        {
            "date": "2026-03-23",
            "huge_pred_rows": 0,
            "propless_n": 250,
            "propless_over_p95": 0.03,
            "minutes_12_20_n": 220,
            "minutes_12_20_over_p95": 0.03,
        },
        {
            "date": "2026-03-24",
            "huge_pred_rows": 0,
            "propless_n": 250,
            "propless_over_p95": 0.04,
            "minutes_12_20_n": 220,
            "minutes_12_20_over_p95": 0.04,
        },
        {
            "date": "2026-03-22",
            "huge_pred_rows": 0,
            "propless_n": 250,
            "propless_over_p95": 0.035,
            "minutes_12_20_n": 220,
            "minutes_12_20_over_p95": 0.035,
        },
    ]

    alerts, diag = _compute_alerts(
        today_row=today,
        history_rows=history,
        baseline_window_days=14,
        drift_abs_band=0.01,
        drift_std_mult=2.0,
        min_bucket_n=100,
    )

    assert "huge_pred_rows" in alerts
    assert "propless_over_p95_drift" in alerts
    assert "minutes_12_20_over_p95_drift" in alerts
    assert diag["propless_over_p95"]["status"] == "ok"
    assert diag["minutes_12_20_over_p95"]["status"] == "ok"
