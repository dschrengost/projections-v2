from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from projections.cli import eval_minutes_bundles as eval_cli
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


def test_score_bundle_uses_full_eval_frame_for_play_prob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "player_id": [1, 2],
            "team_id": [10, 11],
            "game_date": [pd.Timestamp("2026-02-01").date(), pd.Timestamp("2026-02-01").date()],
            "actual_minutes": [12.0, 0.0],
            "plays_target": [1, 0],
            "feat_one": [0.1, 0.2],
            "feat_two": [1.0, 0.5],
            "prior_play_prob": [0.9, 0.1],
        }
    )
    bundle = {
        "feature_columns": ["feat_one", "feat_two"],
        "quantiles": object(),
        "bucket_offsets": {"__global__": {"d10": 0.0, "d90": 0.0, "n": 0}},
        "bucket_mode": "none",
        "conformal_mode": "tail-deltas",
        "play_probability": object(),
    }

    def _fake_predict_quantiles(_quantiles, X):
        n = len(X)
        return {
            0.1: np.full(n, 3.0),
            0.5: np.full(n, 6.0),
            0.9: np.full(n, 9.0),
        }

    def _fake_apply_conformal(df, *_args, **_kwargs):
        out = df.copy()
        out["p10_adj"] = out["p10_pred"]
        out["p50_adj"] = out["p50_pred"]
        out["p90_adj"] = out["p90_pred"]
        return out

    captured: dict[str, list[str]] = {}

    def _fake_predict_play_probability(_artifacts, X, *, _calibrated: bool = True):
        captured["columns"] = list(X.columns)
        return np.full(len(X), 0.7)

    monkeypatch.setattr(eval_cli.ml.modeling, "predict_quantiles", _fake_predict_quantiles)
    monkeypatch.setattr(eval_cli.ml, "apply_conformal", _fake_apply_conformal)
    monkeypatch.setattr(eval_cli.ml, "predict_play_probability", _fake_predict_play_probability)

    scored = eval_cli.score_bundle_on_eval_dataset(
        eval_df,
        bundle=bundle,
        bundle_label="test",
    )

    assert scored["play_prob"].tolist() == pytest.approx([0.7, 0.7])
    assert "columns" in captured
    assert "prior_play_prob" in captured["columns"]
