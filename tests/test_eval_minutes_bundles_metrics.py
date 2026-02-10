from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from projections.cli import eval_minutes_bundles as eval_cli
from projections.cli.eval_minutes_bundles import compute_head_to_head_metrics


def _mock_pred_eval_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    p50_template = np.array([34.0, 32.0, 30.0, 28.0, 26.0, 20.0, 16.0, 12.0, 8.0, 4.0], dtype=float)
    p_play_template = np.array([0.96, 0.93, 0.90, 0.86, 0.82, 0.65, 0.50, 0.30, 0.10, 0.02], dtype=float)

    pred_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    game_date = pd.Timestamp("2026-02-01").date()
    for team_id, home_flag, player_start in ((10, 1, 1000), (11, 0, 2000)):
        for idx in range(10):
            player_id = player_start + idx
            is_out = idx >= 8
            starter = idx < 5
            p50 = float(p50_template[idx])
            p10 = max(0.0, p50 - 4.0)
            p90 = p50 + 6.0
            pred_rows.append(
                {
                    "game_id": "g1",
                    "team_id": team_id,
                    "player_id": player_id,
                    "game_date": game_date,
                    "actual_minutes": 0.0 if is_out else max(1.0, p50 - 2.0),
                    "plays_target": 0 if is_out else 1,
                    "play_prob": float(p_play_template[idx]),
                    "pred_p10_minutes": p10,
                    "pred_p50_minutes": p50,
                    "pred_p90_minutes": p90,
                    "bundle_label": "retrain",
                }
            )
            eval_rows.append(
                {
                    "game_id": "g1",
                    "team_id": team_id,
                    "player_id": player_id,
                    "status": "OUT" if is_out else "Available",
                    "is_out": 1 if is_out else 0,
                    "lineup_role": "out" if is_out else ("projected_starter" if starter else "bench"),
                    "starter_flag": 1 if starter else 0,
                    "is_projected_starter": starter,
                    "is_confirmed_starter": False,
                    "home_flag": home_flag,
                    "spread_home": -6.0,
                    "total": 230.5,
                }
            )
    return pd.DataFrame(pred_rows), pd.DataFrame(eval_rows)


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


def test_apply_occupancy_sparse_layer_zeroes_out_players_and_keeps_team_totals() -> None:
    pred_df, eval_df = _mock_pred_eval_frames()
    out, diagnostics = eval_cli.apply_occupancy_sparse_layer(pred_df, eval_df)

    is_out = eval_df["is_out"].astype(int).eq(1).to_numpy(dtype=bool)
    assert np.allclose(out.loc[is_out, "pred_p50_minutes"].to_numpy(dtype=float), 0.0)
    assert np.allclose(out.loc[is_out, "play_prob"].to_numpy(dtype=float), 0.0)

    sums = out.groupby(["game_id", "team_id"], as_index=False)["pred_p50_minutes"].sum()
    assert sums["pred_p50_minutes"].tolist() == pytest.approx([240.0, 240.0], abs=1e-6)

    assert len(diagnostics) == 2
    assert {"n_eligible", "bench_share_pred", "team_minutes_sum"}.issubset(diagnostics.columns)


def test_apply_occupancy_sparse_layer_applies_starter_play_prob_floor() -> None:
    pred_df, eval_df = _mock_pred_eval_frames()
    out, _diagnostics = eval_cli.apply_occupancy_sparse_layer(
        pred_df,
        eval_df,
        occupancy_scale=20.0,
        starter_floor=0.90,
    )

    starter_active = (
        eval_df["starter_flag"].astype(int).eq(1)
        & eval_df["is_out"].astype(int).eq(0)
        & out["pred_p50_minutes"].gt(0.0)
    )
    assert starter_active.any()
    assert (out.loc[starter_active, "play_prob"] >= 0.90).all()


def test_apply_occupancy_sparse_layer_keeps_intervals_monotonic() -> None:
    pred_df, eval_df = _mock_pred_eval_frames()
    out, _diagnostics = eval_cli.apply_occupancy_sparse_layer(pred_df, eval_df)

    p10 = out["pred_p10_minutes"].to_numpy(dtype=float)
    p50 = out["pred_p50_minutes"].to_numpy(dtype=float)
    p90 = out["pred_p90_minutes"].to_numpy(dtype=float)

    assert (p10 >= 0.0).all()
    assert (p10 <= p50 + 1e-9).all()
    assert (p50 <= p90 + 1e-9).all()
