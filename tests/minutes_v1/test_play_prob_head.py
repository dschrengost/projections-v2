from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from projections.cli import score_minutes_v1
from projections.minutes_v1 import modeling
from projections.models import minutes_lgbm


def _toy_minutes_frame() -> pd.DataFrame:
    base_ts = pd.Timestamp("2025-01-01T00:00:00Z")
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2", "g3"],
            "player_id": [1, 2, 3, 4, 5],
            "team_id": [10, 10, 11, 11, 12],
            "status": ["QUESTIONABLE"] * 5,
            "injury_snapshot_missing": [1] * 5,
            "minutes": [0.0, 22.0, 30.0, 0.0, 15.0],
            "feature_as_of_ts": base_ts + pd.to_timedelta(np.arange(5), unit="h"),
            "game_date": (base_ts + pd.to_timedelta(np.arange(5), unit="h")).tz_convert(None),
            "feat_one": [0.1, 0.3, 0.7, 0.2, 0.4],
            "feat_two": [1.0, 0.5, 0.2, 0.1, 0.6],
            "starter_flag": [0, 1, 1, 0, 1],
            "pos_bucket": ["G", "G", "F", "F", "C"],
        }
    )


def _fit_toy_bundle(df: pd.DataFrame) -> dict:
    feature_cols = ["feat_one", "feat_two"]
    quantiles = modeling.train_lightgbm_quantiles(
        df[feature_cols],
        df["minutes"],
        random_state=0,
        params={"n_estimators": 30, "num_leaves": 15, "min_data_in_leaf": 1, "learning_rate": 0.1},
    )
    play_prob_artifacts = minutes_lgbm._train_play_probability_model(
        df[feature_cols],
        (df["minutes"] > 0).astype(int),
        random_state=0,
    )
    return {
        "feature_columns": feature_cols,
        "quantiles": quantiles,
        "calibrator": None,
        "bucket_offsets": {"__global__": {"d10": 0.0, "d90": 0.0, "n": len(df)}},
        "conformal_mode": "tail-deltas",
        "bucket_mode": "none",
        "play_probability": play_prob_artifacts,
        "play_prob_enabled": True,
    }


def test_play_prob_head_outputs_probabilities_and_metrics() -> None:
    df = _toy_minutes_frame()
    bundle = _fit_toy_bundle(df)

    scored_with = score_minutes_v1._score_rows(df, bundle, enable_play_prob_head=True)
    scored_without = score_minutes_v1._score_rows(df, bundle, enable_play_prob_head=False)

    assert "play_prob" in scored_with.columns
    assert scored_with["play_prob"].between(0.0, 1.0).all()
    for col in ("minutes_p10", "minutes_p50", "minutes_p90"):
        np.testing.assert_allclose(scored_with[col], scored_without[col])

    scored_with["p50"] = scored_with["minutes_p50"]
    scored_with["plays_target"] = (scored_with["minutes"] > 0).astype(int)
    metrics = minutes_lgbm._compute_playable_subset_metrics(
        scored_with,
        target_col="minutes",
        minutes_threshold=0.0,
    )
    assert metrics["val_play_prob_brier_playable"] is not None
    assert metrics["val_play_prob_ece_playable"] is not None


def test_resolved_bundle_scoring_includes_play_prob(tmp_path: Path) -> None:
    df = _toy_minutes_frame()
    bundle = _fit_toy_bundle(df)
    run_dir = tmp_path / "artifacts" / "minutes_lgbm" / "lgbm_full_v1_play_head_test"
    run_dir.mkdir(parents=True)
    joblib.dump(bundle, run_dir / "lgbm_quantiles.joblib")

    loaded_bundle, resolved_dir, _, run_id = score_minutes_v1._resolve_bundle_artifacts(
        None,
        score_minutes_v1.DEFAULT_BUNDLE_CONFIG,
        override_run_id=run_dir.name,
        artifact_root=run_dir.parent,
    )
    assert resolved_dir == run_dir
    assert run_id == run_dir.name

    scored = score_minutes_v1._score_rows(df, loaded_bundle, enable_play_prob_head=True)
    assert "play_prob" in scored.columns
    assert scored["play_prob"].notna().all()
