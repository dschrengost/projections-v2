"""Guardrail tests for rotshare scoring path.

These protect against "double reconciliation" (rotshare already sums to 240).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import projections.cli.score_minutes_v1 as score_cli
from projections.minutes_v1.rotation_share import train_rotation_share_model


def _raise_called(name: str):
    def _inner(*_args, **_kwargs):  # noqa: ANN001 - pytest monkeypatch target
        raise AssertionError(f"{name} should not be called for rotshare scoring")

    return _inner


def test_rotshare_bypasses_upside_and_reconcile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Minimal rotshare bundle.
    X_train = pd.DataFrame(
        {
            "feat": np.linspace(0.0, 1.0, 40),
            "is_out": np.zeros(40, dtype=int),
        }
    )
    y_train = pd.Series(np.where(np.arange(40) < 20, 12.0, 0.0), dtype=float)
    artifacts = train_rotation_share_model(
        X_train,
        y_train,
        random_state=3,
        play_params={"n_estimators": 20},
        share_params={"n_estimators": 20},
    )
    bundle_dir = tmp_path / "rotshare_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, bundle_dir / "rotation_share_model.joblib")
    (bundle_dir / "meta.json").write_text("{}", encoding="utf-8")

    # Minimal features parquet for a single day.
    day = date(2025, 1, 1)
    tip = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    features = pd.DataFrame(
        {
            "game_date": [day] * 10,
            "tip_ts": [tip] * 10,
            "game_id": [1] * 10,
            "team_id": [100] * 10,
            "player_id": list(range(1, 11)),
            "feat": np.linspace(0.0, 1.0, 10),
            "is_out": np.zeros(10, dtype=int),
            "starter_flag": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )
    features_path = bundle_dir / "features.parquet"
    features.to_parquet(features_path, index=False)

    # Guardrails: these should be bypassed for rotshare bundles even if enabled/requested.
    monkeypatch.setattr(score_cli, "apply_upside_adjustment", _raise_called("apply_upside_adjustment"))
    monkeypatch.setattr(score_cli, "reconcile_minutes_p50_all", _raise_called("reconcile_minutes_p50_all"))

    out_root = tmp_path / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    score_cli.score_minutes_range_to_parquet(
        day,
        day,
        features_path=features_path,
        bundle_dir=bundle_dir,
        artifact_root=out_root,
        injuries_root=out_root,  # avoid reading real data roots
        schedule_root=out_root,
        promotion_prior_enabled=False,
        reconcile_team_minutes="p50",  # should be forced to none
        enable_upside_adjustment=True,  # should be forced off
    )


def test_rotshare_mc_quantiles_produces_monotonic_tails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Minimal rotshare bundle.
    X_train = pd.DataFrame(
        {
            "feat": np.linspace(0.0, 1.0, 60),
            "is_out": np.zeros(60, dtype=int),
        }
    )
    y_train = pd.Series(np.where(np.arange(60) < 30, 12.0, 0.0), dtype=float)
    artifacts = train_rotation_share_model(
        X_train,
        y_train,
        random_state=7,
        play_params={"n_estimators": 20},
        share_params={"n_estimators": 20},
        in_rotation_minutes_threshold=10.0,
    )
    bundle_dir = tmp_path / "rotshare_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, bundle_dir / "rotation_share_model.joblib")
    (bundle_dir / "meta.json").write_text("{}", encoding="utf-8")

    day = date(2025, 1, 2)
    tip = datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)
    features = pd.DataFrame(
        {
            "game_date": [day] * 12,
            "tip_ts": [tip] * 12,
            "game_id": [2] * 12,
            "team_id": [200] * 12,
            "player_id": list(range(1, 13)),
            "feat": np.linspace(0.0, 1.0, 12),
            "is_out": np.zeros(12, dtype=int),
            "starter_flag": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    features_path = bundle_dir / "features.parquet"
    features.to_parquet(features_path, index=False)

    monkeypatch.setattr(score_cli, "apply_upside_adjustment", _raise_called("apply_upside_adjustment"))
    monkeypatch.setattr(score_cli, "reconcile_minutes_p50_all", _raise_called("reconcile_minutes_p50_all"))

    out_root = tmp_path / "out"
    out_root.mkdir(parents=True, exist_ok=True)
    scored = score_cli.score_minutes_range_to_parquet(
        day,
        day,
        features_path=features_path,
        bundle_dir=bundle_dir,
        artifact_root=out_root,
        injuries_root=out_root,
        schedule_root=out_root,
        promotion_prior_enabled=False,
        reconcile_team_minutes="none",
        enable_upside_adjustment=False,
        rotshare_quantiles_mode="mc",
        rotshare_n_worlds=500,
        rotshare_concentration=40.0,
        rotshare_seed=11,
        rotshare_min_active_players=5,
    )

    assert {"minutes_p10", "minutes_p50", "minutes_p90"}.issubset(scored.columns)
    assert (scored["minutes_p10"] <= scored["minutes_p50"] + 1e-9).all()
    assert (scored["minutes_p50"] <= scored["minutes_p90"] + 1e-9).all()
    assert (scored[["minutes_p10", "minutes_p50", "minutes_p90"]] >= 0.0).all().all()
    assert (scored[["minutes_p10", "minutes_p50", "minutes_p90"]] <= 48.0 + 1e-9).all().all()
