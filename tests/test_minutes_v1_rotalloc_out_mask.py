from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from typer.testing import CliRunner

from projections.cli.score_minutes_v1 import app as score_v1_app
from projections.minutes_alloc import rotalloc_production
from projections.minutes_v1 import modeling
from projections.minutes_v1.artifacts import compute_feature_hash


runner = CliRunner()


class _FakeQuantileModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        return np.full(len(df), self.value, dtype=float)


class _FakeRotClf:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        feat_a = pd.to_numeric(X.get("feat_a"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        p = np.clip(0.05 + 0.9 * feat_a, 0.0, 1.0)
        return np.vstack([1.0 - p, p]).T


class _FakeMuReg:
    def predict(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        feat_b = pd.to_numeric(X.get("feat_b"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.clip(20.0 + 5.0 * feat_b, 0.0, None)


def _write_live_features(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    features_run_dir = data_root / "live" / "features_minutes_v1" / "2025-12-01" / "run=test_run"
    features_run_dir.mkdir(parents=True)

    # Use 12 players to ensure >= 9 eligible after 1 OUT and cutoff filtering
    n_players = 12
    df = pd.DataFrame(
        {
            "game_id": [1001] * n_players,
            "player_id": list(range(1, n_players + 1)),
            "team_id": [10] * n_players,
            "opponent_team_id": [20] * n_players,
            "season": ["2025"] * n_players,
            "player_name": [f"Player {i}" for i in range(1, n_players + 1)],
            "team_name": ["Team 10"] * n_players,
            "team_tricode": ["T10"] * n_players,
            "opponent_team_name": ["Team 20"] * n_players,
            "opponent_team_tricode": ["T20"] * n_players,
            "minutes": np.linspace(30.0, 5.0, n_players),
            "game_date": ["2025-12-01"] * n_players,
            "starter_flag_label": [1] * 5 + [0] * (n_players - 5),
            "ramp_flag": [0] * n_players,
            "tip_ts": ["2025-12-01T23:00:00Z"] * n_players,
            "feature_as_of_ts": ["2025-12-01T18:00:00Z"] * n_players,
            "injury_as_of_ts": ["2025-12-01T17:00:00Z"] * n_players,
            "odds_as_of_ts": ["2025-12-01T19:00:00Z"] * n_players,
            "blowout_index": [1.0] * n_players,
            "status": ["OK"] * n_players,
            # Ensure all players have p_rot > p_cutoff (0.15) for sufficient eligible set
            "feat_a": np.linspace(0.15, 0.95, n_players),
            "feat_b": np.linspace(0.2, 0.8, n_players),
        }
    )
    df.to_parquet(features_run_dir / "features.parquet", index=False)
    return data_root, features_run_dir


def _write_stub_minutes_bundle(tmp_path: Path, data_root: Path) -> Path:
    bundle_dir = tmp_path / "minutes_bundle"
    bundle_dir.mkdir(parents=True)

    feature_cols = ["feat_a", "feat_b"]
    imputer = SimpleImputer(strategy="mean")
    features_df = pd.read_parquet(
        data_root / "live" / "features_minutes_v1" / "2025-12-01" / "run=test_run" / "features.parquet"
    )
    imputer.fit(features_df[feature_cols])

    models = {q: _FakeQuantileModel(20 + 5 * q) for q in (0.1, 0.5, 0.9)}
    quantiles = modeling.QuantileArtifacts(models=models, imputer=imputer)
    calibrator = modeling.ConformalIntervalCalibrator(alpha_low=0.1, alpha_high=0.1)
    calibrator.fit(np.array([10.0, 12.0]), np.array([9.0, 11.0]), np.array([13.0, 15.0]))
    bundle = {
        "feature_columns": feature_cols,
        "quantiles": quantiles,
        "calibrator": calibrator,
        "bucket_offsets": {"__global__": {"d10": 0.0, "d90": 0.0, "n": len(features_df)}},
        "bucket_mode": "none",
        "conformal_mode": "tail-deltas",
        "play_probability": None,
    }
    joblib.dump(bundle, bundle_dir / "lgbm_quantiles.joblib")
    meta = {"feature_hash": compute_feature_hash(feature_cols)}
    (bundle_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return bundle_dir


def _write_stub_rotalloc_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "rotalloc_bundle"
    models_dir = bundle_dir / "models"
    models_dir.mkdir(parents=True)

    (models_dir / "feature_columns.json").write_text(json.dumps(["feat_a", "feat_b"]), encoding="utf-8")
    joblib.dump(_FakeRotClf(), models_dir / "rot8_classifier.joblib")
    joblib.dump(_FakeMuReg(), models_dir / "minutes_regressor.joblib")
    (bundle_dir / "promote_config.json").write_text(
        json.dumps(
            {
                "allocator": {
                    "a": 1.5,
                    "mu_power": 1.5,
                    "p_cutoff": 0.15,
                    "use_expected_k": True,
                    "k_min": 5,
                    "k_max": 10,
                    "cap_max": 48.0,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle_dir


def test_rotalloc_respects_post_scoring_out_mask(tmp_path: Path) -> None:
    data_root, features_run_dir = _write_live_features(tmp_path)

    # ESPN injuries mark Player 2 OUT, but features status remains OK.
    espn_dir = data_root / "silver" / "espn_injuries" / "date=2025-12-01"
    espn_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "as_of_ts": ["2025-12-01T18:00:00Z"],
            "player_name": ["Player 2"],
            "status": ["OUT"],
        }
    ).to_parquet(espn_dir / "injuries.parquet", index=False)

    minutes_bundle_dir = _write_stub_minutes_bundle(tmp_path, data_root)
    rotalloc_bundle_dir = _write_stub_rotalloc_bundle(tmp_path)

    bundle_config = tmp_path / "minutes_current_run.json"
    bundle_config.write_text(
        json.dumps(
            {
                "bundle_dir": str(minutes_bundle_dir),
                "minutes_alloc_mode": "rotalloc_expk",
                "rotalloc_bundle_dir": str(rotalloc_bundle_dir),
            }
        ),
        encoding="utf-8",
    )

    daily_root = tmp_path / "artifacts" / "minutes_v1" / "daily"
    injuries_root = tmp_path / "data" / "bronze" / "injuries_raw"
    schedule_root = tmp_path / "data" / "silver" / "schedule"

    # Mock versioned config to not exist so test uses bundle-specific config
    # (avoids production guardrails triggering on small test data)
    fake_versioned_path = tmp_path / "nonexistent" / "config.json"

    # Clear FAIL_HARD env var to prevent guardrail exceptions on synthetic test data
    import os

    env_override = {"PROJECTIONS_DATA_ROOT": str(data_root), "PROJECTIONS_ROTALLOC_FAIL_HARD": ""}
    with (
        mock.patch.object(rotalloc_production, "VERSIONED_PROD_CONFIG", fake_versioned_path),
        mock.patch.dict(os.environ, env_override, clear=False),
    ):
        result = runner.invoke(
            score_v1_app,
            [
                "--date",
                "2025-12-01",
                "--mode",
                "live",
                "--features-path",
                str(features_run_dir),
                "--bundle-dir",
                str(minutes_bundle_dir),
                "--bundle-config",
                str(bundle_config),
                "--artifact-root",
                str(daily_root),
                "--injuries-root",
                str(injuries_root),
                "--schedule-root",
                str(schedule_root),
                "--disable-promotion-prior",
            ],
            env=env_override,
        )
    assert result.exit_code == 0, result.output

    parquet_path = daily_root / "2025-12-01" / "run=test_run" / "minutes.parquet"
    assert parquet_path.exists()
    frame = pd.read_parquet(parquet_path)

    status_upper = frame["status"].astype(str).str.upper()
    out_rows = frame.loc[status_upper == "OUT"]
    assert not out_rows.empty
    assert (pd.to_numeric(out_rows["minutes_p50"], errors="coerce").fillna(0.0) == 0.0).all()

    play_prob = pd.to_numeric(frame["play_prob"], errors="coerce").fillna(1.0)
    active = (status_upper != "OUT") & (play_prob > 0.0)
    totals = (
        frame.loc[active]
        .assign(minutes_p50=pd.to_numeric(frame.loc[active, "minutes_p50"], errors="coerce").fillna(0.0))
        .groupby(["game_id", "team_id"])["minutes_p50"]
        .sum()
    )
    assert not totals.empty
    assert float((totals - 240.0).abs().max()) < 1e-6

