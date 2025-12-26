"""Tests for RotAlloc config loading and drift detection.

These tests verify:
1. The loader uses promote_config.json values verbatim
2. Env var overrides are tracked in overrides_applied
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

from projections.minutes_alloc.rotalloc_production import (
    ENV_ROTALLOC_K_MAX,
    ENV_ROTALLOC_P_CUTOFF,
    RotAllocAllocatorConfig,
    score_rotalloc_minutes,
)


class _FakeRotClf:
    """Fake classifier returning uniform rotation probabilities."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])


class _FakeMuReg:
    """Fake regressor returning reasonable conditional minutes."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 25.0)


def _write_stub_rotalloc_bundle(tmp_path: Path, allocator_params: dict) -> Path:
    """Create a minimal rotalloc bundle with specified allocator params."""
    bundle_dir = tmp_path / "rotalloc_bundle"
    models_dir = bundle_dir / "models"
    models_dir.mkdir(parents=True)

    (models_dir / "feature_columns.json").write_text(
        json.dumps(["feat_a", "feat_b"]), encoding="utf-8"
    )
    joblib.dump(_FakeRotClf(), models_dir / "rot8_classifier.joblib")
    joblib.dump(_FakeMuReg(), models_dir / "minutes_regressor.joblib")

    promote_config = {"allocator": allocator_params}
    (bundle_dir / "promote_config.json").write_text(
        json.dumps(promote_config, indent=2), encoding="utf-8"
    )
    return bundle_dir


def _make_test_features() -> pd.DataFrame:
    """Create minimal feature DataFrame for testing."""
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1],
            "team_id": [10, 10, 10, 10, 10],
            "player_id": [101, 102, 103, 104, 105],
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [0.5, 0.6, 0.7, 0.8, 0.9],
            "status": ["ACT", "ACT", "ACT", "ACT", "ACT"],
            "play_prob": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_loader_uses_promote_config_verbatim(tmp_path: Path) -> None:
    """Verify that score_rotalloc_minutes uses promote_config values when no versioned config exists."""
    from projections.minutes_alloc import rotalloc_production
    
    expected_params = {
        "a": 2.0,
        "mu_power": 1.8,
        "p_cutoff": 0.25,
        "use_expected_k": True,
        "k_min": 7,
        "k_max": 12,
        "cap_max": 46.0,
    }
    bundle_dir = _write_stub_rotalloc_bundle(tmp_path, expected_params)
    features = _make_test_features()

    # Ensure no env overrides are set and mock versioned config to not exist
    env_vars_to_clear = [ENV_ROTALLOC_P_CUTOFF, ENV_ROTALLOC_K_MAX]
    
    # Mock VERSIONED_PROD_CONFIG to a non-existent path so bundle config is used
    fake_versioned_path = tmp_path / "nonexistent" / "config.json"
    
    with mock.patch.object(rotalloc_production, "VERSIONED_PROD_CONFIG", fake_versioned_path):
        with mock.patch.dict(os.environ, {}, clear=False):
            for var in env_vars_to_clear:
                os.environ.pop(var, None)

            _, allocator_cfg, _ = score_rotalloc_minutes(features, bundle_dir=bundle_dir)

    # Verify each param matches promote_config exactly
    assert allocator_cfg.a == expected_params["a"]
    assert allocator_cfg.mu_power == expected_params["mu_power"]
    assert allocator_cfg.p_cutoff == expected_params["p_cutoff"]
    assert allocator_cfg.use_expected_k == expected_params["use_expected_k"]
    assert allocator_cfg.k_min == expected_params["k_min"]
    assert allocator_cfg.k_max == expected_params["k_max"]
    assert allocator_cfg.cap_max == expected_params["cap_max"]
    # No overrides should be applied
    assert allocator_cfg.overrides_applied is None


def test_env_override_recorded_in_config(tmp_path: Path) -> None:
    """Verify that env var overrides are tracked in overrides_applied."""
    base_params = {
        "a": 1.5,
        "mu_power": 1.5,
        "p_cutoff": 0.2,
        "use_expected_k": True,
        "k_min": 8,
        "k_max": 11,
        "cap_max": 48.0,
    }
    bundle_dir = _write_stub_rotalloc_bundle(tmp_path, base_params)
    features = _make_test_features()

    # Set env var override
    override_p_cutoff = "0.35"
    override_k_max = "9"

    with mock.patch.dict(
        os.environ,
        {
            ENV_ROTALLOC_P_CUTOFF: override_p_cutoff,
            ENV_ROTALLOC_K_MAX: override_k_max,
        },
    ):
        _, allocator_cfg, _ = score_rotalloc_minutes(features, bundle_dir=bundle_dir)

    # Verify the overridden values are used
    assert allocator_cfg.p_cutoff == float(override_p_cutoff)
    assert allocator_cfg.k_max == int(override_k_max)

    # Verify overrides are recorded
    assert allocator_cfg.overrides_applied is not None
    assert "p_cutoff" in allocator_cfg.overrides_applied
    assert allocator_cfg.overrides_applied["p_cutoff"]["from"] == base_params["p_cutoff"]
    assert allocator_cfg.overrides_applied["p_cutoff"]["to"] == float(override_p_cutoff)
    assert "k_max" in allocator_cfg.overrides_applied
    assert allocator_cfg.overrides_applied["k_max"]["from"] == base_params["k_max"]
    assert allocator_cfg.overrides_applied["k_max"]["to"] == int(override_k_max)


def test_no_override_when_env_not_set(tmp_path: Path) -> None:
    """Verify that overrides_applied is None when no env vars are set."""
    base_params = {
        "a": 1.5,
        "mu_power": 1.5,
        "p_cutoff": 0.2,
        "use_expected_k": True,
        "k_min": 8,
        "k_max": 11,
        "cap_max": 48.0,
    }
    bundle_dir = _write_stub_rotalloc_bundle(tmp_path, base_params)
    features = _make_test_features()

    # Ensure env vars are not set
    env_patch = {
        ENV_ROTALLOC_P_CUTOFF: None,
        ENV_ROTALLOC_K_MAX: None,
    }
    with mock.patch.dict(os.environ, {}, clear=False):
        for var in env_patch:
            os.environ.pop(var, None)

        _, allocator_cfg, _ = score_rotalloc_minutes(features, bundle_dir=bundle_dir)

    assert allocator_cfg.overrides_applied is None
    assert allocator_cfg.p_cutoff == base_params["p_cutoff"]
    assert allocator_cfg.k_max == base_params["k_max"]
