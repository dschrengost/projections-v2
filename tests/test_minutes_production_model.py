"""Sanity checks around the production minutes bundle."""

from __future__ import annotations

from pathlib import Path

from projections.minutes_v1.production import load_production_minutes_bundle, resolve_production_run_dir


def test_resolve_production_run_dir_points_at_default() -> None:
    run_dir, run_id = resolve_production_run_dir()
    assert run_dir.exists(), f"Production run dir missing: {run_dir}"
    assert run_id == "lgbm_full_v1_no_p_play_20251202"


def test_load_production_bundle_has_features() -> None:
    bundle = load_production_minutes_bundle()
    feature_cols = bundle.get("feature_columns")
    assert feature_cols, "Production bundle missing feature columns"
    assert Path(bundle.get("run_dir", "")).exists()
    assert bundle.get("run_id") == "lgbm_full_v1_no_p_play_20251202"
