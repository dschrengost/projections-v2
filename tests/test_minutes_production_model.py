"""Sanity checks around the production minutes bundle."""

from __future__ import annotations

import json
from pathlib import Path

from projections.minutes_v1.production import load_production_minutes_bundle, resolve_production_run_dir


def _read_current_config() -> dict:
    path = Path("config/minutes_current_run.json").resolve()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def test_resolve_production_run_dir_points_at_default() -> None:
    run_dir, run_id = resolve_production_run_dir()
    assert run_dir.exists(), f"Production run dir missing: {run_dir}"
    cfg = _read_current_config()
    if str(cfg.get("mode") or "single").lower() == "dual":
        assert run_id == cfg.get("late_run_id")
    elif cfg.get("run_id") is not None:
        assert run_id == cfg.get("run_id")
    else:
        assert run_id is not None


def test_load_production_bundle_has_features() -> None:
    loaded = load_production_minutes_bundle()
    if loaded.get("mode") == "dual":
        early = loaded.get("early_bundle") or {}
        late = loaded.get("late_bundle") or {}
        assert early.get("feature_columns"), "Early bundle missing feature columns"
        assert late.get("feature_columns"), "Late bundle missing feature columns"
        assert Path(early.get("run_dir", "")).exists()
        assert Path(late.get("run_dir", "")).exists()
    else:
        bundle = loaded
        feature_cols = bundle.get("feature_columns")
        assert feature_cols, "Production bundle missing feature columns"
        assert Path(bundle.get("run_dir", "")).exists()
        assert bundle.get("run_id") is not None
