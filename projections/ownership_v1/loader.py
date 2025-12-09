"""Bundle loading for ownership_v1 models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb

from projections.paths import data_path


@dataclass
class OwnershipBundle:
    """Container for a trained ownership model and its metadata."""
    model: lgb.Booster
    feature_cols: list[str]
    meta: dict[str, Any]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ownership_bundle(
    run_id: str, 
    base_artifacts_root: Path | str | None = None
) -> OwnershipBundle:
    """
    Load model, feature_cols.json, and meta.json for an ownership_v1 run.
    
    Args:
        run_id: The run identifier (e.g., "poc_001")
        base_artifacts_root: Optional override for artifacts root directory.
            Defaults to data_path() which uses PROJECTIONS_DATA_ROOT env var.
    
    Returns:
        OwnershipBundle with loaded model and metadata.
    """
    root = Path(base_artifacts_root) if base_artifacts_root else data_path()
    run_dir = root / "artifacts" / "ownership_v1" / "runs" / run_id
    
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    
    feature_path = run_dir / "feature_cols.json"
    meta_path = run_dir / "meta.json"
    model_path = run_dir / "model.txt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    feature_cols_payload = _load_json(feature_path)
    meta = _load_json(meta_path)
    
    feature_cols = feature_cols_payload.get("feature_cols") or meta.get("feature_cols")
    if not feature_cols:
        raise ValueError("feature_cols missing from artifacts.")
    
    model = lgb.Booster(model_file=str(model_path))
    
    return OwnershipBundle(
        model=model, 
        feature_cols=list(feature_cols), 
        meta=meta
    )


__all__ = ["OwnershipBundle", "load_ownership_bundle"]
