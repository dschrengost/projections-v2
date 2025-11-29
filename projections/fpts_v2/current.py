"""Loader for the current production fpts_v2 bundle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb


@dataclass
class FPTSBundle:
    model: lgb.Booster
    feature_cols: list[str]
    meta: dict[str, Any]
    metrics: dict[str, Any]


def _get_repo_root() -> Path:
    """Walk up from this file until we find the repo root (directory containing config/)."""

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root (no config/ directory found).")


def _get_data_root() -> Path:
    """Resolve PROJECTIONS_DATA_ROOT with a default of <repo_root>/../projections-data."""

    env_root = os.environ.get("PROJECTIONS_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (_get_repo_root() / ".." / "projections-data").resolve()


def _load_current_run_id(config_path: Optional[Path] = None) -> str:
    """Read config/fpts_current_run.json to obtain the active run_id."""

    config_file = (config_path or (_get_repo_root() / "config" / "fpts_current_run.json")).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"fpts_current_run.json not found at {config_file}")
    try:
        payload = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {config_file}: {exc}") from exc
    run_id = payload.get("run_id")
    if not run_id:
        raise RuntimeError(f"Missing 'run_id' in {config_file}")
    return str(run_id)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON at {path}: {exc}") from exc


def load_fpts_bundle(run_id: str, *, data_root: Optional[Path] = None) -> FPTSBundle:
    """
    Load a specific fpts_v2 bundle by run_id.

    Expects LightGBM model.txt plus feature_cols.json, meta.json, and (optionally) metrics.json
    under artifacts/fpts_v2/runs/<run_id> inside the data root.
    """

    base_dir = (data_root or _get_data_root()) / "artifacts" / "fpts_v2" / "runs" / run_id
    if not base_dir.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {base_dir}")

    model_path = base_dir / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing LightGBM model at {model_path}")
    model = lgb.Booster(model_file=str(model_path))

    feature_cols_path = base_dir / "feature_cols.json"
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing feature_cols.json at {feature_cols_path}")
    feature_payload = _read_json(feature_cols_path)
    feature_cols = feature_payload.get("feature_cols")
    if not feature_cols:
        raise RuntimeError(f"feature_cols.json missing 'feature_cols' at {feature_cols_path}")

    meta_path = base_dir / "meta.json"
    meta: dict[str, Any] = _read_json(meta_path) if meta_path.exists() else {}

    metrics_path = base_dir / "metrics.json"
    metrics: dict[str, Any] = _read_json(metrics_path) if metrics_path.exists() else {}

    return FPTSBundle(model=model, feature_cols=list(feature_cols), meta=meta, metrics=metrics)


def load_current_fpts_bundle(*, config_path: Optional[Path] = None, data_root: Optional[Path] = None) -> FPTSBundle:
    """
    Convenience helper: load the bundle pointed to by config/fpts_current_run.json.
    """

    run_id = _load_current_run_id(config_path=config_path)
    return load_fpts_bundle(run_id, data_root=data_root)


__all__ = [
    "FPTSBundle",
    "load_fpts_bundle",
    "load_current_fpts_bundle",
    "_load_current_run_id",
    "_get_data_root",
    "_get_repo_root",
]
