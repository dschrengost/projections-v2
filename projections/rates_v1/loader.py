from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb

from projections.paths import data_path


@dataclass
class RatesBundle:
    models: dict[str, lgb.Booster]
    feature_cols: list[str]
    meta: dict[str, Any]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rates_bundle(
    run_id: str, base_artifacts_root: Path | str | None = None
) -> RatesBundle:
    """
    Load models, feature_cols.json, and meta.json for a rates_v1 run.
    """
    root = Path(base_artifacts_root) if base_artifacts_root else data_path()
    run_dir = root / "artifacts" / "rates_v1" / "runs" / run_id
    feature_path = run_dir / "feature_cols.json"
    meta_path = run_dir / "meta.json"
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    feature_cols_payload = _load_json(feature_path)
    meta = _load_json(meta_path)
    feature_cols = feature_cols_payload.get("feature_cols") or meta.get("feature_cols")
    if not feature_cols:
        raise ValueError("feature_cols missing from artifacts.")
    targets = meta.get("targets")
    if not targets:
        raise ValueError("targets missing from meta.json.")
    models: dict[str, lgb.Booster] = {}
    for target in targets:
        model_path = run_dir / f"model_{target}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model for target={target} at {model_path}")
        models[target] = lgb.Booster(model_file=str(model_path))
    return RatesBundle(models=models, feature_cols=list(feature_cols), meta=meta)
