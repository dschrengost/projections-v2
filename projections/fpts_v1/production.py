"""Inference helpers for the fantasy points per minute model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FptsModelBundle:
    """Serialized artifacts required for inference."""

    model: Any
    imputer: Any
    feature_columns: Sequence[str]
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ProductionFptsBundle:
    """Resolved production bundle + metadata."""

    bundle: FptsModelBundle
    run_dir: Path
    run_id: str
    scoring_system: str


DEFAULT_PRODUCTION_RUN_ID = "fpts_lgbm_v0"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/fpts_lgbm")
DEFAULT_PRODUCTION_CONFIG = Path("config/fpts_current_run.json")
ENV_RUN_ID = "FPTS_PRODUCTION_RUN_ID"
ENV_RUN_DIR = "FPTS_PRODUCTION_DIR"
ENV_CONFIG_PATH = "FPTS_PRODUCTION_CONFIG"
ENV_SCORING_SYSTEM = "FPTS_PRODUCTION_SCORING"


def _expand(path: Path) -> Path:
    return path.expanduser().resolve()


def _run_dir(run_id: str, artifact_root: Path) -> Path:
    run_dir = (artifact_root / run_id).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def load_fpts_model(
    run_id: str,
    *,
    artifact_root: Path | None = None,
    bundle_dir: Path | None = None,
) -> FptsModelBundle:
    """Load a trained FPTS-per-minute LightGBM bundle."""

    if bundle_dir is not None:
        run_dir = bundle_dir.expanduser().resolve()
    else:
        root = artifact_root or DEFAULT_ARTIFACT_ROOT
        run_dir = _run_dir(run_id, root)
    payload = joblib.load(run_dir / "model.joblib")
    return FptsModelBundle(
        model=payload["model"],
        imputer=payload["imputer"],
        feature_columns=payload["feature_columns"],
        metadata=payload.get("metadata"),
    )


def _prepare_features(bundle: FptsModelBundle, features: pd.DataFrame) -> np.ndarray:
    missing = [col for col in bundle.feature_columns if col not in features.columns]
    if missing:
        raise KeyError(
            f"Feature frame missing required columns: {', '.join(sorted(missing))}"
        )
    matrix = features[bundle.feature_columns]
    return bundle.imputer.transform(matrix)


def predict_fpts_per_min(bundle: FptsModelBundle, features: pd.DataFrame) -> pd.Series:
    """Return per-minute FPTS predictions for the provided slate dataframe."""

    transformed = _prepare_features(bundle, features)
    preds = bundle.model.predict(transformed)
    return pd.Series(preds, index=features.index, name="fpts_per_min_pred")


def predict_fpts(
    bundle: FptsModelBundle,
    slate_df: pd.DataFrame,
    *,
    minutes_col: str = "minutes_p50",
) -> pd.DataFrame:
    """Predict per-minute + total fantasy points for a slate DataFrame."""

    per_min = predict_fpts_per_min(bundle, slate_df)
    minutes = pd.to_numeric(slate_df.get(minutes_col), errors="coerce").fillna(0.0)
    total = per_min * minutes
    return pd.DataFrame(
        {
            "fpts_per_min_pred": per_min,
            "proj_fpts": total,
        },
        index=slate_df.index,
    )


def _resolve_production_bundle(
    config_path: Path | None = None,
) -> tuple[Path, str, str]:
    env_dir = os.environ.get(ENV_RUN_DIR)
    env_run = os.environ.get(ENV_RUN_ID)
    env_scoring = (os.environ.get(ENV_SCORING_SYSTEM) or "dk").lower()
    if env_dir:
        run_dir = _expand(Path(env_dir))
        return run_dir, env_run or run_dir.name, env_scoring

    candidate = os.environ.get(ENV_CONFIG_PATH)
    config_file = (
        Path(candidate).expanduser()
        if candidate
        else (config_path or DEFAULT_PRODUCTION_CONFIG)
    )
    config_file = _expand(config_file)
    if config_file.exists():
        try:
            payload = json.loads(config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - operator error
            raise RuntimeError(f"Invalid production config JSON at {config_file}: {exc}") from exc
        run_id = str(payload.get("run_id") or env_run or DEFAULT_PRODUCTION_RUN_ID)
        scoring = str(payload.get("scoring_system") or env_scoring or "dk").lower()
        bundle_dir = payload.get("bundle_dir")
        artifact_root = payload.get("artifact_root")
        if bundle_dir:
            run_dir = Path(bundle_dir)
            if not run_dir.is_absolute():
                run_dir = _expand(run_dir)
            else:
                run_dir = _expand(run_dir)
            if payload.get("run_id") is None and env_run is None:
                run_id = run_dir.name
        else:
            root = _expand(Path(artifact_root)) if artifact_root else _expand(DEFAULT_ARTIFACT_ROOT)
            run_dir = root / run_id
        return run_dir, run_id, scoring

    default_root = _expand(DEFAULT_ARTIFACT_ROOT)
    run_id = env_run or DEFAULT_PRODUCTION_RUN_ID
    return default_root / run_id, run_id, env_scoring


@lru_cache(maxsize=1)
def load_production_fpts_bundle(
    *,
    config_path: Path | None = None,
) -> ProductionFptsBundle:
    """Load the run marked as production in config/fpts_current_run.json."""

    run_dir, run_id, scoring = _resolve_production_bundle(config_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"Production FPTS bundle missing at {run_dir}")
    bundle = load_fpts_model(run_id, bundle_dir=run_dir)
    metadata = bundle.metadata or {}
    resolved_run_id = str(metadata.get("run_id") or run_id)
    resolved_scoring = str(metadata.get("scoring_system") or scoring)
    return ProductionFptsBundle(
        bundle=bundle,
        run_dir=run_dir,
        run_id=resolved_run_id,
        scoring_system=resolved_scoring,
    )


__all__ = [
    "FptsModelBundle",
    "ProductionFptsBundle",
    "load_fpts_model",
    "load_production_fpts_bundle",
    "predict_fpts",
    "predict_fpts_per_min",
]
