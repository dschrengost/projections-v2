"""Inference helpers for FPTS models.

This module supports legacy fpts_v1 models that predict fantasy points **per
minute** as well as fpts_v2 LightGBM bundles trained to predict **total DK
fantasy points per game** (target = ``dk_fpts_actual`` in training). The
``predict_fpts`` helper detects fpts_v2-style bundles and skips the extra
minutes multiplication so totals are not double-scaled.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import json
import joblib
import pandas as pd
import lightgbm as lgb
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
    joblib_path = run_dir / "model.joblib"
    if joblib_path.exists():
        payload = joblib.load(joblib_path)
        return FptsModelBundle(
            model=payload["model"],
            imputer=payload["imputer"],
            feature_columns=payload["feature_columns"],
            metadata=payload.get("metadata"),
        )

    # Fallback: fpts_v2-style LightGBM bundle (model.txt + feature_cols.json/meta.json)
    txt_path = run_dir / "model.txt"
    if txt_path.exists():
        model = lgb.Booster(model_file=str(txt_path))
        meta_path = run_dir / "meta.json"
        feature_cols_path = run_dir / "feature_cols.json"
        metadata = {}
        feature_columns: list[str] = []
        try:
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        try:
            if feature_cols_path.exists():
                payload = json.loads(feature_cols_path.read_text(encoding="utf-8"))
                feature_columns = list(payload.get("feature_cols") or [])
        except Exception:
            feature_columns = []
        if not feature_columns:
            feature_columns = list(metadata.get("feature_cols") or [])
        if not feature_columns:
            raise FileNotFoundError(
                f"feature columns not found for bundle {run_id} under {run_dir}"
            )

        class _PassthroughImputer:
            def transform(self, X):
                df = pd.DataFrame(X, columns=feature_columns)
                return df.fillna(0).to_numpy()

        return FptsModelBundle(
            model=model,
            imputer=_PassthroughImputer(),
            feature_columns=feature_columns,
            metadata=metadata,
        )

    raise FileNotFoundError(
        f"Run directory {run_dir} is missing both model.joblib and model.txt"
    )


def _prepare_features(bundle: FptsModelBundle, features: pd.DataFrame) -> np.ndarray:
    missing = [col for col in bundle.feature_columns if col not in features.columns]
    if missing:
        # Backward-compat: tolerate missing engineered features by filling zeros.
        features = features.copy()
        for col in missing:
            features[col] = 0.0
    matrix = features[bundle.feature_columns]
    return bundle.imputer.transform(matrix)


def predict_fpts_per_min(bundle: FptsModelBundle, features: pd.DataFrame) -> pd.Series:
    """Return per-minute FPTS predictions for the provided slate dataframe."""

    transformed = _prepare_features(bundle, features)
    num_iter = getattr(bundle.model, "best_iteration", None)
    num_iter = num_iter if num_iter and num_iter > 0 else None
    preds = bundle.model.predict(transformed, num_iteration=num_iter)
    return pd.Series(preds, index=features.index, name="fpts_per_min_pred")


def _predict_total_fpts(bundle: FptsModelBundle, features: pd.DataFrame) -> pd.Series:
    """Return total FPTS predictions for bundles trained on total DK FPTS."""

    transformed = _prepare_features(bundle, features)
    num_iter = getattr(bundle.model, "best_iteration", None)
    num_iter = num_iter if num_iter and num_iter > 0 else None
    preds = bundle.model.predict(transformed, num_iteration=num_iter)
    return pd.Series(preds, index=features.index, name="proj_fpts")


def _is_total_fpts_bundle(bundle: FptsModelBundle) -> bool:
    """Heuristic to detect fpts_v2 bundles that output total FPTS.

    Current fpts_v2 meta.json uses run_tag like ``fpts_v2_*`` and the training
    target is ``dk_fpts_actual`` (total fantasy points). Any bundle whose
    metadata/run_id/run_tag contains ``fpts_v2`` is treated as total-FPTS.
    """

    meta = bundle.metadata or {}
    tag = str(meta.get("run_tag") or meta.get("run_id") or "").lower()
    return "fpts_v2" in tag


def predict_fpts(
    bundle: FptsModelBundle,
    slate_df: pd.DataFrame,
    *,
    minutes_col: str = "minutes_p50",
) -> pd.DataFrame:
    """Predict fantasy points for a slate DataFrame.

    - For legacy fpts_v1 bundles (per-minute head), total = per_min * minutes.
    - For fpts_v2 bundles (total-FPTS head), total comes directly from the
      model and per-minute is derived as total / minutes when available.
    """

    minutes = pd.to_numeric(slate_df.get(minutes_col), errors="coerce").fillna(0.0)
    if _is_total_fpts_bundle(bundle):
        total = _predict_total_fpts(bundle, slate_df)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_min = total / minutes.replace({0.0: np.nan})
        per_min = per_min.fillna(0.0)
    else:
        per_min = predict_fpts_per_min(bundle, slate_df)
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
    """
    Load the run marked as production in config/fpts_current_run.json.

    The repo default currently pins production to the stage0 total-FPTS run
    (fpts_v2_stage0_20251129_062655) via that config file; override via env or
    a different config path if needed.
    """

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
