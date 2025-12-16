from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb


def _resolve_data_root(data_root: Optional[Path | str] = None) -> Path:
    if data_root:
        return Path(data_root).expanduser().resolve()
    env_root = os.environ.get("PROJECTIONS_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path.cwd() / "data").resolve()


@dataclass
class FptsBundle:
    model: lgb.Booster
    feature_cols: list[str]
    meta: dict[str, Any]


@dataclass
class ResidualBucket:
    name: str
    is_starter: int
    min_minutes: float
    max_minutes: float | None
    sigma: float
    nu: int
    n: int


@dataclass
class ResidualModel:
    buckets: list[ResidualBucket]
    sigma_default: float
    nu_default: int


def load_fpts_bundle(run_id: str, *, data_root: Optional[Path | str] = None) -> FptsBundle:
    root = _resolve_data_root(data_root)
    base_dir = root / "artifacts" / "fpts_v2" / "runs" / run_id
    if not base_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {base_dir}")

    model_path = base_dir / "model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing LightGBM model at {model_path}")
    model = lgb.Booster(model_file=str(model_path))

    meta_path = base_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    feature_cols_path = base_dir / "feature_cols.json"
    feature_cols: list[str] = []
    if feature_cols_path.exists():
        payload = json.loads(feature_cols_path.read_text(encoding="utf-8"))
        feature_cols = list(payload.get("feature_cols") or [])
    if not feature_cols:
        feature_cols = list(meta.get("feature_cols") or [])
    if not feature_cols:
        raise RuntimeError("feature_cols not found in feature_cols.json or meta.json")

    return FptsBundle(model=model, feature_cols=feature_cols, meta=meta)


def load_residual_model(run_id: str, *, data_root: Optional[Path | str] = None) -> ResidualModel:
    root = _resolve_data_root(data_root)
    residual_path = root / "artifacts" / "fpts_v2" / "runs" / run_id / "residual_model.json"
    if not residual_path.exists():
        raise FileNotFoundError(f"Residual model not found at {residual_path}")
    data = json.loads(residual_path.read_text(encoding="utf-8"))
    buckets_raw = data.get("buckets") or []
    if not buckets_raw:
        raise RuntimeError("Residual model has no buckets")
    buckets: list[ResidualBucket] = []
    for item in buckets_raw:
        min_minutes = float(item["min_minutes"])
        max_minutes = float(item["max_minutes"]) if item.get("max_minutes") is not None else None
        if max_minutes is not None and min_minutes >= max_minutes:
            raise RuntimeError(f"Invalid bucket minutes range for {item}")
        nu = int(item.get("nu", 0))
        if nu <= 0:
            raise RuntimeError(f"Invalid nu for bucket {item}")
        buckets.append(
            ResidualBucket(
                name=item["name"],
                is_starter=int(item["is_starter"]),
                min_minutes=min_minutes,
                max_minutes=max_minutes,
                sigma=float(item["sigma"]),
                nu=nu,
                n=int(item.get("n", 0)),
            )
        )
    sigma_default = float(data.get("sigma_default", 0.0))
    nu_default = int(data.get("nu_default", 0))
    if nu_default <= 0:
        raise RuntimeError("Invalid nu_default in residual model")
    return ResidualModel(buckets=buckets, sigma_default=sigma_default, nu_default=nu_default)


def load_fpts_and_residual(
    run_id: str, *, data_root: Optional[Path | str] = None
) -> tuple[FptsBundle, ResidualModel]:
    bundle = load_fpts_bundle(run_id, data_root=data_root)
    residual = load_residual_model(run_id, data_root=data_root)
    return bundle, residual


__all__ = [
    "FptsBundle",
    "ResidualBucket",
    "ResidualModel",
    "load_fpts_bundle",
    "load_residual_model",
    "load_fpts_and_residual",
]
