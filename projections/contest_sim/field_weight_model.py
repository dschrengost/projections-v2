"""Field-lineup popularity model calibrated from historical DK results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from projections.paths import data_path

__all__ = [
    "FieldWeightModel",
    "FieldWeightModelBucket",
    "default_field_weight_model_path",
    "load_field_weight_model",
    "save_field_weight_model",
]


@dataclass(frozen=True)
class FieldWeightModelBucket:
    """A simple linear model on standardized features.

    log_weight = intercept + dot(coef, (x - mean) / std)
    """

    intercept: float
    coef: Dict[str, float]
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]

    def predict_log_weight(self, features: Mapping[str, float]) -> float:
        total = float(self.intercept)
        for name, w in self.coef.items():
            x = float(features.get(name, 0.0))
            mean = float(self.feature_means.get(name, 0.0))
            std = float(self.feature_stds.get(name, 1.0) or 1.0)
            z = (x - mean) / std
            total += float(w) * z
        return float(total)

    def predict_weight(self, features: Mapping[str, float], *, clamp_exp: float = 20.0) -> float:
        """Return exp(log_weight) with a safety clamp."""
        log_w = self.predict_log_weight(features)
        log_w = float(np.clip(log_w, -clamp_exp, clamp_exp))
        return float(math.exp(log_w))


@dataclass(frozen=True)
class FieldWeightModel:
    """Container for bucketed lineup popularity models."""

    buckets: Dict[str, FieldWeightModelBucket]
    meta: Dict[str, Any]

    def get_bucket(self, bucket: str) -> FieldWeightModelBucket:
        if bucket in self.buckets:
            return self.buckets[bucket]
        # Fall back to the first bucket deterministically.
        if self.buckets:
            key = sorted(self.buckets.keys())[0]
            return self.buckets[key]
        raise ValueError("FieldWeightModel has no buckets")

    def predict_weights(
        self,
        features_list: Iterable[Mapping[str, float]],
        *,
        bucket: str,
        clamp_exp: float = 20.0,
    ) -> list[float]:
        model = self.get_bucket(bucket)
        return [model.predict_weight(feats, clamp_exp=clamp_exp) for feats in features_list]


def default_field_weight_model_path(version: str = "v1") -> Path:
    """Default learned model path under projections-data."""
    return data_path("gold", f"field_weight_model_{version}.json")


def save_field_weight_model(model: FieldWeightModel, path: Path) -> None:
    payload = {
        "_meta": {
            **model.meta,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        "buckets": {
            name: {
                "intercept": bucket.intercept,
                "coef": bucket.coef,
                "feature_means": bucket.feature_means,
                "feature_stds": bucket.feature_stds,
            }
            for name, bucket in model.buckets.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_field_weight_model(path: Optional[Path] = None, *, version: str = "v1") -> FieldWeightModel:
    model_path = path or default_field_weight_model_path(version=version)
    with open(model_path) as f:
        payload = json.load(f)
    meta = payload.get("_meta", {})
    buckets_raw = payload.get("buckets", {})
    buckets: Dict[str, FieldWeightModelBucket] = {}
    for name, data in buckets_raw.items():
        buckets[name] = FieldWeightModelBucket(
            intercept=float(data.get("intercept", 0.0)),
            coef={k: float(v) for k, v in (data.get("coef") or {}).items()},
            feature_means={k: float(v) for k, v in (data.get("feature_means") or {}).items()},
            feature_stds={k: float(v) for k, v in (data.get("feature_stds") or {}).items()},
        )
    return FieldWeightModel(buckets=buckets, meta=dict(meta))

