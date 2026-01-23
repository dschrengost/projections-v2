from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch

from projections.models.minutes_features import MinutesFeatureSpec
from projections.models.minutes_nn import MinutesPreprocessorState

from .model import RotationMinutesHurdleMLP

logger = logging.getLogger(__name__)


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_git_sha_short() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()[:12]


def compute_schema_hash(spec: MinutesFeatureSpec) -> str:
    payload = {"continuous": spec.continuous, "categorical": spec.categorical}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return sha256(blob).hexdigest()


@dataclass(frozen=True)
class RMHBundle:
    model: RotationMinutesHurdleMLP
    feature_spec: MinutesFeatureSpec
    preprocessor: MinutesPreprocessorState
    schema_hash: str
    config: dict[str, Any]
    metrics: dict[str, Any]


def save_bundle(
    run_dir: Path,
    *,
    model: RotationMinutesHurdleMLP,
    feature_spec: MinutesFeatureSpec,
    preprocessor: MinutesPreprocessorState,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    schema_hash = compute_schema_hash(feature_spec)
    meta = {
        "git_sha": _get_git_sha_short(),
        "schema_hash": schema_hash,
        "feature_spec": feature_spec.to_metadata(),
        "model": {
            "class": "RotationMinutesHurdleMLP",
            # Needed to reconstruct the module at inference time.
            "n_continuous": len(feature_spec.continuous),
            "cat_cols": feature_spec.categorical,
            "cat_cardinalities": [
                len(preprocessor.categorical[col]) + 1
                for col in feature_spec.categorical
            ],
            "emb_dim": int(config.get("emb_dim", 16)),
            "hidden_dims": list(config.get("hidden_dims", [])),
            "dropout": float(config.get("dropout", 0.0)),
        },
    }

    _json_dump(run_dir / "config.json", config)
    _json_dump(
        run_dir / "schema.json",
        {"schema_hash": schema_hash, "feature_spec": feature_spec.to_metadata()},
    )
    _json_dump(run_dir / "preprocessor.json", preprocessor.to_json_dict())
    _json_dump(run_dir / "metrics.json", metrics)
    _json_dump(run_dir / "meta.json", meta)

    torch.save({"state_dict": model.state_dict()}, run_dir / "model.pt")


def load_bundle(run_dir: Path, *, map_location: str | None = None) -> RMHBundle:
    config = _json_load(run_dir / "config.json")
    schema_payload = _json_load(run_dir / "schema.json")
    preprocessor_payload = _json_load(run_dir / "preprocessor.json")
    meta = _json_load(run_dir / "meta.json")
    metrics = _json_load(run_dir / "metrics.json")

    feature_spec = MinutesFeatureSpec(
        continuous=list(schema_payload["feature_spec"]["continuous"]),
        categorical=list(schema_payload["feature_spec"]["categorical"]),
    )
    preprocessor = MinutesPreprocessorState.from_json_dict(preprocessor_payload)

    checkpoint = torch.load(run_dir / "model.pt", map_location=map_location or "cpu")
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(
            f"missing state_dict in RMH bundle checkpoint: {run_dir / 'model.pt'}"
        )
    if "delta_head.weight" not in state_dict:
        raise KeyError("missing delta_head.weight in RMH bundle state_dict")
    delta_weight = state_dict["delta_head.weight"]
    if not hasattr(delta_weight, "shape"):
        raise TypeError(
            "RMH bundle delta_head.weight is not a tensor; cannot infer delta_out."
        )
    delta_out = int(delta_weight.shape[0])
    if delta_out == 2:
        logger.info(
            "[rmh] legacy delta_out=2 checkpoint detected; "
            "mapping to v1.1 quantiles via linear interpolation between q10/q50/q90."
        )

    model_cfg = meta["model"]
    model = RotationMinutesHurdleMLP(
        n_continuous=int(model_cfg["n_continuous"]),
        cat_cardinalities=[int(x) for x in model_cfg.get("cat_cardinalities", [])],
        emb_dim=int(model_cfg["emb_dim"]),
        hidden_dims=[int(x) for x in model_cfg.get("hidden_dims", [])],
        dropout=float(model_cfg["dropout"]),
        delta_out=delta_out,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return RMHBundle(
        model=model,
        feature_spec=feature_spec,
        preprocessor=preprocessor,
        schema_hash=str(schema_payload["schema_hash"]),
        config=config,
        metrics=metrics,
    )


__all__ = [
    "RMHBundle",
    "compute_schema_hash",
    "load_bundle",
    "save_bundle",
]
