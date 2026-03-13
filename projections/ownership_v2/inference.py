from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from projections.ownership_v2.slate_transformer import (
    OwnershipSlateTransformer,
    OwnershipSlateTransformerConfig,
)
from projections.paths import data_path


@dataclass(frozen=True)
class OwnershipSlateTransformerBundle:
    """Loaded ownership_v2 artifact bundle."""

    model: OwnershipSlateTransformer
    config: OwnershipSlateTransformerConfig
    feature_columns: list[str]
    meta: dict[str, Any]
    run_dir: Path


def _load_feature_columns(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        cols = payload
    elif isinstance(payload, dict):
        cols = payload.get("columns")
    else:
        cols = None
    if not isinstance(cols, list) or not cols:
        raise ValueError(f"invalid feature column payload: {path}")
    return [str(col) for col in cols]


def load_ownership_transformer_bundle(
    run_id: str,
    *,
    base_artifacts_root: Path | str | None = None,
    map_location: str | torch.device = "cpu",
) -> OwnershipSlateTransformerBundle:
    """Load ownership_v2 transformer artifacts for live inference."""

    root = Path(base_artifacts_root) if base_artifacts_root else data_path()
    run_dir = root / "artifacts" / "ownership_v2" / "runs" / str(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    config_path = run_dir / "config.json"
    model_path = run_dir / "model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"missing ownership_v2 config: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"missing ownership_v2 model: {model_path}")

    config = OwnershipSlateTransformerConfig.load(config_path)
    feature_cols = _load_feature_columns(run_dir / "feature_columns.json")
    if feature_cols is not None and feature_cols != list(config.feature_columns):
        raise ValueError(
            "feature_columns.json does not match config.feature_columns: "
            f"run_dir={run_dir}"
        )
    feature_columns = list(config.feature_columns)
    if not feature_columns:
        raise ValueError(f"ownership_v2 config has empty feature_columns: {config_path}")
    if len(config.feature_mean) != len(feature_columns) or len(config.feature_std) != len(feature_columns):
        raise ValueError(
            "feature_mean/std length mismatch vs feature_columns in ownership_v2 config: "
            f"run_dir={run_dir}"
        )

    model = OwnershipSlateTransformer(config)
    state = torch.load(model_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state_dict = state["state_dict"]
    elif isinstance(state, dict):
        state_dict = state
    else:
        raise ValueError(f"unrecognized checkpoint format at {model_path}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = meta_payload if isinstance(meta_payload, dict) else {}
    else:
        meta = {}

    return OwnershipSlateTransformerBundle(
        model=model,
        config=config,
        feature_columns=feature_columns,
        meta=meta,
        run_dir=run_dir,
    )


def predict_ownership_transformer_slate(
    frame: pd.DataFrame,
    *,
    bundle: OwnershipSlateTransformerBundle,
) -> pd.DataFrame:
    """Score one slate with ownership_v2 and return pct + raw diagnostics."""

    if frame.empty:
        raise ValueError("frame must be non-empty")

    missing = [col for col in bundle.feature_columns if col not in frame.columns]
    if missing:
        raise KeyError(f"missing required ownership_v2 features: {missing}")

    raw = (
        frame[bundle.feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )
    if raw.ndim != 2:
        raise ValueError("expected a 2D feature matrix")

    mean = np.asarray(bundle.config.feature_mean, dtype=np.float32)
    std = np.asarray(bundle.config.feature_std, dtype=np.float32)
    safe_std = np.where(std <= 1e-6, 1.0, std).astype(np.float32)
    standardized = (raw - mean[None, :]) / safe_std[None, :]

    x = torch.from_numpy(standardized).unsqueeze(0)
    valid_mask = torch.ones((1, standardized.shape[0]), dtype=torch.bool)
    with torch.no_grad():
        out = bundle.model(x, valid_mask)

    pred_pct = out["pred_pct"].squeeze(0).detach().cpu().numpy().astype(float)
    probs = out["probs"].squeeze(0).detach().cpu().numpy().astype(float)
    logits = out["logits"].squeeze(0).detach().cpu().numpy().astype(float)

    return pd.DataFrame(
        {
            "pred_own_pct": pred_pct,
            # Share-space probability before target-sum scaling.
            "pred_own_pct_raw": probs * 100.0,
            "pred_own_logit": logits,
        },
        index=frame.index,
    )

