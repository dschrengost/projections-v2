from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, Field, field_validator


class RMHConfig(BaseModel):
    """Configuration for RMH_v1 training/eval/inference.

    Required fields are kept intentionally small; anything not specified in the
    design doc is a tunable convenience.
    """

    # --- Data / IO ---
    train_dataset_dir: Path = Field(..., description="Directory containing features.parquet and labels.parquet.")
    val_dataset_dir: Path | None = Field(
        None, description="Optional dataset directory for evaluation/quick validation."
    )
    artifact_dir: Path = Field(Path("artifacts/rotation_minutes_hurdle_v1"), description="Artifact root directory.")
    run_id: str | None = Field(None, description="Run id (defaults to timestamp in train CLI).")
    limit_rows: int | None = Field(None, description="Optional row limit (dev/smoke only).")

    # --- Targets ---
    play_threshold: float = Field(5.0, description="T: y_play = 1[minutes >= T]. v1.1 default is 5 (rotation threshold).")

    # --- Quantiles (v1.1 expands to 7 quantiles) ---
    quantiles: list[float] = Field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        description="Conditional quantiles for minutes head. Must be sorted ascending and include 0.50.",
    )
    quantile_weights: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        description="Weights for pinball loss per quantile. Optional tail upweighting (e.g., 1.2× on q05/q95).",
    )

    # --- Recency weighting ---
    half_life_play_days: float = Field(30.0, description="Half-life in days for play head weights.")
    half_life_minutes_days: float = Field(30.0, description="Half-life in days for minutes head weights.")

    # --- Model / optimizer ---
    emb_dim: int = 16
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256])
    dropout: float = 0.10
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 25
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # --- Loss balancing (dimensionless) ---
    play_loss_weight: float = 1.0
    minutes_loss_weight: float = 1.0

    @field_validator("quantiles")
    @classmethod
    def _validate_quantiles(cls, v: list[float]) -> list[float]:
        if sorted(v) != list(v):
            raise ValueError("quantiles must be sorted ascending.")
        if any(q <= 0.0 or q >= 1.0 for q in v):
            raise ValueError("quantiles must all be in (0, 1).")
        if 0.50 not in [round(x, 2) for x in v]:
            raise ValueError("quantiles must include 0.50 (median) for non-crossing parameterization.")
        return v

    @field_validator("quantile_weights")
    @classmethod
    def _validate_quantile_weights(cls, v: list[float], info) -> list[float]:
        quantiles = info.data.get("quantiles") or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        if len(v) != len(quantiles):
            raise ValueError("quantile_weights length must match quantiles length.")
        if any(w <= 0 for w in v):
            raise ValueError("quantile_weights must all be > 0.")
        return v

    @field_validator("play_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("play_threshold must be > 0 (T=1 is default).")
        return v

    @field_validator("half_life_play_days", "half_life_minutes_days")
    @classmethod
    def _validate_half_life(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("half-life must be > 0 days.")
        return v

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def to_json_dict(self) -> dict[str, Any]:
        # Pydantic uses Path objects; serialize to strings for readability.
        payload = self.model_dump()
        for key in ("train_dataset_dir", "val_dataset_dir", "artifact_dir"):
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
        payload["device_resolved"] = self.resolved_device()
        return payload


def load_config(path: Path) -> RMHConfig:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw) or {}
    elif path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        raise ValueError(f"Unsupported config file extension: {path}")
    return RMHConfig.model_validate(payload)


__all__ = ["RMHConfig", "load_config"]

