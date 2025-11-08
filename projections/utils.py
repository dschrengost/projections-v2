"""Utility helpers shared across the projections pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataPaths:
    """Container for important data directories used by the pipeline."""

    raw: Path
    external: Path
    interim: Path
    processed: Path


def load_yaml_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file (defaults to config/settings.yaml)."""

    config_path = Path(path) if path else PROJECT_ROOT / "config" / "settings.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file missing at {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_data_paths(cfg: Mapping[str, Any]) -> DataPaths:
    """Build strongly-typed data paths from a parsed config mapping."""

    data_root = Path(cfg.get("data_dir", PROJECT_ROOT / "data")).resolve()
    return DataPaths(
        raw=(Path(cfg.get("data", {}).get("raw_dir", data_root / "raw"))).resolve(),
        external=(
            Path(cfg.get("data", {}).get("external_dir", data_root / "external"))
        ).resolve(),
        interim=(
            Path(cfg.get("data", {}).get("interim_dir", data_root / "interim"))
        ).resolve(),
        processed=(
            Path(cfg.get("data", {}).get("processed_dir", data_root / "processed"))
        ).resolve(),
    )


def ensure_directory(path: Path) -> Path:
    """Create the directory if necessary and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_path(root: Path, stem: str, suffix: str = ".csv") -> Path:
    """Create a timestamped file path helper."""

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return root / f"{stem}_{ts}{suffix}"


def set_seeds(seed: int) -> None:
    """Set deterministic seeds across common ML libraries."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - optional dependency in tests
        pass
