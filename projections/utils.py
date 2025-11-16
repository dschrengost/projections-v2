"""Utility helpers shared across the projections pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from projections import paths

PROJECT_ROOT = paths.get_project_root()


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

    data_root = Path(cfg.get("data_dir", paths.get_data_root())).resolve()
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


def asof_left_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: Sequence[str],
    left_time_col: str,
    right_time_col: str,
    direction: str = "backward",
    tolerance: str | pd.Timedelta | None = None,
    suffixes: tuple[str, str] = ("", "_ref"),
) -> pd.DataFrame:
    """Merge the latest ``right`` row (by timestamp) into ``left`` per key group.

    Parameters
    ----------
    left, right:
        Dataframes to merge. ``left`` retains all rows; ``right`` is searched
        for the latest snapshot per key whose timestamp is <= the left timestamp.
    on:
        Columns that must match exactly between ``left`` and ``right``.
    left_time_col / right_time_col:
        Timestamp columns used for the as-of comparison. Both are coerced to UTC.
    direction:
        Merge direction for :func:`pandas.merge_asof` (defaults to ``backward``).
    tolerance:
        Optional maximum allowed time delta between ``left`` and ``right`` rows.
        Accepts pandas Timedelta, offset string (e.g., ``"2h"``), or ``None``.
    suffixes:
        Column suffixes applied when ``right`` introduces overlapping names.

    Returns
    -------
    pandas.DataFrame
        ``left`` dataframe with ``right`` columns appended.
    """

    if not on:
        raise ValueError("Parameter 'on' must include at least one column name")
    for column in on:
        if column not in left.columns:
            raise ValueError(f"Left dataframe missing join column '{column}'")
        if column not in right.columns:
            raise ValueError(f"Right dataframe missing join column '{column}'")
    if left_time_col not in left.columns:
        raise ValueError(f"Left dataframe missing time column '{left_time_col}'")
    if right_time_col not in right.columns:
        raise ValueError(f"Right dataframe missing time column '{right_time_col}'")

    left_work = left.copy()
    right_work = right.copy()

    def _coerce_timestamp(df: pd.DataFrame, column: str) -> pd.Series:
        series = pd.to_datetime(df[column], utc=True, errors="coerce")
        if series.isna().any():
            raise ValueError(f"Column '{column}' contains invalid timestamps")
        return series

    left_work[left_time_col] = _coerce_timestamp(left_work, left_time_col)
    right_work[right_time_col] = _coerce_timestamp(right_work, right_time_col)

    base_sort_left = list(on) + [left_time_col]
    base_sort_right = list(on) + [right_time_col]
    left_work = left_work.reset_index(drop=True)
    left_work["__asof_idx"] = np.arange(len(left_work))
    left_sorted = left_work.sort_values(base_sort_left, kind="mergesort")
    left_sorted = left_sorted.sort_values(left_time_col, kind="mergesort")
    right_sorted = right_work.sort_values(base_sort_right, kind="mergesort")
    right_sorted = right_sorted.sort_values(right_time_col, kind="mergesort")

    tol_value: pd.Timedelta | None
    if tolerance is None:
        tol_value = None
    elif isinstance(tolerance, pd.Timedelta):
        tol_value = tolerance
    else:
        tol_value = pd.to_timedelta(tolerance)

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        by=list(on),
        left_on=left_time_col,
        right_on=right_time_col,
        direction=direction,
        tolerance=tol_value,
        suffixes=suffixes,
    )
    merged = merged.sort_values("__asof_idx").drop(columns="__asof_idx")
    merged.index = left.index
    return merged
