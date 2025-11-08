"""Data loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import DataPaths, ensure_directory


def load_raw_minutes(paths: DataPaths, source_filename: str) -> pd.DataFrame:
    """Load the immutable raw data file.

    Parameters
    ----------
    paths:
        Dataclass containing important data directories.
    source_filename:
        CSV file name stored in ``data/raw`` (or overridden via config).
    """

    file_path = paths.raw / source_filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"Expected raw data at {file_path}. "
            "Download or copy the dataset into data/raw first."
        )
    return pd.read_csv(file_path)


def clean_minutes(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Basic cleaning shared across experiments.

    Parameters
    ----------
    df:
        Raw minutes dataframe.
    columns:
        Optional list of columns to retain. When omitted the original schema
        is preserved aside from duplicates.
    """

    cleaned = df.drop_duplicates().copy()
    cleaned.columns = [col.strip().lower().replace(" ", "_") for col in cleaned.columns]
    if columns:
        missing = set(columns) - set(cleaned.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {', '.join(sorted(missing))}")
        cleaned = cleaned[list(columns)]
    return cleaned


def write_interim(df: pd.DataFrame, paths: DataPaths, filename: str) -> Path:
    """Persist an interim dataset for faster iteration."""

    dest_dir = ensure_directory(paths.interim)
    target = dest_dir / filename
    df.to_csv(target, index=False)
    return target


def load_interim(paths: DataPaths, filename: str) -> pd.DataFrame:
    """Load an interim CSV file from the configured directory."""

    file_path = paths.interim / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Interim dataset missing at {file_path}")
    return pd.read_csv(file_path)
