"""Tests for data layer helpers."""

from pathlib import Path

import pandas as pd

from projections import data
from projections.utils import DataPaths, ensure_directory


def _paths(tmp_path: Path) -> DataPaths:
    return DataPaths(
        raw=ensure_directory(tmp_path / "raw"),
        external=ensure_directory(tmp_path / "external"),
        interim=ensure_directory(tmp_path / "interim"),
        processed=ensure_directory(tmp_path / "processed"),
    )


def test_clean_minutes_normalizes_column_names():
    df = pd.DataFrame(
        {"Player Name": ["A", "A"], "Minutes ": [30, 30], "Game Date": ["2023-01-01", "2023-01-01"]}
    )
    cleaned = data.clean_minutes(df, columns=["player_name", "minutes", "game_date"])
    assert list(cleaned.columns) == ["player_name", "minutes", "game_date"]
    assert len(cleaned) == 1  # duplicates removed


def test_write_interim_persists_file(tmp_path):
    paths = _paths(tmp_path)
    df = pd.DataFrame({"player_id": [1], "minutes": [30]})
    target = data.write_interim(df, paths, "sample.csv")
    assert target.exists()
