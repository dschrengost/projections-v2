from __future__ import annotations

from pathlib import Path

import pandas as pd

import projections.cli.build_minutes_live as build_minutes_live


def test_load_label_sources_api_exists() -> None:
    assert hasattr(build_minutes_live, "_load_label_sources")


def test_load_label_sources_override_path_returns_expected_shape(tmp_path: Path, monkeypatch) -> None:
    required_cols = [
        "game_id",
        "player_id",
        "team_id",
        "player_name",
        "season",
        "game_date",
        "minutes",
        "starter_flag",
        "starter_flag_label",
        "source",
        "label_frozen_ts",
    ]
    labels = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [101],
            "team_id": [1610612751],
            "player_name": ["Test Player"],
            "season": ["2025-26"],
            "game_date": ["2025-12-15"],
            "minutes": [12.0],
            "starter_flag": [1],
            "starter_flag_label": [1],
            "source": ["test"],
            "label_frozen_ts": [pd.Timestamp("2025-12-15T00:00:00Z")],
        }
    )

    monkeypatch.setattr(build_minutes_live, "_read_parquet_tree", lambda *_: labels.copy())

    warnings: list[str] = []
    override_path = tmp_path / "labels.parquet"
    labels_source_df, label_source = build_minutes_live._load_label_sources(
        data_root=tmp_path,
        season_value=2025,
        override_path=override_path,
        warnings=warnings,
    )

    assert isinstance(labels_source_df, pd.DataFrame)
    assert isinstance(label_source, str)
    assert set(required_cols).issubset(labels_source_df.columns)
    assert label_source == str(override_path)
