"""Schema parity test: live builder schema == training builder schema.

These tests verify that the shared builders produce consistent outputs
regardless of whether they're called from live or training paths.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from projections.builders import (
    FeatureBuildConfig,
    InjuriesResolver,
)


def _write_bronze_injuries(
    tmp_path: Path, season: int, target_day: date, df: pd.DataFrame
) -> None:
    """Write injuries to bronze location in flat file format."""
    bronze_dir = tmp_path / "bronze" / "injuries_raw" / f"season={season}" / f"date={target_day}"
    bronze_dir.mkdir(parents=True)
    df.to_parquet(bronze_dir / "injuries.parquet")


def test_injuries_resolver_schema_consistent(tmp_path: Path) -> None:
    """Test that InjuriesResolver produces consistent schema in both modes."""
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100, 22500101]

    injuries_data = pd.DataFrame(
        {
            "game_id": [22500100, 22500101],
            "player_id": [101, 103],
            "team_id": [1610612751, 1610612744],
            "status": ["OUT", "QUESTIONABLE"],
            "as_of_ts": ["2025-12-15T17:00:00Z", "2025-12-15T17:00:00Z"],
            "restriction_flag": [False, False],
            "ramp_flag": [False, False],
            "days_since_return": [0, 0],
            "games_since_return": [0, 0],
        }
    )

    _write_bronze_injuries(tmp_path, season, target_day, injuries_data)

    resolver = InjuriesResolver(data_root=tmp_path, season=season)

    tip_lookup = {
        22500100: pd.Timestamp("2025-12-15T19:00:00Z"),
        22500101: pd.Timestamp("2025-12-15T23:00:00Z"),
    }

    # Live mode: use feature_as_of_ts
    live_result = resolver.resolve(
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=pd.Timestamp("2025-12-15T18:30:00Z"),
        backfill_mode=False,
    )

    # Training mode: use tip_ts as ceiling
    training_result = resolver.resolve(
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=pd.Timestamp("2025-12-15T18:30:00Z"),
        backfill_mode=True,
    )

    # Both should produce the same columns
    live_cols = set(live_result.injuries.columns)
    training_cols = set(training_result.injuries.columns)
    assert live_cols == training_cols, f"Column mismatch: {live_cols ^ training_cols}"

    # Both should have resolved the same games
    assert live_result.games_with_injuries == training_result.games_with_injuries


def test_injuries_resolver_handles_missing_columns_gracefully(tmp_path: Path) -> None:
    """Test that resolver works with minimal columns."""
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100]

    # Minimal injuries with only essential columns
    injuries_data = pd.DataFrame(
        {
            "game_id": [22500100],
            "player_id": [101],
            "status": ["OUT"],
            "as_of_ts": ["2025-12-15T17:00:00Z"],
        }
    )

    _write_bronze_injuries(tmp_path, season, target_day, injuries_data)

    resolver = InjuriesResolver(data_root=tmp_path, season=season)
    tip_lookup = {22500100: pd.Timestamp("2025-12-15T19:00:00Z")}

    result = resolver.resolve(
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=pd.Timestamp("2025-12-15T18:30:00Z"),
    )

    assert len(result.injuries) == 1
    assert result.injuries.iloc[0]["status"] == "OUT"


def test_config_requires_as_of_ts() -> None:
    """Test that FeatureBuildConfig requires explicit as_of_ts."""
    # This test validates the design - as_of_ts is required, preventing now() usage
    with pytest.raises(TypeError):
        # Missing as_of_ts should fail
        FeatureBuildConfig(  # type: ignore[call-arg]
            data_root=Path("/tmp"),
            season=2025,
            target_day=date(2025, 12, 15),
            # as_of_ts is missing - should fail
        )


def test_config_accepts_explicit_as_of_ts() -> None:
    """Test that FeatureBuildConfig works with explicit as_of_ts."""
    config = FeatureBuildConfig(
        data_root=Path("/tmp"),
        season=2025,
        target_day=date(2025, 12, 15),
        as_of_ts=pd.Timestamp("2025-12-15T18:30:00Z"),
    )

    assert config.as_of_ts == pd.Timestamp("2025-12-15T18:30:00Z")
    assert config.season == 2025
    assert config.target_day == date(2025, 12, 15)
    assert config.backfill_mode is False  # Default


def test_backfill_mode_flag() -> None:
    """Test that backfill_mode flag is correctly set."""
    live_config = FeatureBuildConfig(
        data_root=Path("/tmp"),
        season=2025,
        target_day=date(2025, 12, 15),
        as_of_ts=pd.Timestamp("2025-12-15T18:30:00Z"),
        backfill_mode=False,
    )

    training_config = FeatureBuildConfig(
        data_root=Path("/tmp"),
        season=2025,
        target_day=date(2025, 12, 15),
        as_of_ts=pd.Timestamp("2025-12-15T18:30:00Z"),
        backfill_mode=True,
    )

    assert live_config.backfill_mode is False
    assert training_config.backfill_mode is True
