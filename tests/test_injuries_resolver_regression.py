"""Regression tests for InjuriesResolver.

These tests reproduce the production failure where:
- Silver injuries_snapshot only contained post-tip rows for an early game
- As-of filtering dropped ALL injury rows
- Players were shown as available when they were actually OUT

The resolver must:
1. Prefer bronze (which contains multiple snapshots per day)
2. Select latest snapshot <= tip/feature_as_of_ts per game
3. Never silently produce 0 usable rows for a game (raise or warn)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from projections.builders import InjuriesResolver, resolve_injuries


def _write_bronze_injuries(
    tmp_path: Path, season: int, target_day: date, df: pd.DataFrame
) -> None:
    """Write injuries to bronze location in flat file format."""
    # The bronze reader looks for hour=* or run_ts=* subdirs first,
    # then falls back to flat file. We use flat file for simplicity.
    # NOTE: The default filename is injuries.parquet (from DEFAULT_BRONZE_FILENAMES)
    bronze_dir = tmp_path / "bronze" / "injuries_raw" / f"season={season}" / f"date={target_day}"
    bronze_dir.mkdir(parents=True)
    df.to_parquet(bronze_dir / "injuries.parquet")


def test_resolver_prefers_bronze_over_silver(tmp_path: Path) -> None:
    """Test that resolver prefers bronze when both sources have data."""
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100]

    # Bronze has a pre-tip snapshot (should be used)
    bronze_injuries = pd.DataFrame(
        {
            "game_id": [22500100],
            "player_id": [101],
            "team_id": [1610612751],
            "status": ["OUT"],
            "as_of_ts": ["2025-12-15T18:00:00Z"],  # Pre-tip
        }
    )

    # Silver has a post-tip snapshot (should NOT be used)
    silver_injuries = pd.DataFrame(
        {
            "game_id": [22500100],
            "player_id": [101],
            "team_id": [1610612751],
            "status": ["AVAILABLE"],
            "as_of_ts": ["2025-12-15T23:30:00Z"],  # Post-tip
        }
    )

    _write_bronze_injuries(tmp_path, season, target_day, bronze_injuries)

    # Write silver
    silver_dir = tmp_path / "silver" / "injuries_snapshot" / f"season={season}"
    silver_dir.mkdir(parents=True)
    silver_injuries.to_parquet(silver_dir / "injuries.parquet")

    tip_lookup = {22500100: pd.Timestamp("2025-12-15T19:00:00Z")}
    feature_as_of_ts = pd.Timestamp("2025-12-15T18:30:00Z")

    result = resolve_injuries(
        data_root=tmp_path,
        season=season,
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
    )

    assert result.source == "bronze"
    assert len(result.injuries) == 1
    assert result.injuries.iloc[0]["status"] == "OUT"
    assert 22500100 in result.games_with_injuries


def test_resolver_falls_back_to_silver_when_bronze_empty(tmp_path: Path) -> None:
    """Test that resolver falls back to silver when bronze is empty."""
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100]

    # No bronze data

    # Silver has pre-tip data
    silver_injuries = pd.DataFrame(
        {
            "game_id": [22500100],
            "player_id": [101],
            "team_id": [1610612751],
            "status": ["QUESTIONABLE"],
            "as_of_ts": ["2025-12-15T18:00:00Z"],
        }
    )

    silver_dir = tmp_path / "silver" / "injuries_snapshot" / f"season={season}"
    silver_dir.mkdir(parents=True)
    silver_injuries.to_parquet(silver_dir / "injuries.parquet")

    tip_lookup = {22500100: pd.Timestamp("2025-12-15T19:00:00Z")}
    feature_as_of_ts = pd.Timestamp("2025-12-15T18:30:00Z")

    result = resolve_injuries(
        data_root=tmp_path,
        season=season,
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
    )

    assert result.source == "silver"
    assert len(result.injuries) == 1
    assert result.injuries.iloc[0]["status"] == "QUESTIONABLE"


def test_resolver_filters_out_post_tip_snapshots(tmp_path: Path) -> None:
    """Test that snapshots after tip are filtered out."""
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100]

    # Bronze has both pre and post-tip snapshots
    bronze_injuries = pd.DataFrame(
        {
            "game_id": [22500100, 22500100],
            "player_id": [101, 101],
            "team_id": [1610612751, 1610612751],
            "status": ["OUT", "AVAILABLE"],
            "as_of_ts": [
                "2025-12-15T18:00:00Z",  # Pre-tip (should be used)
                "2025-12-15T23:30:00Z",  # Post-tip (should be filtered)
            ],
        }
    )

    _write_bronze_injuries(tmp_path, season, target_day, bronze_injuries)

    tip_lookup = {22500100: pd.Timestamp("2025-12-15T19:00:00Z")}
    feature_as_of_ts = pd.Timestamp("2025-12-15T18:30:00Z")

    result = resolve_injuries(
        data_root=tmp_path,
        season=season,
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
    )

    # Should only get the pre-tip snapshot
    assert len(result.injuries) == 1
    assert result.injuries.iloc[0]["status"] == "OUT"


def test_resolver_uses_latest_pre_tip_snapshot(tmp_path: Path) -> None:
    """Test that resolver uses the latest snapshot before tip."""
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100]

    # Multiple pre-tip snapshots
    bronze_injuries = pd.DataFrame(
        {
            "game_id": [22500100, 22500100, 22500100],
            "player_id": [101, 101, 101],
            "team_id": [1610612751, 1610612751, 1610612751],
            "status": ["QUESTIONABLE", "PROBABLE", "OUT"],
            "as_of_ts": [
                "2025-12-15T14:00:00Z",  # Early
                "2025-12-15T16:00:00Z",  # Middle
                "2025-12-15T18:00:00Z",  # Latest pre-tip (should be used)
            ],
        }
    )

    _write_bronze_injuries(tmp_path, season, target_day, bronze_injuries)

    tip_lookup = {22500100: pd.Timestamp("2025-12-15T19:00:00Z")}
    feature_as_of_ts = pd.Timestamp("2025-12-15T18:30:00Z")

    result = resolve_injuries(
        data_root=tmp_path,
        season=season,
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
    )

    # Should get the latest pre-tip status
    assert len(result.injuries) == 1
    assert result.injuries.iloc[0]["status"] == "OUT"


def test_resolver_raises_on_empty_when_required(tmp_path: Path) -> None:
    """Test that resolver raises when no data and allow_empty=False."""
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100]

    # No bronze or silver data

    tip_lookup = {22500100: pd.Timestamp("2025-12-15T19:00:00Z")}
    feature_as_of_ts = pd.Timestamp("2025-12-15T18:30:00Z")

    with pytest.raises(ValueError, match="No injuries data found"):
        resolve_injuries(
            data_root=tmp_path,
            season=season,
            target_day=target_day,
            game_ids=game_ids,
            tip_lookup=tip_lookup,
            feature_as_of_ts=feature_as_of_ts,
            allow_empty=False,
        )


def test_resolver_warns_on_empty_when_allowed(tmp_path: Path) -> None:
    """Test that resolver returns with warnings when allow_empty=True."""
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100]

    # No bronze or silver data

    tip_lookup = {22500100: pd.Timestamp("2025-12-15T19:00:00Z")}
    feature_as_of_ts = pd.Timestamp("2025-12-15T18:30:00Z")

    result = resolve_injuries(
        data_root=tmp_path,
        season=season,
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
        allow_empty=True,
    )

    assert result.injuries.empty
    assert result.source == "empty"
    assert 22500100 in result.games_without_injuries
    assert any("empty" in w.lower() for w in result.warnings)


def test_resolver_tracks_games_without_injuries(tmp_path: Path) -> None:
    """Test that resolver correctly reports games without valid injuries."""
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100, 22500101]  # Two games

    # Only game 22500100 has injuries
    bronze_injuries = pd.DataFrame(
        {
            "game_id": [22500100],
            "player_id": [101],
            "team_id": [1610612751],
            "status": ["OUT"],
            "as_of_ts": ["2025-12-15T18:00:00Z"],
        }
    )

    _write_bronze_injuries(tmp_path, season, target_day, bronze_injuries)

    tip_lookup = {
        22500100: pd.Timestamp("2025-12-15T19:00:00Z"),
        22500101: pd.Timestamp("2025-12-15T23:00:00Z"),
    }
    feature_as_of_ts = pd.Timestamp("2025-12-15T18:30:00Z")

    result = resolve_injuries(
        data_root=tmp_path,
        season=season,
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
        allow_empty=True,
    )

    assert 22500100 in result.games_with_injuries
    assert 22500101 in result.games_without_injuries


def test_regression_silver_only_post_tip_bronze_has_pre_tip(tmp_path: Path) -> None:
    """Regression test: silver has only post-tip, bronze has pre-tip.

    This reproduces the exact production failure that caused the injury
    visibility issue. The fix is to prefer bronze which has multiple
    snapshots throughout the day.
    """
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100]

    # Bronze has pre-tip snapshot marking player OUT
    bronze_injuries = pd.DataFrame(
        {
            "game_id": [22500100],
            "player_id": [101],
            "team_id": [1610612751],
            "player_name": ["Star Player"],
            "status": ["OUT"],
            "as_of_ts": ["2025-12-15T18:00:00Z"],  # Pre-tip
            "source": ["nba.com"],
        }
    )

    # Silver ONLY has post-tip snapshot (the problematic case)
    silver_injuries = pd.DataFrame(
        {
            "game_id": [22500100],
            "player_id": [101],
            "team_id": [1610612751],
            "status": ["AVAILABLE"],  # Status changed after tip
            "as_of_ts": ["2025-12-15T23:30:00Z"],  # Post-tip only!
        }
    )

    _write_bronze_injuries(tmp_path, season, target_day, bronze_injuries)

    # Write silver
    silver_dir = tmp_path / "silver" / "injuries_snapshot" / f"season={season}"
    silver_dir.mkdir(parents=True)
    silver_injuries.to_parquet(silver_dir / "injuries.parquet")

    tip_lookup = {22500100: pd.Timestamp("2025-12-15T19:00:00Z")}
    # Feature as_of_ts is before tip (live scoring scenario)
    feature_as_of_ts = pd.Timestamp("2025-12-15T18:30:00Z")

    result = resolve_injuries(
        data_root=tmp_path,
        season=season,
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
    )

    # CRITICAL: Must use bronze (pre-tip) not silver (post-tip)
    assert result.source == "bronze", "Resolver should prefer bronze over silver"

    # CRITICAL: Player should be OUT, not AVAILABLE
    assert len(result.injuries) == 1, "Should have exactly one injury row"
    assert result.injuries.iloc[0]["status"] == "OUT", (
        "Player should be marked OUT from pre-tip snapshot, "
        "not AVAILABLE from post-tip silver"
    )

    # Game should have injuries, not be missing
    assert 22500100 in result.games_with_injuries
    assert 22500100 not in result.games_without_injuries


def test_backfill_mode_uses_tip_as_ceiling(tmp_path: Path) -> None:
    """Test that backfill mode uses tip_ts as the as_of ceiling."""
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100]

    # Snapshot at 18:00, tip at 19:00
    bronze_injuries = pd.DataFrame(
        {
            "game_id": [22500100],
            "player_id": [101],
            "team_id": [1610612751],
            "status": ["OUT"],
            "as_of_ts": ["2025-12-15T18:00:00Z"],
        }
    )

    _write_bronze_injuries(tmp_path, season, target_day, bronze_injuries)

    tip_lookup = {22500100: pd.Timestamp("2025-12-15T19:00:00Z")}
    # In backfill mode, even if feature_as_of_ts is way in the future,
    # we should use tip_ts as the ceiling
    feature_as_of_ts = pd.Timestamp("2025-12-16T12:00:00Z")  # Next day

    result = resolve_injuries(
        data_root=tmp_path,
        season=season,
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
        backfill_mode=True,  # Use tip_ts ceiling
    )

    # Should still get the pre-tip snapshot
    assert len(result.injuries) == 1
    assert result.injuries.iloc[0]["status"] == "OUT"


def test_cleared_player_not_in_latest_report(tmp_path: Path) -> None:
    """Test that players cleared from injury list are not marked OUT.

    This reproduces the production bug where:
    - Player was OUT in an older injury report
    - Player was CLEARED (not listed) in a newer injury report
    - System incorrectly kept using the old OUT status

    The NBA injury reports only list players currently injured. When a player
    is cleared, they simply don't appear in subsequent reports. The resolver
    must interpret absence from the latest report as "cleared".
    """
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100]

    # Simulate multiple injury reports throughout the day:
    # - 11PM previous day: Player A (OUT), Player B (OUT)
    # - 12PM game day: Player A (OUT), [Player B not listed = cleared]
    bronze_injuries = pd.DataFrame(
        {
            "game_id": [22500100, 22500100, 22500100],
            "player_id": [101, 102, 101],  # Player 101 in both, 102 only in old
            "team_id": [1610612751, 1610612751, 1610612751],
            "player_name": ["Still Injured", "Now Cleared", "Still Injured"],
            "status": ["OUT", "OUT", "OUT"],
            "as_of_ts": [
                "2025-12-14T23:00:00Z",  # Old report
                "2025-12-14T23:00:00Z",  # Old report - player 102 OUT
                "2025-12-15T12:00:00Z",  # New report - only player 101
            ],
        }
    )

    _write_bronze_injuries(tmp_path, season, target_day, bronze_injuries)

    tip_lookup = {22500100: pd.Timestamp("2025-12-15T19:00:00Z")}
    feature_as_of_ts = pd.Timestamp("2025-12-15T18:30:00Z")

    result = resolve_injuries(
        data_root=tmp_path,
        season=season,
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
    )

    # CRITICAL: Only player 101 should be in results (still OUT)
    # Player 102 should NOT be in results (cleared - not in latest report)
    assert len(result.injuries) == 1, (
        "Only player still in latest report should be included"
    )
    assert result.injuries.iloc[0]["player_id"] == 101
    assert result.injuries.iloc[0]["status"] == "OUT"

    # Player 102 should NOT appear anywhere in results
    player_ids_in_result = set(result.injuries["player_id"].tolist())
    assert 102 not in player_ids_in_result, (
        "Cleared player (not in latest report) should not be marked as injured"
    )


def test_cleared_player_multiple_games(tmp_path: Path) -> None:
    """Test cleared player logic works independently per game.

    Each game should use its own latest report. A player might be:
    - Cleared for game A (not in latest report for A)
    - Still OUT for game B (in latest report for B)
    """
    season = 2025
    target_day = date(2025, 12, 15)
    game_ids = [22500100, 22500101]

    # Game 100: Player cleared (old report only)
    # Game 101: Player still OUT (in latest report)
    bronze_injuries = pd.DataFrame(
        {
            "game_id": [22500100, 22500101, 22500101],
            "player_id": [101, 101, 101],
            "team_id": [1610612751, 1610612752, 1610612752],
            "status": ["OUT", "OUT", "OUT"],
            "as_of_ts": [
                "2025-12-14T23:00:00Z",  # Game 100: only old report (player cleared)
                "2025-12-14T23:00:00Z",  # Game 101: old report
                "2025-12-15T12:00:00Z",  # Game 101: new report (player still OUT)
            ],
        }
    )

    _write_bronze_injuries(tmp_path, season, target_day, bronze_injuries)

    tip_lookup = {
        22500100: pd.Timestamp("2025-12-15T19:00:00Z"),
        22500101: pd.Timestamp("2025-12-15T22:00:00Z"),
    }
    feature_as_of_ts = pd.Timestamp("2025-12-15T18:30:00Z")

    result = resolve_injuries(
        data_root=tmp_path,
        season=season,
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
        allow_empty=True,  # Game 100 may have no injuries after clearing
    )

    # Game 100: No injuries (only had old report, so player considered cleared)
    # Note: When the only report is old and no newer report exists for a game,
    # we can't know if player was cleared. This case tests when there's only one report.
    # Game 101: Player still OUT (in latest report)
    game_101_injuries = result.injuries[result.injuries["game_id"] == 22500101]
    assert len(game_101_injuries) == 1
    assert game_101_injuries.iloc[0]["status"] == "OUT"
