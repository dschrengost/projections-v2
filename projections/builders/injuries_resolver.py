"""Injuries history resolver for feature building.

This module provides a unified approach to resolving injury snapshots for both
live and training feature builds. It enforces strict as_of_ts semantics:

- Feature code never calls now(); it accepts as_of_ts from caller
- Live builder supplies as_of_ts based on configured offset from tip
- Training builder supplies historical as_of_ts (typically tip_ts)
- Both use identical resolution logic

The resolver prefers bronze (multiple snapshots per day) over silver (single
snapshot per day), and raises errors when no valid snapshots can be resolved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from projections.etl import storage as bronze_storage

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass
class InjuriesResolutionResult:
    """Result of injuries resolution with metadata."""

    injuries: pd.DataFrame
    """Resolved injuries snapshot (latest as_of_ts <= feature_as_of_ts per game)."""

    source: str
    """Source of data: 'bronze', 'silver', or 'empty'."""

    games_with_injuries: set[int]
    """Set of game IDs that have valid injury rows."""

    games_without_injuries: set[int]
    """Set of game IDs that had no valid injury rows."""

    warnings: list[str] = field(default_factory=list)
    """Warnings encountered during resolution."""


class InjuriesResolver:
    """Resolver for injuries snapshots with bronze/silver fallback.

    This class encapsulates the logic for loading and resolving injuries
    from either bronze (preferred) or silver sources, applying strict
    as_of_ts filtering to prevent data leakage.

    Usage:
        resolver = InjuriesResolver(data_root=data_root, season=2025)
        result = resolver.resolve(
            target_day=date(2025, 12, 15),
            game_ids=[22500100, 22500101],
            tip_lookup={22500100: pd.Timestamp("2025-12-15T23:00:00Z"), ...},
            feature_as_of_ts=pd.Timestamp("2025-12-15T22:30:00Z"),
        )
    """

    def __init__(
        self,
        data_root: Path,
        season: int,
        *,
        bronze_root: Path | None = None,
        silver_root: Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.season = season
        self.bronze_root = bronze_root
        self.silver_root = silver_root or (self.data_root / "silver" / "injuries_snapshot")

    def resolve(
        self,
        target_day: date,
        game_ids: list[int],
        tip_lookup: Mapping[int, pd.Timestamp],
        feature_as_of_ts: pd.Timestamp,
        *,
        backfill_mode: bool = False,
        allow_empty: bool = False,
    ) -> InjuriesResolutionResult:
        """Resolve injuries for the given games with strict as_of_ts filtering.

        Args:
            target_day: The target game date.
            game_ids: List of game IDs to resolve injuries for.
            tip_lookup: Mapping of game_id -> tip_ts for each game.
            feature_as_of_ts: The as_of_ts ceiling for feature building.
            backfill_mode: If True, use tip_ts as the ceiling instead of
                feature_as_of_ts (for historical backfills).
            allow_empty: If True, don't raise on empty results (with warning).

        Returns:
            InjuriesResolutionResult with resolved injuries and metadata.

        Raises:
            ValueError: If no valid injury rows are found and allow_empty=False.
        """
        warnings: list[str] = []
        game_id_set = set(game_ids)

        # Try bronze first (has multiple snapshots per day)
        bronze_df = self._load_bronze(target_day)
        if not bronze_df.empty:
            source = "bronze"
            warnings.append(
                f"[injuries] Loaded {len(bronze_df)} rows from bronze "
                f"injuries_raw for day={target_day.isoformat()}."
            )
            raw_injuries = bronze_df
        else:
            # Fallback to silver
            silver_df = self._load_silver()
            if not silver_df.empty:
                source = "silver"
                warnings.append(
                    f"[injuries] Bronze empty for day={target_day.isoformat()}; "
                    f"falling back to silver injuries_snapshot ({len(silver_df)} rows)."
                )
                raw_injuries = silver_df
            else:
                source = "empty"
                warnings.append(
                    f"[injuries] WARNING: Both bronze and silver injuries are empty "
                    f"for season={self.season}, day={target_day.isoformat()}."
                )
                raw_injuries = pd.DataFrame()

        if raw_injuries.empty:
            if not allow_empty:
                raise ValueError(
                    f"No injuries data found for season={self.season}, "
                    f"day={target_day.isoformat()}. Check bronze/silver sources."
                )
            return InjuriesResolutionResult(
                injuries=pd.DataFrame(),
                source=source,
                games_with_injuries=set(),
                games_without_injuries=game_id_set,
                warnings=warnings,
            )

        # Filter to requested game_ids
        if "game_id" in raw_injuries.columns:
            raw_injuries = raw_injuries[
                raw_injuries["game_id"].isin(game_id_set)
            ].copy()

        if raw_injuries.empty:
            if not allow_empty:
                raise ValueError(
                    f"No injuries data for game_ids={sorted(game_id_set)} "
                    f"in season={self.season}."
                )
            return InjuriesResolutionResult(
                injuries=pd.DataFrame(),
                source=source,
                games_with_injuries=set(),
                games_without_injuries=game_id_set,
                warnings=warnings,
            )

        # Apply as_of_ts filtering: keep rows where as_of_ts <= ceiling
        resolved = self._filter_by_asof(
            raw_injuries,
            tip_lookup=tip_lookup,
            feature_as_of_ts=feature_as_of_ts,
            backfill_mode=backfill_mode,
            warnings=warnings,
        )

        # Determine which games have/lack injuries
        if not resolved.empty and "game_id" in resolved.columns:
            games_with = set(
                pd.to_numeric(resolved["game_id"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
        else:
            games_with = set()
        games_without = game_id_set - games_with

        if games_without:
            warnings.append(
                f"[injuries] {len(games_without)} game(s) have no valid injury rows "
                f"after as_of filtering: {sorted(games_without)[:5]}..."
            )

        if resolved.empty and not allow_empty:
            latest_ts = pd.to_datetime(
                raw_injuries.get("as_of_ts"), utc=True, errors="coerce"
            )
            latest_str = latest_ts.max().isoformat() if not latest_ts.dropna().empty else "NA"
            raise ValueError(
                f"Injury snapshot is empty after as_of filtering. "
                f"feature_as_of_ts={feature_as_of_ts.isoformat()} "
                f"latest_injury_as_of_ts={latest_str}. "
                f"This may indicate bronze/silver only contains post-tip snapshots."
            )

        return InjuriesResolutionResult(
            injuries=resolved,
            source=source,
            games_with_injuries=games_with,
            games_without_injuries=games_without,
            warnings=warnings,
        )

    def _load_bronze(self, target_day: date) -> pd.DataFrame:
        """Load injuries from bronze partitions for the target day."""
        return bronze_storage.read_bronze_day(
            "injuries_raw",
            self.data_root,
            self.season,
            target_day,
            include_runs=False,
            prefer_history=True,
            bronze_root=self.bronze_root,
        )

    def _load_silver(self) -> pd.DataFrame:
        """Load injuries from silver snapshot for the season."""
        if self.silver_root is None:
            return pd.DataFrame()

        silver_path = self.silver_root / f"season={self.season}"
        if not silver_path.exists():
            return pd.DataFrame()

        parquet_files = list(silver_path.glob("**/*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        frames = [pd.read_parquet(p) for p in parquet_files]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _filter_by_asof(
        self,
        df: pd.DataFrame,
        tip_lookup: Mapping[int, pd.Timestamp],
        feature_as_of_ts: pd.Timestamp,
        backfill_mode: bool,
        warnings: list[str],
    ) -> pd.DataFrame:
        """Filter injuries to latest snapshot per game where as_of_ts <= ceiling.

        Important: Players who appeared in older reports but are NOT in the latest
        report for a game are considered cleared (removed from injury list). This
        handles the NBA's reporting pattern where cleared players simply disappear
        from subsequent reports rather than being explicitly marked as available.
        """
        if df.empty or "as_of_ts" not in df.columns:
            return df

        df = df.copy()
        df["as_of_ts"] = pd.to_datetime(df["as_of_ts"], utc=True, errors="coerce")
        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
        
        # Determine the ordering column: prefer report_ts, then derive from source URL
        if "report_ts" in df.columns and df["report_ts"].notna().any():
            df["_order_ts"] = pd.to_datetime(df["report_ts"], utc=True, errors="coerce")
        elif "source" in df.columns and df["source"].notna().any():
            # Derive report_ts from source URL for backwards compatibility
            from projections.minutes_v1.snapshots import _parse_report_ts_from_source
            df["_order_ts"] = df["source"].apply(_parse_report_ts_from_source)
        else:
            df["_order_ts"] = df["as_of_ts"]

        # Fall back to as_of_ts where _order_ts is NaT (e.g., non-standard source URLs)
        mask = df["_order_ts"].isna()
        df.loc[mask, "_order_ts"] = df.loc[mask, "as_of_ts"]

        result_frames: list[pd.DataFrame] = []
        for game_id in df["game_id"].dropna().unique():
            game_id_int = int(game_id)
            game_rows = df[df["game_id"] == game_id_int].copy()

            # Determine ceiling for this game
            if backfill_mode:
                # In backfill mode, use tip_ts as ceiling (simulate pre-tip state)
                ceiling = tip_lookup.get(game_id_int)
                if ceiling is None:
                    # Fall back to feature_as_of_ts if tip not known
                    ceiling = feature_as_of_ts
            else:
                # In live mode, use the minimum of tip_ts and feature_as_of_ts
                tip_ts = tip_lookup.get(game_id_int)
                if tip_ts is not None:
                    ceiling = min(tip_ts, feature_as_of_ts)
                else:
                    ceiling = feature_as_of_ts

            # Filter rows where as_of_ts <= ceiling
            valid_mask = game_rows["as_of_ts"] <= ceiling
            valid_rows = game_rows[valid_mask]

            if valid_rows.empty:
                # Log but continue - games_without will capture this
                continue

            # Find the latest report for this game (by _order_ts)
            # Players who were in older reports but NOT in the latest report are considered cleared
            latest_report_ts = valid_rows["_order_ts"].max()
            latest_report_rows = valid_rows[valid_rows["_order_ts"] == latest_report_ts]

            # Only keep players who appear in the latest report
            # This handles the case where a player was OUT in an older report but cleared (not listed) in newer ones
            if "player_id" in latest_report_rows.columns:
                players_in_latest = set(latest_report_rows["player_id"].dropna().unique())
                valid_rows = valid_rows[valid_rows["player_id"].isin(players_in_latest)]

            # Keep latest snapshot per player for this game, ordered by actual report time
            valid_rows = valid_rows.sort_values("_order_ts", ascending=False, na_position="last")
            if "player_id" in valid_rows.columns:
                valid_rows = valid_rows.drop_duplicates(
                    subset=["game_id", "player_id"], keep="first"
                )

            result_frames.append(valid_rows.drop(columns=["_order_ts"], errors="ignore"))

        if not result_frames:
            return pd.DataFrame()

        return pd.concat(result_frames, ignore_index=True)


def resolve_injuries(
    data_root: Path,
    season: int,
    target_day: date,
    game_ids: list[int],
    tip_lookup: Mapping[int, pd.Timestamp],
    feature_as_of_ts: pd.Timestamp,
    *,
    backfill_mode: bool = False,
    allow_empty: bool = False,
) -> InjuriesResolutionResult:
    """Convenience function for resolving injuries.

    See InjuriesResolver.resolve() for full documentation.
    """
    resolver = InjuriesResolver(data_root=data_root, season=season)
    return resolver.resolve(
        target_day=target_day,
        game_ids=game_ids,
        tip_lookup=tip_lookup,
        feature_as_of_ts=feature_as_of_ts,
        backfill_mode=backfill_mode,
        allow_empty=allow_empty,
    )
