"""Shared feature builder for minutes_v1.

This module provides unified feature building logic for both live and training
pipelines. The key principle is that feature code NEVER calls now() - all
temporal state is injected via as_of_ts from the caller.

Live vs Training:
- Live: as_of_ts = run_as_of_ts (offset from tip or explicit param)
- Training: as_of_ts = tip_ts (historical cutoff per game)

Both paths use identical resolution logic and produce identical schema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from projections.builders.injuries_resolver import InjuriesResolver, InjuriesResolutionResult
from projections.minutes_v1.features import MinutesFeatureBuilder
from projections.etl import storage as bronze_storage

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass
class FeatureBuildConfig:
    """Configuration for feature building.

    This config encapsulates all parameters needed for building features,
    ensuring that both live and training paths use consistent settings.
    """

    data_root: Path
    """Root directory containing bronze/silver/gold data."""

    season: int
    """NBA season year (e.g., 2025 for 2025-26 season)."""

    as_of_ts: pd.Timestamp
    """The temporal cutoff for feature building. Features only use data
    available as of this timestamp. NEVER set to now() - always pass
    from caller."""

    target_day: date
    """The target game date for which features are built."""

    backfill_mode: bool = False
    """If True, use tip_ts as ceiling for each game (historical mode).
    If False, use as_of_ts (live mode)."""

    # Optional path overrides
    schedule_path: Path | None = None
    injuries_path: Path | None = None
    odds_path: Path | None = None
    roster_path: Path | None = None
    roles_path: Path | None = None
    archetype_path: Path | None = None
    coach_path: Path | None = None


@dataclass
class FeatureBuildResult:
    """Result of feature building with metadata."""

    features: pd.DataFrame
    """The built features DataFrame."""

    injuries_result: InjuriesResolutionResult
    """Metadata about injuries resolution."""

    warnings: list[str] = field(default_factory=list)
    """Warnings encountered during building."""


class SharedFeaturesBuilder:
    """Unified feature builder for both live and training paths.

    This class provides a single code path for building minutes features,
    enforcing strict as_of_ts semantics to prevent data leakage.

    Key invariants:
    - Feature code NEVER calls now()
    - as_of_ts is always injected from caller
    - Live and training produce identical schema
    - Injuries are resolved via InjuriesResolver (bronze preferred)

    Usage:
        config = FeatureBuildConfig(
            data_root=Path("/data"),
            season=2025,
            as_of_ts=pd.Timestamp("2025-12-15T22:30:00Z"),
            target_day=date(2025, 12, 15),
        )
        builder = SharedFeaturesBuilder(config)
        result = builder.build(labels_df, schedule_df, game_ids)
    """

    def __init__(self, config: FeatureBuildConfig) -> None:
        self.config = config
        self.injuries_resolver = InjuriesResolver(
            data_root=config.data_root,
            season=config.season,
        )

    def build(
        self,
        labels: pd.DataFrame,
        schedule: pd.DataFrame,
        game_ids: list[int],
        *,
        roster: pd.DataFrame | None = None,
        odds: pd.DataFrame | None = None,
        coach: pd.DataFrame | None = None,
        roles: pd.DataFrame | None = None,
        archetype_deltas: pd.DataFrame | None = None,
    ) -> FeatureBuildResult:
        """Build features for the given labels and schedule.

        Args:
            labels: Labels DataFrame with game_id, player_id, minutes, etc.
            schedule: Schedule DataFrame with game_id, tip_ts, etc.
            game_ids: List of game IDs to build features for.
            roster: Optional roster DataFrame (loaded if not provided).
            odds: Optional odds DataFrame (loaded if not provided).
            coach: Optional coach tenure DataFrame.
            roles: Optional roles DataFrame.
            archetype_deltas: Optional archetype deltas DataFrame.

        Returns:
            FeatureBuildResult with features and metadata.
        """
        warnings: list[str] = []
        cfg = self.config

        # Build tip lookup for injuries resolution
        tip_lookup = self._build_tip_lookup(schedule, game_ids)

        # Resolve injuries using the shared resolver
        injuries_result = self.injuries_resolver.resolve(
            target_day=cfg.target_day,
            game_ids=game_ids,
            tip_lookup=tip_lookup,
            feature_as_of_ts=cfg.as_of_ts,
            backfill_mode=cfg.backfill_mode,
            allow_empty=True,  # Handle gracefully, log warnings
        )
        warnings.extend(injuries_result.warnings)

        # Log injury resolution status
        if injuries_result.games_without_injuries:
            missing_count = len(injuries_result.games_without_injuries)
            logger.warning(
                f"[features] {missing_count} game(s) have no injury data "
                f"after resolution. Vacancy features may be affected."
            )

        # Load other inputs if not provided
        if roster is None:
            roster = self._load_roster()
        if odds is None:
            odds = self._load_odds(game_ids)
        if coach is None:
            coach = self._load_coach()
        if roles is None:
            roles = self._load_roles()
        if archetype_deltas is None:
            archetype_deltas = self._load_archetype_deltas()

        # Filter inputs by as_of_ts
        filtered_odds = self._filter_snapshot_by_asof(
            odds, "as_of_ts", tip_lookup, "odds"
        )
        filtered_roster = self._filter_snapshot_by_asof(
            roster, "as_of_ts", tip_lookup, "roster"
        )

        # Build features using MinutesFeatureBuilder
        builder = MinutesFeatureBuilder(
            schedule=schedule[schedule["game_id"].isin(game_ids)],
            injuries_snapshot=injuries_result.injuries,
            odds_snapshot=filtered_odds,
            roster_nightly=filtered_roster,
            coach_tenure=coach,
            archetype_roles=roles,
            archetype_deltas=archetype_deltas,
        )

        features = builder.build(labels)

        # Ensure consistent schema columns
        features = self._ensure_schema_columns(features)

        return FeatureBuildResult(
            features=features,
            injuries_result=injuries_result,
            warnings=warnings,
        )

    def _build_tip_lookup(
        self, schedule: pd.DataFrame, game_ids: list[int]
    ) -> dict[int, pd.Timestamp]:
        """Build mapping of game_id -> tip_ts from schedule."""
        tip_lookup: dict[int, pd.Timestamp] = {}
        if schedule.empty or "tip_ts" not in schedule.columns:
            return tip_lookup

        for _, row in schedule.iterrows():
            game_id = row.get("game_id")
            tip_ts = row.get("tip_ts")
            if pd.notna(game_id) and pd.notna(tip_ts):
                game_id_int = int(game_id)
                if game_id_int in game_ids:
                    tip_lookup[game_id_int] = pd.to_datetime(tip_ts, utc=True)

        return tip_lookup

    def _filter_snapshot_by_asof(
        self,
        df: pd.DataFrame,
        time_col: str,
        tip_lookup: dict[int, pd.Timestamp],
        label: str,
    ) -> pd.DataFrame:
        """Filter snapshot DataFrame to rows where time_col <= ceiling.

        For each game, the ceiling is min(tip_ts, as_of_ts) in live mode,
        or tip_ts in backfill mode.
        """
        if df.empty or time_col not in df.columns:
            return df

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

        if "game_id" not in df.columns:
            # Global filter by as_of_ts
            return df[df[time_col] <= self.config.as_of_ts]

        df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce")

        result_frames: list[pd.DataFrame] = []
        for game_id in df["game_id"].dropna().unique():
            game_id_int = int(game_id)
            game_rows = df[df["game_id"] == game_id_int]

            # Determine ceiling
            if self.config.backfill_mode:
                ceiling = tip_lookup.get(game_id_int, self.config.as_of_ts)
            else:
                tip_ts = tip_lookup.get(game_id_int)
                ceiling = min(tip_ts, self.config.as_of_ts) if tip_ts else self.config.as_of_ts

            valid_rows = game_rows[game_rows[time_col] <= ceiling]
            if not valid_rows.empty:
                result_frames.append(valid_rows)

        return pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()

    def _load_roster(self) -> pd.DataFrame:
        """Load roster from default path."""
        cfg = self.config
        roster_path = cfg.roster_path or (
            cfg.data_root / "silver" / "roster_nightly" / f"season={cfg.season}"
        )
        return self._read_parquet_tree(roster_path)

    def _load_odds(self, game_ids: list[int]) -> pd.DataFrame:
        """Load odds from default path."""
        cfg = self.config
        odds_path = cfg.odds_path or (
            cfg.data_root / "silver" / "odds_snapshot" / f"season={cfg.season}"
        )
        df = self._read_parquet_tree(odds_path)
        if "game_id" in df.columns:
            df = df[df["game_id"].isin(game_ids)]
        return df

    def _load_coach(self) -> pd.DataFrame | None:
        """Load coach tenure from static file."""
        cfg = self.config
        coach_path = cfg.coach_path or (cfg.data_root / "static" / "coach_tenure.csv")
        if coach_path.exists():
            return pd.read_csv(coach_path)
        return None

    def _load_roles(self) -> pd.DataFrame | None:
        """Load minutes roles artifact."""
        cfg = self.config
        roles_path = cfg.roles_path or (
            cfg.data_root / "gold" / "minutes_roles" / f"season={cfg.season}" / "roles.parquet"
        )
        if roles_path.exists():
            return pd.read_parquet(roles_path)
        return None

    def _load_archetype_deltas(self) -> pd.DataFrame | None:
        """Load archetype deltas artifact."""
        cfg = self.config
        archetype_path = cfg.archetype_path or (
            cfg.data_root
            / "gold"
            / "features_minutes_v1"
            / f"season={cfg.season}"
            / "archetype_deltas.parquet"
        )
        if archetype_path.exists():
            return pd.read_parquet(archetype_path)
        return None

    def _read_parquet_tree(self, path: Path) -> pd.DataFrame:
        """Read all parquet files under a path."""
        if not path.exists():
            return pd.DataFrame()

        if path.is_file():
            return pd.read_parquet(path)

        parquet_files = list(path.glob("**/*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        frames = [pd.read_parquet(p) for p in parquet_files]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _ensure_schema_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent schema columns exist."""
        # Key columns that must be present
        required_cols = [
            "game_id",
            "player_id",
            "team_id",
            "game_date",
            "player_name",
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NA

        # Ensure injury-related columns have defaults
        injury_cols = [
            "status",
            "is_out",
            "injury_snapshot_missing",
            "injury_as_of_ts",  # canonical singular form
        ]
        for col in injury_cols:
            if col not in df.columns:
                if col == "is_out":
                    df[col] = 0
                elif col == "injury_snapshot_missing":
                    df[col] = 1
                else:
                    df[col] = pd.NA

        return df


def build_features_unified(
    config: FeatureBuildConfig,
    labels: pd.DataFrame,
    schedule: pd.DataFrame,
    game_ids: list[int],
    **kwargs,
) -> FeatureBuildResult:
    """Convenience function for unified feature building.

    See SharedFeaturesBuilder.build() for full documentation.
    """
    builder = SharedFeaturesBuilder(config)
    return builder.build(labels, schedule, game_ids, **kwargs)


def build_shared_features(
    *,
    data_root: Path,
    season: int,
    target_day: date,
    as_of_ts: pd.Timestamp,
    labels: pd.DataFrame,
    schedule: pd.DataFrame,
    game_ids: list[int],
    backfill_mode: bool = False,
    roster: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
    coach: pd.DataFrame | None = None,
    roles: pd.DataFrame | None = None,
    archetype_deltas: pd.DataFrame | None = None,
) -> FeatureBuildResult:
    """Greppable top-level entry point for shared feature building.

    This is the canonical function for building minutes features in both
    live and training paths. It uses InjuriesResolver under the hood
    to prefer bronze injuries (multiple snapshots) over silver.

    Args:
        data_root: Root directory containing bronze/silver/gold data.
        season: NBA season year (e.g., 2025 for 2024-25 season).
        target_day: The target game date.
        as_of_ts: Temporal cutoff - features only use data available as of
            this timestamp. NEVER pass now() - always inject from caller.
        labels: Labels DataFrame with game_id, player_id, minutes, etc.
        schedule: Schedule DataFrame with game_id, tip_ts, etc.
        game_ids: List of game IDs to build features for.
        backfill_mode: If True, use tip_ts as ceiling (training).
            If False, use as_of_ts (live).
        roster: Optional roster DataFrame.
        odds: Optional odds DataFrame.
        coach: Optional coach tenure DataFrame.
        roles: Optional roles DataFrame.
        archetype_deltas: Optional archetype deltas DataFrame.

    Returns:
        FeatureBuildResult with features DataFrame and metadata.

    Example:
        result = build_shared_features(
            data_root=Path("/data"),
            season=2025,
            target_day=date(2025, 12, 15),
            as_of_ts=pd.Timestamp("2025-12-15T18:30:00Z"),
            labels=labels_df,
            schedule=schedule_df,
            game_ids=[22500100, 22500101],
        )
    """
    config = FeatureBuildConfig(
        data_root=data_root,
        season=season,
        target_day=target_day,
        as_of_ts=as_of_ts,
        backfill_mode=backfill_mode,
    )
    builder = SharedFeaturesBuilder(config)
    return builder.build(
        labels=labels,
        schedule=schedule,
        game_ids=game_ids,
        roster=roster,
        odds=odds,
        coach=coach,
        roles=roles,
        archetype_deltas=archetype_deltas,
    )
