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

import numpy as np
import pandas as pd

try:
    from unidecode import unidecode
except ImportError:
    def unidecode(s: str) -> str:
        return s  # Fallback: no unicode normalization

from projections.builders.injuries_resolver import InjuriesResolver, InjuriesResolutionResult
from projections.features.dnp_history import (
    DNP_HISTORY_FEATURE_COLUMNS,
    compute_dnp_history_features_for_live,
)
from projections.minutes_v1.features import MinutesFeatureBuilder
from projections.minutes_v1.pos import canonical_pos_bucket_series
from projections.names import normalize_player_name

logger = logging.getLogger(__name__)

_OUT_LIKE_STATUS_VALUES: set[str] = {"OUT", "O", "Q", "QUESTIONABLE", "DOUBTFUL", "D", "INACTIVE"}
_VACANCY_FEATURE_COLUMNS: tuple[str, ...] = (
    "vac_min_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
)
_TEAM_CONTEXT_COLUMNS: tuple[str, ...] = (
    "team_pace_szn",
    "team_off_rtg_szn",
    "team_def_rtg_szn",
)
_OPP_CONTEXT_COLUMNS: tuple[str, ...] = (
    "opp_pace_szn",
    "opp_def_rtg_szn",
)


# Use the central normalize_player_name for consistent matching across all data sources.
# This handles suffixes (Jr, III, etc.), accents, and aliases.
_normalize_name_for_matching = normalize_player_name


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

        # Merge Rotowire starter signals into roster (faster lineup confirmations)
        roster = self._load_and_merge_rotowire_starters(roster, warnings=warnings)

        # Filter inputs by as_of_ts
        filtered_odds = self._filter_snapshot_by_asof(
            odds, "as_of_ts", tip_lookup, "odds"
        )
        filtered_roster = self._filter_snapshot_by_asof(
            roster, "as_of_ts", tip_lookup, "roster"
        )

        # Build features using MinutesFeatureBuilder.
        #
        # IMPORTANT: labels includes both historical rows (for history-based features like
        # starter/rest) and live stubs. We must provide schedule rows for *all* label game_ids
        # so MinutesFeatureBuilder can attach tip_ts and compute shift/rolling features.
        schedule_for_builder = schedule
        if "game_id" in labels.columns and "game_id" in schedule.columns:
            label_game_ids = (
                pd.to_numeric(labels["game_id"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            if label_game_ids:
                schedule_for_builder = schedule.loc[schedule["game_id"].isin(label_game_ids)].copy()

        builder = MinutesFeatureBuilder(
            schedule=schedule_for_builder,
            injuries_snapshot=injuries_result.injuries,
            odds_snapshot=filtered_odds,
            roster_nightly=filtered_roster,
            coach_tenure=coach,
            archetype_roles=roles,
            archetype_deltas=archetype_deltas,
        )

        features = builder.build(labels)
        # From this point onward we only need the target slate rows; history rows were used
        # to compute PIT-safe history features (trend/role/rest).
        if "game_date" in features.columns:
            game_dates = pd.to_datetime(features["game_date"], errors="coerce").dt.normalize()
            target_day = pd.Timestamp(cfg.target_day).normalize()
            features = features.loc[game_dates == target_day].copy()
        if "game_id" in features.columns and game_ids:
            gid = pd.to_numeric(features["game_id"], errors="coerce").astype("Int64")
            features = features.loc[gid.isin(set(game_ids))].copy()

        # Recompute trend features (roll_mean_5, min_last1, etc.) using historical labels.
        # This is critical for live builds where raw features may be missing player-level history.
        features = self._attach_trend_features(
            features,
            labels_source=labels,
            warnings=warnings,
        )

        # Attach team-level dispersion prior using historical boxscore minutes, since the
        # live label rows have minutes=NA and cannot self-compute a "prior game" value.
        features = self._attach_team_minutes_dispersion_prior(
            features,
            labels_source=labels,
            warnings=warnings,
        )

        # Enrich with rates_training_base and vacancy context features.
        features = self._attach_team_context(features, warnings=warnings)
        features = self._attach_vacancy_features(
            features,
            labels_source=labels,
            injuries_snapshot=injuries_result.injuries,
            roster_nightly=filtered_roster,
            warnings=warnings,
        )

        # Attach DNP history features (consecutive_active_dnp, etc.)
        features = self._attach_dnp_history_features(
            features,
            labels_source=labels,
            schedule=schedule,
            warnings=warnings,
        )

        # Ensure consistent schema columns
        features = self._ensure_schema_columns(features)

        return FeatureBuildResult(
            features=features,
            injuries_result=injuries_result,
            warnings=warnings,
        )

    def _attach_team_minutes_dispersion_prior(
        self,
        df: pd.DataFrame,
        *,
        labels_source: pd.DataFrame,
        warnings: list[str],
    ) -> pd.DataFrame:
        """Attach team_minutes_dispersion_prior for live inference.

        Intended meaning (matches projections.minutes_v1.features._team_dispersion):
        - For each team, compute the per-game stddev of player minutes ("dispersion")
          from historical boxscore minutes.
        - For a target slate date, use the most recent prior game's dispersion as
          team_minutes_dispersion_prior for all players on that team.

        Why this exists:
        - Live label rows have minutes=NA, so computing a "prior game" within the
          live slice yields NaN for every team. We instead compute the prior from
          history labels and merge by team_id.
        """

        if df.empty:
            return df
        if labels_source.empty:
            warnings.append("[dispersion] labels_source is empty; leaving dispersion prior as-is.")
            return df
        if "team_id" not in df.columns:
            warnings.append("[dispersion] df missing team_id; leaving dispersion prior as-is.")
            return df

        merged = df.copy()
        if "team_minutes_dispersion_prior" not in merged.columns:
            merged["team_minutes_dispersion_prior"] = np.nan

        if "game_date" in merged.columns:
            game_dates = pd.to_datetime(merged["game_date"], errors="coerce").dt.normalize()
            target_day = pd.Timestamp(self.config.target_day).normalize()
            target_mask = game_dates == target_day
            if not bool(target_mask.any()):
                return merged
        else:
            target_day = pd.Timestamp(self.config.target_day).normalize()
            target_mask = pd.Series(True, index=merged.index, dtype=bool)

        target_slice = merged.loc[target_mask].copy()

        required = {"team_id", "game_id", "game_date", "minutes"}
        if not required.issubset(labels_source.columns):
            missing = sorted(required - set(labels_source.columns))
            warnings.append(
                f"[dispersion] labels_source missing columns {missing}; leaving dispersion prior as-is."
            )
            return merged

        history = labels_source.loc[:, ["team_id", "game_id", "game_date", "minutes"]].copy()
        history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce").dt.normalize()
        history = history.loc[history["game_date"] < target_day].copy()
        if history.empty:
            warnings.append(
                f"[dispersion] no historical label rows before {target_day.date()}; using fallback dispersion prior."
            )
            fill_val = 0.0
            values = pd.to_numeric(target_slice["team_minutes_dispersion_prior"], errors="coerce")
            target_slice["team_minutes_dispersion_prior"] = values.fillna(fill_val).astype(float)
            merged.loc[target_mask, "team_minutes_dispersion_prior"] = target_slice["team_minutes_dispersion_prior"]
            return merged

        for col in ("team_id", "game_id"):
            history[col] = pd.to_numeric(history[col], errors="coerce").astype("Int64")
        history = history.dropna(subset=["team_id", "game_id", "game_date"])
        if history.empty:
            warnings.append(
                f"[dispersion] historical label rows missing ids before {target_day.date()}; using fallback dispersion prior."
            )
            values = pd.to_numeric(target_slice["team_minutes_dispersion_prior"], errors="coerce")
            target_slice["team_minutes_dispersion_prior"] = values.fillna(0.0).astype(float)
            merged.loc[target_mask, "team_minutes_dispersion_prior"] = target_slice["team_minutes_dispersion_prior"]
            return merged

        history["minutes"] = pd.to_numeric(history["minutes"], errors="coerce")
        history = history.dropna(subset=["minutes"])
        if history.empty:
            warnings.append(
                f"[dispersion] historical minutes all NaN before {target_day.date()}; using fallback dispersion prior."
            )
            values = pd.to_numeric(target_slice["team_minutes_dispersion_prior"], errors="coerce")
            target_slice["team_minutes_dispersion_prior"] = values.fillna(0.0).astype(float)
            merged.loc[target_mask, "team_minutes_dispersion_prior"] = target_slice["team_minutes_dispersion_prior"]
            return merged

        def _nanstd_or_zero(series: pd.Series) -> float:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
            if np.isnan(values).all():
                return 0.0
            return float(np.nanstd(values, ddof=0))

        team_game = (
            history.groupby(["team_id", "game_id", "game_date"])["minutes"]
            .agg(_nanstd_or_zero)
            .reset_index(name="team_minutes_dispersion")
        )
        if team_game.empty:
            warnings.append(
                f"[dispersion] failed to compute team-game dispersion before {target_day.date()}; using fallback."
            )
            values = pd.to_numeric(target_slice["team_minutes_dispersion_prior"], errors="coerce")
            target_slice["team_minutes_dispersion_prior"] = values.fillna(0.0).astype(float)
            merged.loc[target_mask, "team_minutes_dispersion_prior"] = target_slice["team_minutes_dispersion_prior"]
            return merged

        team_game = team_game.sort_values("game_date")
        last_by_team = team_game.groupby("team_id", as_index=False).tail(1)
        priors = last_by_team.loc[:, ["team_id", "team_minutes_dispersion"]].rename(
            columns={"team_minutes_dispersion": "team_minutes_dispersion_prior"}
        )

        target_slice = target_slice.merge(priors, on="team_id", how="left", suffixes=("", "_hist"))
        if "team_minutes_dispersion_prior_hist" in target_slice.columns:
            existing = pd.to_numeric(target_slice["team_minutes_dispersion_prior"], errors="coerce")
            hist = pd.to_numeric(target_slice["team_minutes_dispersion_prior_hist"], errors="coerce")
            target_slice["team_minutes_dispersion_prior"] = existing.combine_first(hist)
            target_slice.drop(columns=["team_minutes_dispersion_prior_hist"], inplace=True)

        values = pd.to_numeric(target_slice["team_minutes_dispersion_prior"], errors="coerce")
        fill_val = float(values.mean(skipna=True)) if not values.dropna().empty else 0.0
        target_slice["team_minutes_dispersion_prior"] = values.fillna(fill_val).astype(float)
        merged.loc[target_mask, "team_minutes_dispersion_prior"] = target_slice["team_minutes_dispersion_prior"]
        return merged

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

    def _load_and_merge_rotowire_starters(
        self,
        roster: pd.DataFrame,
        *,
        warnings: list[str],
    ) -> pd.DataFrame:
        """Load Rotowire lineups and merge starter flags into roster.

        Rotowire provides faster lineup confirmations than stats.nba.com.
        This method:
        1. Loads Rotowire lineups for the target date
        2. Extracts confirmed and projected starters
        3. Upgrades is_projected_starter and is_confirmed_starter flags in roster

        Args:
            roster: Roster DataFrame to augment with starter signals.
            warnings: List to append any warnings encountered.

        Returns:
            Roster DataFrame with upgraded starter flags.
        """
        if roster.empty:
            return roster

        cfg = self.config
        roster = roster.copy()
        # Drop NBA.com lineup signals; Rotowire is the single source of truth for starters.
        for column in ("lineup_role", "lineup_status", "lineup_roster_status"):
            if column in roster.columns:
                roster[column] = pd.NA
        if "lineup_timestamp" in roster.columns:
            roster["lineup_timestamp"] = pd.NaT
        if "is_projected_starter" in roster.columns:
            roster["is_projected_starter"] = False
        if "is_confirmed_starter" in roster.columns:
            roster["is_confirmed_starter"] = False

        rotowire_path = (
            cfg.data_root
            / "silver"
            / "rotowire_lineups"
            / f"date={cfg.target_day}"
            / "lineups.parquet"
        )

        if not rotowire_path.exists():
            logger.debug(f"Rotowire lineups not found at {rotowire_path}")
            return roster

        try:
            rotowire_df = pd.read_parquet(rotowire_path)
            if rotowire_df.empty or "lineup_role" not in rotowire_df.columns:
                logger.debug("Rotowire lineups file is empty or missing lineup_role column")
                return roster

            # Filter to starters (both confirmed and projected)
            starter_roles = rotowire_df["lineup_role"].isin(["confirmed_starter", "projected_starter"])
            rotowire_starters = rotowire_df[starter_roles]

            if rotowire_starters.empty:
                logger.debug("No starters found in Rotowire lineups")
                return roster

            # Separate confirmed from projected
            confirmed_mask = rotowire_starters["lineup_role"] == "confirmed_starter"
            rotowire_confirmed_names = set(
                rotowire_starters.loc[confirmed_mask, "player_name"]
                .astype(str)
                .map(_normalize_name_for_matching)
                .unique()
            )
            rotowire_projected_names = set(
                rotowire_starters.loc[~confirmed_mask, "player_name"]
                .astype(str)
                .map(_normalize_name_for_matching)
                .unique()
            )

            logger.info(
                f"Loaded {len(rotowire_starters)} starters from Rotowire "
                f"({len(rotowire_confirmed_names)} confirmed, {len(rotowire_projected_names)} projected)"
            )

            # Upgrade starter flags in roster
            if "player_name" not in roster.columns:
                warnings.append("[rotowire] Roster missing player_name column; skipping starter merge")
                return roster

            name_normalized = roster["player_name"].map(_normalize_name_for_matching)
            all_starter_names = rotowire_confirmed_names | rotowire_projected_names
            rotowire_match = name_normalized.isin(all_starter_names)

            # Avoid conflicts: do not mark players as starters if they're already marked out
            eligible = rotowire_match.copy()
            if "active_flag" in roster.columns:
                eligible = eligible & roster["active_flag"].fillna(False).astype(bool)
            if "lineup_role" in roster.columns:
                role_norm = (
                    roster["lineup_role"]
                    .astype("string", copy=False)
                    .str.strip()
                    .str.lower()
                    .fillna("")
                )
                eligible = eligible & ~role_norm.eq("out")

            if eligible.any():
                # Initialize columns if missing
                if "is_confirmed_starter" not in roster.columns:
                    roster["is_confirmed_starter"] = False
                if "is_projected_starter" not in roster.columns:
                    roster["is_projected_starter"] = False
                if "lineup_role" not in roster.columns:
                    roster["lineup_role"] = pd.NA

                # Upgrade is_projected_starter for all starters (confirmed or projected)
                roster.loc[eligible, "is_projected_starter"] = True

                # Upgrade is_confirmed_starter only for confirmed starters
                confirmed_eligible = eligible & name_normalized.isin(rotowire_confirmed_names)
                roster.loc[confirmed_eligible, "is_confirmed_starter"] = True
                roster.loc[eligible, "lineup_role"] = "projected_starter"
                roster.loc[confirmed_eligible, "lineup_role"] = "confirmed_starter"

                projected_count = int(eligible.sum())
                confirmed_count = int(confirmed_eligible.sum())
                logger.info(
                    f"Upgraded {projected_count} players to projected starters, "
                    f"{confirmed_count} to confirmed based on Rotowire"
                )

            return roster

        except Exception as exc:
            warnings.append(f"Failed to load Rotowire lineups: {exc}")
            logger.warning(f"Failed to load Rotowire lineups: {exc}")
            return roster

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
        # Track which live-critical columns were present on input so we can expose
        # diagnostic "missing" flags without changing core feature semantics.
        input_columns = set(df.columns)
        starter_flag_provided = "starter_flag" in input_columns
        starter_signal_cols = [
            col
            for col in ("is_confirmed_starter", "is_projected_starter", "is_starter")
            if col in input_columns
        ]
        has_starter_signal = bool(starter_signal_cols)
        team_ctx_provided = set(_TEAM_CONTEXT_COLUMNS).issubset(input_columns)
        opp_ctx_provided = set(_OPP_CONTEXT_COLUMNS).issubset(input_columns)
        vac_ctx_provided = set(_VACANCY_FEATURE_COLUMNS).issubset(input_columns)

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

        # Live scoring bundle expects these columns to exist, even if upstream
        # enrichment sources are unavailable.
        if "starter_flag" not in df.columns or (
            has_starter_signal
            and pd.to_numeric(df["starter_flag"], errors="coerce").fillna(0).astype(int).sum() == 0
        ):
            # Derive starter_flag from the best available starter signals.
            #
            # Rules (tested in tests/test_features_builder_schema_defaults.py):
            # - If starter_flag is missing: derive from signals when present, else default to 0.
            # - If starter_flag exists but is all zeros AND a starter signal exists: override with signal.
            if has_starter_signal:
                starter_flag = pd.Series(False, index=df.index, dtype=bool)
                for col in starter_signal_cols:
                    series = df[col]
                    if pd.api.types.is_bool_dtype(series):
                        starter_flag = starter_flag | series.fillna(False)
                    else:
                        starter_flag = starter_flag | pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
                df["starter_flag"] = starter_flag.astype(int)
            else:
                df["starter_flag"] = 0
        else:
            df["starter_flag"] = pd.to_numeric(df["starter_flag"], errors="coerce").fillna(0).astype(int)

        # ------------------------------------------------------------------
        # Starter fallback: ensure each team has 5 pre-tip starters when possible.
        #
        # Live lineup sources can be incomplete (late posting, team-specific feed gaps).
        # When a team has <5 starters flagged, derive a best-effort starter set using
        # history-driven signals (starter_prev_game_asof, recent_start_pct_10) and
        # minutes priors (roll_mean_10) while excluding OUT players.
        # ------------------------------------------------------------------
        if {"game_id", "team_id", "player_id"}.issubset(df.columns):
            if "is_projected_starter" not in df.columns:
                df["is_projected_starter"] = False
            if "is_confirmed_starter" not in df.columns:
                df["is_confirmed_starter"] = False
            if "lineup_role" not in df.columns:
                df["lineup_role"] = pd.NA

            # Avoid pandas FutureWarning about silent downcasting on fillna for object dtype.
            df["is_projected_starter"] = (
                df["is_projected_starter"].astype("boolean", copy=False).fillna(False).astype(bool)
            )
            df["is_confirmed_starter"] = (
                df["is_confirmed_starter"].astype("boolean", copy=False).fillna(False).astype(bool)
            )

            is_out = pd.to_numeric(df.get("is_out", 0), errors="coerce").fillna(0).astype(int)
            not_out = is_out == 0
            # Never treat OUT players as starters.
            df["starter_flag"] = (
                (pd.to_numeric(df["starter_flag"], errors="coerce").fillna(0).astype(int) > 0) & not_out
            ).astype(int)

            def _numeric_series(name: str, default: float) -> pd.Series:
                if name not in df.columns:
                    return pd.Series(default, index=df.index, dtype="float64")
                return pd.to_numeric(df[name], errors="coerce").fillna(default).astype("float64")

            prev_start = _numeric_series("starter_prev_game_asof", 0.0)
            recent_pct = _numeric_series("recent_start_pct_10", 0.0)
            roll_mean_10 = _numeric_series("roll_mean_10", 0.0)
            player_id_num = pd.to_numeric(df["player_id"], errors="coerce").fillna(0).astype(int)

            needs_fill = 0
            for (game_id, team_id), group in df.groupby(["game_id", "team_id"], sort=False):
                idx = group.index
                # Unit tests and some edge pipelines may pass partial team frames.
                # Only attempt 5-starter enforcement when the team slice has enough players.
                if len(idx) < 5:
                    continue
                cur = (
                    group["is_confirmed_starter"].fillna(False).astype(bool)
                    | group["is_projected_starter"].fillna(False).astype(bool)
                    | (pd.to_numeric(group["starter_flag"], errors="coerce").fillna(0).astype(int) > 0)
                ) & not_out.loc[idx].astype(bool)
                n = int(cur.sum())
                if n == 5:
                    continue
                needs_fill += 1

                # Build a deterministic ranking for eligible players.
                rank_frame = pd.DataFrame(
                    {
                        "cur": cur.astype(int),
                        "not_out": not_out.loc[idx].astype(int),
                        "prev_start": prev_start.loc[idx].astype(float),
                        "recent_pct": recent_pct.loc[idx].astype(float),
                        "roll_mean_10": roll_mean_10.loc[idx].astype(float),
                        "player_id": player_id_num.loc[idx].astype(int),
                    },
                    index=idx,
                )

                eligible = rank_frame["not_out"] == 1
                if int(eligible.sum()) < 5:
                    continue

                if n >= 5:
                    # Too many starters: keep the top 5 by ranking, prioritizing confirmed starters.
                    confirmed = group["is_confirmed_starter"].fillna(False).astype(bool) & not_out.loc[idx].astype(bool)
                    keep = pd.Series(False, index=idx)
                    if int(confirmed.sum()) >= 5:
                        order = rank_frame.loc[confirmed].sort_values(
                            ["prev_start", "recent_pct", "roll_mean_10", "player_id"],
                            ascending=[False, False, False, True],
                        )
                        keep.loc[order.head(5).index] = True
                    else:
                        keep.loc[confirmed.index] = confirmed.astype(bool)
                        remaining = 5 - int(confirmed.sum())
                        pool = rank_frame.loc[eligible & ~confirmed].sort_values(
                            ["prev_start", "recent_pct", "roll_mean_10", "player_id"],
                            ascending=[False, False, False, True],
                        )
                        keep.loc[pool.head(remaining).index] = True

                    df.loc[idx, "starter_flag"] = keep.astype(int)
                    df.loc[idx, "is_projected_starter"] = keep.astype(bool)
                else:
                    # Not enough starters: fill up to 5 using ranking among eligible non-starters.
                    need = 5 - n
                    pool = rank_frame.loc[eligible & (rank_frame["cur"] == 0)].sort_values(
                        ["prev_start", "recent_pct", "roll_mean_10", "player_id"],
                        ascending=[False, False, False, True],
                    )
                    add_idx = pool.head(need).index
                    if len(add_idx) == 0:
                        continue
                    df.loc[add_idx, "is_projected_starter"] = True
                    df.loc[add_idx, "starter_flag"] = 1
                    if "lineup_role" in df.columns:
                        role_series = df.loc[add_idx, "lineup_role"].astype("string").fillna("")
                        empty = role_series.eq("") | role_series.str.lower().isin({"none", "nan"})
                        df.loc[add_idx[empty.to_numpy()], "lineup_role"] = "projected_starter"

            if needs_fill:
                logger.info(f"[starters] filled starters for {needs_fill} team(s) with incomplete lineup signals.")

        # ------------------------------------------------------------------
        # Lineup semantics: fill categorical lineup fields expected by RMH
        # training and downstream consumers.
        #
        # Training distribution (rotation_train_v1_boxscore_20260112):
        # - lineup_role: bench / confirmed_starter / projected_starter / out
        # - lineup_status: Confirmed for all non-projected, Expected for projected
        # - lineup_roster_status: Active for non-out, Inactive for out
        #
        # Live feeds are often sparse (starters only), so we derive a complete
        # per-player set using starter flags + OUT/active_flag.
        # ------------------------------------------------------------------
        if {"game_id", "team_id", "player_id"}.issubset(df.columns):
            if "lineup_role" not in df.columns:
                df["lineup_role"] = pd.NA
            if "lineup_status" not in df.columns:
                df["lineup_status"] = pd.NA
            if "lineup_roster_status" not in df.columns:
                df["lineup_roster_status"] = pd.NA

            # Determine which rows need role filling.
            role_series = df["lineup_role"].astype("string").fillna("").str.strip()
            role_missing = role_series.eq("") | role_series.str.lower().isin({"none", "nan"})

            # Use active_flag when available (captures roster inactive cases not on injury report).
            if "active_flag" in df.columns:
                active_flag = df["active_flag"].fillna(False).astype(bool)
            else:
                active_flag = pd.Series(True, index=df.index, dtype=bool)

            is_out = pd.to_numeric(df.get("is_out", 0), errors="coerce").fillna(0).astype(int)
            out_like = (is_out == 1) | (~active_flag)

            confirmed = df.get("is_confirmed_starter", False)
            confirmed = confirmed.fillna(False).astype(bool) if isinstance(confirmed, pd.Series) else pd.Series(False, index=df.index, dtype=bool)
            projected = df.get("is_projected_starter", False)
            projected = projected.fillna(False).astype(bool) if isinstance(projected, pd.Series) else pd.Series(False, index=df.index, dtype=bool)
            starter_flag = pd.to_numeric(df.get("starter_flag", 0), errors="coerce").fillna(0).astype(int) > 0

            derived_role = pd.Series("bench", index=df.index, dtype="string[pyarrow]")
            derived_role.loc[out_like] = "out"
            derived_role.loc[~out_like & confirmed] = "confirmed_starter"
            derived_role.loc[~out_like & ~confirmed & (projected | starter_flag)] = "projected_starter"
            df.loc[role_missing, "lineup_role"] = derived_role.loc[role_missing]

            # lineup_roster_status: Active vs Inactive (derived from role/out_like).
            roster_status_series = df["lineup_roster_status"].astype("string").fillna("").str.strip()
            roster_missing = roster_status_series.eq("") | roster_status_series.str.lower().isin({"none", "nan"})
            derived_roster_status = pd.Series("Active", index=df.index, dtype="string[pyarrow]")
            derived_roster_status.loc[df["lineup_role"].astype("string").str.lower() == "out"] = "Inactive"
            df.loc[roster_missing, "lineup_roster_status"] = derived_roster_status.loc[roster_missing]

            # lineup_status: Expected only for projected starters, otherwise Confirmed.
            status_series = df["lineup_status"].astype("string").fillna("").str.strip()
            status_missing = status_series.eq("") | status_series.str.lower().isin({"none", "nan"})
            derived_status = pd.Series("Confirmed", index=df.index, dtype="string[pyarrow]")
            derived_status.loc[df["lineup_role"].astype("string").str.lower() == "projected_starter"] = "Expected"
            df.loc[status_missing, "lineup_status"] = derived_status.loc[status_missing]

        for col in _TEAM_CONTEXT_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        for col in _OPP_CONTEXT_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        for col in _VACANCY_FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        # Diagnostic flags (bool dtype, row-aligned) used by tests and health checks.
        df["starter_flag_missing"] = pd.Series(
            not starter_flag_provided and not has_starter_signal, index=df.index, dtype=bool
        )
        df["team_ctx_missing"] = pd.Series(not team_ctx_provided, index=df.index, dtype=bool)
        df["opp_ctx_missing"] = pd.Series(not opp_ctx_provided, index=df.index, dtype=bool)
        df["vac_ctx_missing"] = pd.Series(not vac_ctx_provided, index=df.index, dtype=bool)

        return df

    def _attach_trend_features(
        self,
        df: pd.DataFrame,
        *,
        labels_source: pd.DataFrame,
        warnings: list[str],
    ) -> pd.DataFrame:
        """Recompute trend features from historical labels.

        Trend features like roll_mean_5, min_last1, min_last3 require player-level
        boxscore history. For live builds, this history comes from labels_source
        (the combined history + live labels). We compute the latest trend for each
        player from games BEFORE target_day and merge onto the live slice.

        This is critical for live Prefect builds where the MinutesFeatureBuilder
        may not have access to sufficient player history.
        """
        if df.empty:
            return df

        trend_cols = [
            "min_last1",
            "min_last3",
            "min_last5",
            "sum_min_7d",
            "roll_mean_3",
            "roll_mean_5",
            "roll_mean_10",
            "roll_iqr_5",
            "z_vs_10",
        ]

        # Check if labels_source has required columns
        required = {"player_id", "game_date", "minutes"}
        if not required.issubset(labels_source.columns):
            missing = sorted(required - set(labels_source.columns))
            warnings.append(f"[trend] labels_source missing columns {missing}; skipping trend recomputation.")
            return df

        try:
            target_day = pd.Timestamp(self.config.target_day).normalize()

            history_work = labels_source.copy()
            history_work["game_date"] = pd.to_datetime(history_work["game_date"]).dt.normalize()
            # Only use history BEFORE target day
            history_work = history_work[history_work["game_date"] < target_day].copy()
            if history_work.empty:
                warnings.append(f"[trend] No historical labels before {target_day.date()}; trend features will be NaN.")
                return df

            history_work["player_id"] = pd.to_numeric(history_work["player_id"], errors="coerce").astype("Int64")
            history_work = history_work.dropna(subset=["player_id"])
            history_work.sort_values(["player_id", "game_date"], inplace=True)

            cutoff_7d = target_day - pd.Timedelta(days=7)
            latest_by_player: list[dict[str, object]] = []

            for pid, group in history_work.groupby("player_id"):
                minutes = pd.to_numeric(group["minutes"], errors="coerce")
                dates = pd.to_datetime(group["game_date"]).dt.normalize()
                if minutes.empty:
                    continue

                # Filter out 0-minute games (DNP/injury) for rolling averages
                played_mask = minutes > 0
                played_minutes = minutes[played_mask]

                # Use last played game for min_last1 (not DNP games)
                last_minutes = played_minutes.iloc[-1] if not played_minutes.empty else 0.0

                # Rolling averages over games actually played
                last3 = played_minutes.tail(3).mean() if len(played_minutes) >= 1 else np.nan
                last5 = played_minutes.tail(5).mean() if len(played_minutes) >= 1 else np.nan
                mean3 = last3
                mean5 = last5
                mean10 = played_minutes.tail(10).mean() if len(played_minutes) >= 1 else np.nan
                iqr5 = (
                    played_minutes.tail(5).quantile(0.75) - played_minutes.tail(5).quantile(0.25)
                    if len(played_minutes.tail(5)) >= 2
                    else 0.0
                )

                # sum_min_7d uses all games in window (including DNPs, as it's schedule-aware)
                recent_window = minutes[dates >= cutoff_7d]
                sum7 = float(recent_window.sum()) if not recent_window.empty else 0.0

                # z-score over played games
                last10_played = played_minutes.tail(10)
                mu10 = last10_played.mean() if not last10_played.empty else 0.0
                std10 = last10_played.std(ddof=0) if not last10_played.empty else 0.0
                z10 = float((last_minutes - mu10) / std10) if std10 and std10 > 0 else 0.0

                latest_by_player.append(
                    {
                        "player_id": pid,
                        "min_last1": float(last_minutes),
                        "min_last3": float(last3) if pd.notna(last3) else np.nan,
                        "min_last5": float(last5) if pd.notna(last5) else np.nan,
                        "sum_min_7d": float(sum7),
                        "roll_mean_3": float(mean3) if pd.notna(mean3) else np.nan,
                        "roll_mean_5": float(mean5) if pd.notna(mean5) else np.nan,
                        "roll_mean_10": float(mean10) if pd.notna(mean10) else np.nan,
                        "roll_iqr_5": float(iqr5) if pd.notna(iqr5) else 0.0,
                        "z_vs_10": z10,
                    }
                )

            if not latest_by_player:
                warnings.append("[trend] No player history computed; trend features will be NaN.")
                return df

            trend_frame = pd.DataFrame(latest_by_player)
            trend_frame["player_id"] = pd.to_numeric(trend_frame["player_id"], errors="coerce").astype("Int64")

            # Merge onto features, preferring computed values over existing
            merged = df.copy()
            merged["player_id"] = pd.to_numeric(merged["player_id"], errors="coerce").astype("Int64")
            merged = merged.merge(trend_frame, on="player_id", how="left", suffixes=("", "_recomp"))

            for col in trend_cols:
                recomputed = f"{col}_recomp"
                if recomputed in merged.columns:
                    merged[col] = pd.to_numeric(merged[recomputed], errors="coerce").combine_first(
                        pd.to_numeric(merged.get(col), errors="coerce")
                    )
                    merged.drop(columns=[recomputed], inplace=True)

            return merged

        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"[trend] recompute failed: {exc}")
            return df

    def _load_team_context_from_rates_training_base(
        self,
        *,
        team_ids: set[int],
        warnings: list[str],
        max_days_back: int = 14,
    ) -> pd.DataFrame:
        cfg = self.config
        if not team_ids:
            return pd.DataFrame(columns=["team_id", *_TEAM_CONTEXT_COLUMNS])

        root = cfg.data_root / "gold" / "rates_training_base" / f"season={cfg.season}"
        if not root.exists():
            warnings.append(f"[team-context] Missing rates_training_base root at {root}.")
            return pd.DataFrame(columns=["team_id", *_TEAM_CONTEXT_COLUMNS])

        target_day = pd.Timestamp(cfg.target_day).normalize()
        candidates: list[tuple[pd.Timestamp, Path]] = []
        for day_dir in root.glob("game_date=*"):
            try:
                day_str = day_dir.name.split("=", 1)[1]
                day = pd.Timestamp(day_str).normalize()
            except Exception:
                continue
            if day < target_day:
                pq_path = day_dir / "rates_training_base.parquet"
                if pq_path.exists():
                    candidates.append((day, pq_path))

        if not candidates:
            warnings.append("[team-context] No prior rates_training_base partitions available.")
            return pd.DataFrame(columns=["team_id", *_TEAM_CONTEXT_COLUMNS])

        # rates_training_base is partitioned by game_date and only contains teams/players
        # active that day. For a slate date, many teams did not play on the immediately
        # prior date, so we need to backfill per-team from earlier partitions.
        candidates.sort(key=lambda item: item[0], reverse=True)
        remaining = set(team_ids)
        frames: list[pd.DataFrame] = []
        required_cols = ["team_id", *_TEAM_CONTEXT_COLUMNS]

        for idx, (_day, pq_path) in enumerate(candidates):
            if not remaining or idx >= max_days_back:
                break
            try:
                day_df = pd.read_parquet(pq_path, columns=required_cols)
            except Exception:  # noqa: BLE001
                # If column projection fails (older parquet), fall back to full read.
                try:
                    day_df = pd.read_parquet(pq_path)
                except Exception:  # noqa: BLE001
                    continue
                if not set(required_cols).issubset(day_df.columns):
                    continue
                day_df = day_df.loc[:, required_cols]

            if day_df.empty:
                continue
            day_df["team_id"] = pd.to_numeric(day_df["team_id"], errors="coerce").astype("Int64")
            day_df = day_df.dropna(subset=["team_id"])
            if day_df.empty:
                continue
            day_df = day_df.loc[day_df["team_id"].astype(int).isin(remaining)].copy()
            if day_df.empty:
                continue

            grouped = day_df.groupby("team_id", as_index=False)[list(_TEAM_CONTEXT_COLUMNS)].mean()
            frames.append(grouped)
            found = set(grouped["team_id"].dropna().astype(int).tolist())
            remaining -= found

        if not frames:
            warnings.append(
                "[team-context] Failed to load any usable rates_training_base partitions for team context."
            )
            return pd.DataFrame(columns=["team_id", *_TEAM_CONTEXT_COLUMNS])

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.dropna(subset=["team_id"]).drop_duplicates(subset=["team_id"], keep="first")
        combined["team_id"] = combined["team_id"].astype(int)
        for col in _TEAM_CONTEXT_COLUMNS:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
        return combined.reset_index(drop=True)

    def _attach_team_context(self, df: pd.DataFrame, *, warnings: list[str]) -> pd.DataFrame:
        if df.empty:
            return df

        merged = df.copy()
        for col in _TEAM_CONTEXT_COLUMNS:
            if col not in merged.columns:
                merged[col] = np.nan
        for col in _OPP_CONTEXT_COLUMNS:
            if col not in merged.columns:
                merged[col] = np.nan

        if "team_id" in merged.columns:
            team_ids = set(
                pd.to_numeric(merged["team_id"], errors="coerce").dropna().astype(int).tolist()
            )
        else:
            team_ids = set()
        if "opponent_team_id" in merged.columns:
            opp_ids = set(
                pd.to_numeric(merged["opponent_team_id"], errors="coerce").dropna().astype(int).tolist()
            )
        else:
            opp_ids = set()
        context_team_ids = team_ids | opp_ids

        context = self._load_team_context_from_rates_training_base(
            team_ids=context_team_ids, warnings=warnings
        )
        if not context.empty and "team_id" in merged.columns:
            merged = merged.merge(context, on="team_id", how="left", suffixes=("", "_ctx"))
            for col in _TEAM_CONTEXT_COLUMNS:
                ctx_col = f"{col}_ctx"
                if ctx_col in merged.columns:
                    merged[col] = pd.to_numeric(merged[ctx_col], errors="coerce").combine_first(
                        pd.to_numeric(merged[col], errors="coerce")
                    )
                    merged.drop(columns=[ctx_col], inplace=True)

            if "opponent_team_id" in merged.columns:
                opp_context = context.rename(
                    columns={
                        "team_id": "opponent_team_id",
                        "team_pace_szn": "opp_pace_szn",
                        "team_def_rtg_szn": "opp_def_rtg_szn",
                    }
                )
                opp_cols = ["opponent_team_id", *_OPP_CONTEXT_COLUMNS]
                opp_cols = [col for col in opp_cols if col in opp_context.columns]
                if len(opp_cols) > 1:
                    merged = merged.merge(
                        opp_context.loc[:, opp_cols],
                        on="opponent_team_id",
                        how="left",
                        suffixes=("", "_oppctx"),
                    )
                    for col in _OPP_CONTEXT_COLUMNS:
                        ctx_col = f"{col}_oppctx"
                        if ctx_col in merged.columns:
                            merged[col] = pd.to_numeric(merged[ctx_col], errors="coerce").combine_first(
                                pd.to_numeric(merged[col], errors="coerce")
                            )
                            merged.drop(columns=[ctx_col], inplace=True)

        # Fill remaining NaNs with a conservative fallback aligned with training enrichment:
        # use the within-slate mean when available, else a generic 100.0.
        for col in (*_TEAM_CONTEXT_COLUMNS, *_OPP_CONTEXT_COLUMNS):
            values = pd.to_numeric(merged[col], errors="coerce")
            mean_val = float(values.mean(skipna=True)) if not values.dropna().empty else 100.0
            merged[col] = values.fillna(mean_val).astype(float)

        return merged

    def _attach_vacancy_features(
        self,
        df: pd.DataFrame,
        *,
        labels_source: pd.DataFrame,
        injuries_snapshot: pd.DataFrame,
        roster_nightly: pd.DataFrame,
        warnings: list[str],
    ) -> pd.DataFrame:
        if df.empty:
            return df
        if injuries_snapshot.empty:
            return df

        required = {"game_id", "player_id", "status"}
        if not required.issubset(injuries_snapshot.columns):
            missing = sorted(required - set(injuries_snapshot.columns))
            warnings.append(
                f"[vacancy] injuries_snapshot missing required columns {missing}; leaving vacancy features at defaults."
            )
            return df

        injuries = injuries_snapshot.copy()
        injuries["game_id"] = pd.to_numeric(injuries["game_id"], errors="coerce").astype("Int64")
        injuries["player_id"] = pd.to_numeric(injuries["player_id"], errors="coerce").astype("Int64")
        injuries = injuries.dropna(subset=["game_id", "player_id"])
        if injuries.empty:
            return df

        status = injuries["status"].astype(str).str.upper().str.strip()
        injuries = injuries.loc[status.isin(_OUT_LIKE_STATUS_VALUES)].copy()
        if injuries.empty:
            return df

        # Map team_id / listed_pos from roster if needed.
        roster_cols = [
            col for col in ("game_id", "player_id", "team_id", "listed_pos") if col in roster_nightly.columns
        ]
        if roster_cols:
            roster_map = roster_nightly.loc[:, roster_cols].dropna(subset=["game_id", "player_id"]).copy()
        else:
            roster_map = pd.DataFrame()
        if not roster_map.empty:
            roster_map["game_id"] = pd.to_numeric(roster_map["game_id"], errors="coerce").astype("Int64")
            roster_map["player_id"] = pd.to_numeric(roster_map["player_id"], errors="coerce").astype("Int64")
            if "team_id" in roster_map.columns:
                roster_map["team_id"] = pd.to_numeric(roster_map["team_id"], errors="coerce").astype("Int64")
            roster_map = roster_map.dropna(subset=["game_id", "player_id"]).drop_duplicates(
                subset=["game_id", "player_id"], keep="last"
            )
            injuries = injuries.merge(
                roster_map,
                on=["game_id", "player_id"],
                how="left",
                suffixes=("", "_roster"),
            )
            if "team_id" in injuries.columns and "team_id_roster" in injuries.columns:
                injuries["team_id"] = injuries["team_id"].fillna(injuries["team_id_roster"])
            elif "team_id_roster" in injuries.columns and "team_id" not in injuries.columns:
                injuries = injuries.rename(columns={"team_id_roster": "team_id"})
            if "listed_pos" in injuries.columns and "listed_pos_roster" in injuries.columns:
                injuries["listed_pos"] = injuries["listed_pos"].fillna(injuries["listed_pos_roster"])
            elif "listed_pos_roster" in injuries.columns and "listed_pos" not in injuries.columns:
                injuries = injuries.rename(columns={"listed_pos_roster": "listed_pos"})
            injuries.drop(columns=["team_id_roster", "listed_pos_roster"], inplace=True, errors="ignore")

        if "team_id" not in injuries.columns:
            warnings.append("[vacancy] Missing team_id mapping; leaving vacancy features at defaults.")
            return df

        injuries["team_id"] = pd.to_numeric(injuries["team_id"], errors="coerce").astype("Int64")
        injuries = injuries.dropna(subset=["team_id"])
        if injuries.empty:
            return df
        injuries["team_id"] = injuries["team_id"].astype(int)

        labels_required = {"player_id", "game_date", "minutes"}
        if not labels_required.issubset(labels_source.columns):
            missing = sorted(labels_required - set(labels_source.columns))
            warnings.append(
                f"[vacancy] labels_source missing columns {missing}; leaving vacancy features at defaults."
            )
            return df

        target_day = pd.Timestamp(self.config.target_day).normalize()
        history = labels_source.loc[:, ["player_id", "game_date", "minutes"]].copy()
        history["game_date"] = pd.to_datetime(history["game_date"]).dt.normalize()
        history = history.loc[history["game_date"] < target_day].copy()
        if history.empty:
            return df

        history["player_id"] = pd.to_numeric(history["player_id"], errors="coerce").astype("Int64")
        history = history.dropna(subset=["player_id"])
        if history.empty:
            return df

        out_player_ids = set(injuries["player_id"].dropna().astype(int).unique().tolist())
        if not out_player_ids:
            return df
        history = history.loc[history["player_id"].astype(int).isin(out_player_ids)].copy()
        if history.empty:
            return df

        history["minutes"] = pd.to_numeric(history["minutes"], errors="coerce").fillna(0.0).astype(float)
        player_hist = (
            history.groupby("player_id", as_index=False)["minutes"]
            .sum()
            .rename(columns={"minutes": "hist_minutes_szn"})
        )
        player_hist["player_id"] = player_hist["player_id"].astype(int)
        player_hist["hist_minutes_szn"] = player_hist["hist_minutes_szn"].astype(float)

        injuries = injuries.merge(player_hist, on="player_id", how="left")
        injuries["hist_minutes_szn"] = injuries["hist_minutes_szn"].fillna(0.0).astype(float)

        if "listed_pos" in injuries.columns:
            injuries["pos_bucket"] = canonical_pos_bucket_series(injuries["listed_pos"])
        else:
            injuries["pos_bucket"] = "UNK"

        pivot = (
            injuries.pivot_table(
                index=["game_id", "team_id"],
                columns="pos_bucket",
                values="hist_minutes_szn",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
        )
        pivot.columns.name = None
        pivot = pivot.rename(
            columns={
                "G": "vac_min_guard_szn",
                "W": "vac_min_wing_szn",
                "BIG": "vac_min_big_szn",
            }
        )
        for col in ("vac_min_guard_szn", "vac_min_wing_szn", "vac_min_big_szn"):
            if col not in pivot.columns:
                pivot[col] = 0.0
        pivot["vac_min_szn"] = (
            pd.to_numeric(pivot["vac_min_guard_szn"], errors="coerce").fillna(0.0)
            + pd.to_numeric(pivot["vac_min_wing_szn"], errors="coerce").fillna(0.0)
            + pd.to_numeric(pivot["vac_min_big_szn"], errors="coerce").fillna(0.0)
        ).astype(float)

        merged = df.merge(
            pivot.loc[:, ["game_id", "team_id", *_VACANCY_FEATURE_COLUMNS]],
            on=["game_id", "team_id"],
            how="left",
        )
        for col in _VACANCY_FEATURE_COLUMNS:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0).astype(float)
        return merged

    def _attach_dnp_history_features(
        self,
        df: pd.DataFrame,
        *,
        labels_source: pd.DataFrame,
        schedule: pd.DataFrame,
        warnings: list[str],
    ) -> pd.DataFrame:
        """Attach DNP history features (consecutive_active_dnp, etc.) from historical labels.

        These features track active-but-DNP patterns critical for availability modeling.
        The rotation model uses these to down-weight dust players with DNP-CD streaks.
        """
        if df.empty:
            return df

        # Validate required columns in labels_source
        required = {"player_id", "team_id", "game_date", "minutes"}
        if not required.issubset(labels_source.columns):
            missing = sorted(required - set(labels_source.columns))
            warnings.append(f"[dnp_history] labels_source missing columns {missing}; skipping.")
            # Initialize with defaults
            for col in DNP_HISTORY_FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0 if "streak" in col or "consecutive" in col or "games_since" in col else 0.0
            return df

        try:
            # Prepare current game DataFrame (today's players)
            current_df = df.copy()

            # Prepare historical DataFrame (prior games with realized minutes)
            target_day = pd.Timestamp(self.config.target_day).normalize()
            historical = labels_source.copy()
            historical["game_date"] = pd.to_datetime(historical["game_date"]).dt.normalize()
            historical = historical[historical["game_date"] < target_day].copy()

            if historical.empty:
                warnings.append(f"[dnp_history] No historical data before {target_day.date()}; using defaults.")
                for col in DNP_HISTORY_FEATURE_COLUMNS:
                    if col not in df.columns:
                        df[col] = 0 if "streak" in col or "consecutive" in col or "games_since" in col else 0.0
                return df

            # ------------------------------------------------------------------
            # Training parity for DNP history requires OUT games to be present in
            # the historical frame. Boxscore labels can omit OUT players entirely,
            # so we build a per-team schedule spine and fill minutes=0 when absent.
            #
            # Steps:
            # 1) enumerate all prior team games for each (player_id, team_id) in current_df
            # 2) merge boxscore minutes (default 0 if missing)
            # 3) derive `is_out` from injuries snapshot as-of tip_ts per game
            # ------------------------------------------------------------------
            if not {"player_id", "team_id"}.issubset(current_df.columns):
                warnings.append("[dnp_history] current_df missing player_id/team_id; using defaults.")
                for col in DNP_HISTORY_FEATURE_COLUMNS:
                    if col not in df.columns:
                        df[col] = 0 if "streak" in col or "consecutive" in col or "games_since" in col else 0.0
                return df

            pairs = current_df.loc[:, ["player_id", "team_id"]].copy()
            pairs["player_id"] = pd.to_numeric(pairs["player_id"], errors="coerce").astype("Int64")
            pairs["team_id"] = pd.to_numeric(pairs["team_id"], errors="coerce").astype("Int64")
            pairs = pairs.dropna(subset=["player_id", "team_id"]).drop_duplicates()
            if pairs.empty:
                return df

            sched_required = {"game_id", "game_date", "tip_ts", "home_team_id", "away_team_id"}
            if not sched_required.issubset(schedule.columns):
                warnings.append(
                    f"[dnp_history] schedule missing columns {sorted(sched_required - set(schedule.columns))}; using defaults."
                )
                for col in DNP_HISTORY_FEATURE_COLUMNS:
                    if col not in df.columns:
                        df[col] = 0 if "streak" in col or "consecutive" in col or "games_since" in col else 0.0
                return df

            sched = schedule.copy()
            sched["game_date"] = pd.to_datetime(sched["game_date"], errors="coerce").dt.normalize()
            sched["tip_ts"] = pd.to_datetime(sched["tip_ts"], utc=True, errors="coerce")

            home = sched.loc[:, ["game_id", "game_date", "tip_ts", "home_team_id"]].rename(
                columns={"home_team_id": "team_id"}
            )
            away = sched.loc[:, ["game_id", "game_date", "tip_ts", "away_team_id"]].rename(
                columns={"away_team_id": "team_id"}
            )
            team_games = pd.concat([home, away], ignore_index=True)
            team_games["team_id"] = pd.to_numeric(team_games["team_id"], errors="coerce").astype("Int64")
            team_games["game_id"] = pd.to_numeric(team_games["game_id"], errors="coerce").astype("Int64")
            team_games = team_games.dropna(subset=["team_id", "game_id", "game_date"])
            team_games = team_games.loc[team_games["game_date"] < target_day].copy()
            team_games = team_games.loc[team_games["team_id"].isin(pairs["team_id"].unique())].copy()
            if team_games.empty:
                return df

            history_spine = pairs.merge(team_games, on="team_id", how="inner")
            if history_spine.empty:
                return df

            # Avoid counting games before the player is first observed with this team in labels history.
            hist_labels = historical.copy()
            hist_labels["player_id"] = pd.to_numeric(hist_labels["player_id"], errors="coerce").astype("Int64")
            hist_labels["team_id"] = pd.to_numeric(hist_labels["team_id"], errors="coerce").astype("Int64")
            first_seen = (
                hist_labels.dropna(subset=["player_id", "team_id"])
                .groupby(["player_id", "team_id"], as_index=False)["game_date"]
                .min()
                .rename(columns={"game_date": "first_seen_game_date"})
            )
            history_spine = history_spine.merge(first_seen, on=["player_id", "team_id"], how="left")
            history_spine = history_spine.dropna(subset=["first_seen_game_date"]).copy()
            history_spine = history_spine.loc[history_spine["game_date"] >= history_spine["first_seen_game_date"]].copy()
            history_spine = history_spine.drop(columns=["first_seen_game_date"], errors="ignore")
            if history_spine.empty:
                return df

            # Merge realized minutes from labels; missing => minutes=0.
            minutes = hist_labels.loc[:, ["game_id", "player_id", "team_id", "minutes"]].copy()
            minutes["game_id"] = pd.to_numeric(minutes["game_id"], errors="coerce").astype("Int64")
            minutes = minutes.dropna(subset=["game_id", "player_id", "team_id"])
            minutes["minutes"] = pd.to_numeric(minutes["minutes"], errors="coerce").fillna(0.0).astype(float)
            minutes = minutes.drop_duplicates(subset=["game_id", "player_id", "team_id"], keep="last")
            history_spine = history_spine.merge(minutes, on=["game_id", "player_id", "team_id"], how="left")
            history_spine["minutes"] = pd.to_numeric(history_spine["minutes"], errors="coerce").fillna(0.0).astype(float)

            injuries_root = self.config.data_root / "silver" / "injuries_snapshot" / f"season={self.config.season}"
            injuries_files = sorted(injuries_root.glob("**/*.parquet")) if injuries_root.exists() else []
            if injuries_files:
                injuries_frames: list[pd.DataFrame] = []
                for path in injuries_files:
                    try:
                        injuries_frames.append(
                            pd.read_parquet(
                                path,
                                columns=[
                                    "game_id",
                                    "player_id",
                                    "status",
                                    "as_of_ts",
                                    "restriction_flag",
                                    "ramp_flag",
                                    "games_since_return",
                                    "days_since_return",
                                ],
                            )
                        )
                    except Exception:  # noqa: BLE001
                        injuries_frames.append(pd.read_parquet(path))
                injuries_snapshot = (
                    pd.concat(injuries_frames, ignore_index=True) if injuries_frames else pd.DataFrame()
                )
            else:
                injuries_snapshot = pd.DataFrame()

            if not injuries_snapshot.empty:
                from projections.features import availability as availability_features

                # Normalize join keys so the schedule spine can match the injury snapshot.
                for col in ("game_id", "player_id"):
                    if col in injuries_snapshot.columns:
                        injuries_snapshot[col] = pd.to_numeric(injuries_snapshot[col], errors="coerce").astype("Int64")
                injuries_snapshot = injuries_snapshot.dropna(subset=["game_id", "player_id"])

                for col in ("game_id", "player_id"):
                    if col in history_spine.columns:
                        history_spine[col] = pd.to_numeric(history_spine[col], errors="coerce").astype("Int64")
                history_spine = history_spine.dropna(subset=["game_id", "player_id"])

                prepared = availability_features.prepare_injuries_snapshot(injuries_snapshot)
                history_spine = availability_features.attach_availability_features(
                    history_spine, prepared_injuries=prepared, injuries_snapshot=injuries_snapshot
                )
            else:
                # No injuries history available; fall back to conservative "active" interpretation.
                history_spine["is_out"] = 0

            # Call the live inference function
            result = compute_dnp_history_features_for_live(
                current_game_df=current_df,
                historical_df=history_spine,
                game_date_col="game_date",
                player_id_col="player_id",
                team_id_col="team_id",
                is_out_col="is_out",
                minutes_col="minutes",
            )

            # Copy computed columns to original DataFrame
            for col in DNP_HISTORY_FEATURE_COLUMNS:
                if col in result.columns:
                    df[col] = result[col].values

            return df

        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"[dnp_history] Failed to compute features: {exc}")
            for col in DNP_HISTORY_FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0 if "streak" in col or "consecutive" in col or "games_since" in col else 0.0
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
