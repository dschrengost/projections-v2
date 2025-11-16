"""Minutes V1 minimal feature builder."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from projections.archetypes import (
    compute_team_role_counts,
    compute_team_missing_totals,
    prepare_injury_availability,
)
from projections.features import availability as availability_features
from projections.features import depth as depth_features
from projections.features import game_env as game_env_features
from projections.features import return_ramp as return_ramp_features
from projections.features import role as role_features
from projections.features import rest as rest_features
from projections.features import trend as trend_features
from projections.minutes_v1.pos import canonical_pos_bucket_series


@dataclass
class MinutesFeatureBuilder:
    """Build the Minutes V1 quick-start feature set."""

    schedule: pd.DataFrame
    injuries_snapshot: pd.DataFrame
    odds_snapshot: pd.DataFrame
    roster_nightly: pd.DataFrame
    coach_tenure: pd.DataFrame | None = None
    archetype_roles: pd.DataFrame | None = None
    archetype_deltas: pd.DataFrame | None = None

    def build(self, labels: pd.DataFrame) -> pd.DataFrame:
        base = self._attach_schedule(labels)
        base = self._attach_injuries(base)
        base = self._attach_odds(base)
        base = self._attach_depth(base)
        base = self._player_history_features(base)
        base = self._attach_archetype_features(base)
        base = self._team_dispersion(base)
        base = self._finalize(base)
        return base

    def _attach_schedule(self, labels: pd.DataFrame) -> pd.DataFrame:
        schedule = self.schedule.copy()
        schedule["tip_ts"] = pd.to_datetime(schedule["tip_ts"], utc=True)
        schedule["schedule_game_date"] = pd.to_datetime(schedule["game_date"]).dt.normalize()
        schedule = schedule.drop(columns=["game_date"])
        if "season" in schedule.columns:
            schedule = schedule.rename(columns={"season": "schedule_season"})
        merged = labels.merge(schedule, on="game_id", how="left")
        missing_mask = merged["tip_ts"].isna()
        if missing_mask.any():
            missing_games = (
                merged.loc[missing_mask, "game_id"].dropna().astype(str).unique().tolist()
            )
            warnings.warn(
                "Dropping rows with missing schedule/tip_ts for games: "
                + ", ".join(missing_games[:10])
                + ("..." if len(missing_games) > 10 else "")
            )
            merged = merged.loc[~missing_mask].copy()
            if merged.empty:
                raise ValueError("Schedule join removed all rows due to missing tip_ts.")
        if "schedule_game_date" in merged:
            schedule_dates = pd.to_datetime(merged.pop("schedule_game_date")).dt.normalize()
            if "game_date" in merged:
                base_dates = pd.to_datetime(merged["game_date"]).dt.normalize()
                merged["game_date"] = base_dates.fillna(schedule_dates)
            else:
                merged["game_date"] = schedule_dates
        else:
            merged["game_date"] = pd.to_datetime(merged["game_date"]).dt.normalize()
        if "schedule_season" in merged:
            if "season" in merged:
                merged["season"] = merged["season"].fillna(merged["schedule_season"])
            else:
                merged["season"] = merged["schedule_season"]
            merged.drop(columns=["schedule_season"], inplace=True)
        merged["home_flag"] = (merged["team_id"] == merged["home_team_id"]).astype(int)
        merged["opponent_team_id"] = np.where(
            merged["team_id"] == merged["home_team_id"], merged["away_team_id"], merged["home_team_id"]
        )
        if {"home_team_name", "away_team_name"}.issubset(merged.columns):
            merged["team_name"] = np.where(
                merged["home_flag"] == 1, merged["home_team_name"], merged["away_team_name"]
            )
            merged["opponent_team_name"] = np.where(
                merged["home_flag"] == 1, merged["away_team_name"], merged["home_team_name"]
            )
        else:
            merged["team_name"] = merged.get("team_name", merged["team_id"])
            merged["opponent_team_name"] = merged.get("opponent_team_name", merged["opponent_team_id"])
        merged["team_name"] = merged["team_name"].fillna(merged["team_id"])
        merged["opponent_team_name"] = merged["opponent_team_name"].fillna(merged["opponent_team_id"])
        if {"home_team_tricode", "away_team_tricode"}.issubset(merged.columns):
            merged["team_tricode"] = np.where(
                merged["home_flag"] == 1, merged["home_team_tricode"], merged["away_team_tricode"]
            )
            merged["opponent_team_tricode"] = np.where(
                merged["home_flag"] == 1, merged["away_team_tricode"], merged["home_team_tricode"]
            )
        else:
            merged["team_tricode"] = merged.get("team_tricode", merged["team_name"])
            merged["opponent_team_tricode"] = merged.get(
                "opponent_team_tricode", merged["opponent_team_name"]
            )
        merged["team_tricode"] = merged["team_tricode"].fillna(merged["team_name"])
        merged["opponent_team_tricode"] = merged["opponent_team_tricode"].fillna(
            merged["opponent_team_name"]
        )
        return merged

    def _attach_injuries(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared_injuries = None
        if self.injuries_snapshot is not None and not self.injuries_snapshot.empty:
            prepared_injuries = availability_features.prepare_injuries_snapshot(
                self.injuries_snapshot
            )
        merged = availability_features.attach_availability_features(
            df, prepared_injuries=prepared_injuries, injuries_snapshot=self.injuries_snapshot
        )
        merged = return_ramp_features.attach_return_ramp_features(merged)
        validate_injury_as_of(merged)
        return merged

    def _attach_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        return game_env_features.attach_game_environment_features(df, self.odds_snapshot)

    def _attach_depth(self, df: pd.DataFrame) -> pd.DataFrame:
        return depth_features.attach_depth_features(df, self.roster_nightly)

    def _player_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.sort_values(["player_id", "game_date"]).copy()
        working = trend_features.attach_trend_features(working)
        working = role_features.attach_role_features(working)
        working = rest_features.attach_rest_features(working)
        return working

    def _attach_archetype_features(self, df: pd.DataFrame) -> pd.DataFrame:
        defaults = self._archetype_feature_defaults()
        required_cols = {"season", "game_id", "player_id", "team_id"}
        if not required_cols.issubset(df.columns):
            return self._ensure_archetype_defaults(df, defaults)
        if self.injuries_snapshot is None or self.injuries_snapshot.empty:
            return self._ensure_archetype_defaults(df, defaults)
        if self.archetype_roles is None or self.archetype_roles.empty:
            return self._ensure_archetype_defaults(df, defaults)

        injuries_with_team = self._injury_snapshot_with_team_ids()
        if injuries_with_team.empty:
            return self._ensure_archetype_defaults(df, defaults)
        injuries_prepared = prepare_injury_availability(injuries_with_team)

        roles_df = self.archetype_roles.copy()
        roles_df["season"] = roles_df["season"].astype("string")
        deltas_df = (
            self.archetype_deltas.copy()
            if self.archetype_deltas is not None and not self.archetype_deltas.empty
            else pd.DataFrame(columns=["season", "role_p", "role_t", "delta_per_missing"])
        )
        if not deltas_df.empty and "season" in deltas_df.columns:
            deltas_df["season"] = deltas_df["season"].astype("string")

        working = df.copy()
        working["_season_str"] = working["season"].astype("string")
        season_games: dict[str, list[int]] = {}
        season_key = working.loc[:, ["game_id", "_season_str"]].dropna()
        season_key["game_id"] = pd.to_numeric(season_key["game_id"], errors="coerce").astype("Int64")
        season_key = season_key.dropna(subset=["game_id", "_season_str"])
        if not season_key.empty:
            for season_value, group in season_key.groupby("_season_str"):
                game_ids = group["game_id"].dropna().astype(int).unique().tolist()
                if game_ids:
                    season_games[str(season_value)] = game_ids

        season_frames: list[pd.DataFrame] = []
        for season_value, game_ids in season_games.items():
            season_roles = roles_df[roles_df["season"] == season_value]
            if season_roles.empty or not game_ids:
                continue
            season_subset = working[working["_season_str"] == season_value]
            if season_subset.empty:
                continue
            season_injuries = injuries_prepared[injuries_prepared["game_id"].isin(game_ids)].copy()
            if season_injuries.empty:
                continue
            try:
                team_role_counts = compute_team_role_counts(
                    season_injuries,
                    season_roles,
                    season_label=season_value,
                )
            except ValueError:
                continue
            team_missing_totals = compute_team_missing_totals(season_injuries)
            if "season" in deltas_df.columns:
                season_deltas = deltas_df[deltas_df["season"] == season_value]
            else:
                season_deltas = deltas_df
            season_features = self._aggregate_season_archetype_features(
                season_subset,
                season_roles,
                season_deltas,
                team_role_counts,
                team_missing_totals,
            )
            if not season_features.empty:
                season_frames.append(season_features)

        if season_frames:
            features = pd.concat(season_frames, ignore_index=True)
            for column in ("game_id", "player_id", "team_id"):
                features[column] = pd.to_numeric(features[column], errors="coerce").astype("Int64")
                working[column] = pd.to_numeric(working[column], errors="coerce").astype("Int64")
            working = working.merge(
                features,
                on=["game_id", "player_id", "team_id"],
                how="left",
            )
        working.drop(columns=["_season_str"], inplace=True)
        return self._ensure_archetype_defaults(working, defaults)

    def _archetype_feature_defaults(self) -> dict[str, float | int]:
        return {
            "arch_delta_sum": 0.0,
            "arch_delta_same_pos": 0.0,
            "arch_delta_max_role": 0.0,
            "arch_delta_min_role": 0.0,
            "arch_missing_same_pos_count": 0,
            "arch_missing_total_count": 0,
        }

    def _ensure_archetype_defaults(
        self, df: pd.DataFrame, defaults: dict[str, float | int]
    ) -> pd.DataFrame:
        working = df.copy()
        for column, default in defaults.items():
            if column not in working.columns:
                working[column] = default
            else:
                working[column] = working[column].fillna(default)
        if "arch_missing_same_pos_count" in working.columns:
            working["arch_missing_same_pos_count"] = working["arch_missing_same_pos_count"].astype(int)
        if "arch_missing_total_count" in working.columns:
            working["arch_missing_total_count"] = working["arch_missing_total_count"].astype(int)
        for column in ("arch_delta_sum", "arch_delta_same_pos", "arch_delta_max_role", "arch_delta_min_role"):
            if column in working.columns:
                working[column] = working[column].astype(float)
        return working

    def _injury_snapshot_with_team_ids(self) -> pd.DataFrame:
        if self.injuries_snapshot is None or self.injuries_snapshot.empty:
            return pd.DataFrame(columns=["game_id", "player_id", "team_id"])
        injuries = self.injuries_snapshot.copy()
        if "team_id" not in injuries.columns:
            if self.roster_nightly is None or self.roster_nightly.empty:
                warnings.warn(
                    "Cannot attach archetype features without team IDs in injuries snapshot.",
                    RuntimeWarning,
                )
                return pd.DataFrame(columns=["game_id", "player_id", "team_id"])
            lookup = (
                self.roster_nightly.loc[:, ["game_id", "player_id", "team_id"]]
                .dropna(subset=["game_id", "player_id", "team_id"])
                .copy()
            )
            lookup["game_id"] = pd.to_numeric(lookup["game_id"], errors="coerce").astype("Int64")
            lookup["player_id"] = pd.to_numeric(lookup["player_id"], errors="coerce").astype("Int64")
            lookup["team_id"] = pd.to_numeric(lookup["team_id"], errors="coerce").astype("Int64")
            lookup = lookup.dropna(subset=["game_id", "player_id", "team_id"])
            lookup = lookup.drop_duplicates(subset=["game_id", "player_id"], keep="last")
            injuries["game_id"] = pd.to_numeric(injuries["game_id"], errors="coerce").astype("Int64")
            injuries["player_id"] = pd.to_numeric(injuries["player_id"], errors="coerce").astype("Int64")
            injuries = injuries.merge(
                lookup,
                on=["game_id", "player_id"],
                how="left",
            )
        injuries = injuries.dropna(subset=["game_id", "player_id", "team_id"])
        return injuries

    def _aggregate_season_archetype_features(
        self,
        season_subset: pd.DataFrame,
        season_roles: pd.DataFrame,
        season_deltas: pd.DataFrame,
        team_role_counts: pd.DataFrame,
        team_missing_totals: pd.DataFrame,
    ) -> pd.DataFrame:
        if season_subset.empty or season_roles.empty:
            return pd.DataFrame(columns=["game_id", "player_id", "team_id"])
        base = season_subset.loc[:, ["game_id", "player_id", "team_id"]].copy()
        base["game_id"] = pd.to_numeric(base["game_id"], errors="coerce").astype("Int64")
        base["player_id"] = pd.to_numeric(base["player_id"], errors="coerce").astype("Int64")
        base["team_id"] = pd.to_numeric(base["team_id"], errors="coerce").astype("Int64")
        base = base.dropna(subset=["game_id", "player_id", "team_id"])
        if base.empty:
            return base
        base["game_id"] = base["game_id"].astype(int)
        base["player_id"] = base["player_id"].astype(int)
        base["team_id"] = base["team_id"].astype(int)

        role_cols = ["player_id", "role_key", "position_group", "starter_tier"]
        role_lookup = (
            season_roles.loc[:, role_cols]
            .dropna(subset=["player_id"])
            .drop_duplicates(subset=["player_id"], keep="last")
            .rename(
                columns={
                    "role_key": "role_p",
                    "position_group": "position_group_p",
                    "starter_tier": "starter_tier_p",
                }
            )
        )
        base = base.merge(role_lookup, on="player_id", how="left")
        base = base.merge(team_role_counts, on=["game_id", "team_id"], how="left")
        missing_totals = team_missing_totals
        if missing_totals.empty:
            missing_totals = pd.DataFrame(
                columns=["game_id", "team_id", "arch_missing_total_count"]
            )
        base = base.merge(missing_totals, on=["game_id", "team_id"], how="left")

        if season_deltas is None or season_deltas.empty:
            delta_frame = pd.DataFrame(columns=["role_p", "role_t", "delta_per_missing"])
        else:
            delta_frame = season_deltas.loc[:, ["role_p", "role_t", "delta_per_missing"]]
        base = base.merge(delta_frame, on=["role_p", "role_t"], how="left")

        base["starter_tier_p"] = base["starter_tier_p"].fillna("unknown")
        base["position_group_p"] = base["position_group_p"].astype("string")
        base["missing_role_t_count"] = base["missing_role_t_count"].fillna(0).astype(int)
        base["delta_per_missing"] = base["delta_per_missing"].fillna(0.0)
        role_valid = (
            base["role_p"].notna()
            & base["position_group_p"].notna()
            & (base["starter_tier_p"].str.lower() != "unknown")
        )
        base.loc[~role_valid, "delta_per_missing"] = 0.0
        base["contrib"] = base["delta_per_missing"] * base["missing_role_t_count"]
        same_pos_flag = (
            base["position_group_p"].notna()
            & base["position_group_t"].notna()
            & (base["position_group_p"] == base["position_group_t"])
        )
        base["same_pos_contrib"] = np.where(same_pos_flag, base["contrib"], 0.0)
        base["same_pos_missing"] = np.where(same_pos_flag, base["missing_role_t_count"], 0)
        base["arch_missing_total_count"] = base["arch_missing_total_count"].fillna(0).astype(int)

        grouped = base.groupby(["game_id", "player_id", "team_id"], as_index=False).agg(
            arch_delta_sum=("contrib", "sum"),
            arch_delta_same_pos=("same_pos_contrib", "sum"),
            arch_missing_same_pos_count=("same_pos_missing", "sum"),
            arch_missing_total_count=("arch_missing_total_count", "max"),
            arch_delta_max_role=("contrib", "max"),
            arch_delta_min_role=("contrib", "min"),
        )
        grouped["arch_missing_same_pos_count"] = grouped["arch_missing_same_pos_count"].fillna(0).astype(int)
        grouped["arch_missing_total_count"] = grouped["arch_missing_total_count"].fillna(0).astype(int)
        for column in ("arch_delta_sum", "arch_delta_same_pos", "arch_delta_max_role", "arch_delta_min_role"):
            grouped[column] = grouped[column].fillna(0.0)
        return grouped


    def _team_dispersion(self, df: pd.DataFrame) -> pd.DataFrame:
        team_dispersion = (
            df.groupby(["team_id", "game_id", "game_date"])["minutes"]
            .agg(lambda s: float(np.nanstd(s.to_numpy(), ddof=0)))
            .reset_index(name="team_minutes_dispersion")
        )
        team_dispersion["team_minutes_dispersion"] = team_dispersion["team_minutes_dispersion"].fillna(0.0)
        team_dispersion["team_minutes_dispersion_prior"] = (
            team_dispersion.sort_values("game_date")
            .groupby("team_id")["team_minutes_dispersion"]
            .shift(1)
        )
        return df.merge(
            team_dispersion[["team_id", "game_id", "team_minutes_dispersion_prior"]],
            on=["team_id", "game_id"],
            how="left",
        )

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        asof_candidates = pd.concat(
            [
                df.get("injury_as_of_ts"),
                df.get("odds_as_of_ts"),
                df.get("roster_as_of_ts"),
            ],
            axis=1,
        )
        df["feature_as_of_ts"] = asof_candidates.max(axis=1)
        df["feature_as_of_ts"] = df["feature_as_of_ts"].fillna(df["tip_ts"])
        df.loc[df["feature_as_of_ts"] > df["tip_ts"], "feature_as_of_ts"] = df["tip_ts"]
        if "pos_bucket" not in df.columns:
            archetype_series = df.get("archetype")
            if archetype_series is None:
                archetype_series = pd.Series("UNK", index=df.index)
            df["pos_bucket"] = canonical_pos_bucket_series(archetype_series)
        else:
            df["pos_bucket"] = canonical_pos_bucket_series(df["pos_bucket"])
        return df


def validate_injury_as_of(df: pd.DataFrame) -> None:
    """Ensure injury snapshots respect tip-time anti-leak constraints."""

    if "injury_as_of_ts" not in df or "tip_ts" not in df:
        return
    injury_ts = pd.to_datetime(df["injury_as_of_ts"], utc=True)
    tip_ts = pd.to_datetime(df["tip_ts"], utc=True)
    mask = injury_ts.notna()
    if (injury_ts[mask] > tip_ts[mask]).any():
        raise RuntimeError("Detected injury_as_of_ts after tip_ts")
    if "injury_snapshot_missing" in df:
        missing_flag = df["injury_snapshot_missing"].fillna(0).astype(int)
        missing_mask = injury_ts.isna()
        if (missing_mask & (missing_flag == 0)).any():
            raise RuntimeError("Missing injury snapshot rows must set injury_snapshot_missing=1")
