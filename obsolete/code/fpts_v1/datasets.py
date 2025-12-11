"""Dataset builder for the fantasy-points-per-minute model."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Literal
import re

import numpy as np
import pandas as pd

from projections.fpts_v1.scoring import SCORING_SYSTEMS
from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest
from projections.minutes_v1.logs import (
    default_minutes_run_id,
    prediction_logs_base,
    prediction_logs_candidates,
)
from projections.minutes_v1.pos import canonical_pos_bucket_series

MinutesSource = Literal["predicted", "actual"]

_PREDICTION_COLS = (
    "minutes_p10",
    "minutes_p50",
    "minutes_p90",
    "play_prob",
    "feature_as_of_ts",
    "run_as_of_ts",
    "log_timestamp",
)


def _normalize_day(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.tz_localize(None).normalize()


def _season_from_day(day: pd.Timestamp) -> int:
    return int(day.year)


def _iter_days(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += pd.Timedelta(days=1)


_ISO_DURATION = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?")


def _parse_minutes_iso(raw: str | None) -> float:
    if not raw:
        return 0.0
    match = _ISO_DURATION.fullmatch(str(raw).strip().upper())
    if not match:
        return 0.0
    hours = float(match.group(1) or 0)
    minutes = float(match.group(2) or 0)
    seconds = float(match.group(3) or 0)
    return hours * 60.0 + minutes + seconds / 60.0


def _coerce_ts(value: str | datetime | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _weighted_player_rolling_avg(
    values: pd.Series,
    weights: pd.Series,
    group_key: pd.Series,
    *,
    window: int,
    min_weight: float,
) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    weighted = numeric_values * numeric_weights
    rolling_weighted = (
        weighted.groupby(group_key)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    rolling_weights = (
        numeric_weights.groupby(group_key)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    denom = rolling_weights.replace(0.0, np.nan)
    avg = rolling_weighted / denom
    return avg.where(rolling_weights >= min_weight)


def _seasonal_cumulative_avg(
    values: pd.Series,
    weights: pd.Series,
    player_ids: pd.Series,
    seasons: pd.Series,
    *,
    min_weight: float,
) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    weighted = numeric_values * numeric_weights
    shifted_weighted = weighted.groupby([player_ids, seasons]).shift(1)
    shifted_weights = numeric_weights.groupby([player_ids, seasons]).shift(1)
    cum_weighted = shifted_weighted.groupby([player_ids, seasons]).cumsum()
    cum_weights = shifted_weights.groupby([player_ids, seasons]).cumsum()
    denom = cum_weights.replace(0.0, np.nan)
    avg = cum_weighted / denom
    return avg.where(cum_weights >= min_weight)


@dataclass
class FptsDatasetBuilder:
    """Join gold minutes features, boxscore stats, and minutes predictions."""

    data_root: Path
    scoring_system: str = "dk"
    minutes_source: MinutesSource = "predicted"
    history_days: int = 60
    feature_root: Path | None = None
    prediction_logs_root: Path | None = None
    boxscores_root: Path | None = None
    minutes_run_id: str | None = None
    _stats_columns: tuple[str, ...] = (
        "game_id",
        "player_id",
        "team_id",
        "game_date",
        "tip_ts",
        "season",
        "actual_minutes",
        "actual_fpts",
        "fpts_per_min_game",
        "usage_per_min_game",
        "assists_per_min_game",
        "rebounds_per_min_game",
    )

    def __post_init__(self) -> None:
        self.data_root = self.data_root.expanduser().resolve()
        self.feature_root = (
            self.feature_root or self.data_root / "gold" / "features_minutes_v1"
        )
        self.boxscores_root = (
            self.boxscores_root or self.data_root / "bronze" / "boxscores_raw"
        )
        normalized_source = str(self.minutes_source).strip().lower()
        if normalized_source not in ("predicted", "actual"):
            raise ValueError("--minutes-source must be 'predicted' or 'actual'")
        self.minutes_source = normalized_source  # type: ignore[assignment]

        if self.minutes_source == "predicted":
            if self.prediction_logs_root is None:
                base_prediction_root = prediction_logs_base(self.data_root)
            else:
                base_prediction_root = self.prediction_logs_root
            self.prediction_logs_root = base_prediction_root.expanduser().resolve()
            resolved_minutes_run_id = self.minutes_run_id or default_minutes_run_id()
            candidate_paths: list[Path] = []
            for path in prediction_logs_candidates(
                run_id=resolved_minutes_run_id, data_root=self.data_root
            ):
                if path not in candidate_paths:
                    candidate_paths.append(path)
            if self.prediction_logs_root not in candidate_paths:
                candidate_paths.insert(0, self.prediction_logs_root)
            self._prediction_log_candidates = candidate_paths
            self.minutes_run_id = resolved_minutes_run_id
        else:
            self.prediction_logs_root = None
            self._prediction_log_candidates = []
        self._scoring_fn = SCORING_SYSTEMS.get(self.scoring_system.lower())
        if self._scoring_fn is None:
            raise ValueError(
                f"Unsupported scoring system '{self.scoring_system}'. "
                f"Available: {', '.join(sorted(SCORING_SYSTEMS))}"
            )
        # minutes_source validated above

    def build(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        start = _normalize_day(start_date)
        end = _normalize_day(end_date)
        if end < start:
            raise ValueError("end date must be on/after start date")

        history_start = start - pd.Timedelta(days=max(self.history_days, 7))
        features = self._load_feature_frame(start, end)
        stats = self._load_stat_lines(history_start, end)
        stats = self._attach_player_priors(stats)
        slice_mask = (stats["game_date"] >= start) & (stats["game_date"] <= end)
        label_frame = stats.loc[
            slice_mask & (stats["actual_minutes"] > 0.0)
        ].copy()
        if label_frame.empty:
            raise ValueError("No label rows matched the requested date window.")

        predictions = self._load_minutes_predictions(label_frame, start, end)
        dataset = label_frame.merge(
            features,
            on=["game_id", "player_id", "team_id"],
            how="left",
        )
        dataset = dataset.merge(
            predictions,
            on=["game_id", "player_id"],
            how="left",
        )
        dataset = self._finalize(dataset)
        dataset.sort_values(["game_date", "game_id", "player_id"], inplace=True)
        return dataset.reset_index(drop=True)

    def _empty_stats_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self._stats_columns)

    def _future_placeholder_rows(
        self,
        features_df: pd.DataFrame,
        slate_day: pd.Timestamp,
    ) -> pd.DataFrame:
        required = ("game_id", "player_id", "team_id")
        working = features_df.copy()
        for key in required:
            working[key] = pd.to_numeric(working.get(key), errors="coerce")
        mask = working["player_id"].notna() & working["game_id"].notna()
        working = working.loc[mask].copy()
        if working.empty:
            return self._empty_stats_frame()
        seasons = working.get("season")
        if seasons is None:
            seasons = pd.Series(str(slate_day.year), index=working.index)
        tip_ts = pd.to_datetime(working.get("tip_ts"), errors="coerce")
        raw_team = working.get("team_id")
        if raw_team is None:
            raw_team = pd.Series(np.nan, index=working.index)
        team_ids = pd.to_numeric(raw_team, errors="coerce")
        placeholder = pd.DataFrame(
            {
                "game_id": working["game_id"].astype("Int64"),
                "player_id": working["player_id"].astype("Int64"),
                "team_id": team_ids.astype("Int64"),
                "game_date": slate_day,
                "tip_ts": tip_ts,
                "season": seasons.astype(str),
                "actual_minutes": 0.0,
                "actual_fpts": 0.0,
                "fpts_per_min_game": 0.0,
                "usage_per_min_game": 0.0,
                "assists_per_min_game": 0.0,
                "rebounds_per_min_game": 0.0,
            }
        )
        return placeholder

    def enrich_live_features(
        self,
        slate_day: date | datetime,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach rolling priors to live features for inference."""

        slate_ts = _normalize_day(slate_day)
        history_start = slate_ts - pd.Timedelta(days=max(self.history_days, 7))
        history_end = slate_ts - pd.Timedelta(days=1)
        if history_end < history_start:
            history_end = history_start
        try:
            stats = self._load_stat_lines(history_start, history_end)
        except FileNotFoundError:
            stats = self._empty_stats_frame()
        future_rows = self._future_placeholder_rows(features_df, slate_ts)
        stats_combined = pd.concat([stats, future_rows], ignore_index=True, sort=False)
        if stats_combined.empty:
            return features_df
        enriched = self._attach_player_priors(stats_combined)
        target_rows = enriched.loc[enriched["game_date"] == slate_ts].copy()
        if target_rows.empty:
            return features_df
        prior_columns = [
            col
            for col in target_rows.columns
            if "_prior" in col
            or col in {"season_minutes_prior", "season_games_played_prior", "fpts_baseline_per_min"}
        ]
        join_cols = ["game_id", "player_id", "team_id"]
        subset = target_rows.loc[:, join_cols + prior_columns]
        for key in join_cols:
            features_df[key] = pd.to_numeric(features_df.get(key), errors="coerce")
            subset[key] = pd.to_numeric(subset.get(key), errors="coerce")
        return features_df.merge(subset, on=join_cols, how="left")

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def _feature_partitions(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> Iterable[Path]:
        seen: set[tuple[int, int]] = set()
        for day in _iter_days(start, end):
            key = (_season_from_day(day), int(day.month))
            if key in seen:
                continue
            seen.add(key)
            season, month = key
            path = (
                self.feature_root
                / f"season={season}"
                / f"month={month:02d}"
                / "features.parquet"
            )
            if path.exists():
                yield path
            else:
                warnings.warn(f"[fpts] missing feature partition: {path}", RuntimeWarning)

    def _load_feature_frame(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for partition in self._feature_partitions(start, end):
            frame = pd.read_parquet(partition)
            frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
            frames.append(frame)
        if not frames:
            raise FileNotFoundError(
                f"No feature partitions found under {self.feature_root} "
                f"covering {start.date()}–{end.date()}."
            )
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = combined[
            (combined["game_date"] >= start) & (combined["game_date"] <= end)
        ].copy()
        combined = deduplicate_latest(
            combined,
            key_cols=KEY_COLUMNS,
            order_cols=["feature_as_of_ts"],
        )
        drop_cols = ["minutes", "starter_flag_label", "game_date", "season", "tip_ts"]
        combined.drop(columns=[col for col in drop_cols if col in combined.columns], inplace=True)
        bool_like = [
            "restriction_flag",
            "ramp_flag",
            "home_flag",
            "is_projected_starter",
            "is_confirmed_starter",
            "is_out",
            "is_q",
            "is_prob",
        ]
        for col in bool_like:
            if col in combined.columns:
                combined[col] = combined[col].fillna(0).astype(int)
        if "pos_bucket" in combined.columns:
            combined["pos_bucket"] = canonical_pos_bucket_series(combined["pos_bucket"])
        return combined

    # ------------------------------------------------------------------
    # Boxscore ingestion
    # ------------------------------------------------------------------

    def _bronze_paths(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> Iterable[Path]:
        seen: set[tuple[int, str]] = set()
        for day in _iter_days(start, end):
            season = _season_from_day(day)
            date_token = day.date().isoformat()
            key = (season, date_token)
            if key in seen:
                continue
            seen.add(key)
            candidate = (
                self.boxscores_root
                / f"season={season}"
                / f"date={date_token}"
                / "boxscores_raw.parquet"
            )
            if candidate.exists():
                yield candidate

    def _load_stat_lines(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for path in self._bronze_paths(start, end):
            bronze = pd.read_parquet(path)
            for row in bronze.itertuples():
                payload = json.loads(row.payload)
                tip_ts = _coerce_ts(payload.get("game_time_utc") or payload.get("game_time_local"))
                if tip_ts is None:
                    continue
                game_date = tip_ts.tz_convert("America/New_York").tz_localize(None).normalize()
                if not (start <= game_date <= end):
                    continue
                game_id = int(str(payload.get("game_id", row.game_id)).zfill(10))
                for side in ("home", "away"):
                    team = payload.get(side)
                    if not team:
                        continue
                    team_id = int(team.get("team_id") or team.get("teamId") or 0)
                    for player in team.get("players", []):
                        stats = player.get("statistics") or {}
                        minutes = _parse_minutes_iso(stats.get("minutes"))
                        fpts = self._scoring_fn(
                            {
                                "points": stats.get("points"),
                                "rebounds_total": stats.get("reboundsTotal")
                                or stats.get("rebounds"),
                                "assists": stats.get("assists"),
                                "steals": stats.get("steals"),
                                "blocks": stats.get("blocks"),
                                "turnovers": stats.get("turnovers"),
                                "three_pointers_made": stats.get("threePointersMade")
                                or stats.get("threePointersMadeTotal"),
                            }
                        )
                        fg_attempts = float(stats.get("fieldGoalsAttempted") or 0.0)
                        ft_attempts = float(stats.get("freeThrowsAttempted") or 0.0)
                        turnovers = float(stats.get("turnovers") or 0.0)
                        usage = 0.0
                        if minutes > 0:
                            usage = (fg_attempts + 0.44 * ft_attempts + turnovers) / minutes
                        assists = float(stats.get("assists") or 0.0)
                        rebounds = float(stats.get("reboundsTotal") or stats.get("rebounds") or 0.0)
                        records.append(
                            {
                                "game_id": game_id,
                                "player_id": int(player.get("person_id") or player.get("personId") or 0),
                                "team_id": team_id,
                                "game_date": game_date,
                                "tip_ts": tip_ts,
                                "season": str(_season_from_day(game_date)),
                                "actual_minutes": minutes,
                                "actual_fpts": fpts,
                                "fpts_per_min_game": fpts / minutes if minutes > 0 else np.nan,
                                "usage_per_min_game": usage,
                                "assists_per_min_game": assists / minutes if minutes > 0 else 0.0,
                                "rebounds_per_min_game": rebounds / minutes if minutes > 0 else 0.0,
                            }
                        )
        if not records:
            raise FileNotFoundError(
                f"No boxscore stat lines found under {self.boxscores_root} "
                f"covering {start.date()}–{end.date()}."
            )
        stats = pd.DataFrame.from_records(records)
        stats.sort_values(["player_id", "tip_ts"], inplace=True)
        return stats

    # ------------------------------------------------------------------
    # Player priors
    # ------------------------------------------------------------------

    def _attach_player_priors(self, stats: pd.DataFrame) -> pd.DataFrame:
        working = stats.copy()
        grouped = working.groupby("player_id", group_keys=False)
        minutes_shifted = grouped["actual_minutes"].shift(1).fillna(0.0)
        per_min_shifted = grouped["fpts_per_min_game"].shift(1).fillna(0.0)
        usage_shifted = grouped["usage_per_min_game"].shift(1).fillna(0.0)
        assists_shifted = grouped["assists_per_min_game"].shift(1).fillna(0.0)
        rebounds_shifted = grouped["rebounds_per_min_game"].shift(1).fillna(0.0)

        working["fpts_per_min_prior_5"] = _weighted_player_rolling_avg(
            per_min_shifted,
            minutes_shifted,
            working["player_id"],
            window=5,
            min_weight=30.0,
        )
        working["fpts_per_min_prior_10"] = _weighted_player_rolling_avg(
            per_min_shifted,
            minutes_shifted,
            working["player_id"],
            window=10,
            min_weight=60.0,
        )
        working["usage_per_min_prior_5"] = _weighted_player_rolling_avg(
            usage_shifted,
            minutes_shifted,
            working["player_id"],
            window=5,
            min_weight=30.0,
        )
        working["usage_per_min_prior_10"] = _weighted_player_rolling_avg(
            usage_shifted,
            minutes_shifted,
            working["player_id"],
            window=10,
            min_weight=60.0,
        )
        working["assist_per_min_prior_5"] = _weighted_player_rolling_avg(
            assists_shifted,
            minutes_shifted,
            working["player_id"],
            window=5,
            min_weight=30.0,
        )
        working["rebound_per_min_prior_5"] = _weighted_player_rolling_avg(
            rebounds_shifted,
            minutes_shifted,
            working["player_id"],
            window=5,
            min_weight=30.0,
        )

        working["fpts_per_min_prior_season"] = _seasonal_cumulative_avg(
            per_min_shifted,
            minutes_shifted,
            working["player_id"],
            working["season"],
            min_weight=180.0,
        )
        working["usage_per_min_prior_season"] = _seasonal_cumulative_avg(
            usage_shifted,
            minutes_shifted,
            working["player_id"],
            working["season"],
            min_weight=180.0,
        )

        baseline = float(
            working.loc[working["actual_minutes"] > 0, "fpts_per_min_game"].mean()
        )
        if not math.isfinite(baseline):
            baseline = 1.0
        for col in (
            "fpts_per_min_prior_5",
            "fpts_per_min_prior_10",
            "fpts_per_min_prior_season",
        ):
            working[col] = working[col].fillna(baseline)
        for col in (
            "usage_per_min_prior_5",
            "usage_per_min_prior_10",
            "usage_per_min_prior_season",
            "assist_per_min_prior_5",
            "rebound_per_min_prior_5",
        ):
            working[col] = working[col].fillna(0.0)

        working["fpts_baseline_per_min"] = working["fpts_per_min_prior_10"].fillna(
            working["fpts_per_min_prior_5"]
        )
        working["fpts_baseline_per_min"] = working["fpts_baseline_per_min"].fillna(
            baseline
        )

        season_groups = working.groupby(["player_id", "season"])
        played_prior = (
            (working["actual_minutes"] > 0)
            .groupby([working["player_id"], working["season"]])
            .shift(1)
            .fillna(0)
        )
        working["season_games_played_prior"] = (
            played_prior.astype(int)
            .groupby([working["player_id"], working["season"]])
            .cumsum()
        )
        minutes_prior = (
            working["actual_minutes"]
            .groupby([working["player_id"], working["season"]])
            .shift(1)
            .fillna(0.0)
        )
        working["season_minutes_prior"] = (
            pd.to_numeric(minutes_prior, errors="coerce").fillna(0.0)
            .groupby([working["player_id"], working["season"]])
            .cumsum()
        )
        return working

    # ------------------------------------------------------------------
    # Minutes predictions
    # ------------------------------------------------------------------

    def _prediction_files(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> Iterable[Path]:
        if self.minutes_source != "predicted":
            return []
        attempted: list[Path] = []
        for root in self._prediction_log_candidates:
            attempted.append(root)
            files = list(self._collect_prediction_files(root, start, end))
            if files:
                yield from files
                return
        raise FileNotFoundError(
            "No minutes prediction logs found under any of the following directories:\n"
            f"{chr(10).join(str(path) for path in attempted)}\n"
            f"covering {start.date()}–{end.date()}. "
            "Backfill minutes_v1 predictions or rerun with --minutes-source actual for training with realized minutes."
        )

    def _collect_prediction_files(
        self,
        root: Path,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[Path]:
        seen: set[tuple[int, int]] = set()
        files: list[Path] = []
        for day in _iter_days(start, end):
            key = (_season_from_day(day), int(day.month))
            if key in seen:
                continue
            seen.add(key)
            season, month = key
            partition_dir = root / f"season={season}" / f"month={month:02d}"
            if not partition_dir.exists():
                continue
            files.extend(sorted(partition_dir.glob("*.parquet")))
        return files

    def _load_minutes_predictions(
        self,
        label_frame: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        if self.minutes_source == "actual":
            result = label_frame[["game_id", "player_id", "actual_minutes"]].copy()
            result.drop_duplicates(["game_id", "player_id"], inplace=True)
            result["minutes_p10_pred"] = result["actual_minutes"]
            result["minutes_p50_pred"] = result["actual_minutes"]
            result["minutes_p90_pred"] = result["actual_minutes"]
            result["play_prob_pred"] = 1.0
            result.drop(columns=["actual_minutes"], inplace=True)
            return result

        frames: list[pd.DataFrame] = []
        for file_path in self._prediction_files(start, end):
            frame = pd.read_parquet(file_path)
            missing_cols = [col for col in ("game_date", "minutes_p50") if col not in frame.columns]
            if missing_cols:
                continue
            frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
            frame = frame[
                (frame["game_date"] >= start) & (frame["game_date"] <= end)
            ].copy()
            if frame.empty:
                continue
            for col in ("run_as_of_ts", "log_timestamp", "feature_as_of_ts"):
                if col in frame.columns:
                    frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
            frames.append(frame.loc[:, ["game_id", "player_id"] + list(_PREDICTION_COLS)])

        if not frames:
            fallback_root = (
                self._prediction_log_candidates[0]
                if getattr(self, "_prediction_log_candidates", None)
                else self.prediction_logs_root
            )
            raise FileNotFoundError(
                f"No minutes prediction logs found under {fallback_root} "
                f"covering {start.date()}–{end.date()}."
            )
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["game_id"] = pd.to_numeric(combined["game_id"], errors="coerce").astype("Int64")
        combined["player_id"] = pd.to_numeric(combined["player_id"], errors="coerce").astype("Int64")
        combined.dropna(subset=["game_id", "player_id"], inplace=True)
        combined["selection_ts"] = combined["run_as_of_ts"].fillna(combined["log_timestamp"])
        combined.sort_values(["game_id", "player_id", "selection_ts"], inplace=True)
        deduped = combined.groupby(["game_id", "player_id"], as_index=False).tail(1)
        deduped.rename(
            columns={
                "minutes_p10": "minutes_p10_pred",
                "minutes_p50": "minutes_p50_pred",
                "minutes_p90": "minutes_p90_pred",
                "play_prob": "play_prob_pred",
            },
            inplace=True,
        )
        return deduped[["game_id", "player_id", "minutes_p10_pred", "minutes_p50_pred", "minutes_p90_pred", "play_prob_pred"]]

    # ------------------------------------------------------------------
    # Final touches
    # ------------------------------------------------------------------

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        fallback_minutes = working.get("actual_minutes", pd.Series(0.0, index=working.index))
        missing_minutes = working["minutes_p50_pred"].isna()
        if missing_minutes.any():
            warnings.warn(
                f"[fpts] {int(missing_minutes.sum())} rows missing minutes predictions; "
                "falling back to actual minutes.",
                RuntimeWarning,
            )
            working.loc[missing_minutes, "minutes_p10_pred"] = fallback_minutes[missing_minutes]
            working.loc[missing_minutes, "minutes_p50_pred"] = fallback_minutes[missing_minutes]
            working.loc[missing_minutes, "minutes_p90_pred"] = fallback_minutes[missing_minutes]
            working.loc[missing_minutes, "play_prob_pred"] = 1.0

        for col in ("minutes_p10_pred", "minutes_p50_pred", "minutes_p90_pred", "play_prob_pred"):
            working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)
        working["minutes_volatility_pred"] = (
            working["minutes_p90_pred"] - working["minutes_p10_pred"]
        ).clip(lower=0.0)
        working["minutes_range_ratio"] = np.where(
            working["minutes_p50_pred"] > 0,
            working["minutes_volatility_pred"] / working["minutes_p50_pred"].clip(lower=1e-3),
            0.0,
        )

        if "pos_bucket" in working.columns:
            pos_series = canonical_pos_bucket_series(working["pos_bucket"])
            working["pos_is_guard"] = (pos_series == "G").astype(int)
            working["pos_is_wing"] = (pos_series == "W").astype(int)
            working["pos_is_big"] = (pos_series == "B").astype(int)
        lineup_role = working.get("lineup_role")
        if lineup_role is not None:
            working["lineup_role_tier"] = (
                lineup_role.fillna("")
                .str.lower()
                .map({"starter": 2, "bench": 1})
                .fillna(0)
                .astype(int)
            )

        status_series = working.get("status", pd.Series("", index=working.index))
        normalized_status = status_series.fillna("").str.lower()
        out_mask = normalized_status.str.startswith("out")
        questionable_mask = normalized_status.str.startswith("q")
        out_flags = out_mask.astype(int)
        q_flags = questionable_mask.astype(int)
        team_out = out_flags.groupby(
            [working["game_id"], working["team_id"]]
        ).transform("sum")
        team_q = q_flags.groupby(
            [working["game_id"], working["team_id"]]
        ).transform("sum")
        working["teammate_out_count"] = (team_out - out_flags).clip(lower=0)
        working["teammate_questionable_count"] = (team_q - q_flags).clip(lower=0)

        prior_contrib = (
            working["fpts_per_min_prior_10"].fillna(working["fpts_per_min_prior_5"])
            * working["minutes_p50_pred"].fillna(0.0)
        )
        out_usage = pd.Series(
            np.where(out_mask, prior_contrib, 0.0), index=working.index
        )
        team_out_usage = out_usage.groupby(
            [working["game_id"], working["team_id"]]
        ).transform("sum")
        working["teammate_out_usage_sum"] = (team_out_usage - out_usage).clip(lower=0.0)

        if "pos_bucket" in working.columns:
            same_pos_out = out_flags.groupby(
                [working["game_id"], working["team_id"], working["pos_bucket"]]
            ).transform("sum")
            working["same_pos_teammate_out_count"] = (
                same_pos_out - out_flags
            ).clip(lower=0)

        if {"total", "spread_home", "home_flag"}.issubset(working.columns):
            total = pd.to_numeric(working["total"], errors="coerce")
            spread_home = pd.to_numeric(working["spread_home"], errors="coerce")
            home_flag = working["home_flag"].fillna(0).astype(int)
            working["team_implied_total"] = np.where(
                home_flag == 1,
                total / 2 - spread_home / 2,
                total / 2 + spread_home / 2,
            )
            working["opponent_implied_total"] = np.where(
                home_flag == 1,
                total / 2 + spread_home / 2,
                total / 2 - spread_home / 2,
            )

        working["fpts_per_min_actual"] = working["actual_fpts"] / working["actual_minutes"].clip(lower=1e-6)
        working = working.dropna(subset=["fpts_per_min_actual"])
        return working


__all__ = ["FptsDatasetBuilder", "MinutesSource"]
