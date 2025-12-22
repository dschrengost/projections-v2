"""Live minutes evaluation dataset + metrics helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from projections import paths
from projections.minutes_v1.logs import prediction_logs_base

LOGGER = logging.getLogger(__name__)
SUPPORTED_SNAPSHOT_MODES = {"last_before_tip"}
_STATUS_OUT_TOKENS = ("OUT", "INACTIVE", "SUSPENDED", "REST", "INJURY", "COVID", "DOUBTFUL", "D")
_STATUS_Q_TOKENS = ("QUESTIONABLE", "GTD", "GAME-TIME", "PROB", "DAY-TO-DAY")
_STATUS_CLEAN_PREFIXES = ("AVA", "ACT", "CLE", "HEA")


def _normalize_day(value: date | pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.tz_localize(None).normalize()


def _iter_months(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    cursor = start.replace(day=1)
    limit = end.replace(day=1)
    while cursor <= limit:
        yield cursor
        cursor += pd.offsets.MonthBegin(1)


def _season_start(day: pd.Timestamp) -> int:
    return day.year if day.month >= 8 else day.year - 1


def _partition_value_from_path(path: Path, prefix: str) -> str | None:
    token = f"{prefix}="
    for part in path.parts:
        if part.startswith(token):
            return part.split("=", 1)[1]
    return None


def _safe_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _normalize_status_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "CLEAN"
    text = str(value).strip().upper()
    if not text:
        return "CLEAN"
    if any(token in text for token in _STATUS_OUT_TOKENS):
        return "OUT"
    if any(token in text for token in _STATUS_Q_TOKENS):
        return "QUESTIONABLE"
    if text in {"ACTIVE", "AVAILABLE", "CLEARED", "HEALTHY", "CLEAN"} or any(
        text.startswith(prefix) for prefix in _STATUS_CLEAN_PREFIXES
    ):
        return "CLEAN"
    return "OTHER"


def _normalize_status_series(series: pd.Series) -> pd.Series:
    normalized = series.map(_normalize_status_value)
    return normalized.astype("string")


def _smape(actual: np.ndarray, preds: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(preds)
    denom = np.where(denom == 0.0, 1e-6, denom)
    return float(np.mean(2.0 * np.abs(actual - preds) / denom))


@dataclass
class SnapshotSummary:
    total_games: int
    snapshot_games: int
    skipped_games: list[dict[str, Any]]
    rows_before_labels: int
    rows_after_labels: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_games": self.total_games,
            "games_with_snapshots": self.snapshot_games,
            "games_skipped": len(self.skipped_games),
            "skipped_games": self.skipped_games,
            "rows_before_label_join": self.rows_before_labels,
            "rows_after_label_join": self.rows_after_labels,
        }


class MinutesLiveEvalDatasetBuilder:
    """Construct a per-game snapshot of minutes predictions joined with labels."""

    def __init__(
        self,
        data_root: Path,
        logs_root: Path | None = None,
        labels_root: Path | None = None,
        schedule_root: Path | None = None,
        snapshot_mode: str = "last_before_tip",
    ) -> None:
        normalized_mode = snapshot_mode.strip().lower()
        if normalized_mode not in SUPPORTED_SNAPSHOT_MODES:
            raise ValueError(
                f"Unsupported snapshot_mode '{snapshot_mode}'. Supported: {', '.join(sorted(SUPPORTED_SNAPSHOT_MODES))}"
            )
        self.snapshot_mode = normalized_mode
        self.data_root = data_root.expanduser().resolve()
        self.logs_root = (logs_root or prediction_logs_base(self.data_root)).expanduser().resolve()
        default_labels = paths.data_path("gold", "labels_minutes_v1")
        self.labels_root = (labels_root or default_labels).expanduser().resolve()
        default_schedule = paths.data_path("silver", "schedule")
        self.schedule_root = (schedule_root or default_schedule).expanduser().resolve()
        self._last_snapshot_summary: SnapshotSummary | None = None

    @property
    def last_snapshot_summary(self) -> dict[str, Any] | None:
        return self._last_snapshot_summary.as_dict() if self._last_snapshot_summary else None

    def build(self, start_date: date, end_date: date) -> pd.DataFrame:
        start = _normalize_day(start_date)
        end = _normalize_day(end_date)
        if end < start:
            raise ValueError("end_date must be on or after start_date")

        schedule_df = self._load_schedule(start, end)
        total_games = int(schedule_df["game_id"].nunique()) if not schedule_df.empty else 0
        if schedule_df.empty:
            LOGGER.warning("[minutes-eval] schedule slice is empty for %s → %s", start.date(), end.date())
            self._last_snapshot_summary = SnapshotSummary(0, 0, [], 0, 0)
            return pd.DataFrame()

        logs_df = self._load_prediction_logs(start, end)
        if logs_df.empty:
            LOGGER.warning("[minutes-eval] no prediction logs found for %s → %s", start.date(), end.date())
            self._last_snapshot_summary = SnapshotSummary(total_games, 0, [], 0, 0)
            return pd.DataFrame()

        schedule_key = schedule_df.drop_duplicates(subset=["game_id", "game_date"])
        logs_with_schedule = logs_df.merge(
            schedule_key,
            on=["game_id", "game_date"],
            how="inner",
            suffixes=("", "_schedule"),
        )
        if logs_with_schedule.empty:
            LOGGER.warning("[minutes-eval] prediction logs did not match schedule window")
            self._last_snapshot_summary = SnapshotSummary(total_games, 0, [], 0, 0)
            return pd.DataFrame()

        snapshots, skipped = self._select_snapshots(schedule_key, logs_with_schedule)
        if snapshots.empty:
            LOGGER.warning("[minutes-eval] snapshot selection produced zero rows")
            self._last_snapshot_summary = SnapshotSummary(total_games, 0, skipped, 0, 0)
            return pd.DataFrame()

        labels_df = self._load_labels(start, end)
        if labels_df.empty:
            LOGGER.warning("[minutes-eval] labels slice is empty for %s → %s", start.date(), end.date())
            self._last_snapshot_summary = SnapshotSummary(
                total_games,
                int(snapshots["game_id"].nunique()),
                skipped,
                len(snapshots),
                0,
            )
            return pd.DataFrame()

        merged = snapshots.merge(
            labels_df,
            on=["game_id", "player_id", "game_date"],
            how="inner",
        )
        merged["actual_minutes"] = pd.to_numeric(merged["actual_minutes"], errors="coerce")
        merged = merged.dropna(subset=["actual_minutes"])

        merged["status_bucket"] = self._derive_status_bucket(merged)
        merged["injury_return_flag"] = self._annotate_injury_return_flag(merged)
        merged.sort_values(["game_date", "game_id", "player_id"], inplace=True)
        merged.reset_index(drop=True, inplace=True)

        merged["snapshot_mode"] = self.snapshot_mode
        rows_before_labels = int(len(snapshots))
        rows_after_labels = int(len(merged))
        self._last_snapshot_summary = SnapshotSummary(
            total_games,
            int(snapshots["game_id"].nunique()),
            skipped,
            rows_before_labels,
            rows_after_labels,
        )
        return merged

    def _load_schedule(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not self.schedule_root.exists():
            LOGGER.warning("[minutes-eval] schedule root %s missing", self.schedule_root)
            return pd.DataFrame()
        keep_cols = {
            "game_id",
            "game_date",
            "tip_ts",
            "season",
            "home_team_id",
            "away_team_id",
        }
        frames: list[pd.DataFrame] = []
        for season_dir in sorted(self.schedule_root.glob("season=*")):
            for month_dir in sorted(season_dir.glob("month=*")):
                sched_path = month_dir / "schedule.parquet"
                if not sched_path.exists():
                    continue
                df = pd.read_parquet(sched_path)
                missing = {"game_id", "game_date"} - set(df.columns)
                if missing:
                    continue
                tip_col = "tip_ts" if "tip_ts" in df.columns else None
                if tip_col is None:
                    if "game_ts" in df.columns:
                        tip_col = "game_ts"
                    elif "game_time_utc" in df.columns:
                        tip_col = "game_time_utc"
                if tip_col is None or tip_col not in df.columns:
                    continue
                subset_cols = [col for col in keep_cols if col in df.columns]
                if tip_col not in subset_cols:
                    subset_cols.append(tip_col)
                work = df.loc[:, subset_cols].copy()
                work.rename(columns={tip_col: "tip_ts"}, inplace=True)
                work["game_id"] = pd.to_numeric(work["game_id"], errors="coerce")
                work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
                work["tip_ts"] = pd.to_datetime(work["tip_ts"], utc=True, errors="coerce")
                work = work.dropna(subset=["game_id", "game_date", "tip_ts"])
                if work.empty:
                    continue
                work["game_id"] = work["game_id"].astype(int)
                work["game_date"] = work["game_date"].dt.tz_localize(None).dt.normalize()
                frames.append(work)
        if not frames:
            return pd.DataFrame()
        schedule = pd.concat(frames, ignore_index=True)
        schedule = schedule[(schedule["game_date"] >= start) & (schedule["game_date"] <= end)].copy()
        if schedule.empty:
            return pd.DataFrame()
        schedule.sort_values(["game_date", "game_id"], inplace=True)
        schedule = schedule.drop_duplicates(subset=["game_id"], keep="last")
        return schedule.reset_index(drop=True)

    def _log_partition_dirs(self, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
        if not self.logs_root.exists():
            return []
        dirs: set[Path] = set()
        for month in _iter_months(start, end):
            season = month.year
            pattern = f"**/season={season}/month={month.month:02d}"
            for candidate in self.logs_root.glob(pattern):
                if candidate.is_dir():
                    dirs.add(candidate.resolve())
        return sorted(dirs)

    def _load_prediction_logs(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        dirs = self._log_partition_dirs(start, end)
        frames: list[pd.DataFrame] = []
        for partition in dirs:
            run_id = _partition_value_from_path(partition, "run")
            for file in sorted(partition.glob("*.parquet")):
                df = pd.read_parquet(file)
                if df.empty:
                    continue
                if "run_id" not in df.columns and run_id is not None:
                    df = df.copy()
                    df["run_id"] = run_id
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        logs = pd.concat(frames, ignore_index=True)
        logs["game_id"] = pd.to_numeric(logs.get("game_id"), errors="coerce")
        logs["player_id"] = pd.to_numeric(logs.get("player_id"), errors="coerce")
        logs["game_date"] = pd.to_datetime(logs.get("game_date"), errors="coerce")
        logs["game_date"] = logs["game_date"].dt.tz_localize(None).dt.normalize()
        run_ts = pd.to_datetime(logs.get("run_as_of_ts"), utc=True, errors="coerce")
        if "log_timestamp" in logs.columns:
            fallback = pd.to_datetime(logs.get("log_timestamp"), utc=True, errors="coerce")
            run_ts = run_ts.fillna(fallback)
        logs["run_as_of_ts"] = run_ts
        logs = logs.dropna(subset=["game_id", "player_id", "game_date"])
        logs["game_id"] = logs["game_id"].astype(int)
        logs["player_id"] = logs["player_id"].astype(int)
        mask = (logs["game_date"] >= start) & (logs["game_date"] <= end)
        logs = logs.loc[mask].copy()
        return logs.reset_index(drop=True)

    def _load_labels(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not self.labels_root.exists():
            LOGGER.warning("[minutes-eval] labels root %s missing", self.labels_root)
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for day in pd.date_range(start, end, freq="D"):
            iso = day.date().isoformat()
            season = _season_start(day)
            season_dir = self.labels_root / f"season={season}"
            day_dir = season_dir / f"game_date={iso}"
            labels_path = day_dir / "labels.parquet"
            if not labels_path.exists():
                continue
            df = pd.read_parquet(labels_path)
            if df.empty:
                continue
            columns = set(df.columns)
            if not {"game_id", "player_id", "game_date"}.issubset(columns):
                continue
            work_columns = [
                col for col in ("game_id", "player_id", "team_id", "game_date", "minutes", "actual_minutes") if col in columns
            ]
            work = df.loc[:, work_columns].copy()
            if "minutes" in work.columns and "actual_minutes" not in work.columns:
                work.rename(columns={"minutes": "actual_minutes"}, inplace=True)
            if "minutes" in work.columns and "actual_minutes" in work.columns:
                work.drop(columns=["minutes"], inplace=True)
            if "actual_minutes" not in work.columns:
                continue
            work["game_id"] = pd.to_numeric(work["game_id"], errors="coerce")
            work["player_id"] = pd.to_numeric(work["player_id"], errors="coerce")
            if "team_id" in work.columns:
                work["team_id"] = pd.to_numeric(work["team_id"], errors="coerce")
            work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
            work["game_date"] = work["game_date"].dt.tz_localize(None).dt.normalize()
            work["actual_minutes"] = pd.to_numeric(work["actual_minutes"], errors="coerce")
            work = work.dropna(subset=["game_id", "player_id", "game_date", "actual_minutes"])
            work["game_id"] = work["game_id"].astype(int)
            work["player_id"] = work["player_id"].astype(int)
            frames.append(work)
        if not frames:
            return pd.DataFrame()
        labels = pd.concat(frames, ignore_index=True)
        labels.sort_values(["game_date", "game_id", "player_id"], inplace=True)
        return labels.reset_index(drop=True)

    def _select_snapshots(
        self,
        schedule_df: pd.DataFrame,
        logs_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        dedup_schedule = schedule_df.drop_duplicates(subset=["game_id", "game_date", "tip_ts"])
        groups = logs_df.groupby("game_id").groups
        snapshots: list[pd.DataFrame] = []
        skipped: list[dict[str, Any]] = []
        for row in dedup_schedule.itertuples(index=False):
            game_id = int(row.game_id)
            indices = groups.get(game_id)
            game_date = row.game_date.date().isoformat()
            if indices is None:
                skipped.append({"game_id": game_id, "game_date": game_date, "reason": "no_logs"})
                continue
            game_logs = logs_df.loc[indices].copy()
            tip_ts = pd.Timestamp(row.tip_ts) if pd.notna(row.tip_ts) else None
            if tip_ts is None or pd.isna(tip_ts):
                skipped.append({"game_id": game_id, "game_date": game_date, "reason": "missing_tip_ts"})
                continue
            valid_mask = game_logs["run_as_of_ts"].notna() & (game_logs["run_as_of_ts"] <= tip_ts)
            valid = game_logs.loc[valid_mask]
            if valid.empty:
                skipped.append({"game_id": game_id, "game_date": game_date, "reason": "no_snapshot_before_tip"})
                continue
            best_ts = valid["run_as_of_ts"].max()
            snapshot = valid.loc[valid["run_as_of_ts"] == best_ts].copy()
            snapshots.append(snapshot)
        if not snapshots:
            return pd.DataFrame(), skipped
        combined = pd.concat(snapshots, ignore_index=True)
        combined.sort_values(["game_date", "game_id", "player_id"], inplace=True)
        return combined.reset_index(drop=True), skipped

    def _derive_status_bucket(self, df: pd.DataFrame) -> pd.Series:
        status_series = df.get("status")
        if status_series is None:
            return pd.Series("CLEAN", index=df.index, dtype="string")
        normalized = _normalize_status_series(status_series)
        return normalized

    def _annotate_injury_return_flag(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=int)
        working = df.copy()
        order_ts = pd.to_datetime(working.get("tip_ts"), utc=True, errors="coerce")
        fallback = pd.to_datetime(working.get("game_date"), errors="coerce")
        order_ts = order_ts.fillna(fallback)
        working["_order_ts"] = order_ts
        working.sort_values(["player_id", "_order_ts", "game_id"], inplace=True)

        def _flag_group(group: pd.DataFrame) -> pd.DataFrame:
            prev_minutes = group["actual_minutes"].shift(1).fillna(0)
            prev_status = group["status_bucket"].shift(1).fillna("CLEAN").str.upper()
            returning = (
                (group["actual_minutes"] > 0)
                & (prev_minutes <= 0)
                & (prev_status.isin(["OUT", "QUESTIONABLE"]))
            ).astype(int)
            group["injury_return_flag"] = returning
            return group

        flagged = working.groupby("player_id", group_keys=False).apply(
            _flag_group, include_groups=False
        )
        flagged.sort_index(inplace=True)
        result = flagged["injury_return_flag"].astype(int)
        return result.reindex(df.index)


def evaluate_minutes_live_run(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        raise ValueError("Evaluation dataframe is empty.")
    actual = _safe_numeric(df.get("actual_minutes"))
    preds = _safe_numeric(df.get("minutes_p50"))
    if preds.isna().all():
        preds = _safe_numeric(df.get("minutes_p50_pred"))
    lower = _safe_numeric(df.get("minutes_p10"))
    upper = _safe_numeric(df.get("minutes_p90"))
    overall = _metric_payload(actual, preds, lower, upper)
    results: dict[str, Any] = {"rows": int(len(df)), "overall": overall, "slices": {}}

    slices: dict[str, list[dict[str, Any]]] = {}
    starter_slice = _starter_slices(df)
    if starter_slice:
        slices["starter_flag"] = starter_slice
    status_slice = _status_slices(df)
    if status_slice:
        slices["status"] = status_slice
    spread_slice = _spread_slices(df)
    if spread_slice:
        slices["spread_home"] = spread_slice
    minutes_slice = _minutes_bucket_slices(df)
    if minutes_slice:
        slices["minutes_p50"] = minutes_slice
    results["slices"] = slices
    rotation_slice = _rotation_slices(df)
    results["rotation_slices"] = rotation_slice
    results["status_slices"] = _status_bucket_slices(df)
    results["injury_return_slices"] = _injury_return_slices(df)
    return results


def _metric_payload(
    actual: pd.Series,
    preds: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> dict[str, float | int]:
    mask = actual.notna() & preds.notna()
    if not mask.any():
        return {
            "rows": 0,
            "mae_minutes": float("nan"),
            "rmse_minutes": float("nan"),
            "smape_minutes": float("nan"),
            "coverage_p10_p90": float("nan"),
            "under_rate_p10": float("nan"),
            "over_rate_p90": float("nan"),
        }
    actual_vals = actual[mask].to_numpy(dtype=float)
    preds_vals = preds[mask].to_numpy(dtype=float)
    errors = actual_vals - preds_vals
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    smape = _smape(actual_vals, preds_vals)
    coverage, under_rate, over_rate = _coverage_metrics(actual, lower, upper)
    cond_mask = actual > 0
    coverage_cond, under_cond, over_cond = _coverage_metrics(actual, lower, upper, mask=cond_mask)
    return {
        "rows": int(mask.sum()),
        "mae_minutes": mae,
        "rmse_minutes": rmse,
        "smape_minutes": smape,
        "coverage_p10_p90": coverage,
        "under_rate_p10": under_rate,
        "over_rate_p90": over_rate,
        "coverage_p10_p90_cond": coverage_cond,
        "under_rate_p10_cond": under_cond,
        "over_rate_p90_cond": over_cond,
    }


def _coverage_metrics(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    mask: pd.Series | None = None,
) -> tuple[float, float, float]:
    if lower.empty or upper.empty:
        return float("nan"), float("nan"), float("nan")
    base_mask = actual.notna() & lower.notna() & upper.notna()
    if mask is not None:
        aligned_mask = mask.reindex(actual.index, fill_value=False)
        base_mask &= aligned_mask.astype(bool)
    if not base_mask.any():
        return float("nan"), float("nan"), float("nan")
    a = actual[base_mask].to_numpy(dtype=float)
    lo = lower[base_mask].to_numpy(dtype=float)
    hi = upper[base_mask].to_numpy(dtype=float)
    covered = (a >= lo) & (a <= hi)
    under = a < lo
    over = a > hi
    return float(np.mean(covered)), float(np.mean(under)), float(np.mean(over))


def _starter_slices(df: pd.DataFrame) -> list[dict[str, Any]]:
    starter_series = df.get("starter_flag")
    if starter_series is None:
        starter_series = df.get("starter_flag_label")
    if starter_series is None:
        return []
    starter_bool = pd.to_numeric(starter_series, errors="coerce").fillna(0).astype(int) > 0
    buckets = {
        "starter": starter_bool,
        "bench": ~starter_bool,
    }
    return _evaluate_slice_buckets(df, buckets)


def _status_bucket(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "healthy"
    text = str(value).strip().lower()
    if not text:
        return "healthy"
    qish_tokens = ("q", "questionable", "probable", "doubt", "gtd")
    if any(token in text for token in qish_tokens):
        return "qish"
    healthy_tokens = ("available", "healthy", "active", "cleared")
    if any(token in text for token in healthy_tokens):
        return "healthy"
    return "other"


def _status_slices(df: pd.DataFrame) -> list[dict[str, Any]]:
    status_series = df.get("status")
    if status_series is None:
        return []
    assignments = status_series.map(_status_bucket)
    buckets = {label: assignments == label for label in ("healthy", "qish", "other")}
    return _evaluate_slice_buckets(df, buckets)


def _spread_slices(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "spread_home" not in df.columns:
        return []
    spread = pd.to_numeric(df["spread_home"], errors="coerce").abs()
    if spread.isna().all():
        return []
    buckets = {
        "<=3": spread <= 3.0,
        "3-8": (spread > 3.0) & (spread <= 8.0),
        ">8": spread > 8.0,
    }
    return _evaluate_slice_buckets(df, buckets)


def _minutes_bucket_slices(df: pd.DataFrame) -> list[dict[str, Any]]:
    minutes_series = df.get("minutes_p50")
    if minutes_series is None:
        minutes_series = df.get("minutes_p50_pred")
    if minutes_series is None:
        return []
    minutes = pd.to_numeric(minutes_series, errors="coerce")
    if minutes.isna().all():
        return []
    buckets = {
        "<20": minutes < 20.0,
        "20-30": (minutes >= 20.0) & (minutes < 30.0),
        ">30": minutes >= 30.0,
    }
    return _evaluate_slice_buckets(df, buckets)


def _rotation_slices(df: pd.DataFrame) -> list[dict[str, Any]]:
    actual = _safe_numeric(df.get("actual_minutes"))
    minutes_series = df.get("minutes_p50")
    if minutes_series is None:
        minutes_series = df.get("minutes_p50_pred")
    if minutes_series is None:
        return []
    minutes = pd.to_numeric(minutes_series, errors="coerce")
    if minutes.isna().all():
        return []
    starter_series = df.get("starter_flag")
    if starter_series is None:
        starter_series = df.get("starter_flag_label")
    starter_bool = (
        pd.to_numeric(starter_series, errors="coerce").fillna(0).astype(int) > 0
        if starter_series is not None
        else pd.Series(False, index=df.index)
    )
    rotation_mask = (actual > 0) & (minutes >= 20.0)
    if not rotation_mask.any():
        return []
    buckets = {
        "rotation_all": rotation_mask,
        "rotation_starters_30_plus": rotation_mask & starter_bool & (minutes >= 30.0),
        "rotation_mid_minutes": rotation_mask & (minutes >= 20.0) & (minutes < 30.0),
    }
    return _evaluate_slice_buckets(df, buckets)


def _status_bucket_slices(df: pd.DataFrame) -> list[dict[str, Any]]:
    status_series = df.get("status_bucket")
    if status_series is None:
        return []
    values = status_series.astype(str).str.upper()
    base_buckets = {
        "CLEAN": values == "CLEAN",
        "QUESTIONABLE": values == "QUESTIONABLE",
        "OUT": values == "OUT",
    }
    taken = pd.Series(False, index=values.index)
    buckets: dict[str, pd.Series] = {}
    for label, mask in base_buckets.items():
        normalized_mask = mask.fillna(False)
        buckets[label] = normalized_mask
        taken |= normalized_mask
    buckets["OTHER"] = ~taken
    return _evaluate_slice_buckets(df, buckets)


def _injury_return_slices(df: pd.DataFrame) -> list[dict[str, Any]]:
    flag = df.get("injury_return_flag")
    if flag is None:
        return []
    mask = pd.to_numeric(flag, errors="coerce").fillna(0).astype(int) > 0
    actual_positive = _safe_numeric(df.get("actual_minutes")).fillna(0) > 0
    buckets = {
        "injury_return": mask & actual_positive,
        "non_injury_return": (~mask) & actual_positive,
    }
    return _evaluate_slice_buckets(df, buckets)
def _evaluate_slice_buckets(
    df: pd.DataFrame,
    buckets: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    actual = _safe_numeric(df.get("actual_minutes"))
    preds = _safe_numeric(df.get("minutes_p50"))
    if preds.isna().all():
        preds = _safe_numeric(df.get("minutes_p50_pred"))
    lower = _safe_numeric(df.get("minutes_p10"))
    upper = _safe_numeric(df.get("minutes_p90"))
    results: list[dict[str, Any]] = []
    for label, mask in buckets.items():
        mask = mask.fillna(False)
        if not mask.any():
            continue
        subset_actual = actual[mask]
        subset_preds = preds[mask]
        subset_lower = lower[mask]
        subset_upper = upper[mask]
        metrics = _metric_payload(subset_actual, subset_preds, subset_lower, subset_upper)
        metrics["bucket"] = label
        results.append(metrics)
    return results
