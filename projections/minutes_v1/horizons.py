"""Utilities for building horizon-based minutes training rows.

These helpers support training separate "early" and "late" minutes models by
selecting a single feature snapshot per (game_id, player_id, horizon) relative
to tip time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import pandas as pd

from projections.minutes_v1.datasets import ensure_columns
from projections.utils import asof_left_join

DEFAULT_HORIZONS_MINUTES: tuple[int, ...] = (360, 180, 90, 60, 30, 15, 0)
DEFAULT_EARLY_HORIZONS_MINUTES: tuple[int, ...] = (360, 180, 90)
DEFAULT_LATE_HORIZONS_MINUTES: tuple[int, ...] = (60, 30, 15, 0)


def _normalize_utc(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _normalize_horizons(horizons_minutes: Sequence[int]) -> list[int]:
    if not horizons_minutes:
        raise ValueError("horizons_minutes must be non-empty")
    normalized: list[int] = []
    for value in horizons_minutes:
        try:
            minutes = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid horizon minutes value: {value!r}") from exc
        if minutes < 0:
            raise ValueError("horizons_minutes must be non-negative")
        normalized.append(minutes)
    return sorted(set(normalized), reverse=True)


def add_time_to_tip_features(
    df: pd.DataFrame,
    *,
    tip_ts_col: str = "tip_ts",
    feature_as_of_col: str = "feature_as_of_ts",
    output_col: str = "time_to_tip_min",
) -> pd.DataFrame:
    """Attach `time_to_tip_min` as (tip_ts - feature_as_of_ts) in minutes."""

    ensure_columns(df, {tip_ts_col})
    working = df.copy()
    tip_ts = pd.to_datetime(working[tip_ts_col], utc=True, errors="coerce")
    if feature_as_of_col in working.columns:
        as_of_ts = pd.to_datetime(working[feature_as_of_col], utc=True, errors="coerce")
    else:
        as_of_ts = tip_ts
    working[output_col] = (tip_ts - as_of_ts).dt.total_seconds() / 60.0
    working[output_col] = pd.to_numeric(working[output_col], errors="coerce").clip(lower=0.0)
    return working


def add_odds_missing_indicator(
    df: pd.DataFrame,
    *,
    spread_col: str = "spread_home",
    total_col: str = "total",
    output_col: str = "odds_missing",
) -> pd.DataFrame:
    """Attach `odds_missing` indicator (1 when spread/total are unavailable)."""

    if spread_col not in df.columns or total_col not in df.columns:
        return df.copy()
    working = df.copy()
    working[output_col] = (working[spread_col].isna() | working[total_col].isna()).astype(int)
    return working


def build_horizon_rows(
    df: pd.DataFrame,
    *,
    horizons_minutes: Sequence[int] = DEFAULT_HORIZONS_MINUTES,
    key_cols: Sequence[str] = ("game_id", "player_id", "team_id"),
    tip_ts_col: str = "tip_ts",
    snapshot_ts_col: str = "feature_as_of_ts",
    max_snapshot_age_hours: float | None = 12.0,
    feature_as_of_col: str = "feature_as_of_ts",
) -> pd.DataFrame:
    """Select exactly one snapshot row per (key, horizon) using as-of semantics.

    Selection rule:
      snapshot_ts = max snapshot_ts such that snapshot_ts <= tip_ts - horizon_min.

    Parameters
    ----------
    df:
        Source feature frame containing multiple snapshots per player/game.
    horizons_minutes:
        Horizons (in minutes before tip) to sample. Higher horizons represent earlier-in-day states.
    key_cols:
        Key columns identifying a player-game row. Defaults to Minutes V1 keys.
    tip_ts_col:
        Tip timestamp column (UTC or parseable to UTC).
    snapshot_ts_col:
        Timestamp column that defines snapshot ordering. Default uses `feature_as_of_ts`.
    max_snapshot_age_hours:
        Optional freshness guard. When set, require selected snapshot_ts >= tip_ts - max_snapshot_age_hours.
    feature_as_of_col:
        Column used for `time_to_tip_min` calculation. Defaults to `feature_as_of_ts`.
    """

    normalized_horizons = _normalize_horizons(horizons_minutes)
    ensure_columns(df, set(key_cols) | {tip_ts_col, snapshot_ts_col})

    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["horizon_min", "time_to_tip_min"])

    working = df.copy()
    working[tip_ts_col] = pd.to_datetime(working[tip_ts_col], utc=True, errors="coerce")
    working[snapshot_ts_col] = pd.to_datetime(working[snapshot_ts_col], utc=True, errors="coerce")
    required = list(key_cols) + [tip_ts_col, snapshot_ts_col]
    working = working.dropna(subset=required).copy()
    if working.empty:
        return pd.DataFrame(columns=list(df.columns) + ["horizon_min", "time_to_tip_min"])

    # Drop any post-tip snapshots (defensive; they should never be used for training).
    working = working.loc[working[snapshot_ts_col] <= working[tip_ts_col]].copy()
    if working.empty:
        return pd.DataFrame(columns=list(df.columns) + ["horizon_min", "time_to_tip_min"])

    # Break ties deterministically when multiple rows share the same snapshot timestamp.
    order_cols = [snapshot_ts_col]
    if "ingested_ts" in working.columns:
        working["ingested_ts"] = pd.to_datetime(working["ingested_ts"], utc=True, errors="coerce")
        order_cols.append("ingested_ts")
    working = working.sort_values(list(key_cols) + order_cols, kind="mergesort")
    working = working.drop_duplicates(subset=list(key_cols) + [snapshot_ts_col], keep="last").reset_index(drop=True)

    base_keys = working.loc[:, list(key_cols) + [tip_ts_col]].drop_duplicates(subset=list(key_cols)).copy()
    if base_keys.empty:
        return pd.DataFrame(columns=list(df.columns) + ["horizon_min", "time_to_tip_min"])

    frames: list[pd.DataFrame] = []
    for horizon in normalized_horizons:
        cutoff = base_keys[tip_ts_col] - pd.Timedelta(minutes=horizon)
        left = base_keys.loc[:, list(key_cols)].copy()
        left["_cutoff_ts"] = cutoff
        left["horizon_min"] = horizon

        selected = asof_left_join(
            left,
            working,
            on=list(key_cols),
            left_time_col="_cutoff_ts",
            right_time_col=snapshot_ts_col,
        )
        selected = selected.drop(columns=["_cutoff_ts"], errors="ignore")
        selected = selected.dropna(subset=[snapshot_ts_col]).copy()
        if selected.empty:
            continue

        if max_snapshot_age_hours is not None:
            min_allowed = selected[tip_ts_col] - pd.Timedelta(hours=float(max_snapshot_age_hours))
            selected = selected.loc[selected[snapshot_ts_col] >= min_allowed].copy()
            if selected.empty:
                continue

        frames.append(selected)

    if not frames:
        return pd.DataFrame(columns=list(df.columns) + ["horizon_min", "time_to_tip_min"])

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = add_time_to_tip_features(out, tip_ts_col=tip_ts_col, feature_as_of_col=feature_as_of_col)

    # Enforce uniqueness for downstream splitting/training.
    dup_key = list(key_cols) + ["horizon_min"]
    if out.duplicated(dup_key).any():
        raise AssertionError("Horizon selection produced duplicate (key, horizon_min) rows.")

    return out


def filter_horizons(df: pd.DataFrame, horizons_minutes: Sequence[int], *, horizon_col: str = "horizon_min") -> pd.DataFrame:
    """Filter a horizonized dataset down to a set of horizons."""

    ensure_columns(df, {horizon_col})
    keep = set(int(x) for x in horizons_minutes)
    return df.loc[df[horizon_col].isin(keep)].copy()


@dataclass(frozen=True)
class TipTimeSplit:
    """Time-ordered splits based on per-game tip timestamps."""

    train_end: pd.Timestamp
    cal_end: pd.Timestamp
    val_end: pd.Timestamp

    @classmethod
    def from_bounds(
        cls,
        *,
        train_end: datetime | str | pd.Timestamp,
        cal_end: datetime | str | pd.Timestamp,
        val_end: datetime | str | pd.Timestamp,
    ) -> "TipTimeSplit":
        train_end_ts = _normalize_utc(train_end)
        cal_end_ts = _normalize_utc(cal_end)
        val_end_ts = _normalize_utc(val_end)
        if not (train_end_ts < cal_end_ts < val_end_ts):
            raise ValueError("Expected train_end < cal_end < val_end for tip time splits.")
        return cls(train_end=train_end_ts, cal_end=cal_end_ts, val_end=val_end_ts)


def assign_game_splits(
    df: pd.DataFrame,
    split: TipTimeSplit,
    *,
    game_id_col: str = "game_id",
    tip_ts_col: str = "tip_ts",
    output_col: str = "split",
) -> pd.DataFrame:
    """Assign each row to train/cal/val, grouping strictly by game_id."""

    ensure_columns(df, {game_id_col, tip_ts_col})
    if df.empty:
        working = df.copy()
        working[output_col] = pd.Series([], dtype="string")
        return working

    working = df.copy()
    working[tip_ts_col] = pd.to_datetime(working[tip_ts_col], utc=True, errors="coerce")
    game_tips = working.loc[:, [game_id_col, tip_ts_col]].dropna().copy()
    if game_tips.empty:
        working[output_col] = "out_of_range"
        return working

    tip_min = game_tips.groupby(game_id_col, dropna=False)[tip_ts_col].min()
    tip_max = game_tips.groupby(game_id_col, dropna=False)[tip_ts_col].max()
    mismatched = (tip_min != tip_max) & tip_min.notna() & tip_max.notna()
    if mismatched.any():
        bad_games = mismatched[mismatched].index.astype(str).tolist()
        raise ValueError(
            "Detected multiple tip_ts values for the same game_id (sample): " + ", ".join(bad_games[:10])
        )

    unique_tips = tip_min.reset_index().rename(columns={tip_ts_col: "game_tip_ts"})
    unique_tips["game_tip_ts"] = pd.to_datetime(unique_tips["game_tip_ts"], utc=True, errors="coerce")
    unique_tips = unique_tips.dropna(subset=["game_tip_ts"]).copy()

    train_end = split.train_end
    cal_end = split.cal_end
    val_end = split.val_end

    def _label(ts: pd.Timestamp) -> str:
        if ts <= train_end:
            return "train"
        if ts <= cal_end:
            return "cal"
        if ts <= val_end:
            return "val"
        return "out_of_range"

    unique_tips[output_col] = unique_tips["game_tip_ts"].apply(_label).astype("string")
    return working.merge(unique_tips[[game_id_col, output_col]], on=game_id_col, how="left")


@dataclass(frozen=True)
class WalkForwardFold:
    """Walk-forward fold definition (train/cal/val cutoffs by tip timestamp)."""

    fold_id: str
    split: TipTimeSplit


def build_walk_forward_folds(
    tip_ts: pd.Series,
    *,
    min_train_days: int,
    cal_days: int,
    val_days: int,
    step_days: int = 7,
) -> list[WalkForwardFold]:
    """Generate simple walk-forward folds using tip-date (UTC) cutoffs.

    This helper is intended for robustness evaluation, not for defining the
    primary production train/cal/val windows.
    """

    if min_train_days <= 0:
        raise ValueError("min_train_days must be positive")
    if cal_days <= 0:
        raise ValueError("cal_days must be positive")
    if val_days <= 0:
        raise ValueError("val_days must be positive")
    if step_days <= 0:
        raise ValueError("step_days must be positive")

    days = pd.to_datetime(tip_ts, utc=True, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    if days.empty:
        return []

    max_day = days.max()
    day_index = pd.Index(days)
    folds: list[WalkForwardFold] = []

    i = min_train_days - 1
    while i < len(days):
        train_end_day = days.iloc[i]
        cal_end_day = train_end_day + pd.Timedelta(days=cal_days)
        val_end_day = cal_end_day + pd.Timedelta(days=val_days)
        if val_end_day > max_day:
            break

        def _eod(day: pd.Timestamp) -> pd.Timestamp:
            ts = pd.Timestamp(day)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        split = TipTimeSplit.from_bounds(
            train_end=_eod(train_end_day),
            cal_end=_eod(cal_end_day),
            val_end=_eod(val_end_day),
        )
        fold_id = f"wf_train_end={train_end_day.date().isoformat()}_val_end={val_end_day.date().isoformat()}"
        folds.append(WalkForwardFold(fold_id=fold_id, split=split))

        target_next = train_end_day + pd.Timedelta(days=step_days)
        next_i = int(day_index.searchsorted(target_next, side="right") - 1)
        if next_i <= i:
            next_i = i + 1
        i = next_i

    return folds
