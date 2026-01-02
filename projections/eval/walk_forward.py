"""Generic walk-forward fold generation + evaluation helpers.

This module is intentionally small and mirrors the folding logic used by
`scripts/walk_forward_minutes.py` (expanding training window with optional
calibration and season-aware skipping), while also supporting in-memory
walk-forward evaluation for other models (e.g. rates).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator, Protocol

import pandas as pd

NBA_OFFSEASON_MONTHS: set[int] = {7, 8, 9}  # July, August, September


@dataclass(frozen=True)
class DateFold:
    """Fold boundaries expressed as datetimes (typically day precision)."""

    fold_id: str
    train_start: datetime
    train_end: datetime
    cal_start: datetime | None
    cal_end: datetime | None
    val_start: datetime
    val_end: datetime

    def to_dict(self) -> dict[str, str | None]:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "cal_start": self.cal_start.isoformat() if self.cal_start else None,
            "cal_end": self.cal_end.isoformat() if self.cal_end else None,
            "val_start": self.val_start.isoformat(),
            "val_end": self.val_end.isoformat(),
        }


def is_offseason(dt: datetime, *, offseason_months: set[int] = NBA_OFFSEASON_MONTHS) -> bool:
    return dt.month in offseason_months


def generate_expanding_date_folds(
    *,
    data_start: datetime,
    data_end: datetime,
    min_train_months: int,
    cal_weeks: int,
    val_weeks: int,
    step_weeks: int,
    season_aware: bool = True,
    offseason_months: set[int] = NBA_OFFSEASON_MONTHS,
    uses_calibration: bool = True,
    fold_id_format: str = "fold_{fold_num:02d}",
) -> list[DateFold]:
    """Generate expanding-window walk-forward folds.

    This mirrors the logic in `scripts/walk_forward_minutes.py`:
    - Train window expands from `data_start`
    - Validation window is a fixed-width trailing window ending at `val_end`
    - Optional calibration window is placed between train and val
    - Folds advance by `step_weeks`
    - Optionally skip folds with off-season validation (July-Sep by default)
    """

    if min_train_months <= 0:
        raise ValueError("min_train_months must be positive")
    if val_weeks <= 0:
        raise ValueError("val_weeks must be positive")
    if step_weeks <= 0:
        raise ValueError("step_weeks must be positive")
    if cal_weeks < 0:
        raise ValueError("cal_weeks must be non-negative")

    folds: list[DateFold] = []

    train_start = data_start
    min_train_end = train_start + timedelta(days=min_train_months * 30)

    fold_num = 1
    if uses_calibration:
        current_val_end = min_train_end + timedelta(weeks=cal_weeks + val_weeks)
    else:
        current_val_end = min_train_end + timedelta(weeks=val_weeks)

    while current_val_end <= data_end:
        val_end = current_val_end
        val_start = val_end - timedelta(weeks=val_weeks)

        if uses_calibration:
            cal_end = val_start - timedelta(days=1)
            cal_start = cal_end - timedelta(weeks=cal_weeks) + timedelta(days=1) if cal_weeks > 0 else cal_end
            train_end = cal_start - timedelta(days=1)
        else:
            cal_start = None
            cal_end = None
            train_end = val_start - timedelta(days=1)

        if train_end < min_train_end:
            current_val_end += timedelta(weeks=step_weeks)
            continue

        if season_aware and (
            is_offseason(val_start, offseason_months=offseason_months)
            or is_offseason(val_end, offseason_months=offseason_months)
        ):
            current_val_end += timedelta(weeks=step_weeks)
            continue

        folds.append(
            DateFold(
                fold_id=fold_id_format.format(fold_num=fold_num),
                train_start=train_start,
                train_end=train_end,
                cal_start=cal_start,
                cal_end=cal_end,
                val_start=val_start,
                val_end=val_end,
            )
        )
        fold_num += 1
        current_val_end += timedelta(weeks=step_weeks)

    return folds


def _ensure_datetime_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Missing required time column: {col}")
    out = pd.to_datetime(df[col], errors="coerce")
    if out.isna().all():
        raise ValueError(f"Column {col} could not be parsed as datetimes (all NaT).")
    return out


def _between_inclusive(ts: pd.Series, start: datetime, end: datetime) -> pd.Series:
    return (ts >= start) & (ts <= end)


@dataclass
class FoldMeta:
    """Metadata passed through the fold runner; may include cal_df when requested."""

    fold: DateFold
    cal_df: pd.DataFrame | None = None
    extras: dict[str, Any] | None = None


class TrainerFn(Protocol):
    def __call__(
        self,
        train_df: pd.DataFrame,
        *,
        train_end_ts: pd.Timestamp,
        fold_meta: FoldMeta,
        seed: int,
    ) -> Any: ...


class PredictFn(Protocol):
    def __call__(self, model: Any, df: pd.DataFrame, *, fold_meta: FoldMeta) -> Any: ...


class EvalFn(Protocol):
    def __call__(self, df: pd.DataFrame, preds: Any, *, fold_meta: FoldMeta) -> dict[str, float]: ...


def iter_time_folds(
    df: pd.DataFrame,
    *,
    time_col: str,
    folds: list[DateFold] | None = None,
    data_start: datetime | None = None,
    data_end: datetime | None = None,
    min_train_months: int = 12,
    cal_weeks: int = 0,
    val_weeks: int = 2,
    step_weeks: int = 4,
    season_aware: bool = True,
    offseason_months: set[int] = NBA_OFFSEASON_MONTHS,
    uses_calibration: bool = True,
    min_train_rows: int = 1,
    min_val_rows: int = 1,
    attach_cal_df: bool = True,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, FoldMeta]]:
    """Yield (train_df, val_df, train_end_ts, fold_meta) for each fold.

    Notes
    -----
    - Slices are inclusive on both ends for each window, matching the day-based
      boundaries produced by `generate_expanding_date_folds`.
    - `train_end_ts` is reported as the fold's `train_end` boundary (converted
      to pandas Timestamp) and is intended for recency-weight computation.
    """

    if folds is None:
        if data_start is None or data_end is None:
            raise ValueError("Provide either folds=... or (data_start=..., data_end=...).")
        folds = generate_expanding_date_folds(
            data_start=data_start,
            data_end=data_end,
            min_train_months=min_train_months,
            cal_weeks=cal_weeks,
            val_weeks=val_weeks,
            step_weeks=step_weeks,
            season_aware=season_aware,
            offseason_months=offseason_months,
            uses_calibration=uses_calibration,
        )

    ts = _ensure_datetime_col(df, time_col)
    working = df.copy()
    working["_wf_time_col"] = ts

    for fold in folds:
        train_mask = _between_inclusive(working["_wf_time_col"], fold.train_start, fold.train_end)
        val_mask = _between_inclusive(working["_wf_time_col"], fold.val_start, fold.val_end)
        train_df = working.loc[train_mask].drop(columns=["_wf_time_col"]).copy()
        val_df = working.loc[val_mask].drop(columns=["_wf_time_col"]).copy()

        if len(train_df) < min_train_rows or len(val_df) < min_val_rows:
            continue

        cal_df = None
        if attach_cal_df and fold.cal_start is not None and fold.cal_end is not None:
            cal_mask = _between_inclusive(working["_wf_time_col"], fold.cal_start, fold.cal_end)
            cal_df = working.loc[cal_mask].drop(columns=["_wf_time_col"]).copy()

        yield train_df, val_df, pd.Timestamp(fold.train_end), FoldMeta(fold=fold, cal_df=cal_df, extras=None)


def run_folds(
    folds_iter: Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, FoldMeta]],
    *,
    trainer_fn: TrainerFn,
    predict_fn: PredictFn,
    eval_fn: EvalFn,
    seed: int,
    on_fold_end: Callable[[int, FoldMeta, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """Run a walk-forward evaluation loop over pre-sliced folds."""

    fold_rows: list[dict[str, Any]] = []
    for i, (train_df, val_df, train_end_ts, fold_meta) in enumerate(folds_iter):
        model = trainer_fn(train_df, train_end_ts=train_end_ts, fold_meta=fold_meta, seed=seed + i)
        preds = predict_fn(model, val_df, fold_meta=fold_meta)
        metrics = eval_fn(val_df, preds, fold_meta=fold_meta)

        row = {
            "fold_id": fold_meta.fold.fold_id,
            "train_end_ts": str(train_end_ts),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "metrics": metrics,
            "fold": fold_meta.fold.to_dict(),
        }
        fold_rows.append(row)
        if on_fold_end is not None:
            on_fold_end(i, fold_meta, metrics)

    return {"n_folds": len(fold_rows), "folds": fold_rows}


def assert_fold_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    time_col: str,
    train_end_ts: pd.Timestamp,
    fold_id: str | None = None,
    key_cols: tuple[str, ...] | None = None,
) -> None:
    """Raise if a fold appears to leak future rows into training or overlap keys.

    Assertions (per user requirement):
    - train_df[time_col].max() <= train_end_ts
    - val_df[time_col].min() > train_end_ts
    - train/val have no overlap on primary keys when available
    """

    tag = f" fold_id={fold_id}" if fold_id else ""
    if time_col not in train_df.columns or time_col not in val_df.columns:
        raise KeyError(f"Missing time_col={time_col} in train/val.{tag}")

    train_times = pd.to_datetime(train_df[time_col], errors="coerce")
    val_times = pd.to_datetime(val_df[time_col], errors="coerce")
    if train_times.isna().any() or val_times.isna().any():
        raise ValueError(
            f"time_col={time_col} contains NaT (train_na={int(train_times.isna().sum())}, "
            f"val_na={int(val_times.isna().sum())}).{tag}"
        )

    train_max = pd.Timestamp(train_times.max())
    val_min = pd.Timestamp(val_times.min())
    end = pd.Timestamp(train_end_ts)
    tz = getattr(train_times.dt, "tz", None)
    if tz is not None and end.tzinfo is None:
        end = end.tz_localize(tz)
    if tz is None and end.tzinfo is not None:
        end = end.tz_localize(None)

    if train_max > end:
        raise ValueError(
            f"Fold integrity violation: train max {time_col}={train_max} > train_end_ts={end}.{tag}"
        )
    if val_min <= end:
        raise ValueError(
            f"Fold integrity violation: val min {time_col}={val_min} <= train_end_ts={end}.{tag}"
        )

    inferred_key_cols: tuple[str, ...] | None = key_cols
    if inferred_key_cols is None:
        for candidate in (("game_id", "player_id"), ("season", "game_id", "player_id")):
            if all(c in train_df.columns for c in candidate) and all(c in val_df.columns for c in candidate):
                inferred_key_cols = candidate
                break

    if inferred_key_cols is None:
        return

    train_keys = train_df.loc[:, list(inferred_key_cols)].dropna().drop_duplicates()
    val_keys = val_df.loc[:, list(inferred_key_cols)].dropna().drop_duplicates()
    if train_keys.empty or val_keys.empty:
        return

    overlap = train_keys.merge(val_keys, on=list(inferred_key_cols), how="inner")
    if not overlap.empty:
        sample = overlap.head(5).to_dict(orient="records")
        raise ValueError(
            f"Fold integrity violation: train/val overlap on keys={inferred_key_cols} "
            f"(n_overlap={len(overlap)}). sample={sample}.{tag}"
        )
