"""Dataset builder for recency-weighted Minutes V1 retraining."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from projections.minutes_v1.artifacts import write_json
from projections.minutes_v1.datasets import KEY_COLUMNS, deduplicate_latest


RETRAIN_DATASET_COLS: tuple[str, ...] = (
    "split",
    "weight_recency",
    "plays_target",
)


@dataclass(frozen=True)
class RetrainWindows:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    cal_start: pd.Timestamp
    cal_end: pd.Timestamp


@dataclass(frozen=True)
class RetrainBuildResult:
    run_id: str
    run_dir: Path
    dataset_path: Path
    meta_path: Path
    windows: RetrainWindows
    label_path: Path
    feature_root: Path


def _normalize_date(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def recency_weight_from_age_days(age_days: pd.Series | np.ndarray, *, half_life_days: float) -> np.ndarray:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")
    age_arr = np.asarray(age_days, dtype=float)
    return np.power(2.0, -age_arr / float(half_life_days))


def _git_sha_or_unknown() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _weight_summary(weights: pd.Series) -> dict[str, float]:
    if weights.empty:
        return {"min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(weights.min()),
        "p05": float(weights.quantile(0.05)),
        "p50": float(weights.quantile(0.50)),
        "p95": float(weights.quantile(0.95)),
        "max": float(weights.max()),
    }


def _load_features_for_season(*, data_root: Path, season: int) -> tuple[pd.DataFrame, Path]:
    season_root = data_root / "gold" / "features_minutes_v1" / f"season={season}"
    if not season_root.exists():
        raise FileNotFoundError(f"Missing features root: {season_root}")
    files = sorted(season_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {season_root}")
    frames = [pd.read_parquet(p) for p in files]
    return pd.concat(frames, ignore_index=True), season_root


def _prepare_joined_frame(
    *,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    windows: RetrainWindows,
    half_life_days: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_feature_cols = {"game_date", "feature_as_of_ts", "tip_ts", *KEY_COLUMNS}
    missing_features = required_feature_cols - set(features_df.columns)
    if missing_features:
        raise ValueError(f"Missing required feature columns: {sorted(missing_features)}")
    required_label_cols = {"game_date", "minutes", *KEY_COLUMNS}
    missing_labels = required_label_cols - set(labels_df.columns)
    if missing_labels:
        raise ValueError(f"Missing required label columns: {sorted(missing_labels)}")

    features = features_df.copy()
    labels = labels_df.copy()
    features["game_date"] = pd.to_datetime(features["game_date"], errors="coerce").dt.normalize()
    labels["game_date"] = pd.to_datetime(labels["game_date"], errors="coerce").dt.normalize()

    features = features[
        (features["game_date"] >= windows.train_start)
        & (features["game_date"] <= windows.cal_end)
    ].copy()
    if features.empty:
        raise ValueError("Feature frame is empty after window filtering.")

    feature_as_of = pd.to_datetime(features["feature_as_of_ts"], utc=True, errors="coerce")
    tip_ts = pd.to_datetime(features["tip_ts"], utc=True, errors="coerce")
    leakage_mask = feature_as_of > tip_ts
    leakage_rows = int(leakage_mask.fillna(False).sum())
    if leakage_rows > 0:
        raise RuntimeError(f"Leakage violation: {leakage_rows} rows have feature_as_of_ts > tip_ts")

    features = deduplicate_latest(features, key_cols=KEY_COLUMNS, order_cols=["feature_as_of_ts"])

    labels_keep = labels[list(KEY_COLUMNS) + ["game_date", "minutes"]].copy()
    labels_keep = labels_keep.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
    joined = features.merge(
        labels_keep.rename(columns={"minutes": "minutes_label", "game_date": "label_game_date"}),
        on=list(KEY_COLUMNS),
        how="left",
    )

    missing_label_mask = joined["minutes_label"].isna()
    dropped_missing_labels = int(missing_label_mask.sum())
    dropped_by_date: dict[str, int] = {}
    if dropped_missing_labels:
        dropped_counts = (
            pd.to_datetime(joined.loc[missing_label_mask, "game_date"], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .value_counts()
            .sort_index()
        )
        dropped_by_date = {str(idx): int(val) for idx, val in dropped_counts.items()}

    joined = joined.loc[~missing_label_mask].copy()
    if joined.empty:
        raise ValueError("No labeled rows remain after dropping missing labels.")

    joined["minutes"] = pd.to_numeric(joined["minutes_label"], errors="coerce").fillna(0.0)
    joined["plays_target"] = (joined["minutes"] > 0.0).astype(int)
    joined.drop(columns=["minutes_label"], inplace=True)

    joined["split"] = pd.NA
    train_mask = (joined["game_date"] >= windows.train_start) & (joined["game_date"] <= windows.train_end)
    cal_mask = (joined["game_date"] >= windows.cal_start) & (joined["game_date"] <= windows.cal_end)
    joined.loc[train_mask, "split"] = "train"
    joined.loc[cal_mask, "split"] = "cal"
    joined = joined[joined["split"].notna()].copy()
    if joined.empty:
        raise ValueError("No rows remain after assigning train/cal splits.")

    joined["weight_recency"] = 1.0
    train_rows_mask = joined["split"] == "train"
    if int(train_rows_mask.sum()) == 0:
        raise ValueError("No train rows found after split assignment.")
    age_days = (
        (windows.train_end - pd.to_datetime(joined.loc[train_rows_mask, "game_date"]).dt.normalize())
        .dt.days.clip(lower=0)
        .astype(int)
    )
    joined.loc[train_rows_mask, "weight_recency"] = recency_weight_from_age_days(
        age_days, half_life_days=half_life_days
    )
    joined["weight_recency"] = pd.to_numeric(joined["weight_recency"], errors="coerce").fillna(1.0)
    joined["weight_recency"] = joined["weight_recency"].where(joined["weight_recency"] > 0.0, 1e-6)

    split_counts = joined["split"].value_counts().to_dict()
    train_weights = joined.loc[joined["split"] == "train", "weight_recency"]
    summary = {
        "leakage_rows": leakage_rows,
        "dropped_missing_labels": dropped_missing_labels,
        "dropped_missing_labels_by_date": dropped_by_date,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "train_weight_summary": _weight_summary(train_weights),
        "dataset_rows": int(len(joined)),
        "dataset_game_date_min": str(pd.to_datetime(joined["game_date"]).min().date()),
        "dataset_game_date_max": str(pd.to_datetime(joined["game_date"]).max().date()),
    }
    return joined, summary


def build_retrain_dataset(
    *,
    data_root: Path,
    run_id: str,
    season: int,
    train_start_date: str | datetime | pd.Timestamp,
    train_end_date: str | datetime | pd.Timestamp,
    cal_start_date: str | datetime | pd.Timestamp,
    cal_end_date: str | datetime | pd.Timestamp,
    half_life_days: float = 35.0,
) -> RetrainBuildResult:
    label_path = data_root / "labels" / f"season={season}" / "boxscore_labels.parquet"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing labels parquet: {label_path}")
    labels = pd.read_parquet(label_path)
    labels["game_date"] = pd.to_datetime(labels["game_date"], errors="coerce").dt.normalize()
    labels = labels.dropna(subset=["game_date"]).copy()
    if labels.empty:
        raise ValueError("Labels frame is empty.")

    requested_train_start = _normalize_date(train_start_date)
    requested_train_end = _normalize_date(train_end_date)
    requested_cal_start = _normalize_date(cal_start_date)
    requested_cal_end = _normalize_date(cal_end_date)

    label_min = labels["game_date"].min()
    label_max = labels["game_date"].max()
    effective_train_start = max(requested_train_start, label_min)
    effective_train_end = min(requested_train_end, label_max)
    effective_cal_start = max(requested_cal_start, label_min)
    effective_cal_end = min(requested_cal_end, label_max)
    if effective_train_start > effective_train_end:
        raise ValueError("Effective train window is empty after label clamping.")
    if effective_cal_start > effective_cal_end:
        raise ValueError("Effective cal window is empty after label clamping.")
    if effective_train_end >= effective_cal_start:
        raise ValueError(
            "Train/cal windows must be non-overlapping for dataset split assignment "
            f"({effective_train_end.date()} vs {effective_cal_start.date()})."
        )
    windows = RetrainWindows(
        train_start=effective_train_start,
        train_end=effective_train_end,
        cal_start=effective_cal_start,
        cal_end=effective_cal_end,
    )

    features, feature_root = _load_features_for_season(data_root=data_root, season=season)
    joined, summary = _prepare_joined_frame(
        features_df=features,
        labels_df=labels,
        windows=windows,
        half_life_days=half_life_days,
    )

    run_dir = data_root / "artifacts" / "minutes_retrain_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = run_dir / "dataset.parquet"
    meta_path = run_dir / "meta.json"
    joined.to_parquet(dataset_path, index=False)

    payload = {
        "run_id": run_id,
        "built_at_utc": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha_or_unknown(),
        "season": season,
        "label_path": str(label_path),
        "feature_root": str(feature_root),
        "requested_windows": {
            "train_start": str(requested_train_start.date()),
            "train_end": str(requested_train_end.date()),
            "cal_start": str(requested_cal_start.date()),
            "cal_end": str(requested_cal_end.date()),
        },
        "effective_windows": {
            "train_start": str(windows.train_start.date()),
            "train_end": str(windows.train_end.date()),
            "cal_start": str(windows.cal_start.date()),
            "cal_end": str(windows.cal_end.date()),
        },
        "label_date_bounds": {
            "min": str(label_min.date()),
            "max": str(label_max.date()),
        },
        "recency_decay": {
            "enabled": True,
            "half_life_days": float(half_life_days),
            "formula": f"2 ** (-age_days / {float(half_life_days):.1f})",
            "train_end_date_anchor": str(windows.train_end.date()),
            "weight_column": "weight_recency",
        },
        "summary": summary,
        "dataset_path": str(dataset_path),
        "columns_added": list(RETRAIN_DATASET_COLS),
    }
    write_json(meta_path, payload)

    return RetrainBuildResult(
        run_id=run_id,
        run_dir=run_dir,
        dataset_path=dataset_path,
        meta_path=meta_path,
        windows=windows,
        label_path=label_path,
        feature_root=feature_root,
    )

