#!/usr/bin/env python3
"""Build a rotation-augmented Minutes V1 training dataset (v1).

Inputs (fixed for this v1 builder):
  - Minutes dataset:
      /home/daniel/projections-data/training/datasets/v1_enriched_20251214/
        - features.parquet
        - labels.parquet
  - Rotation v1 silver:
      <DATA_ROOT>/silver/rotation_v1/player_game_labels/season=*/game_id=*.parquet
      <DATA_ROOT>/silver/rotation_v1/team_game_shape/season=*/game_id=*.parquet

Outputs:
  <DATA_ROOT>/training/datasets/rotation_train_v1_20260103/
    - features.parquet
    - labels.parquet
    - manifest.json

This script does NOT train any model. It only builds a joined dataset.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from projections import paths
from projections.features.dnp_history import (
    DNP_HISTORY_FEATURE_COLUMNS,
    DNPHistoryConfig,
    compute_dnp_history_features,
)
from projections.features.action_props import (
    ACTION_MARKET_FEATURE_COLUMNS,
    attach_action_props_features,
    load_action_props_feature_snapshots_for_date,
)
from projections.rotation.rotation_set_minutes_features_v1 import join_rotation_priors


MINUTES_DATASET_DIRNAME = "v1_enriched_20251214"
DEFAULT_OUT_DIRNAME = "rotation_train_v1_20260103"
MINUTES_BUNDLE_DIR = Path("artifacts/minutes_lgbm/minutes_v1_safe_starter_20251214")

TEAM_TOTAL_MINUTES_COL = "team_total_minutes_from_stints"
PLAYER_MINUTES_FROM_STINTS_COL = "minutes_from_stints"

DROP_FEATURES = {
    "arch_delta_max_role",
    "arch_delta_min_role",
    "arch_delta_same_pos",
    "arch_delta_sum",
    "arch_missing_same_pos_count",
    "arch_missing_total_count",
    "days_since_return",
    "games_since_return",
    "season_phase",
    "starter_flag",
    "starter_prev_game_asof",
    "role_change_rate_10g",
    "rotation_minutes_std_5g",
    # These features only consider games where player got minutes (>0), ignoring DNPs.
    # This is misleading for rotation prediction - use minutes_from_stints_prior_* instead
    # which correctly includes DNP games as 0 minutes.
    "min_last1",
    "min_last3",
    "min_last5",
    "roll_mean_3",
    "roll_mean_5",
    "roll_mean_10",
    "roll_iqr_5",
    "z_vs_10",
}

ODDS_COLS = ("spread_home", "total")

PLAYER_ROTATION_COLS = [
    "num_stints",
    "first_in_time_real",
    "last_out_time_real",
    "max_stint_len_real",
    "minutes_from_stints",
    "started_proxy",
    "time_unit_detected",
]

TEAM_ROTATION_COLS = [
    "depth_6",
    "depth_10",
    "depth_14",
    "effective_n",
    "bench_conc_top1",
    "bench_conc_top2",
    "starter_pool_minutes",
    "bench_pool_minutes",
    "team_total_minutes_from_stints",
]

DEFAULT_MIN_TEAM_MINUTES_FROM_STINTS = 200.0
DEFAULT_MAX_TEAM_MINUTES_GAP = 2.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _season_for_date(day: pd.Timestamp) -> int:
    # Match the convention used elsewhere in the repo (Aug–Jul season boundary).
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _zfill_game_id(series: pd.Series) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce").astype("Int64")
    return coerced.astype("string").str.zfill(10)


def _read_feature_allowlist(bundle_dir: Path) -> list[str]:
    payload = json.loads((bundle_dir / "feature_columns.json").read_text(encoding="utf-8"))
    cols = payload.get("columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError(f"Invalid feature_columns.json at {bundle_dir}")
    return [str(c) for c in cols]


def _infer_label_column(labels_df: pd.DataFrame) -> str:
    candidates = [c for c in labels_df.columns if c.lower() in {"minutes", "min", "target_minutes"}]
    for col in candidates:
        if col in labels_df.columns:
            return col
    numeric = [
        c
        for c in labels_df.columns
        if c not in {"game_id", "team_id", "player_id"}
        and pd.api.types.is_numeric_dtype(labels_df[c])
        and "minute" in c.lower()
    ]
    if len(numeric) == 1:
        return numeric[0]
    raise ValueError(f"Could not infer label column; candidates={candidates}, numeric_minute_like={numeric}")


def _align_labels_to_features(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["game_id", "team_id", "player_id"]
    if not set(key_cols).issubset(features_df.columns):
        raise ValueError(f"Features missing required keys for label alignment: {key_cols}")
    if not set(key_cols).issubset(labels_df.columns):
        raise ValueError(f"Labels missing required keys for label alignment: {key_cols}")

    deduped = labels_df.drop_duplicates(subset=key_cols, keep="last").copy()
    aligned = features_df.loc[:, key_cols].merge(deduped, on=key_cols, how="left", sort=False)

    # Backfill common metadata from features for rows that were missing in labels (e.g. DNP rows).
    for col in ["player_name", "season", "game_date"]:
        if col in aligned.columns and col in features_df.columns:
            aligned[col] = aligned[col].where(aligned[col].notna(), features_df[col])
    return aligned


def _team_game_summary(features_df: pd.DataFrame, labels_df: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    required = {"game_id", "team_id", "player_id"}
    if not required.issubset(features_df.columns) or not required.issubset(labels_df.columns):
        raise ValueError("Both features and labels must contain game_id/team_id/player_id")
    if label_col not in labels_df.columns:
        raise ValueError(f"Labels missing label column: {label_col}")

    # Use a single, positional-aligned frame to avoid groupby index alignment pitfalls
    # (features_df and labels_df may not share identical indices after merges/filters).
    tmp = pd.DataFrame(
        {
            "game_id_norm": _zfill_game_id(features_df["game_id"]).to_numpy(),
            "team_id": pd.to_numeric(features_df["team_id"], errors="coerce").astype("Int64").to_numpy(),
            "rotation_team_missing": pd.to_numeric(
                features_df.get("rotation_team_missing", pd.Series([pd.NA] * len(features_df))), errors="coerce"
            )
            .fillna(1)
            .to_numpy(),
            "stints_team_total": pd.to_numeric(
                features_df.get(TEAM_TOTAL_MINUTES_COL, pd.Series([pd.NA] * len(features_df))), errors="coerce"
            ).to_numpy(),
            "minutes_from_stints": pd.to_numeric(
                features_df.get(PLAYER_MINUTES_FROM_STINTS_COL, pd.Series([pd.NA] * len(features_df))), errors="coerce"
            )
            .fillna(0.0)
            .to_numpy(),
            "label_minutes": pd.to_numeric(labels_df[label_col], errors="coerce").fillna(0.0).to_numpy(),
        }
    )
    grouped = tmp.groupby(["game_id_norm", "team_id"], sort=False)
    out = grouped.agg(
        n_players=("label_minutes", "size"),
        n_pos=("label_minutes", lambda x: int((x > 0).sum())),
        label_sum=("label_minutes", "sum"),
        stints_team_total=("stints_team_total", "mean"),
        minutes_from_stints_sum=("minutes_from_stints", "sum"),
        rotation_team_missing=("rotation_team_missing", "max"),
    ).reset_index()
    out["stints_gap"] = (out["stints_team_total"] - out["minutes_from_stints_sum"]).abs()
    out["label_gap"] = (out["stints_team_total"] - out["label_sum"]).abs()
    return out


def _filter_invalid_team_games(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    label_col: str,
    min_team_minutes_from_stints: float,
    max_team_minutes_gap: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary = _team_game_summary(features_df, labels_df, label_col=label_col)

    rotation_present = summary["rotation_team_missing"].fillna(1).astype(int) == 0
    stints_ok = summary["stints_team_total"].notna() & (summary["stints_team_total"] >= float(min_team_minutes_from_stints))
    coverage_ok = summary["stints_gap"].notna() & (summary["stints_gap"] <= float(max_team_minutes_gap))
    labels_ok = summary["label_gap"].notna() & (summary["label_gap"] <= float(max_team_minutes_gap))

    reason_rotation_missing = ~rotation_present
    reason_stints_low = rotation_present & ~stints_ok
    reason_missing_coverage = rotation_present & stints_ok & ~coverage_ok
    reason_labels_incomplete = rotation_present & stints_ok & coverage_ok & ~labels_ok

    keep = rotation_present & stints_ok & coverage_ok & labels_ok

    kept_pairs = set(
        zip(
            summary.loc[keep, "game_id_norm"].astype(str),
            pd.to_numeric(summary.loc[keep, "team_id"], errors="coerce").fillna(-1).astype(int),
        )
    )

    feat = features_df.copy()
    lab = labels_df.copy()
    feat["game_id_norm"] = _zfill_game_id(feat["game_id"])
    lab["game_id_norm"] = _zfill_game_id(lab["game_id"])
    feat["team_id"] = pd.to_numeric(feat["team_id"], errors="coerce").astype("Int64")
    lab["team_id"] = pd.to_numeric(lab["team_id"], errors="coerce").astype("Int64")

    mask_feat = list(
        zip(feat["game_id_norm"].astype(str), pd.to_numeric(feat["team_id"], errors="coerce").fillna(-1).astype(int))
    )
    mask_lab = list(
        zip(lab["game_id_norm"].astype(str), pd.to_numeric(lab["team_id"], errors="coerce").fillna(-1).astype(int))
    )
    feat_keep = pd.Series([p in kept_pairs for p in mask_feat], index=feat.index)
    lab_keep = pd.Series([p in kept_pairs for p in mask_lab], index=lab.index)

    filtered_features = feat.loc[feat_keep].drop(columns=["game_id_norm"])
    filtered_labels = lab.loc[lab_keep].drop(columns=["game_id_norm"])

    # For kept team-games, missing labels are treated as DNP and filled to 0.
    filtered_labels[label_col] = pd.to_numeric(filtered_labels[label_col], errors="coerce").fillna(0.0).astype("float64")

    def _stats(series: pd.Series) -> dict[str, float | None]:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return {"min": None, "p10": None, "p50": None, "mean": None, "p90": None, "max": None}
        return {
            "min": float(clean.min()),
            "p10": float(clean.quantile(0.1)),
            "p50": float(clean.quantile(0.5)),
            "mean": float(clean.mean()),
            "p90": float(clean.quantile(0.9)),
            "max": float(clean.max()),
        }

    worst_rows = (
        summary.loc[~keep]
        .sort_values(["label_sum", "n_players"], ascending=[True, True])
        .head(20)
        .loc[
            :,
            [
                "game_id_norm",
                "team_id",
                "n_players",
                "n_pos",
                "label_sum",
                "stints_team_total",
                "minutes_from_stints_sum",
                "stints_gap",
                "label_gap",
            ],
        ]
    )
    worst_by_label: list[dict[str, Any]] = []
    for row in worst_rows.to_dict(orient="records"):
        if row.get("team_id") is None or pd.isna(row["team_id"]):
            continue
        worst_by_label.append(
            {
                "game_id_norm": str(row["game_id_norm"]),
                "team_id": int(pd.to_numeric(row["team_id"], errors="coerce")),
                "n_players": int(row["n_players"]),
                "n_pos": int(row["n_pos"]),
                "label_sum": float(row["label_sum"]),
                "stints_team_total": float(row["stints_team_total"]),
                "minutes_from_stints_sum": float(row["minutes_from_stints_sum"]),
                "stints_gap": float(row["stints_gap"]),
                "label_gap": float(row["label_gap"]),
            }
        )

    meta: dict[str, Any] = {
        "thresholds": {
            "min_team_minutes_from_stints": float(min_team_minutes_from_stints),
            "max_team_minutes_gap": float(max_team_minutes_gap),
        },
        "team_games_total": int(len(summary)),
        "team_games_kept": int(keep.sum()),
        "team_games_dropped_by_reason": {
            "rotation_missing": int(reason_rotation_missing.sum()),
            "stints_team_total_too_low": int(reason_stints_low.sum()),
            "missing_player_coverage": int(reason_missing_coverage.sum()),
            "labels_incomplete_vs_stints": int(reason_labels_incomplete.sum()),
        },
        "summaries": {
            "n_players": _stats(summary["n_players"]),
            "stints_team_total": _stats(summary["stints_team_total"]),
            "minutes_from_stints_sum": _stats(summary["minutes_from_stints_sum"]),
            "label_sum": _stats(summary["label_sum"]),
        },
        "summaries_kept": {
            "n_players": _stats(summary.loc[keep, "n_players"]),
            "stints_team_total": _stats(summary.loc[keep, "stints_team_total"]),
            "minutes_from_stints_sum": _stats(summary.loc[keep, "minutes_from_stints_sum"]),
            "label_sum": _stats(summary.loc[keep, "label_sum"]),
        },
        "worst_examples": worst_by_label,
    }
    return filtered_features, filtered_labels, meta

def _resolve_minutes_dataset_dir(data_root: Path, dataset_ref: str | None) -> Path:
    if dataset_ref:
        candidate = Path(dataset_ref).expanduser()
        if candidate.is_absolute() or candidate.exists():
            return candidate.resolve()
        return (data_root / "training" / "datasets" / candidate).resolve()
    return (data_root / "training" / "datasets" / MINUTES_DATASET_DIRNAME).resolve()


def _parse_date_arg(value: str, *, arg_name: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).normalize()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid {arg_name}: {value!r}") from exc


def _resolve_date_window(
    *,
    start_date: str | None,
    end_date: str | None,
    lookback_days: int | None,
    anchor_date: str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start_day = _parse_date_arg(start_date, arg_name="--start-date") if start_date else None
    end_day = _parse_date_arg(end_date, arg_name="--end-date") if end_date else None

    if lookback_days is not None:
        if int(lookback_days) <= 0:
            raise ValueError("--lookback-days must be > 0")
        anchor_day = (
            _parse_date_arg(anchor_date, arg_name="--anchor-date")
            if anchor_date
            else pd.Timestamp.now(tz=timezone.utc).tz_localize(None).normalize()
        )
        lookback_start = anchor_day - pd.Timedelta(days=int(lookback_days))
        start_day = max(start_day, lookback_start) if start_day is not None else lookback_start
        if end_day is None:
            end_day = anchor_day

    if start_day is not None and end_day is not None and end_day < start_day:
        raise ValueError(f"--end-date ({end_day.date()}) is before --start-date ({start_day.date()})")
    return start_day, end_day


def _filter_minutes_dataset_by_game_date(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    start_day: pd.Timestamp | None,
    end_day: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if start_day is None and end_day is None:
        return features_df, labels_df, {"enabled": False}
    if "game_date" not in features_df.columns:
        raise ValueError("Cannot apply date window: minutes features missing required column 'game_date'")

    feat = features_df.copy()
    feat["game_date"] = pd.to_datetime(feat["game_date"], errors="coerce").dt.normalize()
    mask = pd.Series(True, index=feat.index)
    if start_day is not None:
        mask &= feat["game_date"] >= start_day
    if end_day is not None:
        mask &= feat["game_date"] <= end_day
    feat = feat.loc[mask].copy()

    key_cols = ["game_id", "team_id", "player_id"]
    for col in key_cols:
        if col not in feat.columns or col not in labels_df.columns:
            raise ValueError(f"Cannot align labels after date window; missing key column: {col}")
    keys = feat.loc[:, key_cols].drop_duplicates()
    lab = keys.merge(labels_df, on=key_cols, how="left", sort=False)

    meta = {
        "enabled": True,
        "start_date": start_day.date().isoformat() if start_day is not None else None,
        "end_date": end_day.date().isoformat() if end_day is not None else None,
        "rows_features_before": int(len(features_df)),
        "rows_features_after": int(len(feat)),
        "rows_labels_before": int(len(labels_df)),
        "rows_labels_after": int(len(lab)),
        "game_date_min_after": feat["game_date"].min().date().isoformat() if len(feat) else None,
        "game_date_max_after": feat["game_date"].max().date().isoformat() if len(feat) else None,
    }
    return feat, lab, meta


def _load_minutes_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    features_path = dataset_dir / "features.parquet"
    labels_path = dataset_dir / "labels.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing minutes features at {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing minutes labels at {labels_path}")
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)
    return features_df, labels_df, features_path, labels_path


def _discover_rotation_files(root: Path, subdir: str) -> list[Path]:
    base = root / "silver" / "rotation_v1" / subdir
    if not base.exists():
        return []
    return sorted(base.glob("season=*/game_id=*.parquet"))


def _read_parquet_with_columns(path: Path, desired: list[str]) -> pd.DataFrame:
    schema_cols = set(pq.ParquetFile(path).schema.names)
    cols = [c for c in desired if c in schema_cols]
    return pd.read_parquet(path, columns=cols if cols else None)


def _load_rotation_player_labels(data_root: Path) -> pd.DataFrame:
    files = _discover_rotation_files(data_root, "player_game_labels")
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    desired = ["game_id", "team_id", "person_id", *PLAYER_ROTATION_COLS, "time_unit"]
    for path in files:
        frames.append(_read_parquet_with_columns(path, desired))
    df = pd.concat(frames, ignore_index=True)
    if "time_unit_detected" not in df.columns and "time_unit" in df.columns:
        df["time_unit_detected"] = df["time_unit"]
    # Normalize join keys.
    df["game_id_norm"] = _zfill_game_id(df["game_id"])
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").astype("Int64")
    df = df.drop_duplicates(subset=["game_id_norm", "team_id", "person_id"], keep="last")
    return df


def _load_rotation_team_shape(data_root: Path) -> pd.DataFrame:
    files = _discover_rotation_files(data_root, "team_game_shape")
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    desired = ["game_id", "team_id", *TEAM_ROTATION_COLS]
    for path in files:
        frames.append(_read_parquet_with_columns(path, desired))
    df = pd.concat(frames, ignore_index=True)
    df["game_id_norm"] = _zfill_game_id(df["game_id"])
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df = df.drop_duplicates(subset=["game_id_norm", "team_id"], keep="last")
    return df


def _backfill_odds_from_silver_snapshot(df: pd.DataFrame, *, data_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Backfill missing spread_home/total from silver odds_snapshot by game_id.

    Selection rule:
      - use latest odds snapshot per game_id
      - if tip_ts exists in base df, require as_of_ts <= tip_ts
    """

    if df.empty:
        return {"enabled": True, "rows": 0}, df
    if "game_id" not in df.columns:
        raise ValueError("Cannot backfill odds: missing game_id in minutes features")
    if "spread_home" not in df.columns or "total" not in df.columns:
        raise ValueError("Cannot backfill odds: missing spread_home/total columns in minutes features")

    out = df.copy()
    out["game_id_norm"] = _zfill_game_id(out["game_id"])

    spread_before = pd.to_numeric(out["spread_home"], errors="coerce")
    total_before = pd.to_numeric(out["total"], errors="coerce")
    row_missing_before = spread_before.isna() | total_before.isna()
    games_missing = set(out.loc[row_missing_before, "game_id_norm"].dropna().astype(str).unique().tolist())
    if not games_missing:
        out = out.drop(columns=["game_id_norm"], errors="ignore")
        return {
            "enabled": True,
            "rows": int(len(out)),
            "rows_missing_before": int(row_missing_before.sum()),
            "rows_missing_after": int(row_missing_before.sum()),
            "games_missing_before": 0,
            "games_missing_after": 0,
            "games_filled": 0,
            "odds_snapshot_rows_read": 0,
            "odds_snapshot_candidate_rows": 0,
        }, out

    seasons: set[int] = set()
    if "game_date" in out.columns:
        g = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize().dropna()
        seasons = {_season_for_date(pd.Timestamp(v)) for v in g}
    season_dirs = (
        [data_root / "silver" / "odds_snapshot" / f"season={s}" for s in sorted(seasons)]
        if seasons
        else sorted((data_root / "silver" / "odds_snapshot").glob("season=*"))
    )
    odds_files: list[Path] = []
    for season_dir in season_dirs:
        odds_files.extend(sorted(season_dir.glob("month=*/odds_snapshot.parquet")))

    odds_frames: list[pd.DataFrame] = []
    read_rows = 0
    desired = ["game_id", "as_of_ts", "spread_home", "total", "home_line"]
    for path in odds_files:
        schema_cols = set(pq.ParquetFile(path).schema.names)
        cols = [c for c in desired if c in schema_cols]
        if "game_id" not in cols or "as_of_ts" not in cols:
            continue
        odds_part = pd.read_parquet(path, columns=cols)
        read_rows += int(len(odds_part))
        if odds_part.empty:
            continue
        odds_part["game_id_norm"] = _zfill_game_id(odds_part["game_id"])
        odds_part = odds_part.loc[odds_part["game_id_norm"].isin(games_missing)].copy()
        if odds_part.empty:
            continue
        if "spread_home" not in odds_part.columns and "home_line" in odds_part.columns:
            odds_part["spread_home"] = odds_part["home_line"]
        odds_part["spread_home"] = pd.to_numeric(odds_part.get("spread_home"), errors="coerce")
        odds_part["total"] = pd.to_numeric(odds_part.get("total"), errors="coerce")
        odds_part["as_of_ts"] = pd.to_datetime(odds_part["as_of_ts"], utc=True, errors="coerce")
        odds_part = odds_part.dropna(subset=["game_id_norm", "as_of_ts"])
        odds_frames.append(odds_part.loc[:, ["game_id_norm", "as_of_ts", "spread_home", "total"]])

    if not odds_frames:
        out = out.drop(columns=["game_id_norm"], errors="ignore")
        return {
            "enabled": True,
            "rows": int(len(out)),
            "rows_missing_before": int(row_missing_before.sum()),
            "rows_missing_after": int(row_missing_before.sum()),
            "games_missing_before": int(len(games_missing)),
            "games_missing_after": int(len(games_missing)),
            "games_filled": 0,
            "odds_snapshot_rows_read": int(read_rows),
            "odds_snapshot_candidate_rows": 0,
            "warning": "No matching odds_snapshot rows for missing games.",
        }, out

    odds = pd.concat(odds_frames, ignore_index=True)

    tips = out.loc[:, ["game_id_norm"]].drop_duplicates().copy()
    if "tip_ts" in out.columns:
        tip_frame = out.loc[:, ["game_id_norm", "tip_ts"]].dropna(subset=["tip_ts"]).copy()
        tip_frame["tip_ts"] = pd.to_datetime(tip_frame["tip_ts"], utc=True, errors="coerce")
        tip_frame = tip_frame.dropna(subset=["tip_ts"])
        tip_frame = tip_frame.sort_values(["game_id_norm", "tip_ts"]).drop_duplicates(subset=["game_id_norm"], keep="last")
        tips = tips.merge(tip_frame, on="game_id_norm", how="left")
    else:
        tips["tip_ts"] = pd.NaT

    candidates = odds.merge(tips, on="game_id_norm", how="inner")
    has_tip = candidates["tip_ts"].notna()
    eligible = candidates.loc[~has_tip | (candidates["as_of_ts"] <= candidates["tip_ts"])].copy()
    if eligible.empty:
        out = out.drop(columns=["game_id_norm"], errors="ignore")
        return {
            "enabled": True,
            "rows": int(len(out)),
            "rows_missing_before": int(row_missing_before.sum()),
            "rows_missing_after": int(row_missing_before.sum()),
            "games_missing_before": int(len(games_missing)),
            "games_missing_after": int(len(games_missing)),
            "games_filled": 0,
            "odds_snapshot_rows_read": int(read_rows),
            "odds_snapshot_candidate_rows": int(len(candidates)),
            "warning": "Odds rows found but none passed as_of_ts <= tip_ts filter.",
        }, out

    eligible = eligible.sort_values(["game_id_norm", "as_of_ts"]).drop_duplicates(subset=["game_id_norm"], keep="last")
    odds_map = eligible.set_index("game_id_norm")[["spread_home", "total"]]

    fill_spread = out["spread_home"].isna()
    fill_total = out["total"].isna()
    out.loc[fill_spread, "spread_home"] = out.loc[fill_spread, "game_id_norm"].map(odds_map["spread_home"])
    out.loc[fill_total, "total"] = out.loc[fill_total, "game_id_norm"].map(odds_map["total"])

    spread_after = pd.to_numeric(out["spread_home"], errors="coerce")
    total_after = pd.to_numeric(out["total"], errors="coerce")
    row_missing_after = spread_after.isna() | total_after.isna()

    missing_games_after = set(out.loc[row_missing_after, "game_id_norm"].dropna().astype(str).unique().tolist())
    out = out.drop(columns=["game_id_norm"], errors="ignore")

    meta: dict[str, Any] = {
        "enabled": True,
        "rows": int(len(out)),
        "rows_missing_before": int(row_missing_before.sum()),
        "rows_missing_after": int(row_missing_after.sum()),
        "games_missing_before": int(len(games_missing)),
        "games_missing_after": int(len(missing_games_after)),
        "games_filled": int(len(games_missing) - len(missing_games_after)),
        "odds_snapshot_rows_read": int(read_rows),
        "odds_snapshot_candidate_rows": int(len(candidates)),
        "odds_snapshot_eligible_rows": int(len(eligible)),
    }
    return meta, out


def _load_rotation_priors_v1(
    data_root: Path, *, game_id_norm_by_season: dict[int, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load rotation_priors_v1 partitions for the given (season -> game_id_norm) map."""

    root = data_root / "silver" / "rotation_priors_v1"
    team_root = root / "team_game_priors"
    player_root = root / "player_game_priors"
    if not team_root.exists() or not player_root.exists():
        raise FileNotFoundError(f"Missing rotation_priors_v1 roots under {root}")

    def _read_many(dir_root: Path, *, season: int, game_ids: list[str]) -> tuple[pd.DataFrame, list[str]]:
        season_dir = dir_root / f"season={int(season)}"
        frames: list[pd.DataFrame] = []
        missing: list[str] = []
        for gid in game_ids:
            path = season_dir / f"game_id={gid}.parquet"
            if not path.exists():
                missing.append(gid)
                continue
            frames.append(pd.read_parquet(path))
        if not frames:
            return pd.DataFrame(), missing
        return pd.concat(frames, ignore_index=True), missing

    team_frames: list[pd.DataFrame] = []
    player_frames: list[pd.DataFrame] = []
    missing_team: list[str] = []
    missing_player: list[str] = []

    for season, game_ids in sorted(game_id_norm_by_season.items()):
        gids = [str(g) for g in game_ids if str(g).strip()]
        if not gids:
            continue
        team_df, miss_team = _read_many(team_root, season=season, game_ids=gids)
        player_df, miss_player = _read_many(player_root, season=season, game_ids=gids)
        if not team_df.empty:
            team_frames.append(team_df)
        if not player_df.empty:
            player_frames.append(player_df)
        missing_team.extend([f"season={season}:game_id={gid}" for gid in miss_team])
        missing_player.extend([f"season={season}:game_id={gid}" for gid in miss_player])

    team_priors = pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame()
    player_priors = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()

    meta: dict[str, object] = {
        "root": str(root),
        "seasons": sorted(int(s) for s in game_id_norm_by_season.keys()),
        "counts": {
            "team_rows": int(len(team_priors)),
            "player_rows": int(len(player_priors)),
            "missing_team_partitions": int(len(missing_team)),
            "missing_player_partitions": int(len(missing_player)),
        },
        "missing_team_partitions_sample": missing_team[:25],
        "missing_player_partitions_sample": missing_player[:25],
    }

    return team_priors, player_priors, meta


def _apply_feature_pruning(features_df: pd.DataFrame, *, allowlist: list[str]) -> tuple[pd.DataFrame, list[str]]:
    kept = [c for c in allowlist if c not in DROP_FEATURES]
    missing = [c for c in kept if c not in features_df.columns]
    if missing:
        raise ValueError(f"Minutes features missing expected columns: {missing}")
    # Keep all non-feature columns as-is (ids, timestamps, strings), but drop pruned numeric features to avoid selection.
    to_drop = [c for c in DROP_FEATURES if c in features_df.columns]
    pruned = features_df.drop(columns=to_drop)
    return pruned, kept


def _apply_odds_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    for col in ODDS_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required odds column: {col}")
    df = df.copy()
    df["spread_home_missing"] = df["spread_home"].isna()
    df["total_missing"] = df["total"].isna()
    df["spread_home"] = pd.to_numeric(df["spread_home"], errors="coerce").fillna(0.0).astype("float64")
    df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0.0).astype("float64")
    return df


def _discover_action_props_days(props_dir: Path) -> set[str]:
    if not props_dir.exists():
        return set()
    days: set[str] = set()
    for path in props_dir.glob("*.json"):
        prefix = path.name.split("_", 1)[0]
        if len(prefix) != 10:
            continue
        try:
            day = pd.Timestamp(prefix).date().isoformat()
        except Exception:
            continue
        days.add(day)
    return days


def _attach_action_props_training_features(
    df: pd.DataFrame,
    *,
    data_root: Path,
    enabled: bool,
    props_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    out = df.copy()
    existing_prop_cols = [col for col in ACTION_MARKET_FEATURE_COLUMNS if col in out.columns]
    if existing_prop_cols:
        out = out.drop(columns=existing_prop_cols)
    resolved_props_dir = (
        props_dir.expanduser().resolve()
        if props_dir is not None
        else (data_root / "bronze" / "action_network" / "props").resolve()
    )
    meta: dict[str, Any] = {
        "enabled": bool(enabled),
        "source_dir": str(resolved_props_dir),
        "rows": int(len(out)),
    }
    if not enabled:
        out = attach_action_props_features(
            out,
            pd.DataFrame(),
            strict_asof=True,
            as_of_col="feature_as_of_ts",
            tip_col="tip_ts",
            game_date_offsets=(0, -1),
            clamp_late_asof_to_game_date=True,
        )
        meta["reason"] = "disabled_by_flag"
        return meta, out

    available_days = _discover_action_props_days(resolved_props_dir)
    game_days: list[str] = []
    if "game_date" in out.columns:
        game_days = sorted(
            {
                d.date().isoformat()
                for d in pd.to_datetime(out["game_date"], errors="coerce").dropna()
            }
        )
    target_days: set[str] = {d for d in game_days if d in available_days}
    for d in game_days:
        next_day = (pd.Timestamp(d) + pd.Timedelta(days=1)).date().isoformat()
        if next_day in available_days:
            target_days.add(next_day)
    target_days_sorted = sorted(target_days)

    snapshots: list[pd.DataFrame] = []
    failed_days: list[str] = []
    for day in target_days_sorted:
        try:
            snap = load_action_props_feature_snapshots_for_date(
                props_dir=resolved_props_dir,
                game_date=pd.Timestamp(day),
            )
        except Exception:
            failed_days.append(day)
            continue
        if not snap.empty:
            snapshots.append(snap)

    action_snaps = pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()
    out = attach_action_props_features(
        out,
        action_snaps,
        strict_asof=True,
        as_of_col="feature_as_of_ts",
        tip_col="tip_ts",
        game_date_offsets=(0, -1),
        clamp_late_asof_to_game_date=True,
    )
    matched_rows = int(
        pd.to_numeric(out.get("an_has_any_props"), errors="coerce")
        .fillna(0.0)
        .gt(0.0)
        .sum()
    )
    meta.update(
        {
            "game_days": int(len(game_days)),
            "available_days": int(len(available_days)),
            "target_days": int(len(target_days_sorted)),
            "failed_days": failed_days[:25],
            "snapshot_rows": int(len(action_snaps)),
            "matched_rows": matched_rows,
            "coverage_rate": (
                round(float(matched_rows) / float(len(out)), 4) if len(out) > 0 else 0.0
            ),
            "feature_columns_present": [
                col for col in ACTION_MARKET_FEATURE_COLUMNS if col in out.columns
            ],
        }
    )
    return meta, out


def _join_rotation(
    df: pd.DataFrame,
    *,
    player_rotation: pd.DataFrame,
    team_rotation: pd.DataFrame,
) -> pd.DataFrame:
    required_keys = {"game_id", "team_id", "player_id"}
    if not required_keys.issubset(df.columns):
        raise ValueError(f"Minutes dataset missing join keys: {sorted(required_keys - set(df.columns))}")

    out = df.copy()
    out["game_id_norm"] = _zfill_game_id(out["game_id"])
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["person_id"] = out["player_id"]

    if not team_rotation.empty:
        join_cols = ["game_id_norm", "team_id", *TEAM_ROTATION_COLS]
        out = out.merge(
            team_rotation.loc[:, join_cols],
            on=["game_id_norm", "team_id"],
            how="left",
            suffixes=("", "_team"),
        )
    else:
        for col in TEAM_ROTATION_COLS:
            out[col] = pd.NA

    if not player_rotation.empty:
        join_cols = ["game_id_norm", "team_id", "person_id", *PLAYER_ROTATION_COLS]
        out = out.merge(
            player_rotation.loc[:, join_cols],
            on=["game_id_norm", "team_id", "person_id"],
            how="left",
        )
    else:
        for col in PLAYER_ROTATION_COLS:
            out[col] = pd.NA

    # Team-level presence indicates rotation data is available for the game/team.
    if "team_total_minutes_from_stints" in out.columns:
        team_missing = out["team_total_minutes_from_stints"].isna()
    else:
        team_missing = out.get("depth_10", pd.Series([pd.NA] * len(out))).isna()
    out["rotation_team_missing"] = team_missing.astype("int8")
    out["rotation_missing"] = out["rotation_team_missing"]

    # Player-level row missing when team is present but no player stint row exists.
    player_row_missing = out["num_stints"].isna().astype("int8")
    out["rotation_player_row_missing_raw"] = player_row_missing

    fill_mask = (out["rotation_team_missing"] == 0) & (out["rotation_player_row_missing_raw"] == 1)
    out["rotation_player_filled_zero"] = 0
    if fill_mask.any():
        out.loc[fill_mask, "num_stints"] = 0
        out.loc[fill_mask, "minutes_from_stints"] = 0.0
        out.loc[fill_mask, "started_proxy"] = 0
        if "max_stint_len_real" in out.columns:
            out.loc[fill_mask, "max_stint_len_real"] = 0
        if "first_in_time_real" in out.columns:
            out.loc[fill_mask, "first_in_time_real"] = pd.NA
        if "last_out_time_real" in out.columns:
            out.loc[fill_mask, "last_out_time_real"] = pd.NA
        out.loc[fill_mask, "rotation_player_filled_zero"] = 1
    out["rotation_player_filled_zero"] = out["rotation_player_filled_zero"].astype("int8")

    # Remove helper join columns not needed downstream.
    out = out.drop(columns=["game_id_norm", "person_id"])
    return out


def _sample_rows(df: pd.DataFrame, labels: pd.DataFrame, *, max_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if len(df) <= max_rows:
        return df, labels
    sampled = df.sample(n=max_rows, random_state=42)
    key_cols = ["game_id", "team_id", "player_id"]
    sampled_keys = sampled.loc[:, key_cols].copy()
    merged = sampled_keys.merge(labels, on=key_cols, how="left")
    return sampled.reset_index(drop=True), merged.reset_index(drop=True)


def _report(df: pd.DataFrame) -> None:
    def _rate(series: pd.Series) -> float:
        return float(series.mean()) if len(series) else float("nan")

    print("[rotation_train_v1] Report")
    print(f"  Total rows: {len(df):,}")
    if "rotation_team_missing" in df.columns:
        team_missing_rate = _rate(df["rotation_team_missing"])
        print(f"  rotation_team_missing rate: {team_missing_rate:.4f}")
        team_present = (df["rotation_team_missing"] == 0)
        filled_rate = float(df.loc[team_present, "rotation_player_filled_zero"].mean()) if team_present.any() else 0.0
        print(f"  rotation_player_filled_zero rate (team present): {filled_rate:.4f}")
        print(f"  Rows with team present: {int(team_present.sum()):,}")
        print(f"  Rows filled to zero: {int(df['rotation_player_filled_zero'].sum()):,}")
    if "rotation_missing" in df.columns:
        print(f"  rotation_missing rate: {_rate(df['rotation_missing']):.4f}")
    if "spread_home_missing" in df.columns:
        print(f"  spread_home_missing rate: {_rate(df['spread_home_missing']):.4f}")
    if "total_missing" in df.columns:
        print(f"  total_missing rate: {_rate(df['total_missing']):.4f}")

    for col in ["depth_10", "effective_n", "bench_conc_top2"]:
        if col not in df.columns:
            continue
        missing = float(df[col].isna().mean())
        non_missing = pd.to_numeric(df[col], errors="coerce").dropna()
        if non_missing.empty:
            stats = "min/median/max: n/a"
        else:
            stats = f"min/median/max: {non_missing.min():.3f}/{non_missing.median():.3f}/{non_missing.max():.3f}"
        print(f"  {col}: missing_rate={missing:.4f} ({stats})")


def _report_match_diagnostics(
    minutes_df: pd.DataFrame,
    *,
    player_rotation: pd.DataFrame,
    team_rotation: pd.DataFrame,
    sample_size: int = 50000,
) -> None:
    minutes_game_ids = _zfill_game_id(minutes_df["game_id"]).dropna()
    minutes_game_unique = set(minutes_game_ids.unique())

    rotation_game_source = player_rotation if not player_rotation.empty else team_rotation
    rotation_game_ids = rotation_game_source.get("game_id_norm", pd.Series(dtype="string")).dropna()
    rotation_game_unique = set(rotation_game_ids.unique())

    overlap_game = len(minutes_game_unique & rotation_game_unique)

    minutes_pairs = set(
        zip(
            _zfill_game_id(minutes_df["game_id"]),
            pd.to_numeric(minutes_df["team_id"], errors="coerce").astype("Int64"),
        )
    )
    rotation_pairs = set(
        zip(
            team_rotation.get("game_id_norm", pd.Series(dtype="string")),
            team_rotation.get("team_id", pd.Series(dtype="Int64")),
        )
    )
    overlap_pair = len(minutes_pairs & rotation_pairs) if rotation_pairs else 0

    if len(minutes_df) > sample_size:
        sample_df = minutes_df.sample(sample_size, random_state=42)
        sampled = True
    else:
        sample_df = minutes_df
        sampled = False

    minutes_triples = set(
        zip(
            _zfill_game_id(sample_df["game_id"]),
            pd.to_numeric(sample_df["team_id"], errors="coerce").astype("Int64"),
            pd.to_numeric(sample_df["player_id"], errors="coerce").astype("Int64"),
        )
    )
    rotation_triples = set(
        zip(
            player_rotation.get("game_id_norm", pd.Series(dtype="string")),
            player_rotation.get("team_id", pd.Series(dtype="Int64")),
            player_rotation.get("person_id", pd.Series(dtype="Int64")),
        )
    )
    overlap_triple = len(minutes_triples & rotation_triples) if rotation_triples else 0

    print("[rotation_train_v1] Join diagnostics")
    print(f"  Minutes unique game_ids (normalized): {len(minutes_game_unique)}")
    print(f"  Rotation unique game_ids:             {len(rotation_game_unique)}")
    print(f"  Overlap game_ids:                     {overlap_game}")
    print(f"  Overlap (game_id, team_id):           {overlap_pair}")
    suffix = " (sampled)" if sampled else ""
    print(f"  Overlap (game_id, team_id, person_id){suffix}: {overlap_triple}")


def _coerce_bool01(series: pd.Series) -> pd.Series:
    mapping = {True: 1, False: 0}
    coerced = series.map(mapping)
    return pd.to_numeric(coerced, errors="coerce").fillna(0).astype("int8")


def _coerce_rotation_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "started_proxy" in out.columns:
        out["started_proxy"] = _coerce_bool01(out["started_proxy"])
    if "num_stints" in out.columns:
        out["num_stints"] = pd.to_numeric(out["num_stints"], errors="coerce").fillna(0).astype("int16")
    for col in ["rotation_team_missing", "rotation_player_filled_zero", "rotation_missing"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int8")
    for col in ["minutes_from_stints", "max_stint_len_real"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype("float64")
    for col in ["first_in_time_real", "last_out_time_real"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def _write_manifest(
    out_dir: Path,
    *,
    source_features: Path,
    source_labels: Path,
    out_features: Path,
    out_labels: Path,
    rotation_player_root: Path,
    rotation_team_root: Path,
    row_count: int,
    feature_col_count: int,
    label_col_count: int,
    rotation_missing_rate: float,
    require_rotation: bool,
    max_rows: int | None,
    date_window: dict[str, Any] | None = None,
    odds_backfill: dict[str, Any] | None = None,
    action_props: dict[str, Any] | None = None,
    team_game_validation: dict[str, Any] | None = None,
    dnp_history_config: DNPHistoryConfig | None = None,
    rotation_priors_v1: dict[str, object] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "created_at": _utc_now_iso(),
        "source": {
            "minutes_features": str(source_features),
            "minutes_labels": str(source_labels),
        },
        "rotation_v1": {
            "player_game_labels_root": str(rotation_player_root),
            "team_game_shape_root": str(rotation_team_root),
        },
        "outputs": {
            "features": str(out_features),
            "labels": str(out_labels),
        },
        "counts": {
            "rows": int(row_count),
            "feature_columns": int(feature_col_count),
            "label_columns": int(label_col_count),
        },
        "rates": {
            "rotation_missing": float(rotation_missing_rate),
        },
        "options": {
            "require_rotation": bool(require_rotation),
            "max_rows": int(max_rows) if max_rows is not None else None,
        },
    }
    if date_window is not None:
        payload["options"]["date_window"] = date_window
    if odds_backfill is not None:
        payload["odds_backfill"] = odds_backfill
    if action_props is not None:
        payload["action_props"] = action_props
    if team_game_validation is not None:
        payload["team_game_validation"] = team_game_validation
    if dnp_history_config is not None:
        payload["dnp_history_features"] = {
            "config": dnp_history_config.to_dict(),
            "columns": DNP_HISTORY_FEATURE_COLUMNS,
        }
    if rotation_priors_v1 is not None:
        payload["rotation_priors_v1"] = rotation_priors_v1
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--minutes-dataset-dir",
        type=str,
        default=None,
        help=(
            "Minutes dataset source. Can be an absolute path to a dataset directory "
            "(containing features.parquet + labels.parquet) or a dataset name under "
            "$PROJECTIONS_DATA_ROOT/training/datasets/."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(paths.get_data_root() / "training" / "datasets" / DEFAULT_OUT_DIRNAME),
        help="Output directory for the rotation training dataset.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional sampling cap for faster iteration.")
    parser.add_argument("--report-only", action="store_true", help="Print diagnostics and exit (no writes).")
    parser.add_argument(
        "--require-rotation",
        action="store_true",
        help="Filter output rows to those with rotation_missing == 0.",
    )
    parser.add_argument(
        "--min-team-minutes-from-stints",
        type=float,
        default=DEFAULT_MIN_TEAM_MINUTES_FROM_STINTS,
        help="Drop team-games with team_total_minutes_from_stints below this threshold.",
    )
    parser.add_argument(
        "--max-team-minutes-gap",
        type=float,
        default=DEFAULT_MAX_TEAM_MINUTES_GAP,
        help="Max allowed |team_total_minutes_from_stints - (label_sum or minutes_from_stints_sum)| to keep a team-game.",
    )
    parser.add_argument(
        "--no-filter-invalid-team-games",
        action="store_true",
        help="Disable dropping team-games with incomplete player coverage or incomplete labels.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive game_date floor (YYYY-MM-DD) before joins.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional inclusive game_date ceiling (YYYY-MM-DD) before joins.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Optional rolling lookback window in days (e.g. 365).",
    )
    parser.add_argument(
        "--anchor-date",
        type=str,
        default=None,
        help="Anchor date for --lookback-days (YYYY-MM-DD). Defaults to current UTC date.",
    )
    parser.add_argument(
        "--backfill-odds-from-silver",
        dest="backfill_odds_from_silver",
        action="store_true",
        help="Backfill missing spread_home/total from silver/odds_snapshot before applying odds missing flags.",
    )
    parser.add_argument(
        "--no-backfill-odds-from-silver",
        dest="backfill_odds_from_silver",
        action="store_false",
        help="Disable odds backfill from silver/odds_snapshot.",
    )
    parser.set_defaults(backfill_odds_from_silver=True)
    parser.add_argument(
        "--action-props",
        dest="action_props",
        action="store_true",
        help="Attach Action Network player prop features from bronze snapshots.",
    )
    parser.add_argument(
        "--no-action-props",
        dest="action_props",
        action="store_false",
        help="Disable Action Network player prop feature joins.",
    )
    parser.set_defaults(action_props=True)
    parser.add_argument(
        "--action-props-dir",
        type=str,
        default=None,
        help="Optional override for bronze Action props directory.",
    )
    args = parser.parse_args()

    data_root = paths.get_data_root()
    out_dir = Path(args.out_dir).expanduser().resolve()
    bundle_dir = (paths.get_project_root() / MINUTES_BUNDLE_DIR).resolve()
    minutes_dataset_dir = _resolve_minutes_dataset_dir(data_root, args.minutes_dataset_dir)

    allowlist = _read_feature_allowlist(bundle_dir)
    features_df, labels_df, source_features_path, source_labels_path = _load_minutes_dataset(minutes_dataset_dir)

    start_day, end_day = _resolve_date_window(
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        anchor_date=args.anchor_date,
    )
    features_df, labels_df, date_window_meta = _filter_minutes_dataset_by_game_date(
        features_df,
        labels_df,
        start_day=start_day,
        end_day=end_day,
    )
    if date_window_meta.get("enabled"):
        print(
            "[rotation_train_v1] Date window:",
            f"start={date_window_meta.get('start_date')}",
            f"end={date_window_meta.get('end_date')}",
            f"rows_features={date_window_meta.get('rows_features_before')}->{date_window_meta.get('rows_features_after')}",
            f"rows_labels={date_window_meta.get('rows_labels_before')}->{date_window_meta.get('rows_labels_after')}",
        )

    pruned_df, kept_minutes_features = _apply_feature_pruning(features_df, allowlist=allowlist)
    if bool(args.backfill_odds_from_silver):
        odds_backfill_meta, pruned_df = _backfill_odds_from_silver_snapshot(pruned_df, data_root=data_root)
        print(
            "[rotation_train_v1] Odds backfill:",
            f"rows_missing={odds_backfill_meta.get('rows_missing_before')}->{odds_backfill_meta.get('rows_missing_after')}",
            f"games_missing={odds_backfill_meta.get('games_missing_before')}->{odds_backfill_meta.get('games_missing_after')}",
            f"games_filled={odds_backfill_meta.get('games_filled')}",
        )
    else:
        odds_backfill_meta = {"enabled": False}
    pruned_df = _apply_odds_missing_flags(pruned_df)
    action_props_dir = Path(args.action_props_dir) if args.action_props_dir else None
    action_props_meta, pruned_df = _attach_action_props_training_features(
        pruned_df,
        data_root=data_root,
        enabled=bool(args.action_props),
        props_dir=action_props_dir,
    )
    print(
        "[rotation_train_v1] Action props:",
        f"enabled={action_props_meta.get('enabled')}",
        f"snapshot_rows={action_props_meta.get('snapshot_rows', 0)}",
        f"matched_rows={action_props_meta.get('matched_rows', 0)}",
        f"coverage={action_props_meta.get('coverage_rate', 0.0)}",
    )

    player_rotation = _load_rotation_player_labels(data_root)
    team_rotation = _load_rotation_team_shape(data_root)
    if args.report_only:
        _report_match_diagnostics(features_df, player_rotation=player_rotation, team_rotation=team_rotation)
    joined = _join_rotation(pruned_df, player_rotation=player_rotation, team_rotation=team_rotation)

    # Attach rotation_priors_v1 (pre-game priors windows) so the set model can learn
    # who plays vs DNP without relying on same-game rotation features (leakage).
    game_map = joined.loc[:, ["game_id", "game_date"]].drop_duplicates().copy()
    game_map["game_date"] = pd.to_datetime(game_map["game_date"], errors="coerce").dt.normalize()
    game_map = game_map.dropna(subset=["game_date"]).copy()
    if not game_map.empty:
        game_map["season"] = game_map["game_date"].map(_season_for_date).astype(int)
        game_map["game_id_norm"] = _zfill_game_id(game_map["game_id"])
        season_map = (
            game_map.groupby("season", sort=False)["game_id_norm"].apply(lambda s: sorted(set(s.dropna().astype(str))))
        )
        game_id_norm_by_season = {int(season): ids for season, ids in season_map.items()}
        team_priors, player_priors, priors_meta = _load_rotation_priors_v1(
            data_root, game_id_norm_by_season=game_id_norm_by_season
        )
        joined = join_rotation_priors(joined, team_priors=team_priors, player_priors=player_priors)
    else:
        priors_meta = {"warning": "game_date missing; skipping rotation_priors_v1 join"}

    if args.require_rotation:
        joined = joined.loc[joined["rotation_team_missing"] == 0].copy()
    label_col = _infer_label_column(labels_df)
    labels_df = _align_labels_to_features(joined, labels_df)

    # Compute DNP history features using labels for minutes
    # Merge labels temporarily to get minutes for DNP computation
    print("[rotation_train_v1] Computing DNP history features...")
    labels_for_dnp = labels_df[["game_id", "team_id", "player_id", label_col]].copy()
    labels_for_dnp = labels_for_dnp.rename(columns={label_col: "_label_minutes"})
    joined_with_labels = joined.merge(
        labels_for_dnp,
        on=["game_id", "team_id", "player_id"],
        how="left",
    )
    joined_with_labels["_label_minutes"] = (
        pd.to_numeric(joined_with_labels["_label_minutes"], errors="coerce").fillna(0.0)
    )

    # Compute DNP history features
    dnp_config = DNPHistoryConfig()
    joined_with_labels = compute_dnp_history_features(
        joined_with_labels,
        config=dnp_config,
        game_date_col="game_date",
        player_id_col="player_id",
        team_id_col="team_id",
        is_out_col="is_out",
        minutes_col="_label_minutes",
        validate_pit=True,
    )

    # Report DNP feature stats
    for col in DNP_HISTORY_FEATURE_COLUMNS:
        if col in joined_with_labels.columns:
            vals = pd.to_numeric(joined_with_labels[col], errors="coerce")
            print(f"  {col}: missing={vals.isna().mean():.4f}, p50={vals.median():.2f}, mean={vals.mean():.2f}")

    # Drop temporary label column but keep DNP features
    joined = joined_with_labels.drop(columns=["_label_minutes"])
    labels_df = _align_labels_to_features(joined, labels_df)

    team_game_validation: dict[str, Any] | None = None
    if not args.no_filter_invalid_team_games:
        joined, labels_df, team_game_validation = _filter_invalid_team_games(
            joined,
            labels_df,
            label_col=label_col,
            min_team_minutes_from_stints=float(args.min_team_minutes_from_stints),
            max_team_minutes_gap=float(args.max_team_minutes_gap),
        )
        if team_game_validation is not None:
            dropped_by_reason = team_game_validation.get("team_games_dropped_by_reason", {})
            print(
                "[rotation_train_v1] Team-game validation:",
                f"total={team_game_validation['team_games_total']:,}",
                f"kept={team_game_validation['team_games_kept']:,}",
                f"dropped={team_game_validation['team_games_total']-team_game_validation['team_games_kept']:,}",
            )
            print(f"[rotation_train_v1] Dropped by reason: {dropped_by_reason}")
            worst = team_game_validation.get("worst_examples", [])[:5]
            if worst:
                print("[rotation_train_v1] Worst examples (first 5):")
                for row in worst:
                    print(
                        " ",
                        row.get("game_id_norm"),
                        row.get("team_id"),
                        f"n_players={row.get('n_players')}",
                        f"label_sum={row.get('label_sum')}",
                        f"stints_total={row.get('stints_team_total')}",
                        f"stints_gap={row.get('stints_gap')}",
                        f"label_gap={row.get('label_gap')}",
                    )

    if args.max_rows is not None:
        joined, labels_df = _sample_rows(joined, labels_df, max_rows=args.max_rows)

    _report(joined)
    if args.report_only:
        return

    joined = _coerce_rotation_dtypes(joined)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_features_path = out_dir / "features.parquet"
    out_labels_path = out_dir / "labels.parquet"

    joined.to_parquet(out_features_path, index=False)
    labels_df.to_parquet(out_labels_path, index=False)

    rotation_player_root = data_root / "silver" / "rotation_v1" / "player_game_labels"
    rotation_team_root = data_root / "silver" / "rotation_v1" / "team_game_shape"

    rotation_missing_rate = float(joined["rotation_missing"].mean()) if len(joined) else float("nan")
    _write_manifest(
        out_dir,
        source_features=source_features_path,
        source_labels=source_labels_path,
        out_features=out_features_path,
        out_labels=out_labels_path,
        rotation_player_root=rotation_player_root,
        rotation_team_root=rotation_team_root,
        row_count=len(joined),
        feature_col_count=len(joined.columns),
        label_col_count=len(labels_df.columns),
        rotation_missing_rate=rotation_missing_rate,
        require_rotation=bool(args.require_rotation),
        max_rows=args.max_rows,
        date_window=date_window_meta,
        odds_backfill=odds_backfill_meta,
        action_props=action_props_meta,
        team_game_validation=team_game_validation,
        dnp_history_config=dnp_config,
        rotation_priors_v1=priors_meta,
    )

    print(f"[rotation_train_v1] Wrote features -> {out_features_path}")
    print(f"[rotation_train_v1] Wrote labels   -> {out_labels_path}")
    print(f"[rotation_train_v1] Wrote manifest -> {out_dir / 'manifest.json'}")
    print(f"[rotation_train_v1] Kept minutes audited features: {len(kept_minutes_features)}")


if __name__ == "__main__":
    main()
