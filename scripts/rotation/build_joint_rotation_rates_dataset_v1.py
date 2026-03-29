#!/usr/bin/env python3
"""Build a unified rotation+minutes+rates training dataset (v1).

This script joins:
- rotation_train_v1 features/labels (team-set minutes dataset), and
- rates_training_base labels (per-minute stat rates + efficiency labels)

Outputs:
  <out_dir>/
    - features.parquet
    - labels_minutes.parquet
    - labels_rates.parquet
    - team_game_index.parquet
    - manifest.json

The output preserves row alignment across the three parquet files by using
the filtered rotation features frame as the row spine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projections import paths

KEY_COLS = ["game_id", "team_id", "player_id"]
JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]

# Raw count-stat columns produced by build_boxscore_count_labels.py.
COUNT_STAT_COLS = [
    "fga2",
    "fg2m",
    "fga3",
    "fg3m",
    "fta",
    "ftm",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "minutes",   # minutes from boxscore payload (may differ slightly from rotation labels)
    "starter_flag",
    "played",
]

# Feature columns to drop from the rotation feature spine.
# See spec Section 4.7 for rationale.
FEATURE_COLS_DROP = [
    "is_confirmed_starter",         # redundant with is_projected_starter
    "first_in_time_real",           # current-game PBP label, not a pre-game feature
    "last_out_time_real",           # current-game PBP label
    "time_unit_detected",           # current-game PBP label
]

RATE_TARGET_COLS = [
    "fga2_per_min",
    "fga3_per_min",
    "fta_per_min",
    "ast_per_min",
    "tov_per_min",
    "oreb_per_min",
    "dreb_per_min",
    "stl_per_min",
    "blk_per_min",
]
EFFICIENCY_LABEL_COLS = [
    "fg2_pct_label",
    "fg3_pct_label",
    "ft_pct_label",
]
RATES_LABEL_COLS = ["minutes_actual", *RATE_TARGET_COLS, *EFFICIENCY_LABEL_COLS]

TRACKING_CONTEXT_COLS = [
    "track_touches_per_min_szn",
    "track_sec_per_touch_szn",
    "track_pot_ast_per_min_szn",
    "track_drives_per_min_szn",
    "track_drive_fta_per_min_szn",
    "track_drive_pf_per_min_szn",
    "track_paint_touches_per_min_szn",
    "track_fta_per_drive_szn",
    "track_catch_shoot_fg3a_per_min_szn",
    "track_pull_up_fg3a_per_min_szn",
    "track_pull_up_3pa_share_szn",
    "track_role_cluster",
    "track_role_is_low_minutes",
]

DEFAULT_ROTATION_PREFIX = "rotation_train_v1"
DEFAULT_OUT_PREFIX = "joint_rotation_rates_v1"
DEFAULT_MINUTES_FOR_RATES_LOSS = 4.0
# Game-total points per possession (~2.2 to 2.3 in modern NBA scoring).
# estimated_possessions ~= vegas_total / league_ppp
DEFAULT_LEAGUE_PPP = 2.27
# Vegas-first by default. Pace component can be reintroduced after calibration.
DEFAULT_EST_POSSESSIONS_PACE_WEIGHT = 0.0
DEFAULT_EST_POSSESSIONS_CLIP_MIN = 85.0
DEFAULT_EST_POSSESSIONS_CLIP_MAX = 130.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.get_project_root())  # noqa: S603,S607
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _season_for_date(day: pd.Timestamp) -> int:
    return int(day.year) if int(day.month) >= 8 else int(day.year) - 1


def _coerce_season_key(
    series: pd.Series | None,
    *,
    game_date: pd.Series | None = None,
) -> pd.Series:
    if series is None:
        if game_date is None:
            base = pd.Series(dtype="Float64")
        else:
            base = pd.Series(pd.NA, index=game_date.index, dtype="Int64")
    else:
        raw = series.astype("string").str.strip()
        numeric = pd.to_numeric(raw, errors="coerce")
        prefix = pd.to_numeric(raw.str.extract(r"^(\d{4})", expand=False), errors="coerce")
        base = numeric.where(numeric.notna(), prefix).astype("Float64")

    if game_date is not None:
        dates = pd.to_datetime(game_date, errors="coerce").dt.normalize()
        derived = dates.map(lambda d: _season_for_date(pd.Timestamp(d)) if pd.notna(d) else pd.NA)
        base = base.where(base.notna(), pd.to_numeric(derived, errors="coerce"))

    return pd.to_numeric(base, errors="coerce").astype("Int64")


def _zfill_game_id(series: pd.Series) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce").astype("Int64")
    return coerced.astype("string").str.zfill(10)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _resolve_date_window(
    *,
    start_date: str | None,
    end_date: str | None,
    lookback_days: int | None,
    anchor_date: str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if lookback_days is not None:
        if lookback_days <= 0:
            raise ValueError("--lookback-days must be > 0")
        if start_date is not None or end_date is not None:
            raise ValueError("Use either --lookback-days or --start-date/--end-date, not both.")
        anchor_day = pd.Timestamp(anchor_date).normalize() if anchor_date else pd.Timestamp.utcnow().normalize()
        end_day = anchor_day
        start_day = (end_day - pd.Timedelta(days=int(lookback_days) - 1)).normalize()
    else:
        start_day = pd.Timestamp(start_date).normalize() if start_date else None
        end_day = pd.Timestamp(end_date).normalize() if end_date else None

    if start_day is not None and end_day is not None and start_day > end_day:
        raise ValueError("start_date must be <= end_date")

    return start_day, end_day


def _coerce_join_keys(df: pd.DataFrame, *, name: str, require_game_date: bool) -> pd.DataFrame:
    out = df.copy()
    for col in KEY_COLS:
        if col not in out.columns:
            raise ValueError(f"{name} missing required key column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if require_game_date:
        if "game_date" not in out.columns:
            raise ValueError(f"{name} missing required key column: game_date")
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    elif "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()

    missing_key_rows = out[KEY_COLS].isna().any(axis=1)
    if missing_key_rows.any():
        raise ValueError(f"{name} has {int(missing_key_rows.sum())} rows with invalid key ids")
    if require_game_date and out["game_date"].isna().any():
        raise ValueError(f"{name} has {int(out['game_date'].isna().sum())} rows with invalid game_date")
    return out


def _infer_minutes_label_column(labels_df: pd.DataFrame) -> str:
    candidates = [c for c in labels_df.columns if c.lower() in {"minutes", "min", "target_minutes"}]
    for col in candidates:
        if col in labels_df.columns:
            return col
    numeric = [
        c
        for c in labels_df.columns
        if c not in {"game_id", "team_id", "player_id", "game_date"}
        and pd.api.types.is_numeric_dtype(labels_df[c])
        and "minute" in c.lower()
    ]
    if len(numeric) == 1:
        return numeric[0]
    raise ValueError(f"Could not infer minutes label column; candidates={candidates}, numeric_minute_like={numeric}")


def _resolve_rotation_dataset_dir(data_root: Path, value: str | None) -> Path:
    datasets_root = data_root / "training" / "datasets"
    if value:
        direct = Path(value).expanduser()
        if direct.exists():
            return direct.resolve()
        named = datasets_root / value
        if named.exists():
            return named.resolve()
        raise FileNotFoundError(f"Rotation dataset not found: {value}")

    candidates = sorted(p for p in datasets_root.glob(f"{DEFAULT_ROTATION_PREFIX}*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No {DEFAULT_ROTATION_PREFIX}* datasets found under {datasets_root}")
    return candidates[-1].resolve()


def _load_rotation_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    features_path = dataset_dir / "features.parquet"
    labels_path = dataset_dir / "labels.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features parquet: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels parquet: {labels_path}")
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)
    return features_df, labels_df, features_path, labels_path


def _filter_by_game_date(
    df: pd.DataFrame,
    *,
    start_day: pd.Timestamp | None,
    end_day: pd.Timestamp | None,
) -> pd.DataFrame:
    if "game_date" not in df.columns:
        return df
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    if start_day is not None:
        out = out.loc[out["game_date"] >= start_day]
    if end_day is not None:
        out = out.loc[out["game_date"] <= end_day]
    return out


def _align_minutes_labels_to_features(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    minutes_label_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    labels = labels_df.copy()
    if "game_date" in labels.columns:
        labels["game_date"] = pd.to_datetime(labels["game_date"], errors="coerce").dt.normalize()

    dedupe_keys = KEY_COLS + (["game_date"] if "game_date" in labels.columns else [])
    pre_dupes = int(labels.duplicated(subset=dedupe_keys, keep=False).sum())
    labels = labels.drop_duplicates(subset=dedupe_keys, keep="last")

    spine = features_df.loc[:, JOIN_KEYS].copy()
    spine["_row_idx"] = np.arange(len(spine), dtype=np.int64)
    aligned = spine.merge(labels, on=KEY_COLS, how="left", sort=False, suffixes=("", "_label"))
    aligned = aligned.sort_values("_row_idx").drop(columns=["_row_idx"])
    if "game_date_label" in aligned.columns:
        aligned = aligned.drop(columns=["game_date_label"])

    # Keep the canonical game_date from the features spine.
    aligned["game_date"] = pd.to_datetime(aligned["game_date"], errors="coerce").dt.normalize()
    if minutes_label_col in aligned.columns:
        aligned["minutes_label"] = pd.to_numeric(aligned[minutes_label_col], errors="coerce")
    else:
        aligned["minutes_label"] = np.nan

    missing_minutes = int(aligned["minutes_label"].isna().sum())
    stats = {
        "duplicate_rows_in_source_labels": pre_dupes,
        "missing_minutes_labels_after_alignment": missing_minutes,
    }
    return aligned, stats


def _resolve_rates_partition_paths(
    rates_root: Path,
    *,
    game_dates: list[pd.Timestamp],
) -> tuple[list[Path], list[str]]:
    paths_out: list[Path] = []
    missing_dates: list[str] = []
    for day in game_dates:
        day_token = day.date().isoformat()
        season = _season_for_date(day)
        direct = rates_root / f"season={season}" / f"game_date={day_token}" / "rates_training_base.parquet"
        if direct.exists():
            paths_out.append(direct)
            continue
        matches = sorted(rates_root.glob(f"season=*/game_date={day_token}/rates_training_base.parquet"))
        if matches:
            paths_out.extend(matches)
        else:
            missing_dates.append(day_token)

    unique_paths = sorted(set(paths_out))
    return unique_paths, missing_dates


def _load_rates_labels(
    partition_paths: list[Path],
    *,
    context_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    requested_context = [str(c) for c in (context_cols or [])]
    if not partition_paths:
        return (
            pd.DataFrame(columns=JOIN_KEYS + RATES_LABEL_COLS + requested_context),
            {
                "partition_count": 0,
                "rows_loaded": 0,
                "rows_after_dedupe": 0,
                "requested_context_cols": requested_context,
                "context_cols_present": [],
                "context_cols_missing": requested_context,
            },
        )
    frames: list[pd.DataFrame] = []
    for path in partition_paths:
        frame = pd.read_parquet(path)
        for col in KEY_COLS:
            if col not in frame.columns:
                raise ValueError(f"Rates partition {path} missing key column: {col}")
        if "game_date" not in frame.columns:
            # Partition encodes game_date in directory, but we require explicit join column.
            parts = str(path).split("game_date=")
            if len(parts) >= 2:
                token = parts[1].split("/", 1)[0]
                frame["game_date"] = token
            else:
                raise ValueError(f"Rates partition {path} missing game_date and not parseable from path")

        keep = [
            c
            for c in JOIN_KEYS + RATES_LABEL_COLS + ["feature_as_of_ts", "tip_ts"] + requested_context
            if c in frame.columns
        ]
        frames.append(frame.loc[:, keep].copy())

    rates = pd.concat(frames, ignore_index=True)
    rates = _coerce_join_keys(rates, name="rates_training_base", require_game_date=True)
    sort_cols = [c for c in ["feature_as_of_ts", "tip_ts"] if c in rates.columns]
    if sort_cols:
        for col in sort_cols:
            rates[col] = pd.to_datetime(rates[col], errors="coerce")
        rates = rates.sort_values(sort_cols)
    pre_dedupe_rows = int(len(rates))
    rates = rates.drop_duplicates(subset=JOIN_KEYS, keep="last")
    meta = {
        "partition_count": int(len(partition_paths)),
        "rows_loaded": pre_dedupe_rows,
        "rows_after_dedupe": int(len(rates)),
        "requested_context_cols": requested_context,
        "context_cols_present": [c for c in requested_context if c in rates.columns],
        "context_cols_missing": [c for c in requested_context if c not in rates.columns],
    }
    return rates, meta


def _append_rates_context_columns(
    features_df: pd.DataFrame,
    rates_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if rates_df.empty:
        return features_df.copy(), []

    excluded = set(JOIN_KEYS + RATES_LABEL_COLS + ["feature_as_of_ts", "tip_ts"])
    rates_ctx_cols = [c for c in rates_df.columns if c not in excluded and c not in features_df.columns]
    if not rates_ctx_cols:
        return features_df.copy(), []

    spine = features_df.loc[:, JOIN_KEYS].copy()
    spine["_row_idx"] = np.arange(len(spine), dtype=np.int64)
    join_frame = rates_df.loc[:, JOIN_KEYS + rates_ctx_cols].copy()
    merged = spine.merge(join_frame, on=JOIN_KEYS, how="left", sort=False)
    merged = merged.sort_values("_row_idx").drop(columns=["_row_idx"])

    out = features_df.copy()
    for col in rates_ctx_cols:
        out[col] = merged[col].to_numpy()
    return out, rates_ctx_cols


def _load_tracking_roles_window(
    tracking_root: Path,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_cols = ["season", "game_date", "player_id", *TRACKING_CONTEXT_COLS]
    if not tracking_root.exists():
        return (
            pd.DataFrame(columns=base_cols),
            {
                "enabled": True,
                "tracking_root": str(tracking_root),
                "rows_loaded": 0,
                "partitions_loaded": 0,
                "warning": "tracking_roles root missing",
            },
        )

    frames: list[pd.DataFrame] = []
    partitions_loaded = 0
    for season_dir in sorted(tracking_root.glob("season=*")):
        for day_dir in sorted(season_dir.glob("game_date=*")):
            token = day_dir.name.split("=", 1)[-1]
            try:
                day = pd.Timestamp(token).normalize()
            except Exception:
                continue
            if day < start_day or day > end_day:
                continue
            path = day_dir / "tracking_roles.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            keep = [c for c in base_cols if c in frame.columns]
            frames.append(frame.loc[:, keep].copy())
            partitions_loaded += 1

    if not frames:
        return (
            pd.DataFrame(columns=base_cols),
            {
                "enabled": True,
                "tracking_root": str(tracking_root),
                "rows_loaded": 0,
                "partitions_loaded": int(partitions_loaded),
                "window_start": str(start_day.date()),
                "window_end": str(end_day.date()),
                "warning": "no tracking_roles partitions in date window",
            },
        )

    tracking = pd.concat(frames, ignore_index=True)
    tracking["game_date"] = pd.to_datetime(tracking["game_date"], errors="coerce").dt.normalize()
    if "season" in tracking.columns:
        tracking["season"] = _coerce_season_key(tracking["season"], game_date=tracking["game_date"])
    else:
        tracking["season"] = _coerce_season_key(None, game_date=tracking["game_date"])
    tracking["player_id"] = pd.to_numeric(tracking["player_id"], errors="coerce").astype("Int64")
    for col in TRACKING_CONTEXT_COLS:
        if col not in tracking.columns:
            tracking[col] = np.nan
        tracking[col] = pd.to_numeric(tracking[col], errors="coerce")

    tracking = tracking.dropna(subset=["season", "player_id", "game_date"]).copy()
    tracking = tracking.sort_values(["season", "player_id", "game_date"])
    tracking = tracking.drop_duplicates(subset=["season", "player_id", "game_date"], keep="last")

    meta = {
        "enabled": True,
        "tracking_root": str(tracking_root),
        "window_start": str(start_day.date()),
        "window_end": str(end_day.date()),
        "partitions_loaded": int(partitions_loaded),
        "rows_loaded": int(len(tracking)),
        "unique_player_dates": int(tracking[["season", "player_id", "game_date"]].drop_duplicates().shape[0]),
        "context_cols_present": [c for c in TRACKING_CONTEXT_COLS if c in tracking.columns],
    }
    return tracking, meta


def _apply_tracking_context_asof_fallback(
    features_df: pd.DataFrame,
    tracking_roles_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    available_cols = [
        c
        for c in TRACKING_CONTEXT_COLS
        if c in features_df.columns or c in tracking_roles_df.columns
    ]
    out = features_df.copy()
    if not available_cols:
        return out, {"enabled": False, "reason": "no tracking context columns present"}

    for col in available_cols:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    before_cov = {col: float(out[col].notna().mean()) for col in available_cols}
    before_any = float(out[available_cols].notna().any(axis=1).mean())

    if tracking_roles_df.empty:
        for col in available_cols:
            out[f"{col}_missing"] = out[col].isna().astype("int8")
        return out, {
            "enabled": True,
            "rows_total": int(len(out)),
            "before_coverage_any_tracking": before_any,
            "after_coverage_any_tracking": before_any,
            "coverage_by_column_before": before_cov,
            "coverage_by_column_after": before_cov,
            "rows_filled_any_tracking": 0,
            "warning": "tracking_roles source empty; missing flags emitted only",
        }

    if "game_date" not in out.columns or "player_id" not in out.columns:
        raise ValueError("features_df missing required columns for tracking as-of fallback: game_date/player_id")

    spine = out.loc[:, ["game_date", "player_id"]].copy()
    spine["game_date"] = pd.to_datetime(spine["game_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    spine["player_id"] = pd.to_numeric(spine["player_id"], errors="coerce").astype("Int64")
    if "season" in out.columns:
        spine["season"] = _coerce_season_key(out["season"], game_date=spine["game_date"])
    else:
        spine["season"] = _coerce_season_key(None, game_date=spine["game_date"])
    spine["_row_idx"] = np.arange(len(spine), dtype=np.int64)
    spine_valid = spine.dropna(subset=["season", "player_id", "game_date"]).copy()

    hist = tracking_roles_df.copy()
    hist["season"] = _coerce_season_key(hist.get("season"), game_date=hist.get("game_date"))
    hist["player_id"] = pd.to_numeric(hist["player_id"], errors="coerce").astype("Int64")
    hist["game_date"] = pd.to_datetime(hist["game_date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    hist = hist.dropna(subset=["season", "player_id", "game_date"]).copy()
    hist = hist.sort_values(["season", "player_id", "game_date"])
    hist = hist.drop_duplicates(subset=["season", "player_id", "game_date"], keep="last")
    for col in available_cols:
        if col not in hist.columns:
            hist[col] = np.nan
        hist[col] = pd.to_numeric(hist[col], errors="coerce")

    filled_parts: list[pd.DataFrame] = []
    hist_groups = {(int(k[0]), int(k[1])): g for k, g in hist.groupby(["season", "player_id"], sort=False)}
    for key, group in spine_valid.groupby(["season", "player_id"], sort=False):
        season_val = int(key[0])
        player_val = int(key[1])
        left = group.loc[:, ["game_date", "_row_idx"]].sort_values("game_date")
        left["game_date"] = left["game_date"].astype("datetime64[ns]")
        right = hist_groups.get((season_val, player_val))
        if right is None or right.empty:
            tmp = left.copy()
            for col in available_cols:
                tmp[col] = np.nan
        else:
            right_min = right.loc[:, ["game_date", *available_cols]].sort_values("game_date")
            right_min["game_date"] = pd.to_datetime(right_min["game_date"], errors="coerce").astype("datetime64[ns]")
            tmp = pd.merge_asof(left, right_min, on="game_date", direction="backward")
        filled_parts.append(tmp.loc[:, ["_row_idx", *available_cols]])

    if filled_parts:
        filled = pd.concat(filled_parts, ignore_index=True).set_index("_row_idx")
        filled = filled.reindex(pd.Index(np.arange(len(out), dtype=np.int64)))
    else:
        filled = pd.DataFrame(index=pd.Index(np.arange(len(out), dtype=np.int64)), columns=available_cols)

    for col in available_cols:
        out[col] = out[col].where(out[col].notna(), pd.to_numeric(filled[col], errors="coerce").to_numpy())
        out[f"{col}_missing"] = out[col].isna().astype("int8")

    after_cov = {col: float(out[col].notna().mean()) for col in available_cols}
    after_any = float(out[available_cols].notna().any(axis=1).mean())
    rows_filled_any = int(
        ((~features_df.loc[:, [c for c in available_cols if c in features_df.columns]].notna().any(axis=1)) & out[available_cols].notna().any(axis=1))
        .sum()
    ) if any(c in features_df.columns for c in available_cols) else int(out[available_cols].notna().any(axis=1).sum())
    meta = {
        "enabled": True,
        "rows_total": int(len(out)),
        "available_columns": available_cols,
        "source_rows": int(len(hist)),
        "source_player_dates": int(hist[["season", "player_id", "game_date"]].drop_duplicates().shape[0]),
        "before_coverage_any_tracking": before_any,
        "after_coverage_any_tracking": after_any,
        "coverage_by_column_before": before_cov,
        "coverage_by_column_after": after_cov,
        "rows_filled_any_tracking": int(max(rows_filled_any, 0)),
    }
    return out, meta


def _align_rates_labels_to_features(
    features_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    *,
    min_minutes_for_rates_loss: float,
) -> pd.DataFrame:
    spine = features_df.loc[:, JOIN_KEYS].copy()
    spine["_row_idx"] = np.arange(len(spine), dtype=np.int64)

    label_cols_present = [c for c in RATES_LABEL_COLS if c in rates_df.columns]
    rates_payload = rates_df.loc[:, JOIN_KEYS + label_cols_present].copy() if not rates_df.empty else pd.DataFrame()
    aligned = spine.merge(rates_payload, on=JOIN_KEYS, how="left", sort=False)
    aligned = aligned.sort_values("_row_idx").drop(columns=["_row_idx"])

    for col in label_cols_present:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

    rate_cols_present = [c for c in RATE_TARGET_COLS if c in aligned.columns]
    eff_cols_present = [c for c in EFFICIENCY_LABEL_COLS if c in aligned.columns]
    if rate_cols_present:
        aligned["rates_non_null_count"] = aligned[rate_cols_present].notna().sum(axis=1).astype("int16")
    else:
        aligned["rates_non_null_count"] = 0
    if eff_cols_present:
        aligned["efficiency_non_null_count"] = aligned[eff_cols_present].notna().sum(axis=1).astype("int16")
    else:
        aligned["efficiency_non_null_count"] = 0
    aligned["rates_label_available_any"] = (aligned["rates_non_null_count"] > 0).astype("int8")
    aligned["rates_label_available_all_rate_targets"] = (
        aligned["rates_non_null_count"] == int(len(rate_cols_present))
    ).astype("int8")

    minutes_actual = pd.to_numeric(aligned.get("minutes_actual", np.nan), errors="coerce")
    aligned["rates_loss_eligible"] = (
        (minutes_actual >= float(min_minutes_for_rates_loss)) & (aligned["rates_label_available_any"] > 0)
    ).astype("int8")
    return aligned


def _build_team_game_index(
    features_df: pd.DataFrame,
    labels_minutes_df: pd.DataFrame,
    labels_rates_df: pd.DataFrame,
) -> pd.DataFrame:
    base = features_df.loc[:, ["game_id", "team_id", "game_date"]].copy()
    base["game_id_norm"] = _zfill_game_id(base["game_id"])
    base["has_minutes_label"] = pd.to_numeric(labels_minutes_df.get("minutes_label", np.nan), errors="coerce").notna()
    base["has_rates_any"] = pd.to_numeric(labels_rates_df.get("rates_label_available_any", 0), errors="coerce").fillna(0) > 0
    base["has_rates_loss_eligible"] = (
        pd.to_numeric(labels_rates_df.get("rates_loss_eligible", 0), errors="coerce").fillna(0) > 0
    )

    grouped = (
        base.groupby(["game_id_norm", "game_id", "team_id", "game_date"], sort=False)
        .agg(
            n_players=("game_id", "size"),
            n_minutes_labeled=("has_minutes_label", "sum"),
            n_rates_any=("has_rates_any", "sum"),
            n_rates_loss_eligible=("has_rates_loss_eligible", "sum"),
        )
        .reset_index()
    )
    grouped["minutes_label_coverage"] = grouped["n_minutes_labeled"] / grouped["n_players"].clip(lower=1)
    grouped["rates_any_coverage"] = grouped["n_rates_any"] / grouped["n_players"].clip(lower=1)
    grouped["rates_loss_eligible_coverage"] = grouped["n_rates_loss_eligible"] / grouped["n_players"].clip(lower=1)
    return grouped.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)


def _assert_unique_keys(df: pd.DataFrame, *, name: str, keys: list[str]) -> None:
    dupes = df.duplicated(subset=keys, keep=False)
    if dupes.any():
        sample = df.loc[dupes, keys].head(5).to_dict(orient="records")
        raise ValueError(f"{name} has duplicated rows for keys={keys}; sample={sample}")


def _apply_lineup_feature_contract(features_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the lineup/starter feature contract from spec Section 4.7.

    Changes:
    - Derives `lineup_available` per team-game (1 if we have usable pre-tip lineup/starter signal).
    - Renames `is_projected_starter` -> `lineup_starter_announced`.
    - Drops `is_confirmed_starter` (redundant) and current-game PBP labels.
    """
    df = features_df.copy()

    # lineup_available: True for every player on a team-game where we have
    # usable pre-tip lineup/starter context at the as_of_ts cutoff.
    #
    # This intentionally includes the fallback projected-starter path below so
    # train/eval do not treat "projected starter known pre-tip" as equivalent
    # to "no lineup information at all". For DFS we want the model to act on
    # usable pre-tip starter information even before a full official lineup is
    # scraped.
    if "lineup_timestamp" in df.columns:
        has_lineup = df.groupby(["game_id", "team_id"], sort=False)["lineup_timestamp"].transform(
            lambda x: x.notna().any()
        )
        lineup_available_strict = has_lineup.astype(bool)
    else:
        lineup_available_strict = pd.Series(False, index=df.index, dtype=bool)

    # Derive lineup_starter_announced from lineup metadata.
    # Semantics:
    # - treat projected starters as announced starters (same as confirmed)
    # - suppress standalone projected-starter noise when lineup data is absent
    role_norm = (
        df.get("lineup_role", pd.Series("", index=df.index))
        .astype("string", copy=False)
        .fillna("")
        .str.strip()
        .str.lower()
    )
    # Starter announcements must come from explicit starter role or starter flags.
    # `lineup_status` (e.g. "confirmed"/"expected") is feed-level freshness metadata
    # and can be populated for bench players; using it as starter signal marks whole
    # team-games as starters and breaks train/live parity.
    starter_from_lineup = role_norm.isin({"projected_starter", "confirmed_starter"})
    starter_from_flag = (
        pd.to_numeric(df.get("is_projected_starter", 0), errors="coerce")
        .fillna(0)
        .astype(float)
        .gt(0.0)
    )
    starter_hint = starter_from_lineup | starter_from_flag
    if {"feature_as_of_ts", "tip_ts"}.issubset(df.columns):
        feature_as_of = pd.to_datetime(df["feature_as_of_ts"], utc=True, errors="coerce")
        tip_ts = pd.to_datetime(df["tip_ts"], utc=True, errors="coerce")
        safe_pre_tip = feature_as_of.notna() & tip_ts.notna() & (feature_as_of <= tip_ts)
    else:
        safe_pre_tip = pd.Series(False, index=df.index, dtype=bool)
    fallback_lineup = (starter_hint & safe_pre_tip).groupby([df["game_id"], df["team_id"]], sort=False).transform(
        "any"
    )
    lineup_present_for_starter = lineup_available_strict | fallback_lineup
    df["lineup_available"] = lineup_present_for_starter.astype("int8")
    df["lineup_starter_announced"] = (
        (lineup_present_for_starter & (starter_from_lineup | starter_from_flag)).astype("int8")
    )
    if "is_projected_starter" in df.columns:
        df = df.drop(columns=["is_projected_starter"])

    # Drop redundant / PBP-only columns (spec Section 4.7 / 4.6).
    drop = [c for c in FEATURE_COLS_DROP if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    return df


def _apply_game_context_feature_contract(
    features_df: pd.DataFrame,
    *,
    league_ppp: float,
    pace_weight: float,
    clip_min: float,
    clip_max: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add canonical game context features required by spec Section 4.8.

    Produces:
    - vegas_total (from total)
    - vegas_spread (from spread_home)
    - estimated_possessions (blend of pace-based and odds-based estimates)
    """
    if league_ppp <= 0:
        raise ValueError("league_ppp must be > 0")
    if not (0.0 <= pace_weight <= 1.0):
        raise ValueError("estimated_possessions pace_weight must be in [0, 1]")
    if clip_max <= clip_min:
        raise ValueError("estimated_possessions clip_max must be > clip_min")

    df = features_df.copy()

    raw_total = _numeric_series(df, "total")
    raw_spread = _numeric_series(df, "spread_home")

    total_missing_flag = (
        pd.to_numeric(df["total_missing"], errors="coerce").fillna(0).astype(bool)
        if "total_missing" in df.columns
        else pd.Series(False, index=df.index)
    )
    spread_missing_flag = (
        pd.to_numeric(df["spread_home_missing"], errors="coerce").fillna(0).astype(bool)
        if "spread_home_missing" in df.columns
        else pd.Series(False, index=df.index)
    )

    # Treat rows marked by *_missing flags as null for canonical vegas features.
    vegas_total = raw_total.mask(total_missing_flag)
    vegas_spread = raw_spread.mask(spread_missing_flag)

    team_pace = _numeric_series(df, "team_pace_szn")
    opp_pace = _numeric_series(df, "opp_pace_szn")
    poss_from_pace = 0.5 * (team_pace + opp_pace)
    poss_from_vegas = vegas_total / float(league_ppp)

    est_possessions = poss_from_pace.copy()
    est_possessions = est_possessions.where(est_possessions.notna(), poss_from_vegas)
    both = poss_from_pace.notna() & poss_from_vegas.notna()
    est_possessions.loc[both] = (
        float(pace_weight) * poss_from_pace.loc[both]
        + (1.0 - float(pace_weight)) * poss_from_vegas.loc[both]
    )
    est_possessions = est_possessions.clip(lower=float(clip_min), upper=float(clip_max))
    est_missing_mask = est_possessions.isna()
    vegas_non_null = poss_from_vegas.dropna()
    pace_non_null = poss_from_pace.dropna()
    if not vegas_non_null.empty:
        neutral_possessions = float(vegas_non_null.median())
    elif not pace_non_null.empty:
        neutral_possessions = float(pace_non_null.median())
    else:
        neutral_possessions = 0.5 * (float(clip_min) + float(clip_max))
    neutral_possessions = float(np.clip(neutral_possessions, float(clip_min), float(clip_max)))
    est_possessions = est_possessions.fillna(neutral_possessions).astype("float64")

    df["vegas_total"] = vegas_total.astype("float64")
    df["vegas_spread"] = vegas_spread.astype("float64")
    df["estimated_possessions"] = est_possessions
    df["vegas_total_missing"] = df["vegas_total"].isna().astype("int8")
    df["vegas_spread_missing"] = df["vegas_spread"].isna().astype("int8")
    df["estimated_possessions_missing"] = est_missing_mask.astype("int8")

    meta = {
        "league_ppp": float(league_ppp),
        "estimated_possessions_pace_weight": float(pace_weight),
        "estimated_possessions_clip_min": float(clip_min),
        "estimated_possessions_clip_max": float(clip_max),
        "estimated_possessions_neutral_fallback": float(neutral_possessions),
        "vegas_total_coverage": float(df["vegas_total"].notna().mean()),
        "vegas_spread_coverage": float(df["vegas_spread"].notna().mean()),
        "estimated_possessions_raw_coverage": float((~est_missing_mask).mean()),
        "estimated_possessions_final_coverage": float(df["estimated_possessions"].notna().mean()),
        "estimated_possessions_source": {
            "pace_only": int((poss_from_pace.notna() & poss_from_vegas.isna()).sum()),
            "vegas_only": int((poss_from_pace.isna() & poss_from_vegas.notna()).sum()),
            "blended": int(both.sum()),
            "missing": int((poss_from_pace.isna() & poss_from_vegas.isna()).sum()),
        },
    }
    return df, meta


def _load_boxscore_count_labels(
    counts_path: Path,
    game_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load labels_boxscore_counts.parquet and filter to relevant game dates."""
    if not counts_path.exists():
        print(f"[joint_dataset] warning: boxscore count labels not found at {counts_path}; count labels will be null.")
        empty = pd.DataFrame(columns=JOIN_KEYS + COUNT_STAT_COLS + ["count_labels_available"])
        return empty, {"path": str(counts_path), "found": False, "rows_loaded": 0}

    df = pd.read_parquet(counts_path)
    df = _coerce_join_keys(df, name="boxscore_count_labels", require_game_date=True)

    if game_dates:
        date_set = {pd.Timestamp(d).normalize() for d in game_dates}
        min_day = min(date_set) - pd.Timedelta(days=1)
        max_day = max(date_set) + pd.Timedelta(days=1)
        df = df.loc[(df["game_date"] >= min_day) & (df["game_date"] <= max_day)].copy()

    pre_dedupe = len(df)
    df = df.drop_duplicates(subset=JOIN_KEYS, keep="last")
    if len(df) < pre_dedupe:
        print(f"[joint_dataset] count labels dedup: {pre_dedupe} -> {len(df)}")

    meta: dict[str, Any] = {
        "path": str(counts_path),
        "found": True,
        "rows_loaded": int(len(df)),
        "games_loaded": int(df["game_id"].nunique()) if not df.empty else 0,
    }
    return df, meta


def _align_boxscore_count_labels_to_features(
    features_df: pd.DataFrame,
    counts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join count labels onto the feature spine, row-aligned."""
    spine = features_df.loc[:, JOIN_KEYS].copy()
    spine["_row_idx"] = np.arange(len(spine), dtype=np.int64)

    cols_present = [c for c in COUNT_STAT_COLS if c in counts_df.columns]
    counts_payload = counts_df.loc[:, JOIN_KEYS + cols_present].copy() if not counts_df.empty else pd.DataFrame()
    aligned = spine.merge(counts_payload, on=JOIN_KEYS, how="left", sort=False)
    aligned = aligned.sort_values("_row_idx").drop(columns=["_row_idx"])

    # Fallback for known date-shift issues in some historical boxscore partitions:
    # if strict (game_id, team_id, player_id, game_date) join misses, backfill from
    # (game_id, team_id, player_id) when that key maps to a unique row in counts.
    if cols_present and not counts_df.empty:
        anchor_col = cols_present[0]
        missing_mask = aligned[anchor_col].isna()
        if missing_mask.any():
            missing_idx = np.flatnonzero(missing_mask.to_numpy())
            counts_by_id = (
                counts_df.loc[:, KEY_COLS + cols_present]
                .drop_duplicates(subset=KEY_COLS, keep="last")
            )
            fallback_spine = features_df.iloc[missing_idx].loc[:, KEY_COLS].copy()
            fallback = fallback_spine.merge(counts_by_id, on=KEY_COLS, how="left", sort=False)
            for col in cols_present:
                aligned.loc[missing_mask, col] = aligned.loc[missing_mask, col].where(
                    aligned.loc[missing_mask, col].notna(),
                    fallback[col].to_numpy(),
                )

    for col in cols_present:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

    # Availability flag: 1 if at least fga2 is non-null (game was captured in bronze).
    fga2_col = "fga2" if "fga2" in aligned.columns else None
    if fga2_col:
        aligned["count_labels_available"] = aligned[fga2_col].notna().astype("int8")
    else:
        aligned["count_labels_available"] = np.int8(0)

    return aligned


def _coverage_by_column(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in cols:
        if col in df.columns:
            out[col] = float(pd.to_numeric(df[col], errors="coerce").notna().mean())
        else:
            out[col] = float("nan")
    return out


def _write_manifest(
    out_dir: Path,
    *,
    rotation_dataset_dir: Path,
    rotation_features_path: Path,
    rotation_labels_path: Path,
    rates_root: Path,
    rates_partition_paths: list[Path],
    missing_rates_dates: list[str],
    date_window: dict[str, Any],
    minutes_label_col: str,
    features_df: pd.DataFrame,
    labels_minutes_df: pd.DataFrame,
    labels_rates_df: pd.DataFrame,
    labels_boxscore_counts_df: pd.DataFrame,
    team_game_index_df: pd.DataFrame,
    rates_meta: dict[str, Any],
    tracking_roles_meta: dict[str, Any],
    tracking_asof_meta: dict[str, Any],
    alignment_meta: dict[str, Any],
    boxscore_counts_meta: dict[str, Any],
    game_context_meta: dict[str, Any],
    appended_rates_context_cols: list[str],
    args_dict: dict[str, Any],
) -> None:
    rates_cov = _coverage_by_column(labels_rates_df, RATE_TARGET_COLS + EFFICIENCY_LABEL_COLS + ["minutes_actual"])
    counts_cov = _coverage_by_column(labels_boxscore_counts_df, COUNT_STAT_COLS)
    payload: dict[str, Any] = {
        "version": "joint_rotation_rates_v1_dataset",
        "created_at": _utc_now_iso(),
        "git_sha": _git_sha(),
        "inputs": {
            "rotation_dataset_dir": str(rotation_dataset_dir),
            "rotation_features_path": str(rotation_features_path),
            "rotation_labels_path": str(rotation_labels_path),
            "rates_training_base_root": str(rates_root),
            "rates_partition_count": int(len(rates_partition_paths)),
            "rates_partitions_sample": [str(p) for p in rates_partition_paths[:25]],
            "missing_rates_dates": missing_rates_dates,
            "boxscore_counts": boxscore_counts_meta,
        },
        "outputs": {
            "features": str(out_dir / "features.parquet"),
            "labels_minutes": str(out_dir / "labels_minutes.parquet"),
            "labels_rates": str(out_dir / "labels_rates.parquet"),
            "labels_boxscore_counts": str(out_dir / "labels_boxscore_counts.parquet"),
            "team_game_index": str(out_dir / "team_game_index.parquet"),
        },
        "counts": {
            "rows_features": int(len(features_df)),
            "rows_labels_minutes": int(len(labels_minutes_df)),
            "rows_labels_rates": int(len(labels_rates_df)),
            "team_games": int(len(team_game_index_df)),
            "features_columns": int(len(features_df.columns)),
            "labels_minutes_columns": int(len(labels_minutes_df.columns)),
            "labels_rates_columns": int(len(labels_rates_df.columns)),
        },
        "date_window": date_window,
        "minutes": {
            "source_label_column": minutes_label_col,
            "minutes_label_non_null_rate": float(pd.to_numeric(labels_minutes_df["minutes_label"], errors="coerce").notna().mean()),
        },
        "rates": {
            "meta": rates_meta,
            "coverage_by_column": rates_cov,
            "rows_with_any_rate_labels": int(pd.to_numeric(labels_rates_df["rates_label_available_any"], errors="coerce").fillna(0).sum()),
            "rows_with_all_rate_targets": int(
                pd.to_numeric(labels_rates_df["rates_label_available_all_rate_targets"], errors="coerce").fillna(0).sum()
            ),
            "rows_loss_eligible": int(pd.to_numeric(labels_rates_df["rates_loss_eligible"], errors="coerce").fillna(0).sum()),
        },
        "boxscore_counts": {
            "coverage_by_column": counts_cov,
            "rows_with_count_labels": int(
                pd.to_numeric(labels_boxscore_counts_df.get("count_labels_available", 0), errors="coerce").fillna(0).sum()
            ),
            "count_label_join_rate": float(
                pd.to_numeric(labels_boxscore_counts_df.get("count_labels_available", 0), errors="coerce").fillna(0).mean()
            ),
        },
        "lineup_feature_contract": {
            "lineup_available_coverage": float(
                pd.to_numeric(features_df.get("lineup_available", 0), errors="coerce").fillna(0).mean()
            ),
            "lineup_starter_announced_coverage": float(
                pd.to_numeric(features_df.get("lineup_starter_announced", 0), errors="coerce").fillna(0).mean()
            ),
        },
        "game_context_feature_contract": game_context_meta,
        "alignment": alignment_meta,
        "features_appended_from_rates_context": appended_rates_context_cols,
        "tracking_context": {
            "tracking_roles_load": tracking_roles_meta,
            "tracking_asof_fallback": tracking_asof_meta,
        },
        "args": args_dict,
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--rotation-dataset-dir",
        type=str,
        default=None,
        help=(
            "Rotation dataset source directory (contains features.parquet + labels.parquet). "
            "Can be an absolute path or a dataset name under $PROJECTIONS_DATA_ROOT/training/datasets/. "
            "If omitted, the latest rotation_train_v1* dataset is used."
        ),
    )
    parser.add_argument(
        "--rates-training-base-root",
        type=str,
        default=None,
        help="Root for rates training base partitions (default: $PROJECTIONS_DATA_ROOT/gold/rates_training_base).",
    )
    parser.add_argument(
        "--tracking-roles-root",
        type=str,
        default=None,
        help="Root for tracking_roles partitions (default: $PROJECTIONS_DATA_ROOT/gold/tracking_roles).",
    )
    parser.add_argument(
        "--boxscore-counts-dir",
        type=str,
        default=None,
        help=(
            "Directory containing labels_boxscore_counts.parquet "
            "(default: $PROJECTIONS_DATA_ROOT/gold/labels_boxscore_counts). "
            "Build with scripts/rotation/build_boxscore_count_labels.py."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output directory. Defaults to "
            "$PROJECTIONS_DATA_ROOT/training/datasets/joint_rotation_rates_v1_<utc_timestamp>."
        ),
    )
    parser.add_argument("--start-date", type=str, default=None, help="Optional inclusive game_date floor (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default=None, help="Optional inclusive game_date ceiling (YYYY-MM-DD).")
    parser.add_argument("--lookback-days", type=int, default=None, help="Optional rolling lookback window in days.")
    parser.add_argument(
        "--anchor-date",
        type=str,
        default=None,
        help="Anchor date for --lookback-days (YYYY-MM-DD). Defaults to current UTC date.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional deterministic row cap after joins (for fast iteration).",
    )
    parser.add_argument(
        "--drop-rows-missing-any-rates",
        action="store_true",
        help="Drop rows that have no non-null per-minute rate labels after alignment.",
    )
    parser.add_argument(
        "--min-minutes-for-rates-loss",
        type=float,
        default=DEFAULT_MINUTES_FOR_RATES_LOSS,
        help="Eligibility threshold used to mark rows suitable for rates loss.",
    )
    parser.add_argument(
        "--league-ppp",
        type=float,
        default=DEFAULT_LEAGUE_PPP,
        help=(
            "League-average game-total points per possession used for vegas-only possessions fallback "
            "(for example, ~2.25)."
        ),
    )
    parser.add_argument(
        "--estimated-possessions-pace-weight",
        type=float,
        default=DEFAULT_EST_POSSESSIONS_PACE_WEIGHT,
        help=(
            "Blend weight for pace-based possessions estimate when both pace and vegas are available. "
            "0 uses vegas-only, 1 uses pace-only."
        ),
    )
    parser.add_argument(
        "--estimated-possessions-clip-min",
        type=float,
        default=DEFAULT_EST_POSSESSIONS_CLIP_MIN,
        help="Lower clip bound for estimated_possessions.",
    )
    parser.add_argument(
        "--estimated-possessions-clip-max",
        type=float,
        default=DEFAULT_EST_POSSESSIONS_CLIP_MAX,
        help="Upper clip bound for estimated_possessions.",
    )
    parser.add_argument("--report-only", action="store_true", help="Print diagnostics and exit without writing files.")
    args = parser.parse_args()

    data_root = paths.get_data_root()
    rotation_dataset_dir = _resolve_rotation_dataset_dir(data_root, args.rotation_dataset_dir)
    rates_root = (
        Path(args.rates_training_base_root).expanduser().resolve()
        if args.rates_training_base_root
        else (data_root / "gold" / "rates_training_base").resolve()
    )
    tracking_roles_root = (
        Path(args.tracking_roles_root).expanduser().resolve()
        if args.tracking_roles_root
        else (data_root / "gold" / "tracking_roles").resolve()
    )
    counts_path = (
        Path(args.boxscore_counts_dir).expanduser().resolve() / "labels_boxscore_counts.parquet"
        if args.boxscore_counts_dir
        else (data_root / "gold" / "labels_boxscore_counts" / "labels_boxscore_counts.parquet")
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (data_root / "training" / "datasets" / f"{DEFAULT_OUT_PREFIX}_{_utc_now_compact()}").resolve()
    )

    start_day, end_day = _resolve_date_window(
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        anchor_date=args.anchor_date,
    )

    print(f"[joint_dataset] data_root={data_root}")
    print(f"[joint_dataset] rotation_dataset_dir={rotation_dataset_dir}")
    print(f"[joint_dataset] rates_training_base_root={rates_root}")
    print(f"[joint_dataset] tracking_roles_root={tracking_roles_root}")
    print(f"[joint_dataset] boxscore_counts_path={counts_path}")
    if start_day is not None or end_day is not None:
        print(f"[joint_dataset] date_window start={start_day} end={end_day}")

    features_df, labels_df, rotation_features_path, rotation_labels_path = _load_rotation_dataset(rotation_dataset_dir)
    features_df = _coerce_join_keys(features_df, name="rotation_features", require_game_date=True)
    labels_df = _coerce_join_keys(labels_df, name="rotation_labels", require_game_date=False)

    features_before = int(len(features_df))
    labels_before = int(len(labels_df))
    features_df = _filter_by_game_date(features_df, start_day=start_day, end_day=end_day)
    labels_df = _filter_by_game_date(labels_df, start_day=start_day, end_day=end_day)
    print(
        "[joint_dataset] rotation date filter:",
        f"features={features_before}->{len(features_df)}",
        f"labels={labels_before}->{len(labels_df)}",
    )
    if features_df.empty:
        raise ValueError("No rotation feature rows remain after date filtering.")

    _assert_unique_keys(features_df, name="rotation_features(filtered)", keys=JOIN_KEYS)

    # Apply lineup/starter feature contract (spec Section 4.7).
    features_df = _apply_lineup_feature_contract(features_df)
    print(
        "[joint_dataset] lineup contract applied:",
        f"lineup_available={features_df['lineup_available'].mean():.1%}",
        f"lineup_starter_announced={features_df.get('lineup_starter_announced', pd.Series([0])).mean():.1%}",
    )
    features_df, game_context_meta = _apply_game_context_feature_contract(
        features_df,
        league_ppp=float(args.league_ppp),
        pace_weight=float(args.estimated_possessions_pace_weight),
        clip_min=float(args.estimated_possessions_clip_min),
        clip_max=float(args.estimated_possessions_clip_max),
    )
    print(
        "[joint_dataset] game-context contract applied:",
        f"vegas_total={game_context_meta['vegas_total_coverage']:.1%}",
        f"vegas_spread={game_context_meta['vegas_spread_coverage']:.1%}",
        f"estimated_possessions_raw={game_context_meta['estimated_possessions_raw_coverage']:.1%}",
        f"estimated_possessions_final={game_context_meta['estimated_possessions_final_coverage']:.1%}",
    )

    minutes_label_col = _infer_minutes_label_column(labels_df)
    labels_minutes_df, alignment_meta = _align_minutes_labels_to_features(
        features_df, labels_df, minutes_label_col=minutes_label_col
    )

    unique_days = sorted(pd.to_datetime(features_df["game_date"]).dt.normalize().dropna().unique().tolist())
    rates_partition_paths, missing_rates_dates = _resolve_rates_partition_paths(
        rates_root,
        game_dates=[pd.Timestamp(day) for day in unique_days],
    )
    print(
        "[joint_dataset] rates partitions:",
        f"dates={len(unique_days)}",
        f"found_partitions={len(rates_partition_paths)}",
        f"missing_dates={len(missing_rates_dates)}",
    )
    rates_df, rates_meta = _load_rates_labels(
        rates_partition_paths,
        context_cols=TRACKING_CONTEXT_COLS,
    )
    if not rates_df.empty:
        _assert_unique_keys(rates_df, name="rates_labels(deduped)", keys=JOIN_KEYS)

    features_aug_df, appended_rates_context_cols = _append_rates_context_columns(features_df, rates_df)
    tracking_roles_meta: dict[str, Any] = {
        "enabled": False,
        "reason": "no feature dates available",
    }
    tracking_asof_meta: dict[str, Any] = {
        "enabled": False,
        "reason": "tracking fallback not attempted",
    }
    if unique_days:
        unique_min_day = pd.Timestamp(min(unique_days)).normalize()
        unique_max_day = pd.Timestamp(max(unique_days)).normalize()
        tracking_df, tracking_roles_meta = _load_tracking_roles_window(
            tracking_roles_root,
            start_day=(unique_min_day - pd.Timedelta(days=365)).normalize(),
            end_day=unique_max_day,
        )
        features_aug_df, tracking_asof_meta = _apply_tracking_context_asof_fallback(features_aug_df, tracking_df)
        print(
            "[joint_dataset] tracking context:",
            f"rates_context_cols={len([c for c in appended_rates_context_cols if c.startswith('track_')])}",
            f"coverage_before={tracking_asof_meta.get('before_coverage_any_tracking', 0.0):.1%}",
            f"coverage_after={tracking_asof_meta.get('after_coverage_any_tracking', 0.0):.1%}",
            f"rows_filled={tracking_asof_meta.get('rows_filled_any_tracking', 0)}",
        )
    labels_rates_df = _align_rates_labels_to_features(
        features_aug_df,
        rates_df,
        min_minutes_for_rates_loss=float(args.min_minutes_for_rates_loss),
    )

    # Load and align boxscore count labels.
    counts_df, boxscore_counts_meta = _load_boxscore_count_labels(
        counts_path,
        game_dates=[pd.Timestamp(day) for day in unique_days],
    )
    labels_boxscore_counts_df = _align_boxscore_count_labels_to_features(features_aug_df, counts_df)
    count_join_rate = float(labels_boxscore_counts_df["count_labels_available"].mean())
    print(
        "[joint_dataset] boxscore count labels:",
        f"join_rate={count_join_rate:.1%}",
        f"rows_with_counts={int(labels_boxscore_counts_df['count_labels_available'].sum())}",
    )

    if args.drop_rows_missing_any_rates:
        # Keep whole team-game sets intact (do not drop individual rows), otherwise
        # roster-set structure is corrupted and 240-minute allocation training is biased.
        rates_any = pd.to_numeric(labels_rates_df["rates_label_available_any"], errors="coerce").fillna(0).astype(int)
        tg_index = features_aug_df.loc[:, ["game_id", "team_id"]].copy()
        tg_index["rates_any"] = rates_any.to_numpy()
        keep_team_games = (
            tg_index.groupby(["game_id", "team_id"], sort=False)["rates_any"].sum() > 0
        )
        keep_mask = pd.Series(
            list(zip(features_aug_df["game_id"], features_aug_df["team_id"], strict=False)),
            index=features_aug_df.index,
        ).isin(set(keep_team_games[keep_team_games].index.tolist()))
        before_rows = int(len(features_aug_df))
        before_tg = int(
            features_aug_df.loc[:, ["game_id", "team_id"]]
            .drop_duplicates()
            .shape[0]
        )
        features_aug_df = features_aug_df.loc[keep_mask].reset_index(drop=True)
        labels_minutes_df = labels_minutes_df.loc[keep_mask].reset_index(drop=True)
        labels_rates_df = labels_rates_df.loc[keep_mask].reset_index(drop=True)
        after_tg = int(
            features_aug_df.loc[:, ["game_id", "team_id"]]
            .drop_duplicates()
            .shape[0]
        )
        print(
            "[joint_dataset] drop missing rates team-games:",
            f"rows={before_rows}->{len(features_aug_df)}",
            f"team_games={before_tg}->{after_tg}",
        )

    if args.max_rows is not None and args.max_rows > 0 and len(features_aug_df) > args.max_rows:
        keep_n = int(args.max_rows)
        features_aug_df = features_aug_df.iloc[:keep_n].reset_index(drop=True)
        labels_minutes_df = labels_minutes_df.iloc[:keep_n].reset_index(drop=True)
        labels_rates_df = labels_rates_df.iloc[:keep_n].reset_index(drop=True)
        labels_boxscore_counts_df = labels_boxscore_counts_df.iloc[:keep_n].reset_index(drop=True)
        print(f"[joint_dataset] max_rows cap applied: {keep_n}")

    team_game_index_df = _build_team_game_index(features_aug_df, labels_minutes_df, labels_rates_df)

    rates_any = int(pd.to_numeric(labels_rates_df["rates_label_available_any"], errors="coerce").fillna(0).sum())
    rates_all = int(
        pd.to_numeric(labels_rates_df["rates_label_available_all_rate_targets"], errors="coerce").fillna(0).sum()
    )
    loss_eligible = int(pd.to_numeric(labels_rates_df["rates_loss_eligible"], errors="coerce").fillna(0).sum())
    print(
        "[joint_dataset] coverage:",
        f"rows={len(features_aug_df)}",
        f"rows_with_any_rate_labels={rates_any}",
        f"rows_with_all_rate_targets={rates_all}",
        f"rows_rates_loss_eligible={loss_eligible}",
        f"rows_with_count_labels={int(labels_boxscore_counts_df['count_labels_available'].sum())}",
    )

    date_min = pd.to_datetime(features_aug_df["game_date"]).min()
    date_max = pd.to_datetime(features_aug_df["game_date"]).max()
    date_window_meta: dict[str, Any] = {
        "enabled": bool(start_day is not None or end_day is not None),
        "start_date": str(start_day.date()) if start_day is not None else None,
        "end_date": str(end_day.date()) if end_day is not None else None,
        "min_game_date_in_output": str(date_min.date()) if pd.notna(date_min) else None,
        "max_game_date_in_output": str(date_max.date()) if pd.notna(date_max) else None,
        "n_game_dates_in_output": int(pd.to_datetime(features_aug_df["game_date"]).nunique()),
    }

    if args.report_only:
        print("[joint_dataset] report-only mode; no files written.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_features = out_dir / "features.parquet"
    out_labels_minutes = out_dir / "labels_minutes.parquet"
    out_labels_rates = out_dir / "labels_rates.parquet"
    out_labels_boxscore_counts = out_dir / "labels_boxscore_counts.parquet"
    out_team_game_index = out_dir / "team_game_index.parquet"

    features_aug_df.to_parquet(out_features, index=False)
    labels_minutes_df.to_parquet(out_labels_minutes, index=False)
    labels_rates_df.to_parquet(out_labels_rates, index=False)
    labels_boxscore_counts_df.to_parquet(out_labels_boxscore_counts, index=False)
    team_game_index_df.to_parquet(out_team_game_index, index=False)

    _write_manifest(
        out_dir,
        rotation_dataset_dir=rotation_dataset_dir,
        rotation_features_path=rotation_features_path,
        rotation_labels_path=rotation_labels_path,
        rates_root=rates_root,
        rates_partition_paths=rates_partition_paths,
        missing_rates_dates=missing_rates_dates,
        date_window=date_window_meta,
        minutes_label_col=minutes_label_col,
        features_df=features_aug_df,
        labels_minutes_df=labels_minutes_df,
        labels_rates_df=labels_rates_df,
        labels_boxscore_counts_df=labels_boxscore_counts_df,
        team_game_index_df=team_game_index_df,
        rates_meta=rates_meta,
        tracking_roles_meta=tracking_roles_meta,
        tracking_asof_meta=tracking_asof_meta,
        alignment_meta=alignment_meta,
        boxscore_counts_meta=boxscore_counts_meta,
        game_context_meta=game_context_meta,
        appended_rates_context_cols=appended_rates_context_cols,
        args_dict={
            "rotation_dataset_dir": args.rotation_dataset_dir,
            "rates_training_base_root": args.rates_training_base_root,
            "tracking_roles_root": args.tracking_roles_root,
            "boxscore_counts_dir": args.boxscore_counts_dir,
            "out_dir": args.out_dir,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "lookback_days": args.lookback_days,
            "anchor_date": args.anchor_date,
            "max_rows": args.max_rows,
            "drop_rows_missing_any_rates": bool(args.drop_rows_missing_any_rates),
            "min_minutes_for_rates_loss": float(args.min_minutes_for_rates_loss),
            "league_ppp": float(args.league_ppp),
            "estimated_possessions_pace_weight": float(args.estimated_possessions_pace_weight),
            "estimated_possessions_clip_min": float(args.estimated_possessions_clip_min),
            "estimated_possessions_clip_max": float(args.estimated_possessions_clip_max),
        },
    )

    print(f"[joint_dataset] wrote features              -> {out_features}")
    print(f"[joint_dataset] wrote labels_minutes        -> {out_labels_minutes}")
    print(f"[joint_dataset] wrote labels_rates          -> {out_labels_rates}")
    print(f"[joint_dataset] wrote labels_boxscore_counts-> {out_labels_boxscore_counts}")
    print(f"[joint_dataset] wrote team_game_index       -> {out_team_game_index}")
    print(f"[joint_dataset] wrote manifest              -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
