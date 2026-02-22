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

DEFAULT_ROTATION_PREFIX = "rotation_train_v1"
DEFAULT_OUT_PREFIX = "joint_rotation_rates_v1"
DEFAULT_MINUTES_FOR_RATES_LOSS = 4.0


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


def _zfill_game_id(series: pd.Series) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce").astype("Int64")
    return coerced.astype("string").str.zfill(10)


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not partition_paths:
        return pd.DataFrame(columns=JOIN_KEYS + RATES_LABEL_COLS), {"partition_count": 0, "rows_loaded": 0}

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

        keep = [c for c in JOIN_KEYS + RATES_LABEL_COLS + ["feature_as_of_ts", "tip_ts"] if c in frame.columns]
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
    team_game_index_df: pd.DataFrame,
    rates_meta: dict[str, Any],
    alignment_meta: dict[str, Any],
    appended_rates_context_cols: list[str],
    args_dict: dict[str, Any],
) -> None:
    rates_cov = _coverage_by_column(labels_rates_df, RATE_TARGET_COLS + EFFICIENCY_LABEL_COLS + ["minutes_actual"])
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
        },
        "outputs": {
            "features": str(out_dir / "features.parquet"),
            "labels_minutes": str(out_dir / "labels_minutes.parquet"),
            "labels_rates": str(out_dir / "labels_rates.parquet"),
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
        "alignment": alignment_meta,
        "features_appended_from_rates_context": appended_rates_context_cols,
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
    parser.add_argument("--report-only", action="store_true", help="Print diagnostics and exit without writing files.")
    args = parser.parse_args()

    data_root = paths.get_data_root()
    rotation_dataset_dir = _resolve_rotation_dataset_dir(data_root, args.rotation_dataset_dir)
    rates_root = (
        Path(args.rates_training_base_root).expanduser().resolve()
        if args.rates_training_base_root
        else (data_root / "gold" / "rates_training_base").resolve()
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
    rates_df, rates_meta = _load_rates_labels(rates_partition_paths)
    if not rates_df.empty:
        _assert_unique_keys(rates_df, name="rates_labels(deduped)", keys=JOIN_KEYS)

    features_aug_df, appended_rates_context_cols = _append_rates_context_columns(features_df, rates_df)
    labels_rates_df = _align_rates_labels_to_features(
        features_aug_df,
        rates_df,
        min_minutes_for_rates_loss=float(args.min_minutes_for_rates_loss),
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
    out_team_game_index = out_dir / "team_game_index.parquet"

    features_aug_df.to_parquet(out_features, index=False)
    labels_minutes_df.to_parquet(out_labels_minutes, index=False)
    labels_rates_df.to_parquet(out_labels_rates, index=False)
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
        team_game_index_df=team_game_index_df,
        rates_meta=rates_meta,
        alignment_meta=alignment_meta,
        appended_rates_context_cols=appended_rates_context_cols,
        args_dict={
            "rotation_dataset_dir": args.rotation_dataset_dir,
            "rates_training_base_root": args.rates_training_base_root,
            "out_dir": args.out_dir,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "lookback_days": args.lookback_days,
            "anchor_date": args.anchor_date,
            "max_rows": args.max_rows,
            "drop_rows_missing_any_rates": bool(args.drop_rows_missing_any_rates),
            "min_minutes_for_rates_loss": float(args.min_minutes_for_rates_loss),
        },
    )

    print(f"[joint_dataset] wrote features        -> {out_features}")
    print(f"[joint_dataset] wrote labels_minutes  -> {out_labels_minutes}")
    print(f"[joint_dataset] wrote labels_rates    -> {out_labels_rates}")
    print(f"[joint_dataset] wrote team_game_index -> {out_team_game_index}")
    print(f"[joint_dataset] wrote manifest        -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
