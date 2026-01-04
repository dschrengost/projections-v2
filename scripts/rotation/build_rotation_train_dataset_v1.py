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
    "min_last3",
    "min_last5",
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

def _load_minutes_dataset(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    dataset_dir = data_root / "training" / "datasets" / MINUTES_DATASET_DIRNAME
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
    team_game_validation: dict[str, Any] | None,
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
    if team_game_validation is not None:
        payload["team_game_validation"] = team_game_validation
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    args = parser.parse_args()

    data_root = paths.get_data_root()
    out_dir = Path(args.out_dir).expanduser().resolve()
    bundle_dir = (paths.get_project_root() / MINUTES_BUNDLE_DIR).resolve()

    allowlist = _read_feature_allowlist(bundle_dir)
    features_df, labels_df, source_features_path, source_labels_path = _load_minutes_dataset(data_root)

    pruned_df, kept_minutes_features = _apply_feature_pruning(features_df, allowlist=allowlist)
    pruned_df = _apply_odds_missing_flags(pruned_df)

    player_rotation = _load_rotation_player_labels(data_root)
    team_rotation = _load_rotation_team_shape(data_root)
    if args.report_only:
        _report_match_diagnostics(features_df, player_rotation=player_rotation, team_rotation=team_rotation)
    joined = _join_rotation(pruned_df, player_rotation=player_rotation, team_rotation=team_rotation)

    if args.require_rotation:
        joined = joined.loc[joined["rotation_team_missing"] == 0].copy()
    label_col = _infer_label_column(labels_df)
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
        team_game_validation=team_game_validation,
    )

    print(f"[rotation_train_v1] Wrote features -> {out_features_path}")
    print(f"[rotation_train_v1] Wrote labels   -> {out_labels_path}")
    print(f"[rotation_train_v1] Wrote manifest -> {out_dir / 'manifest.json'}")
    print(f"[rotation_train_v1] Kept minutes audited features: {len(kept_minutes_features)}")


if __name__ == "__main__":
    main()
