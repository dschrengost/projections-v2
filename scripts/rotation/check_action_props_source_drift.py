#!/usr/bin/env python3
"""Compare Action Network vs Rotowire-fallback props feature distributions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projections import paths
from projections.features.action_props import (
    ACTION_MARKET_FEATURE_COLUMNS,
    build_action_props_feature_snapshots,
    load_action_props_feature_snapshots_for_date,
    load_rotowire_props_long_from_bronze,
)


JOIN_KEYS = ["game_date", "team_tricode", "player_name_norm"]


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _date_range(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must be <= end_date")
    return list(pd.date_range(start, end, freq="D"))


def _numeric_feature_cols() -> list[str]:
    return [c for c in ACTION_MARKET_FEATURE_COLUMNS if c != "action_props_as_of_ts"]


def _load_day_pair(data_root: Path, day: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_dir = data_root / "bronze" / "action_network" / "props"
    rotowire_root = data_root / "bronze" / "props"
    action = load_action_props_feature_snapshots_for_date(props_dir=action_dir, game_date=day)
    rotowire_long = load_rotowire_props_long_from_bronze(rotowire_props_root=rotowire_root, game_date=day)
    rotowire = build_action_props_feature_snapshots(rotowire_long)
    return action, rotowire


def _merge_on_keys(action_df: pd.DataFrame, rotowire_df: pd.DataFrame) -> pd.DataFrame:
    if action_df.empty or rotowire_df.empty:
        return pd.DataFrame()
    a = action_df.copy()
    r = rotowire_df.copy()
    a["game_date"] = pd.to_datetime(a["game_date"], errors="coerce").dt.normalize()
    r["game_date"] = pd.to_datetime(r["game_date"], errors="coerce").dt.normalize()
    a = a.dropna(subset=JOIN_KEYS).copy()
    r = r.dropna(subset=JOIN_KEYS).copy()
    if a.empty or r.empty:
        return pd.DataFrame()
    return a.merge(r, on=JOIN_KEYS, how="inner", suffixes=("_action", "_rotowire"))


def _column_metrics(merged: pd.DataFrame, col: str) -> dict[str, float]:
    a_col = f"{col}_action"
    r_col = f"{col}_rotowire"
    if a_col not in merged.columns or r_col not in merged.columns:
        return {}
    a = pd.to_numeric(merged[a_col], errors="coerce")
    r = pd.to_numeric(merged[r_col], errors="coerce")
    valid = a.notna() & r.notna()
    if not bool(valid.any()):
        return {}
    av = a.loc[valid].to_numpy(dtype=float)
    rv = r.loc[valid].to_numpy(dtype=float)
    diff = av - rv
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mad = float(np.mean(np.abs(diff)))
    std_ref = float(np.std(av, ddof=0))
    normalized_mad = float(mad / max(std_ref, 1e-6))
    return {
        "n": int(valid.sum()),
        "mean_action": float(np.mean(av)),
        "mean_rotowire": float(np.mean(rv)),
        "mean_abs_diff": mad,
        "rmse": rmse,
        "normalized_mad": normalized_mad,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional projections data root (defaults to PROJECTIONS_DATA_ROOT or repo default).",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output path for drift summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else paths.get_data_root()
    days = _date_range(args.start_date, args.end_date)

    merged_days: list[pd.DataFrame] = []
    day_stats: list[dict[str, Any]] = []
    for day in days:
        action_df, rotowire_df = _load_day_pair(data_root, day)
        merged = _merge_on_keys(action_df, rotowire_df)
        merged_days.append(merged)
        day_stats.append(
            {
                "date": day.date().isoformat(),
                "action_rows": int(len(action_df)),
                "rotowire_rows": int(len(rotowire_df)),
                "matched_rows": int(len(merged)),
            }
        )

    merged_all = pd.concat(merged_days, ignore_index=True) if merged_days else pd.DataFrame()
    by_col: dict[str, dict[str, float]] = {}
    for col in _numeric_feature_cols():
        metrics = _column_metrics(merged_all, col)
        if metrics:
            by_col[col] = metrics

    normalized = [float(v["normalized_mad"]) for v in by_col.values() if "normalized_mad" in v]
    global_drift_score = float(np.mean(normalized)) if normalized else None
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "window": {
            "start_date": str(pd.Timestamp(args.start_date).date()),
            "end_date": str(pd.Timestamp(args.end_date).date()),
            "n_days": int(len(days)),
        },
        "day_stats": day_stats,
        "matched_rows_total": int(len(merged_all)),
        "columns_evaluated": sorted(by_col.keys()),
        "column_metrics": by_col,
        "global_drift_score_normalized_mad": global_drift_score,
    }

    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else (
            data_root
            / "reports"
            / "gtv2_props_source_drift"
            / f"action_vs_rotowire_{_utc_now_compact()}.json"
        )
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "matched_rows_total": int(len(merged_all))}, indent=2))


if __name__ == "__main__":
    main()

