"""Extend a v1_enriched training dataset by appending live feature snapshots + labels.

Usage:
    uv run python scripts/livefill_enriched_dataset.py \
        --base-dir /home/daniel/projections-data/training/datasets/v1_enriched_boxscore_20260218_livefill \
        --out-dir /home/daniel/projections-data/training/datasets/v1_enriched_boxscore_20260331_livefill \
        --start-date 2026-02-16 \
        --end-date 2026-03-29
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _find_latest_run(date_dir: Path) -> Path | None:
    """Return the latest run=*/features.parquet under a date directory."""
    runs = sorted(date_dir.glob("run=*/features.parquet"))
    return runs[-1] if runs else None


def _load_live_features_for_date(live_root: Path, game_date: str) -> pd.DataFrame | None:
    date_dir = live_root / game_date
    if not date_dir.is_dir():
        return None
    feat_path = _find_latest_run(date_dir)
    if feat_path is None:
        return None
    df = pd.read_parquet(feat_path)
    df["_live_run_path"] = str(feat_path)
    return df


def _load_labels_for_date(labels_root: Path, game_date: str, season: int) -> pd.DataFrame | None:
    label_path = labels_root / f"season={season}" / f"game_date={game_date}" / "labels.parquet"
    if not label_path.exists():
        return None
    return pd.read_parquet(label_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True, help="Base enriched dataset directory")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for extended dataset")
    parser.add_argument("--start-date", type=str, required=True, help="Start date for livefill (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date for livefill (YYYY-MM-DD)")
    parser.add_argument("--data-root", type=Path, default=Path("/home/daniel/projections-data"))
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()

    live_root = args.data_root / "live" / "features_minutes_v1"
    labels_root = args.data_root / "gold" / "labels_minutes_v1"

    # Load base dataset
    print(f"[livefill] Loading base features from {args.base_dir}")
    base_feat = pd.read_parquet(args.base_dir / "features.parquet")
    base_lab = pd.read_parquet(args.base_dir / "labels.parquet")
    print(f"[livefill] Base: {len(base_feat)} feature rows, {len(base_lab)} label rows")
    print(f"[livefill] Base date range: {base_feat['game_date'].min().date()} → {base_feat['game_date'].max().date()}")

    base_feat_cols = list(base_feat.columns)
    base_lab_cols = list(base_lab.columns)

    # Iterate over dates
    dates = pd.date_range(args.start_date, args.end_date, freq="D")
    coverage = []
    feat_frames = []
    lab_frames = []

    for day in dates:
        game_date = day.strftime("%Y-%m-%d")
        live_df = _load_live_features_for_date(live_root, game_date)
        label_df = _load_labels_for_date(labels_root, game_date, args.season)

        has_live = live_df is not None and not live_df.empty
        has_labels = label_df is not None and not label_df.empty

        coverage.append({
            "game_date": game_date,
            "live_features": has_live,
            "labels": has_labels,
        })

        if not (has_live and has_labels):
            continue

        # Align live features to base schema: keep shared columns, fill missing with NaN
        for col in base_feat_cols:
            if col not in live_df.columns:
                # Use dtype-appropriate fill: empty string for object cols, NaN for numeric
                base_dtype = base_feat[col].dtype
                if base_dtype == object:
                    live_df[col] = ""
                else:
                    live_df[col] = pd.NA

        # The live snapshot has a 'minutes' column (prior minutes, not label) — keep it
        # Labels have the actual minutes outcome
        live_subset = live_df[base_feat_cols].copy()

        # Align labels
        for col in base_lab_cols:
            if col not in label_df.columns:
                label_df[col] = pd.NA
        label_subset = label_df[base_lab_cols].copy()

        feat_frames.append(live_subset)
        lab_frames.append(label_subset)

    if not feat_frames:
        print("[livefill] No new dates to append!")
        return

    append_feat = pd.concat(feat_frames, ignore_index=True)
    append_lab = pd.concat(lab_frames, ignore_index=True)

    # Normalize game_date to datetime
    append_feat["game_date"] = pd.to_datetime(append_feat["game_date"])
    append_lab["game_date"] = pd.to_datetime(append_lab["game_date"])
    base_feat["game_date"] = pd.to_datetime(base_feat["game_date"])
    base_lab["game_date"] = pd.to_datetime(base_lab["game_date"])

    print(f"[livefill] Appending {len(append_feat)} feature rows, {len(append_lab)} label rows")

    # Deduplicate: drop rows from base that overlap with new data by (game_id, player_id)
    base_max_date = base_feat["game_date"].max()
    append_min_date = append_feat["game_date"].min()
    if append_min_date <= base_max_date:
        # There's overlap — deduplicate by keeping the newer data
        overlap_game_ids = set(append_feat["game_id"].unique())
        base_feat = base_feat[~base_feat["game_id"].isin(overlap_game_ids)].copy()
        base_lab = base_lab[~base_lab["game_id"].isin(overlap_game_ids)].copy()
        print(f"[livefill] After dedup: {len(base_feat)} base feature rows")

    out_feat = pd.concat([base_feat, append_feat], ignore_index=True)
    out_lab = pd.concat([base_lab, append_lab], ignore_index=True)

    # Sort by game_date
    out_feat.sort_values("game_date", inplace=True, ignore_index=True)
    out_lab.sort_values("game_date", inplace=True, ignore_index=True)

    # Convert string-dtype columns to object and fill pd.NA → "" to avoid downstream issues
    for col in out_feat.columns:
        if pd.api.types.is_string_dtype(out_feat[col]):
            out_feat[col] = out_feat[col].astype(object).fillna("")

    print(f"[livefill] Output: {len(out_feat)} feature rows, {len(out_lab)} label rows")
    print(f"[livefill] Date range: {out_feat['game_date'].min().date()} → {out_feat['game_date'].max().date()}")

    # Write
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_feat.to_parquet(args.out_dir / "features.parquet", index=False)
    out_lab.to_parquet(args.out_dir / "labels.parquet", index=False)

    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(args.out_dir / "livefill_coverage.csv", index=False)

    days_joined = int(coverage_df[coverage_df["live_features"] & coverage_df["labels"]].shape[0])
    manifest = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "base_dir": str(args.base_dir),
        "out_dir": str(args.out_dir),
        "window": {"start": args.start_date, "end": args.end_date},
        "counts": {
            "base_features_rows": int(len(base_feat)),
            "base_labels_rows": int(len(base_lab)),
            "append_features_rows": int(len(append_feat)),
            "append_labels_rows": int(len(append_lab)),
            "features_rows_out": int(len(out_feat)),
            "labels_rows_out": int(len(out_lab)),
            "feature_columns": int(len(base_feat_cols)),
            "label_columns": int(len(base_lab_cols)),
        },
        "coverage": {
            "days_total": int(len(dates)),
            "days_with_live": int(coverage_df["live_features"].sum()),
            "days_with_labels": int(coverage_df["labels"].sum()),
            "days_joined": days_joined,
        },
        "date_ranges": {
            "features_min": str(out_feat["game_date"].min()),
            "features_max": str(out_feat["game_date"].max()),
            "labels_min": str(out_lab["game_date"].min()),
            "labels_max": str(out_lab["game_date"].max()),
        },
    }
    (args.out_dir / "manifest_livefill.json").write_text(
        json.dumps(manifest, indent=4), encoding="utf-8"
    )
    print(f"[livefill] Coverage: {days_joined}/{len(dates)} days")
    print(f"[livefill] Wrote to {args.out_dir}")


if __name__ == "__main__":
    main()
