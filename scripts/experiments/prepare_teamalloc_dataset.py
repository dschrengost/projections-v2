#!/usr/bin/env python3
"""Prepare team-allocation training dataset from gold minutes features.

This script prepares a parquet suitable for training the DeepSets team allocator.
It uses the EXACT feature columns from a specified LGBM model artifact to ensure
consistency between the allocator experiment and production minutes model.

Example usage:
    # Using the safe_starter model's feature columns
    PROJECTIONS_DATA_ROOT=/home/daniel/projections-data uv run python -m scripts.experiments.prepare_teamalloc_dataset \
        --model-artifact minutes_v1_safe_starter_20251214 \
        --out-dir artifacts/experiments/team_alloc_dataset

    # Specify custom date range
    PROJECTIONS_DATA_ROOT=/home/daniel/projections-data uv run python -m scripts.experiments.prepare_teamalloc_dataset \
        --model-artifact minutes_v1_safe_starter_20251214 \
        --start-date 2022-10-01 \
        --end-date 2025-04-30 \
        --out-dir artifacts/experiments/team_alloc_dataset

Output Files:
    - minutes_teamalloc_train.parquet: Full dataset with is_ot_team_game flag
    - minutes_teamalloc_train_no_ot.parquet: Dataset with OT team-games dropped
    - dataset_summary.json: Summary statistics and validation results

Notes:
    - Uses existing gold/features_minutes_v1 parquets which already include DNP rows
    - Filters to active roster only (excludes status=OUT players)
    - Deduplicates by (game_id, team_id, player_id) deterministically
    - Labels (minutes) are actual box-score minutes from the features parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Key columns that identify a player-game row
KEY_COLUMNS = ("game_id", "player_id", "team_id")

# Regulation minutes per team
REGULATION_MINUTES = 240.0
OVERTIME_THRESHOLD = 242.0

# Status values that indicate player is OUT (not in active roster)
OUT_STATUS_VALUES = {"OUT", "out", "Out"}


def load_feature_columns(artifact_path: Path) -> list[str]:
    """Load feature columns from LGBM model artifact."""
    feature_file = artifact_path / "feature_columns.json"
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature columns file not found: {feature_file}")

    with open(feature_file) as f:
        data = json.load(f)

    columns = data.get("columns", [])
    if not columns:
        raise ValueError(f"No columns found in {feature_file}")

    return columns


def load_model_metadata(artifact_path: Path) -> dict[str, Any]:
    """Load model metadata from LGBM artifact."""
    meta_file = artifact_path / "meta.json"
    if not meta_file.exists():
        return {}

    with open(meta_file) as f:
        return json.load(f)


def discover_feature_parquets(
    data_root: Path,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[Path]:
    """Discover feature parquet files within optional date range."""
    features_dir = data_root / "gold" / "features_minutes_v1"
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    paths = []
    for season_dir in sorted(features_dir.glob("season=*")):
        for month_dir in sorted(season_dir.glob("month=*")):
            parquet = month_dir / "features.parquet"
            if parquet.exists():
                paths.append(parquet)

    if not paths:
        raise FileNotFoundError(f"No feature parquets found under {features_dir}")

    logger.info(f"Discovered {len(paths)} feature parquet files")
    return paths


def load_features(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate feature parquets."""
    frames = []
    for path in paths:
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(df):,} total rows from {len(paths)} files")
    return df


def filter_date_range(
    df: pd.DataFrame,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    """Filter dataframe to specified date range."""
    if start_date is None and end_date is None:
        return df

    if "game_date" not in df.columns:
        logger.warning("No game_date column found, skipping date filter")
        return df

    working = df.copy()
    working["game_date"] = pd.to_datetime(working["game_date"]).dt.normalize()

    mask = pd.Series(True, index=working.index)
    if start_date is not None:
        mask &= working["game_date"] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= working["game_date"] <= pd.Timestamp(end_date)

    filtered = working[mask]
    logger.info(f"Date filter: {len(df):,} -> {len(filtered):,} rows")
    return filtered


def filter_active_roster(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to active roster only (exclude OUT players).

    We want the candidate set for team allocation to include:
    - Players who played (minutes > 0)
    - Players who were active but DNP (minutes == 0, status != OUT)

    We exclude:
    - Players marked as OUT in injury reports
    """
    if "status" not in df.columns:
        logger.warning("No 'status' column, cannot filter OUT players")
        return df

    # Include rows where status is NOT in OUT_STATUS_VALUES
    is_out = df["status"].isin(OUT_STATUS_VALUES)
    active = df[~is_out]

    n_removed = len(df) - len(active)
    logger.info(f"Filtered OUT players: removed {n_removed:,} rows ({100*n_removed/len(df):.1f}%)")
    return active


def add_overtime_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_ot_team_game flag based on team minute totals."""
    team_totals = df.groupby(["game_id", "team_id"])["minutes"].transform("sum")
    df = df.copy()
    df["is_ot_team_game"] = (team_totals > OVERTIME_THRESHOLD).astype(int)

    n_ot_teams = df[df["is_ot_team_game"] == 1][["game_id", "team_id"]].drop_duplicates().shape[0]
    total_teams = df[["game_id", "team_id"]].drop_duplicates().shape[0]

    logger.info(f"OT team-games: {n_ot_teams:,} ({100*n_ot_teams/total_teams:.1f}%)")
    return df


def validate_and_dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate data and deduplicate by key columns."""
    validation = {
        "rows_before_dedupe": len(df),
        "duplicates_removed": 0,
        "null_id_rows_removed": 0,
        "negative_minutes_rows": 0,
        "warnings": [],
    }

    working = df.copy()

    # Check required columns
    missing = set(KEY_COLUMNS) - set(working.columns)
    if missing:
        raise ValueError(f"Missing required key columns: {missing}")

    if "minutes" not in working.columns:
        raise ValueError("Missing required 'minutes' column")

    # Remove rows with null IDs
    for col in KEY_COLUMNS:
        null_count = working[col].isna().sum()
        if null_count > 0:
            validation["null_id_rows_removed"] += null_count
            working = working.dropna(subset=[col])

    if validation["null_id_rows_removed"] > 0:
        validation["warnings"].append(
            f"Removed {validation['null_id_rows_removed']} rows with null ID columns"
        )

    # Check for negative minutes
    neg_mins = (working["minutes"] < 0).sum()
    if neg_mins > 0:
        validation["negative_minutes_rows"] = neg_mins
        raise ValueError(f"Found {neg_mins} rows with negative minutes")

    # Deduplicate deterministically (keep last by game_date if available)
    n_before = len(working)
    if "game_date" in working.columns:
        working = working.sort_values(list(KEY_COLUMNS) + ["game_date"], kind="mergesort")
    else:
        working = working.sort_values(list(KEY_COLUMNS), kind="mergesort")

    working = working.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
    n_after = len(working)

    validation["duplicates_removed"] = n_before - n_after
    validation["rows_after_dedupe"] = n_after

    if validation["duplicates_removed"] > 0:
        validation["warnings"].append(
            f"Removed {validation['duplicates_removed']} duplicate rows"
        )
        logger.warning(f"Removed {validation['duplicates_removed']} duplicate (game_id, team_id, player_id) rows")

    return working.reset_index(drop=True), validation


def compute_dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics for the dataset."""
    team_games = df.groupby(["game_id", "team_id"])

    # Players per team-game
    players_per_tg = team_games.size()

    # Team totals
    team_totals = team_games["minutes"].sum()

    # Zero-minute rate
    zero_rate = (df["minutes"] == 0).mean()

    stats = {
        "total_rows": len(df),
        "total_team_games": len(players_per_tg),
        "unique_games": df["game_id"].nunique(),
        "unique_players": df["player_id"].nunique(),
        "players_per_team_game": {
            "min": int(players_per_tg.min()),
            "median": float(players_per_tg.median()),
            "mean": float(players_per_tg.mean()),
            "p90": float(players_per_tg.quantile(0.90)),
            "p99": float(players_per_tg.quantile(0.99)),
            "max": int(players_per_tg.max()),
        },
        "team_minutes_total": {
            "mean": float(team_totals.mean()),
            "median": float(team_totals.median()),
            "std": float(team_totals.std()),
            "min": float(team_totals.min()),
            "max": float(team_totals.max()),
        },
        "zero_minute_rate": float(zero_rate),
        "ot_team_game_rate": float(df["is_ot_team_game"].mean()) if "is_ot_team_game" in df.columns else None,
    }

    # Warn if median players per team is low
    if stats["players_per_team_game"]["median"] < 11:
        logger.warning(
            f"WARNING: Median players per team-game is {stats['players_per_team_game']['median']:.1f}, "
            "which may indicate missing bench players"
        )

    return stats


def select_output_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Select ID columns, label, features, and helper columns for output."""
    output_cols = list(KEY_COLUMNS) + ["minutes"]

    # Add feature columns that exist in the dataframe
    missing_features = []
    for col in feature_cols:
        if col in df.columns:
            output_cols.append(col)
        else:
            missing_features.append(col)

    if missing_features:
        logger.warning(f"Missing {len(missing_features)} feature columns: {missing_features[:5]}...")

    # Add helper columns
    if "is_ot_team_game" in df.columns:
        output_cols.append("is_ot_team_game")
    if "game_date" in df.columns:
        output_cols.append("game_date")
    if "status" in df.columns:
        output_cols.append("status")
    if "player_name" in df.columns:
        output_cols.append("player_name")

    # Ensure unique columns
    output_cols = list(dict.fromkeys(output_cols))

    return df[output_cols].copy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare team-allocation training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-artifact",
        type=str,
        required=True,
        help="Name of LGBM model artifact directory (e.g., minutes_v1_safe_starter_20251214)",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/minutes_lgbm"),
        help="Root directory for model artifacts (default: artifacts/minutes_lgbm)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root (defaults to PROJECTIONS_DATA_ROOT or ./data)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for prepared datasets",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--include-out",
        action="store_true",
        help="Include OUT players in candidate set (default: exclude)",
    )

    args = parser.parse_args()

    # Resolve paths
    import os
    data_root = args.data_root
    if data_root is None:
        data_root = Path(os.environ.get("PROJECTIONS_DATA_ROOT", "./data"))

    artifact_path = args.artifacts_root / args.model_artifact
    if not artifact_path.exists():
        logger.error(f"Model artifact not found: {artifact_path}")
        return 1

    # Load feature columns from model artifact
    logger.info(f"Loading feature columns from {artifact_path}")
    try:
        feature_cols = load_feature_columns(artifact_path)
        logger.info(f"Loaded {len(feature_cols)} feature columns")
    except Exception as e:
        logger.error(f"Failed to load feature columns: {e}")
        return 1

    # Load model metadata for reference (unused but available for debugging)
    _ = load_model_metadata(artifact_path)

    # Parse dates
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    # Discover and load feature parquets
    logger.info(f"Loading features from {data_root}")
    try:
        parquet_paths = discover_feature_parquets(data_root, start_date, end_date)
        df = load_features(parquet_paths)
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        return 1

    # Filter date range
    df = filter_date_range(df, start_date, end_date)

    # Filter to active roster (exclude OUT players) unless --include-out
    if not args.include_out:
        df = filter_active_roster(df)
    else:
        logger.info("Including OUT players per --include-out flag")

    # Add overtime flag
    df = add_overtime_flag(df)

    # Validate and dedupe
    df, validation_results = validate_and_dedupe(df)

    # Compute stats
    stats = compute_dataset_stats(df)

    # Select output columns
    df_out = select_output_columns(df, feature_cols)

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Write full dataset
    full_path = args.out_dir / "minutes_teamalloc_train.parquet"
    df_out.to_parquet(full_path, index=False)
    logger.info(f"Wrote {len(df_out):,} rows to {full_path}")

    # Write no-OT dataset
    df_no_ot = df_out[df_out["is_ot_team_game"] == 0].drop(columns=["is_ot_team_game"])
    no_ot_path = args.out_dir / "minutes_teamalloc_train_no_ot.parquet"
    df_no_ot.to_parquet(no_ot_path, index=False)
    logger.info(f"Wrote {len(df_no_ot):,} rows (no OT) to {no_ot_path}")

    # Write summary
    summary = {
        "model_artifact": args.model_artifact,
        "feature_columns": feature_cols,
        "feature_columns_count": len(feature_cols),
        "date_range": {
            "start": args.start_date,
            "end": args.end_date,
        },
        "include_out_players": args.include_out,
        "validation": validation_results,
        "stats": stats,
        "output_files": {
            "full": str(full_path),
            "no_ot": str(no_ot_path),
        },
    }

    summary_path = args.out_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Wrote summary to {summary_path}")

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("TEAM ALLOCATION DATASET PREPARED")
    print("=" * 70)
    print(f"\nModel Artifact: {args.model_artifact}")
    print(f"Feature Columns: {len(feature_cols)}")
    print("\nDataset Statistics:")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Team-games: {stats['total_team_games']:,}")
    print(f"  Unique games: {stats['unique_games']:,}")
    print(f"  Unique players: {stats['unique_players']:,}")
    print("\nPlayers per Team-Game:")
    print(f"  Median: {stats['players_per_team_game']['median']:.1f}")
    print(f"  p90: {stats['players_per_team_game']['p90']:.1f}")
    print(f"  p99: {stats['players_per_team_game']['p99']:.1f}")
    print("\nTeam Minutes Total:")
    print(f"  Mean: {stats['team_minutes_total']['mean']:.1f}")
    print(f"  Median: {stats['team_minutes_total']['median']:.1f}")
    print(f"\nZero-minute rate: {100*stats['zero_minute_rate']:.1f}%")
    if stats["ot_team_game_rate"] is not None:
        print(f"OT team-game rate: {100*stats['ot_team_game_rate']:.1f}%")

    if validation_results["warnings"]:
        print("\nWarnings:")
        for w in validation_results["warnings"]:
            print(f"  - {w}")

    print("\nOutput Files:")
    print(f"  Full dataset: {full_path}")
    print(f"  No-OT dataset: {no_ot_path}")
    print(f"  Summary: {summary_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
