#!/usr/bin/env python3
"""H1 experiment: Sweep flow_scale_clip values at inference time.

Tests whether the flow scale_clip ceiling (default 2.0) suppresses star stat
projections. Loads the same checkpoint with different scale_clip values and
compares predicted stat distributions.

Usage:
    uv run python scripts/experiments/gtv2_flow_clip_sweep.py \
        --checkpoint /home/daniel/projections-data/artifacts/game_transformer_v2/bundle_current \
        --dates 2026-02-10,2026-02-11 \
        --clips 2.0,3.0,4.0 \
        --n-samples 200 \
        --seed 42

Or using date range:
    uv run python scripts/experiments/gtv2_flow_clip_sweep.py \
        --checkpoint bundle_current \
        --date-from 2026-02-01 --date-to 2026-02-11 \
        --clips 2.0,3.0,4.0

Outputs saved to: artifacts/experiments/gtv2_clip_sweep/<run_id>/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure projections package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projections import paths
from projections.rotation.game_transformer_v2 import (
    FLOW_TARGET_COLUMNS_V1,
    GameLevelDataset,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
    flow_target_columns,
)
from projections.rotation.sample_worlds_v2 import (
    project_flow_stats_to_contract,
    sample_worlds_for_batch,
    summarize_worlds_to_projections,
)
from projections.rotation.set_model import zfill_game_id_series


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_checkpoint(value: str) -> Path:
    """Resolve checkpoint path from name or absolute path."""
    p = Path(value).expanduser()
    if p.exists() and (p / "config.json").exists():
        return p.resolve()
    # Try under artifacts
    artifacts = paths.get_data_root() / "artifacts" / "game_transformer_v2"
    candidate = artifacts / value
    if candidate.exists() and (candidate / "config.json").exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {value}")


def _resolve_dataset_dir() -> Path:
    """Find the latest joint_rotation_rates_v1 dataset."""
    root = paths.get_data_root() / "training" / "datasets"
    candidates = sorted(root.glob("joint_rotation_rates_v1*"))
    if not candidates:
        raise FileNotFoundError(f"No joint_rotation_rates_v1* datasets under {root}")
    return candidates[-1].resolve()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_model(
    checkpoint: Path,
    *,
    device: torch.device,
    scale_clip_override: float | None = None,
) -> tuple[GameTransformerV2Config, torch.nn.Module]:
    """Load model with optional scale_clip override."""
    config = GameTransformerV2Config.load(checkpoint / "config.json")
    model = build_game_transformer_v2(config)
    state = torch.load(checkpoint / "model.pt", map_location=device)
    model.load_state_dict(state)

    if scale_clip_override is not None:
        model.flow_head.set_scale_clip(float(scale_clip_override))

    model = model.to(device=device)
    model.eval()
    return config, model


def _run_inference_for_clip(
    checkpoint: Path,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    clip_value: float,
    n_samples: int,
    chunk_size: int,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """Run inference with a specific scale_clip value."""
    _set_seed(seed)

    config, model = _load_model(checkpoint, device=device, scale_clip_override=clip_value)
    ftc = flow_target_columns(include_pf=config.include_pf_in_flow_targets)

    # Build examples
    examples = build_game_level_examples(
        features_df,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        minutes_label_col="minutes_label" if "minutes_label" in features_df.columns else "minutes",
        overflow_protected_prior_play_prob_floor=float(config.overflow_protected_prior_play_prob_floor),
        overflow_protected_prior_minutes_floor=float(config.overflow_protected_prior_minutes_floor),
        overflow_risk_weight_consecutive_active_dnp=float(config.overflow_risk_weight_consecutive_active_dnp),
        overflow_risk_weight_active_but_dnp_rate_last10=float(config.overflow_risk_weight_active_but_dnp_rate_last10),
        overflow_risk_weight_inactive_streak_len=float(config.overflow_risk_weight_inactive_streak_len),
        overflow_keep_weight_prior_play_prob=float(config.overflow_keep_weight_prior_play_prob),
        overflow_keep_weight_prior_minutes=float(config.overflow_keep_weight_prior_minutes),
    )

    loader = DataLoader(
        GameLevelDataset(examples),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_game_level_examples,
    )

    frames: list[pd.DataFrame] = []
    for batch in loader:
        df_batch, _ = sample_worlds_for_batch(
            model,
            batch,
            device=device,
            num_worlds=n_samples,
            chunk_size=chunk_size,
            active_temperature=1.0,
            strict_contracts=False,
        )
        frames.append(df_batch)

    if not frames:
        return pd.DataFrame()

    worlds_df = pd.concat(frames, ignore_index=True)

    # Summarize to projections
    projections = summarize_worlds_to_projections(
        worlds_df,
        sim_profile=f"gtv2_clip{clip_value:.1f}",
    )
    projections["scale_clip"] = clip_value
    return projections


def _parse_dates(args: argparse.Namespace) -> list[str]:
    """Parse date arguments into list of YYYY-MM-DD strings."""
    if args.dates:
        return [d.strip() for d in args.dates.split(",")]

    if args.date_from and args.date_to:
        start = pd.to_datetime(args.date_from)
        end = pd.to_datetime(args.date_to)
        return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end)]

    raise ValueError("Must provide --dates or --date-from/--date-to")


def _generate_comparison_report(
    results_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate comparison report with tier-level bias analysis."""
    # Join with actuals
    labels_df = labels_df.copy()
    labels_df["pts"] = 2 * labels_df["fg2m"] + 3 * labels_df["fg3m"] + labels_df["ftm"]
    labels_df["reb"] = labels_df["oreb"] + labels_df["dreb"]

    joined = results_df.merge(
        labels_df[["game_id", "player_id", "pts", "reb", "ast", "minutes"]].rename(
            columns={"pts": "actual_pts", "reb": "actual_reb", "ast": "actual_ast", "minutes": "actual_minutes"}
        ),
        on=["game_id", "player_id"],
        how="left",
    )

    # Define tiers
    tiers = [
        ("elite_35plus", joined["actual_pts"] >= 35),
        ("star_25_34", (joined["actual_pts"] >= 25) & (joined["actual_pts"] < 35)),
        ("starter_15_24", (joined["actual_pts"] >= 15) & (joined["actual_pts"] < 25)),
        ("role_5_14", (joined["actual_pts"] >= 5) & (joined["actual_pts"] < 15)),
        ("bench_lt5", joined["actual_pts"] < 5),
    ]

    # Compute metrics per clip and tier
    report: dict[str, Any] = {
        "created_at": _utc_now_compact(),
        "clips": sorted(results_df["scale_clip"].unique().tolist()),
        "tier_metrics": {},
        "overall_metrics": {},
    }

    for clip in report["clips"]:
        clip_df = joined[joined["scale_clip"] == clip]
        clip_key = f"clip_{clip:.1f}"
        report["tier_metrics"][clip_key] = {}

        for tier_name, mask in tiers:
            tier_df = clip_df[mask]
            if len(tier_df) == 0:
                continue

            bias = (tier_df["pts_mean"] - tier_df["actual_pts"]).mean()
            mae = (tier_df["pts_mean"] - tier_df["actual_pts"]).abs().mean()
            report["tier_metrics"][clip_key][tier_name] = {
                "n": int(len(tier_df)),
                "actual_mean": float(tier_df["actual_pts"].mean()),
                "pred_mean": float(tier_df["pts_mean"].mean()),
                "bias": float(bias),
                "mae": float(mae),
            }

        # Overall
        overall_bias = (clip_df["pts_mean"] - clip_df["actual_pts"]).mean()
        overall_mae = (clip_df["pts_mean"] - clip_df["actual_pts"]).abs().mean()
        report["overall_metrics"][clip_key] = {
            "n": int(len(clip_df)),
            "bias": float(overall_bias),
            "mae": float(overall_mae),
        }

    # Write JSON
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Write markdown summary
    md_lines = [
        "# GTv2 Flow Scale Clip Sweep Results",
        "",
        f"**Generated**: {report['created_at']}",
        f"**Clips tested**: {report['clips']}",
        "",
        "## Tier-Level Bias (PTS)",
        "",
        "| Tier | " + " | ".join(f"Clip {c:.1f} Bias" for c in report["clips"]) + " |",
        "|------|" + "|".join(["-------:"] * len(report["clips"])) + "|",
    ]

    for tier_name, _ in tiers:
        row = f"| {tier_name} |"
        for clip in report["clips"]:
            clip_key = f"clip_{clip:.1f}"
            tier_data = report["tier_metrics"].get(clip_key, {}).get(tier_name, {})
            if tier_data:
                row += f" {tier_data['bias']:+.1f} |"
            else:
                row += " - |"
        md_lines.append(row)

    md_lines.extend([
        "",
        "## Tier-Level MAE (PTS)",
        "",
        "| Tier | " + " | ".join(f"Clip {c:.1f} MAE" for c in report["clips"]) + " |",
        "|------|" + "|".join(["-------:"] * len(report["clips"])) + "|",
    ])

    for tier_name, _ in tiers:
        row = f"| {tier_name} |"
        for clip in report["clips"]:
            clip_key = f"clip_{clip:.1f}"
            tier_data = report["tier_metrics"].get(clip_key, {}).get(tier_name, {})
            if tier_data:
                row += f" {tier_data['mae']:.1f} |"
            else:
                row += " - |"
        md_lines.append(row)

    md_lines.extend([
        "",
        "## Key Finding",
        "",
        "Compare elite (35+ pts) bias across clips. If higher clips reduce negative bias,",
        "scale_clip is a contributing factor to star under-projection.",
    ])

    md_path = output_dir / "comparison_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="bundle_current",
        help="Path or name of GTv2 checkpoint (default: bundle_current)",
    )
    parser.add_argument(
        "--dates",
        type=str,
        default=None,
        help="Comma-separated dates (YYYY-MM-DD) to evaluate",
    )
    parser.add_argument("--date-from", type=str, default=None, help="Start date for range")
    parser.add_argument("--date-to", type=str, default=None, help="End date for range")
    parser.add_argument(
        "--clips",
        type=str,
        default="2.0,3.0,4.0",
        help="Comma-separated scale_clip values to test (default: 2.0,3.0,4.0)",
    )
    parser.add_argument("--n-samples", type=int, default=200, help="Number of worlds per game (default: 200)")
    parser.add_argument("--chunk-size", type=int, default=50, help="Chunk size for world sampling (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: artifacts/experiments/gtv2_clip_sweep/<run_id>)",
    )
    parser.add_argument(
        "--use-val-split",
        action="store_true",
        help="Use validation split from training dataset instead of live features",
    )
    parser.add_argument("--val-days", type=int, default=14, help="Days for val split (default: 14)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    checkpoint = _resolve_checkpoint(args.checkpoint)
    print(f"Checkpoint: {checkpoint}")

    clips = [float(c.strip()) for c in args.clips.split(",")]
    print(f"Scale clips to test: {clips}")

    # Setup output directory
    run_id = _utc_now_compact()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = paths.get_data_root() / "artifacts" / "experiments" / "gtv2_clip_sweep" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    device = torch.device(args.device)

    # Load dataset
    dataset_dir = _resolve_dataset_dir()
    print(f"Dataset: {dataset_dir}")

    features_df = pd.read_parquet(dataset_dir / "features.parquet")
    labels_minutes = pd.read_parquet(dataset_dir / "labels_minutes.parquet")
    labels_box = pd.read_parquet(dataset_dir / "labels_boxscore_counts.parquet")

    # Merge
    join_keys = ["game_id", "team_id", "player_id", "game_date"]
    label_overlap = [c for c in labels_minutes.columns if c in features_df.columns and c not in join_keys]
    merged = features_df.merge(labels_minutes.drop(columns=label_overlap), on=join_keys, how="left")
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])
    merged["game_date"] = pd.to_datetime(merged["game_date"])

    # Filter to dates
    if args.use_val_split:
        # Use last N days as val
        days = sorted(merged["game_date"].dropna().dt.normalize().unique())
        val_dates = set(days[-args.val_days :])
        filtered = merged[merged["game_date"].dt.normalize().isin(val_dates)].copy()
        print(f"Using val split: {len(val_dates)} days, {len(filtered)} rows")
    else:
        dates = _parse_dates(args)
        date_set = set(pd.to_datetime(dates).normalize())
        filtered = merged[merged["game_date"].dt.normalize().isin(date_set)].copy()
        print(f"Filtering to {len(dates)} dates: {dates}")

    if len(filtered) == 0:
        print("ERROR: No data after filtering")
        sys.exit(1)

    print(f"Games: {filtered['game_id'].nunique()}, Players: {len(filtered)}")

    # Run inference for each clip value
    all_results: list[pd.DataFrame] = []
    for clip in clips:
        print(f"\n{'='*60}")
        print(f"Running inference with scale_clip = {clip}")
        print(f"{'='*60}")

        proj_df = _run_inference_for_clip(
            checkpoint,
            filtered,
            labels_box,
            clip_value=clip,
            n_samples=args.n_samples,
            chunk_size=args.chunk_size,
            seed=args.seed,
            device=device,
        )

        if len(proj_df) > 0:
            # Save per-clip results
            clip_dir = output_dir / f"clip_{clip:.1f}"
            clip_dir.mkdir(parents=True, exist_ok=True)
            proj_df.to_parquet(clip_dir / "projections.parquet", index=False)
            all_results.append(proj_df)
            print(f"Saved {len(proj_df)} projections to {clip_dir}")

    if not all_results:
        print("ERROR: No results generated")
        sys.exit(1)

    # Combine results
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_parquet(output_dir / "all_projections.parquet", index=False)

    # Generate comparison report
    print(f"\n{'='*60}")
    print("Generating comparison report...")
    print(f"{'='*60}")

    report = _generate_comparison_report(
        combined,
        labels_box,
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Tier-level PTS bias by scale_clip")
    print("=" * 80)
    print(f"{'Tier':<20} " + " ".join(f"{'clip=' + str(c):<12}" for c in clips))
    print("-" * 80)

    tier_names = ["elite_35plus", "star_25_34", "starter_15_24", "role_5_14", "bench_lt5"]
    for tier in tier_names:
        row = f"{tier:<20} "
        for clip in clips:
            clip_key = f"clip_{clip:.1f}"
            tier_data = report["tier_metrics"].get(clip_key, {}).get(tier, {})
            if tier_data:
                row += f"{tier_data['bias']:+8.1f}     "
            else:
                row += f"{'N/A':>8}     "
        print(row)

    print("\n" + "=" * 80)
    print(f"Full report: {output_dir / 'comparison_report.md'}")
    print(f"JSON data: {output_dir / 'comparison_report.json'}")
    print("=" * 80)

    # Write run metadata
    meta = {
        "created_at": run_id,
        "checkpoint": str(checkpoint),
        "clips": clips,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "device": args.device,
        "n_games": int(filtered["game_id"].nunique()),
        "n_players": int(len(filtered)),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
