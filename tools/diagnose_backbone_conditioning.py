#!/usr/bin/env python3
"""Diagnose backbone game-environment conditioning.

Checks section 16.6 validation gates from GAME_TRANSFORMER_SPEC.md:
  1. mu_P.std across held-out games must be > 3.0  (was 0.60 in production)
  2. Correlation(vegas_total, backbone_poss_mean) must be > 0.7
  3. Rate spread across teams should be meaningfully larger than production baseline

Usage:
  uv run python tools/diagnose_backbone_conditioning.py \
    --run-dir $PROJECTIONS_DATA_ROOT/training/runs/<run> \
    --dataset-dir $PROJECTIONS_DATA_ROOT/training/datasets/<dataset> \
    --val-days 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from projections.rotation.game_transformer_v2 import (
    GameTransformerV2Config,
    build_game_transformer_v2,
)
from scripts.rotation.eval_game_transformer_v2 import (
    _coerce_join_keys,
    _resolve_dataset_dir,
    _resolve_run_dir,
    _split_val,
)
from scripts.rotation.train_game_transformer_v2 import (
    GameLevelDataset,
    build_game_level_examples,
    collate_game_level_examples,
)


def _extract_backbone_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    vegas_total_idx: int | None,
    estimated_poss_idx: int | None,
) -> pd.DataFrame:
    """Run forward pass and extract per-game backbone outputs."""
    model.eval()
    rows: list[dict[str, float]] = []

    with torch.no_grad():
        for batch in loader:
            player_features = batch["player_features"].to(device=device)
            player_valid_mask = batch["player_valid_mask"].to(device=device)
            game_features = batch["game_features"].to(device=device)
            team_features = batch["team_features"].to(device=device)

            out = model(
                player_features,
                player_valid_mask,
                game_features=game_features,
                team_features=team_features,
                sample_active=False,
                run_flow=False,
                sample_backbone=False,
            )

            gfeat = batch["game_features"].cpu().numpy()
            game_ids = batch["game_id_norm"]

            for b_idx in range(player_features.shape[0]):
                row: dict[str, float] = {
                    "game_id": str(game_ids[b_idx]),
                }

                if vegas_total_idx is not None:
                    row["vegas_total"] = float(gfeat[b_idx, vegas_total_idx])
                if estimated_poss_idx is not None:
                    row["estimated_poss"] = float(gfeat[b_idx, estimated_poss_idx])

                if out.possession is not None:
                    row["mu_P"] = float(out.possession.mu[b_idx].cpu())
                    row["sigma_P"] = float(out.possession.sigma[b_idx].cpu())
                    row["df_P"] = float(out.possession.df[b_idx].cpu())

                if out.backbone is not None:
                    for stat_name, tensor in [
                        ("fta_rate", out.backbone.fta),
                        ("tov_rate", out.backbone.tov),
                        ("oreb_rate", out.backbone.oreb),
                    ]:
                        if tensor is not None:
                            for side, side_label in [(0, "home"), (1, "away")]:
                                row[f"{stat_name}_{side_label}"] = float(
                                    tensor[b_idx, side].cpu()
                                )

                rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose backbone conditioning")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)

    config = GameTransformerV2Config.load(run_dir / "config.json")
    model = build_game_transformer_v2(config)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    device = torch.device(str(args.device))
    model = model.to(device=device)
    model.eval()

    print(f"[config] backbone={config.enable_possession_backbone}, three_pa={config.enable_three_pa_share}")
    if not config.enable_possession_backbone:
        print("WARN: backbone not enabled on this checkpoint — nothing to diagnose.")
        return

    features_df = _coerce_join_keys(
        pd.read_parquet(dataset_dir / "features.parquet"), name="features"
    )
    labels_minutes_df = _coerce_join_keys(
        pd.read_parquet(dataset_dir / "labels_minutes.parquet"), name="labels_minutes"
    )
    join_keys = ["game_id", "team_id", "player_id", "game_date"]
    label_overlap = [
        c for c in labels_minutes_df.columns
        if c in features_df.columns and c not in join_keys
    ]
    labels_for_merge = labels_minutes_df.drop(columns=label_overlap)
    merged = features_df.merge(labels_for_merge, on=join_keys, how="left", validate="one_to_one")

    from projections.rotation.set_model import zfill_game_id_series
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])
    val_df = _split_val(merged, val_days=int(args.val_days))
    print(f"[data] val games={val_df['game_id'].nunique()}, val rows={len(val_df)}")

    examples = build_game_level_examples(
        val_df,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        minutes_label_col="minutes_label" if "minutes_label" in val_df.columns else "minutes",
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
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_game_level_examples,
    )

    vegas_total_idx = None
    if "vegas_total" in config.game_feature_columns:
        vegas_total_idx = int(config.game_feature_columns.index("vegas_total"))
    estimated_poss_idx = None
    if "estimated_possessions" in config.game_feature_columns:
        estimated_poss_idx = int(config.game_feature_columns.index("estimated_possessions"))

    df = _extract_backbone_outputs(
        model,
        loader,
        device=device,
        vegas_total_idx=vegas_total_idx,
        estimated_poss_idx=estimated_poss_idx,
    )

    print(f"\n{'='*70}")
    print("SECTION 16.6 VALIDATION GATES")
    print(f"{'='*70}")

    # Gate 1: mu_P.std across held-out games must be > 3.0
    mu_std = df["mu_P"].std()
    mu_mean = df["mu_P"].mean()
    mu_min = df["mu_P"].min()
    mu_max = df["mu_P"].max()
    gate1_pass = mu_std > 3.0
    print(f"\nGate 1: mu_P.std > 3.0")
    print(f"  mu_P: mean={mu_mean:.2f}, std={mu_std:.2f}, min={mu_min:.2f}, max={mu_max:.2f}")
    print(f"  Production baseline: std=0.60")
    print(f"  Result: {'PASS' if gate1_pass else 'FAIL'} (std={mu_std:.4f})")

    # Gate 2: Correlation(vegas_total, backbone_poss_mean) > 0.7
    if "vegas_total" in df.columns:
        valid = df[["vegas_total", "mu_P"]].dropna()
        if len(valid) > 2:
            corr = valid["vegas_total"].corr(valid["mu_P"])
            gate2_pass = corr > 0.7
            print(f"\nGate 2: corr(vegas_total, mu_P) > 0.7")
            print(f"  Correlation: {corr:.4f}")
            print(f"  Production baseline: r=0.046")
            print(f"  Result: {'PASS' if gate2_pass else 'FAIL'}")
        else:
            print("\nGate 2: SKIP (insufficient data)")
            gate2_pass = False
    else:
        print("\nGate 2: SKIP (no vegas_total in features)")
        gate2_pass = False

    # Bonus: sigma_P distribution
    print(f"\nBonus: sigma_P distribution")
    print(f"  mean={df['sigma_P'].mean():.2f}, std={df['sigma_P'].std():.2f}, "
          f"min={df['sigma_P'].min():.2f}, max={df['sigma_P'].max():.2f}")

    # Bonus: rate spread
    rate_cols = [c for c in df.columns if c.endswith("_home") or c.endswith("_away")]
    if rate_cols:
        print(f"\nBonus: backbone rate spread (all team-sides)")
        for base in ["fta_rate", "tov_rate", "oreb_rate"]:
            home_col = f"{base}_home"
            away_col = f"{base}_away"
            if home_col in df.columns and away_col in df.columns:
                all_vals = pd.concat([df[home_col], df[away_col]])
                print(f"  {base}: mean={all_vals.mean():.4f}, std={all_vals.std():.4f}, "
                      f"range=[{all_vals.min():.4f}, {all_vals.max():.4f}], "
                      f"span={all_vals.max() - all_vals.min():.4f}")

    # Bonus: correlation of estimated_possessions with mu_P
    if "estimated_poss" in df.columns:
        valid = df[["estimated_poss", "mu_P"]].dropna()
        if len(valid) > 2:
            corr_ep = valid["estimated_poss"].corr(valid["mu_P"])
            print(f"\nBonus: corr(estimated_poss, mu_P) = {corr_ep:.4f}")

    print(f"\n{'='*70}")
    overall = gate1_pass and gate2_pass
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    if not gate1_pass:
        print("  -> mu_P.std too low: backbone is still near-unconditional")
    if not gate2_pass:
        print("  -> vegas_total correlation too low: backbone not responding to game env")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
