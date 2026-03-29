#!/usr/bin/env python3
"""Grouped permutation importance for GameTransformerV2 runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from projections.rotation.game_transformer_v2 import (
    GameLevelDataset,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
)
from scripts.rotation.eval_game_transformer_v2 import (
    _active_count_calibration,
    _attach_sparse_context,
    _build_sparse_context_frame,
    _coerce_join_keys,
    _lineup_parity_metrics,
    _possessions_calibration,
    _predict,
    _resolve_dataset_dir,
    _resolve_run_dir,
    _sparse_rotation_metrics,
    _split_val,
)

JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _feature_group(col: str) -> str:
    if col.startswith("ctx_") or "_ctx_same_pos_" in col:
        return "context_priors"
    if col.startswith("an_"):
        return "action_props"
    if col in {"prior_play_prob", "is_out", "status", "is_prob", "is_q"} or "dnp" in col or "inactive" in col:
        return "availability_dnp"
    if col.startswith("lineup_") or "starter" in col:
        return "lineup_starter"
    if col.startswith("team_") or col.startswith("opp_"):
        return "team_opp_ctx"
    if col in {
        "available_B",
        "available_G",
        "available_W",
        "depth_same_pos_active",
        "team_n_not_out",
        "team_n_players",
        "available_G_not_out",
        "available_W_not_out",
        "available_B_not_out",
        "depth_same_pos_not_out",
        "vacated_minutes_prior_20_total",
        "vacated_minutes_prior_20_same_pos",
        "prior_minutes_share_20",
        "role_change_starter_5v20",
        "role_change_minutes_5v20",
        "role_change_starter_5v10",
        "role_change_minutes_5v10",
        "recent_start_pct_10",
        "same_archetype_overlap",
    }:
        return "rotation_opportunity"
    if "_prior_" in col or col.endswith("_prior") or "prior_" in col:
        return "general_priors"
    if col in {
        "home_team_id",
        "away_team_id",
        "home_flag",
        "opponent_team_id",
        "restriction_flag",
        "ramp_flag",
        "spread_home",
        "total",
        "spread_home_missing",
        "total_missing",
        "blowout_index",
        "blowout_risk_score",
        "close_game_score",
        "days_since_last",
        "is_b2b",
        "is_3in4",
        "is_4in6",
    }:
        return "game_schedule_context"
    return "other"


def _group_feature_columns(feature_columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for col in feature_columns:
        groups[_feature_group(col)].append(col)
    return dict(sorted(groups.items()))


def _split_feature_groups(
    feature_groups: dict[str, list[str]],
    split_groups: set[str],
) -> dict[str, list[str]]:
    if not split_groups:
        return feature_groups
    expanded: dict[str, list[str]] = {}
    for group_name, cols in feature_groups.items():
        if group_name not in split_groups:
            expanded[group_name] = cols
            continue
        for col in cols:
            expanded[f"{group_name}::{col}"] = [col]
    return dict(sorted(expanded.items()))


def _evaluate_frame(
    model: torch.nn.Module,
    config: GameTransformerV2Config,
    features_df: pd.DataFrame,
    labels_minutes_df: pd.DataFrame,
    labels_boxscore_counts_df: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    val_days: int,
    active_threshold: float,
    low_minutes_threshold: float,
    sparse_prior_play_prob_max: float,
    sparse_prior_minutes_max: float,
    next_up_actual_min: float,
    next_up_pred_min: float,
) -> dict[str, Any]:
    label_overlap = [c for c in labels_minutes_df.columns if c in features_df.columns and c not in JOIN_KEYS]
    labels_for_merge = labels_minutes_df.drop(columns=label_overlap)
    merged = features_df.merge(labels_for_merge, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged["game_id_norm"] = pd.Series(merged["game_id"], copy=False).astype("string").str.zfill(10)
    val_df = _split_val(merged, val_days=int(val_days))

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
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        collate_fn=collate_game_level_examples,
    )
    estimated_possessions_idx = (
        int(config.game_feature_columns.index("estimated_possessions"))
        if "estimated_possessions" in config.game_feature_columns
        else None
    )
    player_df, team_df = _predict(
        model,
        loader,
        device=device,
        active_threshold=float(active_threshold),
        estimated_possessions_idx=estimated_possessions_idx,
    )
    sparse_context_df = _build_sparse_context_frame(val_df)
    player_eval_df = _attach_sparse_context(player_df, sparse_context_df)

    lineup_parity = _lineup_parity_metrics(player_eval_df)
    active_count_cal = _active_count_calibration(team_df)
    possessions_cal = _possessions_calibration(team_df, labels_boxscore_counts_df)
    sparse_diag = _sparse_rotation_metrics(
        player_eval_df,
        active_threshold=float(active_threshold),
        low_minutes_threshold=float(low_minutes_threshold),
        sparse_prior_play_prob_max=float(sparse_prior_play_prob_max),
        sparse_prior_minutes_max=float(sparse_prior_minutes_max),
        next_up_actual_min=float(next_up_actual_min),
        next_up_pred_min=float(next_up_pred_min),
    )

    overall_minutes_mae = float(
        (pd.to_numeric(player_eval_df["minutes_pred"], errors="coerce").fillna(0.0)
         - pd.to_numeric(player_eval_df["minutes_actual"], errors="coerce").fillna(0.0)).abs().mean()
    )
    starter_sparse_slice = (sparse_diag.get("slices", {}) or {}).get("starter_sparse_prior", {}) or {}
    return {
        "overall_minutes_mae": overall_minutes_mae,
        "lineup_state_parity": lineup_parity,
        "active_count_mae": float((active_count_cal or {}).get("mae", float("nan"))),
        "sparse_next_up_underpred": float(
            ((sparse_diag.get("failure_rates", {}) or {}).get("sparse_next_up_underprediction_rate", float("nan")))
        ),
        "starter_sparse_pred_minutes_mean": float(starter_sparse_slice.get("pred_minutes_mean", float("nan"))),
        "payload": {
            "game_volume_calibration": {
                "active_count": active_count_cal,
                "possessions_proxy": possessions_cal,
            },
            "sparse_rotation_diagnostics": sparse_diag,
        },
    }


def _permute_group(df: pd.DataFrame, cols: list[str], *, seed: int) -> pd.DataFrame:
    out = df.copy()
    present = [c for c in cols if c in out.columns]
    if not present:
        return out
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(out))
    for col in present:
        values = out.iloc[perm][col].to_numpy(copy=True)
        out[col] = pd.Series(values, index=out.index, dtype=out[col].dtype)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--val-days", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active-threshold", type=float, default=1.0)
    parser.add_argument("--low-minutes-threshold", type=float, default=8.0)
    parser.add_argument("--sparse-prior-play-prob-max", type=float, default=0.20)
    parser.add_argument("--sparse-prior-minutes-max", type=float, default=6.0)
    parser.add_argument("--next-up-actual-min", type=float, default=20.0)
    parser.add_argument("--next-up-pred-min", type=float, default=10.0)
    parser.add_argument("--groups", type=str, default=None, help="Comma-separated groups to evaluate. Default: all.")
    parser.add_argument(
        "--split-group",
        action="append",
        default=None,
        help="Group name to explode into per-feature permutation rows. Repeatable.",
    )
    parser.add_argument("--out-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    dataset_dir = _resolve_dataset_dir(args.dataset_dir)

    config = GameTransformerV2Config.load(run_dir / "config.json")
    model = build_game_transformer_v2(config)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    device = torch.device(str(args.device))
    model = model.to(device=device)
    model.eval()

    features_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "features.parquet"), name="features")
    labels_minutes_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_minutes.parquet"), name="labels_minutes")
    labels_boxscore_counts_df = _coerce_join_keys(
        pd.read_parquet(dataset_dir / "labels_boxscore_counts.parquet"),
        name="labels_boxscore_counts",
    )

    feature_groups = _group_feature_columns(list(config.feature_columns))
    if args.groups:
        requested = {c.strip() for c in str(args.groups).split(",") if c.strip()}
        feature_groups = {k: v for k, v in feature_groups.items() if k in requested}
    split_groups = {str(name).strip() for name in (args.split_group or []) if str(name).strip()}
    feature_groups = _split_feature_groups(feature_groups, split_groups)

    baseline = _evaluate_frame(
        model,
        config,
        features_df,
        labels_minutes_df,
        labels_boxscore_counts_df,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        val_days=int(args.val_days),
        active_threshold=float(args.active_threshold),
        low_minutes_threshold=float(args.low_minutes_threshold),
        sparse_prior_play_prob_max=float(args.sparse_prior_play_prob_max),
        sparse_prior_minutes_max=float(args.sparse_prior_minutes_max),
        next_up_actual_min=float(args.next_up_actual_min),
        next_up_pred_min=float(args.next_up_pred_min),
    )

    group_rows: list[dict[str, Any]] = []
    for idx, (group_name, cols) in enumerate(feature_groups.items()):
        permuted = _permute_group(features_df, cols, seed=int(args.seed) + idx + 1)
        scored = _evaluate_frame(
            model,
            config,
            permuted,
            labels_minutes_df,
            labels_boxscore_counts_df,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            val_days=int(args.val_days),
            active_threshold=float(args.active_threshold),
            low_minutes_threshold=float(args.low_minutes_threshold),
            sparse_prior_play_prob_max=float(args.sparse_prior_play_prob_max),
            sparse_prior_minutes_max=float(args.sparse_prior_minutes_max),
            next_up_actual_min=float(args.next_up_actual_min),
            next_up_pred_min=float(args.next_up_pred_min),
        )
        group_rows.append(
            {
                "group": group_name,
                "n_features": int(len(cols)),
                "overall_minutes_mae": float(scored["overall_minutes_mae"]),
                "delta_overall_minutes_mae": float(scored["overall_minutes_mae"] - baseline["overall_minutes_mae"]),
                "active_count_mae": float(scored["active_count_mae"]),
                "delta_active_count_mae": float(scored["active_count_mae"] - baseline["active_count_mae"]),
                "sparse_next_up_underpred": float(scored["sparse_next_up_underpred"]),
                "delta_sparse_next_up_underpred": float(
                    scored["sparse_next_up_underpred"] - baseline["sparse_next_up_underpred"]
                ),
                "starter_sparse_pred_minutes_mean": float(scored["starter_sparse_pred_minutes_mean"]),
                "delta_starter_sparse_pred_minutes_mean": float(
                    scored["starter_sparse_pred_minutes_mean"] - baseline["starter_sparse_pred_minutes_mean"]
                ),
                "feature_columns": cols,
            }
        )

    group_rows = sorted(group_rows, key=lambda row: row["delta_overall_minutes_mae"], reverse=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "val_days": int(args.val_days),
        "baseline": baseline,
        "groups": group_rows,
    }

    out_json = (
        Path(args.out_json).expanduser()
        if args.out_json
        else run_dir / f"feature_group_importance_{_utc_now_compact()}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"saved_importance_json={out_json}")


if __name__ == "__main__":
    main()
