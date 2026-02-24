#!/usr/bin/env python3
"""Evaluate GameTransformerV2 with lineup-state parity and game-volume calibration slices."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from projections import paths
from projections.rotation.game_transformer_v2 import (
    GameLevelDataset,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
)
from projections.rotation.set_model import zfill_game_id_series

JOIN_KEYS = ["game_id", "team_id", "player_id", "game_date"]


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _coerce_join_keys(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["game_id", "team_id", "player_id"]:
        if col not in out.columns:
            raise ValueError(f"{name} missing key column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    if "game_date" not in out.columns:
        raise ValueError(f"{name} missing key column: game_date")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()

    invalid = out[["game_id", "team_id", "player_id", "game_date"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"{name} has invalid key rows: {int(invalid.sum())}")
    return out


def _resolve_dataset_dir(value: str | None) -> Path:
    root = paths.get_data_root() / "training" / "datasets"
    if value:
        p = Path(value).expanduser()
        if p.exists():
            return p.resolve()
        p2 = root / value
        if p2.exists():
            return p2.resolve()
        raise FileNotFoundError(f"Dataset directory not found: {value}")

    candidates = sorted(root.glob("joint_rotation_rates_v1*"))
    if not candidates:
        raise FileNotFoundError(f"No joint_rotation_rates_v1* datasets found under {root}")
    return candidates[-1].resolve()


def _resolve_run_dir(value: str) -> Path:
    p = Path(value).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Run directory not found: {value}")
    if not (p / "config.json").exists() or not (p / "model.pt").exists():
        raise FileNotFoundError(f"Run directory missing config.json/model.pt: {p}")
    return p.resolve()


def _split_val(df: pd.DataFrame, *, val_days: int) -> pd.DataFrame:
    days = sorted(pd.to_datetime(df["game_date"]).dropna().dt.normalize().unique().tolist())
    vd = max(1, int(val_days))
    val_dates = set(days[-vd:])
    return df.loc[pd.to_datetime(df["game_date"]).dt.normalize().isin(val_dates)].copy()


def _flatten_side_np(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("expected shape (B,2,N)")
    return np.concatenate([x[:, 0], x[:, 1]], axis=1)


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    active_threshold: float,
    estimated_possessions_idx: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()

    player_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []

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
            )

            pred_minutes = out.minutes.minutes.detach().cpu().numpy().reshape(-1, 2, 15)
            pred_active = out.active.active_mask.detach().cpu().numpy().reshape(-1, 2, 15)
            valid = batch["player_valid_mask"].cpu().numpy().astype(bool)
            y_minutes = batch["y_minutes"].cpu().numpy()
            lineup_available = batch["lineup_available"].cpu().numpy().astype(bool)
            player_ids = batch["player_ids"].cpu().numpy()
            team_ids = batch["team_ids"].cpu().numpy()
            game_ids = batch["game_id_norm"]
            game_dates = batch["game_date"]

            gfeat = batch["game_features"].cpu().numpy()

            for b_idx in range(pred_minutes.shape[0]):
                est_possessions = (
                    float(gfeat[b_idx, estimated_possessions_idx])
                    if estimated_possessions_idx is not None and gfeat.shape[1] > estimated_possessions_idx
                    else float("nan")
                )
                for side in (0, 1):
                    team_id = int(team_ids[b_idx, side])
                    if team_id <= 0:
                        continue

                    side_valid = valid[b_idx, side]
                    side_pred_active = pred_active[b_idx, side] & side_valid
                    side_actual_active = (y_minutes[b_idx, side] >= float(active_threshold)) & side_valid
                    team_rows.append(
                        {
                            "game_id_norm": str(game_ids[b_idx]),
                            "game_date": str(game_dates[b_idx]),
                            "team_id": team_id,
                            "pred_active_count": int(side_pred_active.sum()),
                            "actual_active_count": int(side_actual_active.sum()),
                            "estimated_possessions": est_possessions,
                        }
                    )

                    for slot in range(15):
                        if not bool(side_valid[slot]):
                            continue
                        player_rows.append(
                            {
                                "game_id_norm": str(game_ids[b_idx]),
                                "game_date": str(game_dates[b_idx]),
                                "team_id": team_id,
                                "player_id": int(player_ids[b_idx, side, slot]),
                                "lineup_available": int(lineup_available[b_idx, side, slot]),
                                "minutes_pred": float(pred_minutes[b_idx, side, slot]),
                                "minutes_actual": float(y_minutes[b_idx, side, slot]),
                                "active_pred": int(bool(pred_active[b_idx, side, slot])),
                                "active_actual": int(bool(y_minutes[b_idx, side, slot] >= float(active_threshold))),
                            }
                        )

    return pd.DataFrame(player_rows), pd.DataFrame(team_rows)


def _lineup_parity_metrics(player_df: pd.DataFrame) -> dict[str, Any]:
    if player_df.empty:
        return {}

    out: dict[str, Any] = {}
    for flag in [1, 0]:
        subset = player_df.loc[player_df["lineup_available"] == flag]
        key = "lineup_available_1" if flag == 1 else "lineup_available_0"
        if subset.empty:
            out[key] = {"n": 0, "minutes_mae": float("nan"), "active_acc": float("nan")}
            continue
        mae = (subset["minutes_pred"] - subset["minutes_actual"]).abs().mean()
        active_acc = (subset["active_pred"] == subset["active_actual"]).mean()
        out[key] = {
            "n": int(len(subset)),
            "minutes_mae": float(mae),
            "active_acc": float(active_acc),
        }

    mae1 = out["lineup_available_1"]["minutes_mae"]
    mae0 = out["lineup_available_0"]["minutes_mae"]
    if np.isfinite(mae1) and np.isfinite(mae0):
        out["minutes_mae_gap_abs"] = float(abs(mae1 - mae0))
    else:
        out["minutes_mae_gap_abs"] = float("nan")
    return out


def _active_count_calibration(team_df: pd.DataFrame) -> dict[str, Any]:
    if team_df.empty:
        return {}

    err = team_df["pred_active_count"] - team_df["actual_active_count"]
    mae = err.abs().mean()
    bias = err.mean()
    rmse = np.sqrt(np.mean(np.square(err)))

    by_pred = (
        team_df.groupby("pred_active_count", dropna=False)
        .agg(n=("actual_active_count", "size"), actual_mean=("actual_active_count", "mean"))
        .reset_index()
        .sort_values("pred_active_count")
    )

    return {
        "n_team_games": int(len(team_df)),
        "mae": float(mae),
        "bias": float(bias),
        "rmse": float(rmse),
        "by_pred_active_count": by_pred.to_dict(orient="records"),
    }


def _actual_game_possessions(labels_boxscore_counts_df: pd.DataFrame) -> pd.DataFrame:
    req = {"game_id", "team_id", "game_date", "fga2", "fga3", "fta", "oreb", "tov"}
    missing = sorted(req - set(labels_boxscore_counts_df.columns))
    if missing:
        raise ValueError(f"labels_boxscore_counts missing columns: {missing}")

    df = labels_boxscore_counts_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    for col in ["fga2", "fga3", "fta", "oreb", "tov"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["fga"] = df["fga2"] + df["fga3"]
    df["team_possessions"] = df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"]

    team_game = (
        df.groupby(["game_id", "team_id", "game_date"], as_index=False)
        .agg(team_possessions=("team_possessions", "sum"))
    )
    game = (
        team_game.groupby(["game_id", "game_date"], as_index=False)
        .agg(actual_possessions=("team_possessions", "mean"), n_teams=("team_id", "nunique"))
    )
    game = game.loc[game["n_teams"] >= 2].copy()
    game["game_id_norm"] = zfill_game_id_series(game["game_id"])
    game["game_date"] = pd.to_datetime(game["game_date"]).dt.date.astype(str)
    return game[["game_id_norm", "game_date", "actual_possessions"]]


def _possessions_calibration(team_df: pd.DataFrame, labels_boxscore_counts_df: pd.DataFrame) -> dict[str, Any]:
    if team_df.empty:
        return {}

    pred_game = (
        team_df.groupby(["game_id_norm", "game_date"], as_index=False)
        .agg(estimated_possessions=("estimated_possessions", "mean"))
    )
    actual_game = _actual_game_possessions(labels_boxscore_counts_df)
    merged = pred_game.merge(actual_game, on=["game_id_norm", "game_date"], how="inner")

    if merged.empty:
        return {
            "n_games": 0,
            "mae": float("nan"),
            "bias": float("nan"),
            "rmse": float("nan"),
            "deciles": [],
        }

    err = merged["estimated_possessions"] - merged["actual_possessions"]
    mae = err.abs().mean()
    bias = err.mean()
    rmse = np.sqrt(np.mean(np.square(err)))

    work = merged.copy()
    if work["estimated_possessions"].nunique() >= 10:
        work["decile"] = pd.qcut(work["estimated_possessions"], q=10, labels=False, duplicates="drop")
        deciles = (
            work.groupby("decile", as_index=False)
            .agg(
                n=("actual_possessions", "size"),
                pred_mean=("estimated_possessions", "mean"),
                actual_mean=("actual_possessions", "mean"),
            )
            .sort_values("decile")
            .to_dict(orient="records")
        )
    else:
        deciles = []

    return {
        "n_games": int(len(merged)),
        "mae": float(mae),
        "bias": float(bias),
        "rmse": float(rmse),
        "deciles": deciles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--val-days", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--active-threshold", type=float, default=1.0)
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-player-preds", type=str, default=None)
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

    features_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "features.parquet"), name="features")
    labels_minutes_df = _coerce_join_keys(pd.read_parquet(dataset_dir / "labels_minutes.parquet"), name="labels_minutes")
    labels_boxscore_counts_df = _coerce_join_keys(
        pd.read_parquet(dataset_dir / "labels_boxscore_counts.parquet"),
        name="labels_boxscore_counts",
    )

    label_overlap = [c for c in labels_minutes_df.columns if c in features_df.columns and c not in JOIN_KEYS]
    labels_for_merge = labels_minutes_df.drop(columns=label_overlap)
    merged = features_df.merge(labels_for_merge, on=JOIN_KEYS, how="left", validate="one_to_one")
    merged["game_id_norm"] = zfill_game_id_series(merged["game_id"])
    val_df = _split_val(merged, val_days=int(args.val_days))

    examples = build_game_level_examples(
        val_df,
        feature_columns=list(config.feature_columns),
        feature_mean=np.asarray(config.feature_mean, dtype=np.float32),
        feature_std=np.asarray(config.feature_std, dtype=np.float32),
        game_feature_columns=list(config.game_feature_columns),
        team_feature_columns=list(config.team_feature_columns),
        minutes_label_col="minutes_label" if "minutes_label" in val_df.columns else "minutes",
    )

    loader = DataLoader(
        GameLevelDataset(examples),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_game_level_examples,
    )

    estimated_possessions_idx = None
    if "estimated_possessions" in config.game_feature_columns:
        estimated_possessions_idx = int(config.game_feature_columns.index("estimated_possessions"))

    player_df, team_df = _predict(
        model,
        loader,
        device=device,
        active_threshold=float(args.active_threshold),
        estimated_possessions_idx=estimated_possessions_idx,
    )

    lineup_parity = _lineup_parity_metrics(player_df)
    active_count_cal = _active_count_calibration(team_df)
    possessions_cal = _possessions_calibration(team_df, labels_boxscore_counts_df)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "val_days": int(args.val_days),
        "active_threshold": float(args.active_threshold),
        "n_player_rows": int(len(player_df)),
        "n_team_rows": int(len(team_df)),
        "lineup_state_parity": lineup_parity,
        "game_volume_calibration": {
            "active_count": active_count_cal,
            "possessions_proxy": possessions_cal,
        },
    }

    if args.out_player_preds:
        out_player = Path(args.out_player_preds).expanduser()
        out_player.parent.mkdir(parents=True, exist_ok=True)
        player_df.to_parquet(out_player, index=False)
        payload["out_player_preds"] = str(out_player)

    out_json = Path(args.out_json).expanduser() if args.out_json else (run_dir / f"eval_slices_{_utc_now_compact()}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"saved_eval_json={out_json}")


if __name__ == "__main__":
    main()
