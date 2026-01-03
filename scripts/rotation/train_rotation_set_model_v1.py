#!/usr/bin/env python3
"""Train a permutation-invariant TEAM-SET minutes model on rotation_train_v1.

This trains on team-games: each example is a (game_id, team_id) with a variable-size set of players.

Model variants:
  - deepsets: φ(player) -> mean pool -> ρ([φ_i, pool]) -> per-player logits
  - settransformer: attention blocks over player embeddings -> per-player logits

Output minutes are constrained to sum to 240 per team-game:
  logits -> softplus -> normalize to 240 (mask-aware)

OT / label totals handling:
  - Labels are rescaled per team-game to sum exactly to 240 minutes.

Example:
  uv run python scripts/rotation/train_rotation_set_model_v1.py \
    --dataset-dir /home/daniel/projections-data/training/datasets/rotation_train_v1_20260103 \
    --model deepsets --epochs 2 --batch-size 32 --lr 1e-3 --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Add project root to path for script execution from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.rotation.set_model import (  # noqa: E402
    RotationSetModelConfig,
    build_model,
    zfill_game_id_series,
)


TEAM_TOTAL_MINUTES = 240.0


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)  # noqa: S603,S607
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _infer_label_column(labels_df: pd.DataFrame) -> str:
    candidates = [c for c in labels_df.columns if c.lower() in {"minutes", "min", "target_minutes"}]
    for c in candidates:
        if c in labels_df.columns:
            return c
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


def _infer_feature_columns(features_df: pd.DataFrame, *, labels_df: pd.DataFrame, label_col: str) -> list[str]:
    cols: list[str] = []
    excluded = {"game_id", "team_id", "player_id", "game_id_norm", label_col}
    excluded.update(set(labels_df.columns))
    for col in features_df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(features_df[col]) or pd.api.types.is_bool_dtype(features_df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns found in features dataframe")
    return cols


def _coerce_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["game_id_norm"] = zfill_game_id_series(out["game_id"])
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    if out["game_id_norm"].isna().any() or out["team_id"].isna().any() or out["player_id"].isna().any():
        raise ValueError("Invalid keys found after coercion")
    return out


def _rescale_labels_to_240(df: pd.DataFrame, *, label_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    sums = out.groupby(["game_id_norm", "team_id"], sort=False)[label_col].sum(min_count=1)
    sums = sums.rename("team_minutes_sum")
    out = out.merge(sums.reset_index(), on=["game_id_norm", "team_id"], how="left")
    if out["team_minutes_sum"].isna().any():
        raise ValueError("Found team-games with missing label totals after grouping")
    bad = out["team_minutes_sum"] <= 0
    if bad.any():
        out = out.loc[~bad].copy()
    scale = TEAM_TOTAL_MINUTES / out["team_minutes_sum"].astype("float64")
    out[label_col] = out[label_col].astype("float64") * scale

    # OT diagnostics (regulation tolerance only for reporting).
    tol_lo, tol_hi = 238.0, 242.0
    unique_sums = sums.dropna()
    ot_rate = float(((unique_sums < tol_lo) | (unique_sums > tol_hi)).mean()) if len(unique_sums) else float("nan")
    payload = {
        "label_rescale": {"type": "per_team_game", "target_sum": TEAM_TOTAL_MINUTES},
        "regulation_tolerance": [tol_lo, tol_hi],
        "team_games_total": int(len(unique_sums)),
        "team_games_ot_rate": ot_rate,
    }
    return out, payload


def _split_by_game_date(df: pd.DataFrame, *, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1)")

    if "game_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        game_dates = df.loc[:, ["game_id_norm", "game_date"]].drop_duplicates()
        game_dates = game_dates.dropna(subset=["game_date"]).sort_values(["game_date", "game_id_norm"])
        sort_key = "game_date"
    else:
        game_dates = df.loc[:, ["game_id_norm"]].drop_duplicates().sort_values(["game_id_norm"])
        game_dates["game_date"] = pd.NaT
        sort_key = "game_id_norm"

    n_games = len(game_dates)
    n_val = max(1, int(round(n_games * val_frac)))
    val_games = set(game_dates.tail(n_val)["game_id_norm"].astype(str))

    train_df = df.loc[~df["game_id_norm"].astype(str).isin(val_games)].copy()
    val_df = df.loc[df["game_id_norm"].astype(str).isin(val_games)].copy()

    meta: dict[str, Any] = {
        "split": {"type": "time", "sort_key": sort_key, "val_frac": float(val_frac)},
        "counts": {"games_total": int(n_games), "games_val": int(n_val)},
    }
    if "game_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        meta["date_ranges"] = {
            "train_min": train_df["game_date"].min().date().isoformat() if len(train_df) else None,
            "train_max": train_df["game_date"].max().date().isoformat() if len(train_df) else None,
            "val_min": val_df["game_date"].min().date().isoformat() if len(val_df) else None,
            "val_max": val_df["game_date"].max().date().isoformat() if len(val_df) else None,
        }
    return train_df, val_df, meta


def _numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    frame = df.loc[:, cols].copy()
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.fillna(0.0)


@dataclass(frozen=True)
class TeamGameExample:
    x: np.ndarray
    y: np.ndarray


class TeamGameDataset(Dataset[TeamGameExample]):
    def __init__(self, examples: list[TeamGameExample]) -> None:
        if not examples:
            raise ValueError("TeamGameDataset requires at least one example")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TeamGameExample:
        return self.examples[idx]


def _collate_team_games(batch: list[TeamGameExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_n = max(ex.x.shape[0] for ex in batch)
    num_features = batch[0].x.shape[1]
    x = torch.zeros((len(batch), max_n, num_features), dtype=torch.float32)
    y = torch.zeros((len(batch), max_n), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_n), dtype=torch.bool)
    for i, ex in enumerate(batch):
        n = ex.x.shape[0]
        x[i, :n] = torch.from_numpy(ex.x)
        y[i, :n] = torch.from_numpy(ex.y)
        mask[i, :n] = True
    return x, y, mask


def _build_team_game_examples(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> list[TeamGameExample]:
    feats = _numeric_frame(df, feature_cols).to_numpy(dtype="float32", copy=False)
    feats = (feats - feature_mean) / feature_std
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0).to_numpy(dtype="float32", copy=False)

    examples: list[TeamGameExample] = []
    for _, idx in df.groupby(["game_id_norm", "team_id"], sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        x = feats[idx_arr]
        y = labels[idx_arr]
        examples.append(TeamGameExample(x=x, y=y))
    return examples


def _masked_mae(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - y).abs() * mask.to(dtype=pred.dtype)
    denom = mask.sum().clamp(min=1)
    return err.sum() / denom


def _evaluate(model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> dict[str, float]:
    model.eval()
    total_abs = 0.0
    total_count = 0.0
    team_maes: list[float] = []
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            pred = model(x, mask)
            abs_err = (pred - y).abs() * mask.to(dtype=pred.dtype)
            total_abs += float(abs_err.sum().item())
            total_count += float(mask.sum().item())
            per_team = abs_err.sum(dim=1) / mask.sum(dim=1).clamp(min=1).to(dtype=abs_err.dtype)
            team_maes.extend(per_team.cpu().numpy().tolist())

    mae = total_abs / max(total_count, 1.0)
    team_mae = float(np.mean(team_maes)) if team_maes else float("nan")
    return {"mae": float(mae), "team_mae": float(team_mae)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/daniel/projections-data/training/datasets/rotation_train_v1_20260103",
        help="Path containing features.parquet and labels.parquet.",
    )
    parser.add_argument("--model", type=str, choices=["deepsets", "settransformer"], default="deepsets")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="artifacts/rotation_set_minutes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--max-team-games", type=int, default=None, help="Optional cap for faster smoke runs.")
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(args.device)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    features_path = dataset_dir / "features.parquet"
    labels_path = dataset_dir / "labels.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features.parquet at {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.parquet at {labels_path}")

    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)
    label_col = _infer_label_column(labels_df)
    target_col = "target_minutes"

    features_df = _coerce_keys(features_df)
    labels_df = _coerce_keys(labels_df)
    labels_keep = labels_df.loc[:, ["game_id_norm", "team_id", "player_id", label_col]].rename(
        columns={label_col: target_col}
    )
    merged = features_df.merge(labels_keep, on=["game_id_norm", "team_id", "player_id"], how="inner")

    merged, ot_meta = _rescale_labels_to_240(merged, label_col=target_col)

    feature_cols = _infer_feature_columns(features_df, labels_df=labels_df, label_col=target_col)
    train_df, val_df, split_meta = _split_by_game_date(merged, val_frac=float(args.val_frac))
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if args.max_team_games is not None:
        max_tg = int(args.max_team_games)
        if max_tg <= 0:
            raise ValueError("--max-team-games must be positive")
        train_groups = list(train_df.groupby(["game_id_norm", "team_id"], sort=False).indices.items())
        val_groups = list(val_df.groupby(["game_id_norm", "team_id"], sort=False).indices.items())
        random.Random(args.seed).shuffle(train_groups)
        random.Random(args.seed).shuffle(val_groups)
        train_keep = train_groups[:max_tg]
        val_keep = val_groups[: max(1, int(round(max_tg * 0.25)))]
        train_idx = np.concatenate([np.asarray(idx, dtype=np.int64) for _, idx in train_keep]) if train_keep else np.array([], dtype=np.int64)
        val_idx = np.concatenate([np.asarray(idx, dtype=np.int64) for _, idx in val_keep]) if val_keep else np.array([], dtype=np.int64)
        train_df = train_df.iloc[train_idx].copy()
        val_df = val_df.iloc[val_idx].copy()
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    train_frame = _numeric_frame(train_df, feature_cols)
    mean = train_frame.mean(axis=0).to_numpy(dtype="float32", copy=False)
    std = train_frame.std(axis=0, ddof=0).to_numpy(dtype="float32", copy=False)
    std = np.where(std < 1e-6, 1.0, std).astype("float32")

    train_examples = _build_team_game_examples(
        train_df, feature_cols=feature_cols, label_col=target_col, feature_mean=mean, feature_std=std
    )
    val_examples = _build_team_game_examples(
        val_df, feature_cols=feature_cols, label_col=target_col, feature_mean=mean, feature_std=std
    )

    train_loader = DataLoader(
        TeamGameDataset(train_examples),
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=_collate_team_games,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    val_loader = DataLoader(
        TeamGameDataset(val_examples),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=_collate_team_games,
        num_workers=0,
    )

    config = RotationSetModelConfig(
        model=args.model,
        feature_columns=feature_cols,
        feature_mean=mean.astype(float).tolist(),
        feature_std=std.astype(float).tolist(),
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, mask)
            loss = _masked_mae(pred, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        train_eval = _evaluate(model, train_loader, device=device)
        val_eval = _evaluate(model, val_loader, device=device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(batches, 1),
            "train_mae": train_eval["mae"],
            "val_mae": val_eval["mae"],
            "val_team_mae": val_eval["team_mae"],
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] loss={row['train_loss']:.4f} train_mae={row['train_mae']:.4f} "
            f"val_mae={row['val_mae']:.4f} val_team_mae={row['val_team_mae']:.4f}"
        )

        if row["val_mae"] < best_val:
            best_val = row["val_mae"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best_state captured")

    run_id = f"{config.version}_{args.model}_{_utc_now_compact()}"
    out_root = Path(args.out_dir)
    run_dir = (out_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, run_dir / "model.pt")
    config.save(run_dir / "config.json")
    (run_dir / "feature_columns.json").write_text(
        json.dumps({"columns": feature_cols}, indent=2, sort_keys=True), encoding="utf-8"
    )

    final_metrics = {"best_val_mae": float(best_val), "final_epoch": int(args.epochs)}
    final_metrics.update({f"last_{k}": v for k, v in history[-1].items() if k != "epoch"})

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "dataset_dir": str(dataset_dir),
        "inputs": {"features": str(features_path), "labels": str(labels_path)},
        "model": config.to_dict(),
        "label_handling": ot_meta,
        "split": split_meta,
        "counts": {
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "team_games_train": int(len(train_examples)),
            "team_games_val": int(len(val_examples)),
        },
        "metrics": final_metrics,
        "history": history,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("\n[rotation_set_minutes] Done")
    print(f"  run_id: {run_id}")
    print(f"  best_val_mae: {best_val:.4f}")
    print(f"  artifacts: {run_dir}")


if __name__ == "__main__":
    main()
