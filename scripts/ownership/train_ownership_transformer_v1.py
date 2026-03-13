"""Train a slate-level transformer for ownership prediction.

This model operates at the full-slate level and allocates ownership with a
masked softmax so predictions always sum to the configured lineup slot target.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from projections.ownership_v2 import (
    DEFAULT_OWNERSHIP_TRANSFORMER_FEATURES,
    OwnershipSlateDataset,
    OwnershipSlateTransformer,
    OwnershipSlateTransformerConfig,
    collate_ownership_slates,
    merge_gtv2_embeddings,
)
from projections.paths import data_path
from projections.ownership_v1.evaluation import evaluate_predictions
from projections.ownership_v1.score import normalize_ownership_to_target_sum
from projections.ownership_v2.slate_transformer import standardize_feature_frame
from scripts.ownership.train_ownership_v1 import prepare_features, split_by_date


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_mae: float
    val_top20_mae: float
    val_top10_bias: float
    val_top10_hit: float
    val_top5_hit: float
    val_spearman: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _ownership_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target_pct: torch.Tensor,
    target_share: torch.Tensor,
    valid_mask: torch.Tensor,
    target_sum_pct: float,
    weight_top: float,
    focus_topk: int,
    focus_topk_weight: float,
    focus_under_weight: float,
) -> torch.Tensor:
    pred_pct = outputs["pred_pct"]
    log_probs = outputs["log_probs"]
    share = target_share.clamp_min(0.0)
    share = share / share.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    ce = -(share * log_probs.masked_fill(~valid_mask, 0.0)).sum(dim=-1).mean()
    weights = torch.ones_like(target_pct)
    weights = weights + (target_pct >= 10.0).to(dtype=weights.dtype) * float(weight_top)
    weights = weights + (target_pct >= 20.0).to(dtype=weights.dtype) * float(weight_top)
    pct_err = torch.abs(pred_pct - target_pct)
    weighted_mae = (pct_err * weights * valid_mask.to(dtype=pct_err.dtype)).sum() / (
        (weights * valid_mask.to(dtype=weights.dtype)).sum().clamp_min(1.0)
    )

    focus_mae = pred_pct.new_tensor(0.0)
    focus_under = pred_pct.new_tensor(0.0)
    if int(focus_topk) > 0 and (float(focus_topk_weight) > 0.0 or float(focus_under_weight) > 0.0):
        k = min(int(focus_topk), int(target_pct.shape[1]))
        top_scores = target_pct.masked_fill(~valid_mask, float("-inf"))
        top_idx = torch.topk(top_scores, k=k, dim=-1).indices
        focus_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        focus_mask.scatter_(1, top_idx, True)
        focus_mask &= valid_mask
        focus_den = focus_mask.to(dtype=pred_pct.dtype).sum().clamp_min(1.0)
        focus_mae = (pct_err * focus_mask.to(dtype=pct_err.dtype)).sum() / focus_den
        focus_under = (torch.relu(target_pct - pred_pct) * focus_mask.to(dtype=pct_err.dtype)).sum() / focus_den

    slate_sum = pred_pct.sum(dim=-1)
    sum_penalty = torch.mean(torch.abs(slate_sum - float(target_sum_pct)))
    return (
        ce
        + 0.15 * weighted_mae
        + float(focus_topk_weight) * focus_mae
        + float(focus_under_weight) * focus_under
        + 0.01 * sum_penalty
    )


def _run_epoch(
    model: OwnershipSlateTransformer,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    target_sum_pct: float,
    top_weight: float,
    focus_topk: int,
    focus_topk_weight: float,
    focus_under_weight: float,
) -> tuple[float, pd.DataFrame]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    losses: list[float] = []
    rows: list[pd.DataFrame] = []

    for batch in loader:
        features = batch["features"].to(device)
        target_pct = batch["target_pct"].to(device)
        target_share = batch["target_share"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(features, valid_mask)
            loss = _ownership_loss(
                outputs,
                target_pct=target_pct,
                target_share=target_share,
                valid_mask=valid_mask,
                target_sum_pct=target_sum_pct,
                weight_top=top_weight,
                focus_topk=focus_topk,
                focus_topk_weight=focus_topk_weight,
                focus_under_weight=focus_under_weight,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

        pred_pct = outputs["pred_pct"].detach().cpu().numpy()
        actual_pct = target_pct.detach().cpu().numpy()
        valid = valid_mask.detach().cpu().numpy().astype(bool)
        for i, slate_id in enumerate(batch["slate_id"]):
            n = int(valid[i].sum())
            slate_df = pd.DataFrame(
                {
                    "slate_id": slate_id,
                    "game_date": batch["game_date"][i],
                    "player_name": batch["player_name"][i][:n],
                    "player_id": batch["player_id"][i][:n],
                    "actual_own_pct": actual_pct[i, :n],
                    "pred_own_pct": pred_pct[i, :n],
                }
            )
            slate_df["pred_own_pct"] = normalize_ownership_to_target_sum(
                slate_df["pred_own_pct"],
                target_sum_pct=float(target_sum_pct),
                cap_pct=100.0,
            ).to_numpy()
            rows.append(slate_df)

    return float(np.mean(losses)) if losses else float("nan"), pd.concat(rows, ignore_index=True)


def _topk_hit(df: pd.DataFrame, k: int) -> float:
    vals: list[float] = []
    for _, group in df.groupby("slate_id", sort=False):
        top_pred = set(group.nlargest(k, "pred_own_pct")["player_name"])
        top_actual = set(group.nlargest(k, "actual_own_pct")["player_name"])
        vals.append(len(top_pred & top_actual) / float(k))
    return float(np.mean(vals)) if vals else float("nan")


def _topk_mae(df: pd.DataFrame, k: int) -> float:
    vals: list[float] = []
    for _, group in df.groupby("slate_id", sort=False):
        top_actual = group.nlargest(k, "actual_own_pct")
        vals.append(float(np.mean(np.abs(top_actual["pred_own_pct"] - top_actual["actual_own_pct"]))))
    return float(np.mean(vals)) if vals else float("nan")


def _topk_bias(df: pd.DataFrame, k: int) -> float:
    vals: list[float] = []
    for _, group in df.groupby("slate_id", sort=False):
        top_actual = group.nlargest(k, "actual_own_pct")
        vals.append(float(np.mean(top_actual["pred_own_pct"] - top_actual["actual_own_pct"])))
    return float(np.mean(vals)) if vals else float("nan")


def _summarize_epoch(epoch: int, train_loss: float, val_loss: float, val_preds: pd.DataFrame) -> EpochResult:
    spearman = float(spearmanr(val_preds["actual_own_pct"], val_preds["pred_own_pct"]).statistic)
    mae = float(np.mean(np.abs(val_preds["pred_own_pct"] - val_preds["actual_own_pct"])))
    return EpochResult(
        epoch=int(epoch),
        train_loss=float(train_loss),
        val_loss=float(val_loss),
        val_mae=float(mae),
        val_top20_mae=float(_topk_mae(val_preds, 20)),
        val_top10_bias=float(_topk_bias(val_preds, 10)),
        val_top10_hit=float(_topk_hit(val_preds, 10)),
        val_top5_hit=float(_topk_hit(val_preds, 5)),
        val_spearman=float(spearman),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a slate-level ownership transformer")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--training-base",
        type=Path,
        default=Path("/tmp/ownership_inhouse_base_full_v3.parquet"),
    )
    parser.add_argument("--val-start-date", type=str, default="2026-01-25")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target-sum-pct", type=float, default=800.0)
    parser.add_argument("--top-weight", type=float, default=2.0)
    parser.add_argument("--focus-topk", type=int, default=10)
    parser.add_argument("--focus-topk-weight", type=float, default=0.0)
    parser.add_argument("--focus-under-weight", type=float, default=0.0)
    parser.add_argument("--gtv2-embeddings-parquet", type=Path, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("ownership_xfmr_%Y%m%dT%H%M%SZ")
    run_dir = data_path() / "artifacts" / "ownership_v2" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(int(args.seed))
    device = _resolve_device(str(args.device))

    df = pd.read_parquet(args.training_base)
    df = prepare_features(
        df,
        compute_historical=True,
        compute_slate_features=True,
        compute_value_leverage=True,
        compute_player_popularity=True,
        compute_interactions=True,
    )

    feature_columns = list(DEFAULT_OWNERSHIP_TRANSFORMER_FEATURES)
    gtv2_cols: list[str] = []
    gtv2_coverage = 0.0
    if args.gtv2_embeddings_parquet is not None:
        gtv2_df = pd.read_parquet(args.gtv2_embeddings_parquet)
        df, gtv2_cols, gtv2_coverage = merge_gtv2_embeddings(df, gtv2_df)
        feature_columns.extend(gtv2_cols)
        print(
            f"[gtv2-enrich] merged {len(gtv2_cols)} GTV2 features "
            f"with coverage={gtv2_coverage:.3f}"
        )

    train_df, val_df = split_by_date(df, args.val_start_date)
    train_df, feature_mean, feature_std = standardize_feature_frame(train_df, feature_columns=feature_columns)
    val_df, _, _ = standardize_feature_frame(
        val_df,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    train_ds = OwnershipSlateDataset(train_df, feature_columns=feature_columns, target_sum_pct=float(args.target_sum_pct))
    val_ds = OwnershipSlateDataset(val_df, feature_columns=feature_columns, target_sum_pct=float(args.target_sum_pct))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_ownership_slates)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_ownership_slates)

    max_players = max(max(item["features"].shape[0] for item in train_ds.examples), max(item["features"].shape[0] for item in val_ds.examples))
    config = OwnershipSlateTransformerConfig(
        feature_columns=feature_columns,
        feature_mean=feature_mean.astype(float).tolist(),
        feature_std=feature_std.astype(float).tolist(),
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        target_sum_pct=float(args.target_sum_pct),
        max_players=int(max_players),
    )
    model = OwnershipSlateTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state: dict[str, torch.Tensor] | None = None
    best_result: EpochResult | None = None
    best_preds: pd.DataFrame | None = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        train_loss, _ = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            target_sum_pct=float(args.target_sum_pct),
            top_weight=float(args.top_weight),
            focus_topk=int(args.focus_topk),
            focus_topk_weight=float(args.focus_topk_weight),
            focus_under_weight=float(args.focus_under_weight),
        )
        val_loss, val_preds = _run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            target_sum_pct=float(args.target_sum_pct),
            top_weight=float(args.top_weight),
            focus_topk=int(args.focus_topk),
            focus_topk_weight=float(args.focus_topk_weight),
            focus_under_weight=float(args.focus_under_weight),
        )
        result = _summarize_epoch(epoch, train_loss, val_loss, val_preds)
        history.append(result.to_dict())
        print(
            f"[epoch {epoch:02d}] train_loss={result.train_loss:.4f} "
            f"val_loss={result.val_loss:.4f} val_mae={result.val_mae:.4f} "
            f"val_top20_mae={result.val_top20_mae:.4f} val_top10_hit={result.val_top10_hit:.4f} "
            f"val_spearman={result.val_spearman:.4f}"
        )

        improved = best_result is None or (
            result.val_top20_mae < best_result.val_top20_mae - 1e-4
            or (
                abs(result.val_top20_mae - best_result.val_top20_mae) <= 1e-4
                and result.val_top10_hit > best_result.val_top10_hit + 1e-4
            )
            or (
                abs(result.val_top20_mae - best_result.val_top20_mae) <= 1e-4
                and abs(result.val_top10_hit - best_result.val_top10_hit) <= 1e-4
                and result.val_mae < best_result.val_mae
            )
        )
        if improved:
            best_result = result
            best_preds = val_preds.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                print(f"[early-stop] patience reached at epoch {epoch}")
                break

    if best_state is None or best_result is None or best_preds is None:
        raise RuntimeError("training did not produce a best checkpoint")

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "model.pt")
    config.save(run_dir / "config.json")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_preds.to_csv(run_dir / "val_predictions.csv", index=False)

    eval_res = evaluate_predictions(
        best_preds,
        slice_name=run_id,
        pred_col="pred_own_pct",
        actual_col="actual_own_pct",
        slate_id_col="slate_id",
        target_sum_pct=float(args.target_sum_pct),
        normalization="none",
    )
    metrics = {
        "best_epoch": best_result.epoch,
        "best_train_loss": best_result.train_loss,
        "best_val_loss": best_result.val_loss,
        "best_val_mae": best_result.val_mae,
        "best_val_top20_mae": best_result.val_top20_mae,
        "best_val_top10_bias": best_result.val_top10_bias,
        "best_val_top10_hit": best_result.val_top10_hit,
        "best_val_top5_hit": best_result.val_top5_hit,
        "best_val_spearman": best_result.val_spearman,
        "top5_hit": _topk_hit(best_preds, 5),
        "top10_hit": _topk_hit(best_preds, 10),
        "top20_hit": _topk_hit(best_preds, 20),
        "evaluation": eval_res.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_base": str(args.training_base),
        "val_start_date": str(args.val_start_date),
        "device": str(device),
        "seed": int(args.seed),
        "focus_topk": int(args.focus_topk),
        "focus_topk_weight": float(args.focus_topk_weight),
        "focus_under_weight": float(args.focus_under_weight),
        "gtv2_embeddings_parquet": None if args.gtv2_embeddings_parquet is None else str(args.gtv2_embeddings_parquet),
        "gtv2_feature_count": int(len(gtv2_cols)),
        "gtv2_coverage": float(gtv2_coverage),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
