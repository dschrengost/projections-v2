"""
Train NN RESIDUAL models for usage shares prediction.

Similar to LGBM residual: trains to predict delta from baseline, then applies shrinkage.
Uses group-aware batching with KL loss over team-games.

Usage:
    uv run python -m scripts.usage_shares_v1.train_nn_residual \
        --data-root /home/daniel/projections-data \
        --start-date 2024-10-22 \
        --end-date 2025-11-28 \
        --val-start 2025-10-29 \
        --targets fga \
        --epochs 30 \
        --seed 1337
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader

from projections.paths import data_path
from projections.usage_shares_v1.features import GROUP_COLS, add_derived_features
from projections.usage_shares_v1.metrics import compute_baseline_log_weights, compute_metrics

app = typer.Typer(add_completion=False, help=__doc__)


# Starterless feature set (matches decision report exactly)
NUMERIC_FEATURES = [
    "minutes_pred_p50",
    "minutes_pred_play_prob",
    "minutes_pred_p50_team_scaled",
    "minutes_pred_team_sum_invalid",
    "minutes_pred_team_rank",
    "position_flags_PG",
    "position_flags_SG",
    "position_flags_SF",
    "position_flags_PF",
    "position_flags_C",
    "spread_close",
    "total_close",
    "team_itt",
    "opp_itt",
    "has_odds",
    "odds_lead_time_minutes",
    "vac_min_szn",
    "vac_fga_szn",
    "vac_min_guard_szn",
    "vac_min_wing_szn",
    "vac_min_big_szn",
    "vac_min_szn_x_minutes_rank",
    "season_fga_per_min",
    "season_fta_per_min",
    "season_tov_per_min",
]


class ResidualMLP(nn.Module):
    """Simple MLP to predict delta from baseline log-weights."""

    def __init__(
        self,
        n_numeric: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = n_numeric

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns delta prediction (B,)."""
        h = self.trunk(x)
        return self.head(h).squeeze(-1)


@dataclass
class GroupBatch:
    """A batch of team-games with variable player counts."""
    features: torch.Tensor  # (B, max_players, n_features)
    baseline_logw: torch.Tensor  # (B, max_players)
    true_shares: torch.Tensor  # (B, max_players)
    mask: torch.Tensor  # (B, max_players) - True for valid players
    n_players: list[int]  # Number of players per group


class GroupedDataset(torch.utils.data.Dataset):
    """Dataset that groups by (game_id, team_id)."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        feature_cols: list[str],
        alpha: float,
        scaler_mean: np.ndarray | None = None,
        scaler_std: np.ndarray | None = None,
    ):
        self.target = target
        self.feature_cols = feature_cols
        self.alpha = alpha

        share_col = f"share_{target}"

        # Group by team-game
        self.groups = []
        for (game_id, team_id), group in df.groupby(GROUP_COLS):
            # Get features
            X = group[feature_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)

            # Compute baseline and true
            baseline_logw = compute_baseline_log_weights(group, target, alpha)
            true_shares = group[share_col].values.astype(np.float32)

            self.groups.append({
                "features": X,
                "baseline_logw": baseline_logw.astype(np.float32),
                "true_shares": true_shares,
                "n_players": len(group),
            })

        # Compute or apply scaler
        if scaler_mean is None:
            all_features = np.vstack([g["features"] for g in self.groups])
            self.scaler_mean = np.nanmean(all_features, axis=0)
            self.scaler_std = np.nanstd(all_features, axis=0) + 1e-8
        else:
            self.scaler_mean = scaler_mean
            self.scaler_std = scaler_std

        # Apply scaling
        for g in self.groups:
            g["features"] = (g["features"] - self.scaler_mean) / self.scaler_std

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.groups[idx]


def collate_groups(batch: list[dict]) -> GroupBatch:
    """Collate variable-length groups into padded tensors."""
    max_players = max(b["n_players"] for b in batch)
    n_features = batch[0]["features"].shape[1]
    B = len(batch)

    features = torch.zeros(B, max_players, n_features)
    baseline_logw = torch.zeros(B, max_players)
    true_shares = torch.zeros(B, max_players)
    mask = torch.zeros(B, max_players, dtype=torch.bool)
    n_players = []

    for i, b in enumerate(batch):
        n = b["n_players"]
        features[i, :n] = torch.from_numpy(b["features"])
        baseline_logw[i, :n] = torch.from_numpy(b["baseline_logw"])
        true_shares[i, :n] = torch.from_numpy(b["true_shares"])
        mask[i, :n] = True
        n_players.append(n)

    return GroupBatch(
        features=features,
        baseline_logw=baseline_logw,
        true_shares=true_shares,
        mask=mask,
        n_players=n_players,
    )


def compute_kl_loss(
    delta_pred: torch.Tensor,  # (B, max_players)
    baseline_logw: torch.Tensor,  # (B, max_players)
    true_shares: torch.Tensor,  # (B, max_players)
    mask: torch.Tensor,  # (B, max_players)
    shrink: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute KL divergence loss and MAE for shares."""
    eps = 1e-9

    # Compute predicted log-weights
    logw_pred = baseline_logw + shrink * delta_pred  # (B, max_players)

    # Masked softmax: set masked positions to -inf before softmax
    logw_pred = logw_pred.masked_fill(~mask, float("-inf"))
    pred_shares = F.softmax(logw_pred, dim=-1)  # (B, max_players)

    # Clamp for numerical stability
    pred_shares = pred_shares.clamp(min=eps)
    true_clamped = true_shares.clamp(min=eps)

    # KL divergence: sum over players, mean over groups
    kl = true_clamped * (torch.log(true_clamped) - torch.log(pred_shares))
    kl = (kl * mask.float()).sum(dim=-1)  # Sum per group
    kl_loss = kl.mean()

    # MAE on shares (masked)
    mae = (torch.abs(pred_shares - true_shares) * mask.float()).sum() / mask.float().sum()

    return kl_loss, mae


def load_training_data(
    data_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load usage_shares_training_base partitions."""
    root = data_root / "gold" / "usage_shares_training_base"
    frames = []
    for season_dir in root.glob("season=*"):
        for day_dir in season_dir.glob("game_date=*"):
            try:
                day_str = day_dir.name.split("=", 1)[1]
                day = pd.Timestamp(day_str).normalize()
            except (ValueError, IndexError):
                continue
            if day < start_date or day > end_date:
                continue
            path = day_dir / "usage_shares_training_base.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


@app.command()
def main(
    data_root: Path = typer.Option(None),
    run_id: str = typer.Option(None),
    start_date: str = typer.Option("2024-10-22"),
    end_date: str = typer.Option("2025-11-28"),
    val_start: str = typer.Option("2025-10-29"),
    val_end: str = typer.Option(None),
    targets: str = typer.Option("fga"),
    alpha: float = typer.Option(0.5),
    epochs: int = typer.Option(30),
    batch_groups: int = typer.Option(32),
    lr: float = typer.Option(3e-4),
    weight_decay: float = typer.Option(1e-4),
    hidden_dims: str = typer.Option("128,64"),
    dropout: float = typer.Option(0.1),
    seed: int = typer.Option(1337),
) -> None:
    """Train NN residual-on-baseline models."""

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    val_start_ts = pd.Timestamp(val_start).normalize()
    val_end_ts = pd.Timestamp(val_end).normalize() if val_end else end
    target_list = [t.strip() for t in targets.split(",")]
    hidden = tuple(int(d) for d in hidden_dims.split(","))
    run_id = run_id or f"nn_residual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"[nn-residual] Device: {device}")

    typer.echo(f"[nn-residual] Loading data from {start.date()} to {end.date()}...")
    df = load_training_data(root, start, end)
    typer.echo(f"[nn-residual] Loaded {len(df):,} rows")

    df = add_derived_features(df)

    # Get available features
    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    typer.echo(f"[nn-residual] Using {len(feature_cols)} features")

    # Artifacts directory
    artifacts_dir = root / "artifacts" / "usage_shares_v1" / "runs" / run_id / "nn_residual"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for target in target_list:
        typer.echo(f"\n[nn-residual] Training {target}...")

        share_col = f"share_{target}"
        valid_col = f"share_{target}_valid"

        # Filter valid rows
        valid_mask = df[valid_col].fillna(True) & df[share_col].notna()
        valid_df = df[valid_mask].copy()
        valid_df = valid_df[np.isfinite(valid_df[target])].copy()

        # Split by date
        train_df = valid_df[valid_df["game_date"] < val_start_ts].copy()
        val_df = valid_df[
            (valid_df["game_date"] >= val_start_ts) & (valid_df["game_date"] <= val_end_ts)
        ].copy()

        typer.echo(f"[nn-residual] Train: {len(train_df):,} rows, Val: {len(val_df):,} rows")

        # Create datasets
        train_ds = GroupedDataset(train_df, target, feature_cols, alpha)
        val_ds = GroupedDataset(
            val_df, target, feature_cols, alpha,
            scaler_mean=train_ds.scaler_mean,
            scaler_std=train_ds.scaler_std,
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_groups, shuffle=True,
            collate_fn=collate_groups, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_groups, shuffle=False,
            collate_fn=collate_groups, num_workers=0,
        )

        # Model
        model = ResidualMLP(
            n_numeric=len(feature_cols),
            hidden_dims=hidden,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Train with shrink=1.0 (will tune shrink later)
        train_shrink = 1.0
        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_kl = 0.0
            train_mae = 0.0
            n_train_batches = 0

            for batch in train_loader:
                batch = GroupBatch(
                    features=batch.features.to(device),
                    baseline_logw=batch.baseline_logw.to(device),
                    true_shares=batch.true_shares.to(device),
                    mask=batch.mask.to(device),
                    n_players=batch.n_players,
                )

                optimizer.zero_grad()

                # Forward: flatten, predict, unflatten
                B, max_p, n_feat = batch.features.shape
                flat_x = batch.features.view(-1, n_feat)
                delta_flat = model(flat_x)
                delta = delta_flat.view(B, max_p)

                kl_loss, mae = compute_kl_loss(
                    delta, batch.baseline_logw, batch.true_shares, batch.mask, train_shrink
                )

                kl_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_kl += kl_loss.item()
                train_mae += mae.item()
                n_train_batches += 1

            scheduler.step()

            # Validate
            model.eval()
            val_kl = 0.0
            val_mae = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = GroupBatch(
                        features=batch.features.to(device),
                        baseline_logw=batch.baseline_logw.to(device),
                        true_shares=batch.true_shares.to(device),
                        mask=batch.mask.to(device),
                        n_players=batch.n_players,
                    )

                    B, max_p, n_feat = batch.features.shape
                    flat_x = batch.features.view(-1, n_feat)
                    delta_flat = model(flat_x)
                    delta = delta_flat.view(B, max_p)

                    kl_loss, mae = compute_kl_loss(
                        delta, batch.baseline_logw, batch.true_shares, batch.mask, train_shrink
                    )

                    val_kl += kl_loss.item()
                    val_mae += mae.item()
                    n_val_batches += 1

            train_kl /= n_train_batches
            train_mae /= n_train_batches
            val_kl /= n_val_batches
            val_mae /= n_val_batches

            if val_kl < best_val_loss:
                best_val_loss = val_kl
                best_epoch = epoch
                torch.save(model.state_dict(), artifacts_dir / f"model_{target}_best.pt")

            if epoch % 5 == 0 or epoch == epochs - 1:
                typer.echo(
                    f"  Epoch {epoch:3d}: train_kl={train_kl:.4f} train_mae={train_mae:.5f} "
                    f"val_kl={val_kl:.4f} val_mae={val_mae:.5f}"
                )

        typer.echo(f"  Best epoch: {best_epoch} with val_kl={best_val_loss:.4f}")

        # Load best model
        model.load_state_dict(torch.load(artifacts_dir / f"model_{target}_best.pt"))
        model.eval()

        # Grid search shrink on val
        typer.echo("[nn-residual] Grid searching shrink...")
        shrink_grid = [0.25, 0.5, 0.75, 1.0]
        best_shrink = 0.5
        best_shrink_mae = float("inf")

        for shrink in shrink_grid:
            # Compute predictions
            all_preds = []
            all_true = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = GroupBatch(
                        features=batch.features.to(device),
                        baseline_logw=batch.baseline_logw.to(device),
                        true_shares=batch.true_shares.to(device),
                        mask=batch.mask.to(device),
                        n_players=batch.n_players,
                    )

                    B, max_p, n_feat = batch.features.shape
                    flat_x = batch.features.view(-1, n_feat)
                    delta_flat = model(flat_x)
                    delta = delta_flat.view(B, max_p)

                    logw_pred = batch.baseline_logw + shrink * delta
                    logw_pred = logw_pred.masked_fill(~batch.mask, float("-inf"))
                    pred_shares = F.softmax(logw_pred, dim=-1)

                    for i in range(B):
                        n = batch.n_players[i]
                        all_preds.extend(pred_shares[i, :n].cpu().numpy())
                        all_true.extend(batch.true_shares[i, :n].cpu().numpy())

            all_preds = np.array(all_preds)
            all_true = np.array(all_true)
            mae = np.abs(all_preds - all_true).mean()

            typer.echo(f"  shrink={shrink}: MAE={mae:.5f}")

            if mae < best_shrink_mae:
                best_shrink_mae = mae
                best_shrink = shrink

        typer.echo(f"[nn-residual] Best shrink: {best_shrink}")

        # Compute final metrics with best shrink
        # Need to predict in original row order - compute row-by-row through DataFrame
        baseline_logw_val = compute_baseline_log_weights(val_df, target, alpha)
        baseline_metrics = compute_metrics(val_df, baseline_logw_val, target, alpha)

        # NN predictions - iterate through groups in sorted order matching DataFrame
        # Track original indices to reconstruct
        all_logw_pred_dict = {}  # idx -> logw_pred
        
        for (game_id, team_id), group in val_df.groupby(GROUP_COLS, sort=True):
            # Get features and scale 
            X = group[feature_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0)
            X = (X - train_ds.scaler_mean) / train_ds.scaler_std
            
            # Compute baseline for this group
            bl_logw = compute_baseline_log_weights(group, target, alpha).astype(np.float32)
            
            # NN delta prediction
            with torch.no_grad():
                X_t = torch.from_numpy(X).to(device)
                delta_pred = model(X_t).cpu().numpy()
            
            # Combine
            logw_pred = bl_logw + best_shrink * delta_pred
            
            # Store by original index
            for idx, lw in zip(group.index, logw_pred):
                all_logw_pred_dict[idx] = lw
        
        # Reconstruct in original order
        nn_logw_pred = np.array([all_logw_pred_dict[idx] for idx in val_df.index])
        nn_metrics = compute_metrics(val_df, nn_logw_pred, target, alpha)

        improvement = (1 - nn_metrics.share_MAE / baseline_metrics.share_MAE) * 100
        typer.echo(
            f"[nn-residual] {target}: MAE={nn_metrics.share_MAE:.5f} "
            f"(baseline={baseline_metrics.share_MAE:.5f}, {improvement:+.1f}%) "
            f"KL={nn_metrics.KL:.4f}"
        )

        all_results[target] = {
            "val": nn_metrics.to_dict(),
            "train": {"note": "see training logs"},
            "best_shrink": best_shrink,
            "best_epoch": best_epoch,
            "val_mae_from_grid": float(best_shrink_mae),
        }

        # Save scaler
        scaler = {
            "mean": train_ds.scaler_mean.tolist(),
            "std": train_ds.scaler_std.tolist(),
            "feature_cols": feature_cols,
        }
        (artifacts_dir / f"scaler_{target}.json").write_text(json.dumps(scaler, indent=2))

    # Save config
    config = {
        "mode": "nn_residual",
        "hidden_dims": list(hidden),
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "feature_cols": feature_cols,
        "alpha": alpha,
        "shrink_values": {t: all_results[t]["best_shrink"] for t in target_list},
    }
    (artifacts_dir / "nn_config.json").write_text(json.dumps(config, indent=2))

    # Save metrics
    (artifacts_dir.parent / "metrics_nn.json").write_text(json.dumps(all_results, indent=2))

    # Save meta
    meta = {
        "run_id": run_id,
        "backend": "nn_residual",
        "date_range": [start.date().isoformat(), end.date().isoformat()],
        "val_range": [val_start_ts.date().isoformat(), val_end_ts.date().isoformat()],
        "targets": target_list,
        "seed": seed,
        "created_at": datetime.now().isoformat(),
    }
    (artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    typer.echo(f"\n[nn-residual] Artifacts saved to {artifacts_dir}")

    # Summary
    typer.echo("\n=== SUMMARY (NN Residual Training) ===")
    for target in target_list:
        if target in all_results:
            m = all_results[target]["val"]
            s = all_results[target]["best_shrink"]
            status = "✓" if m["beats_baseline"] else "✗"
            improvement = (1 - m["share_MAE"] / m["share_MAE_baseline"]) * 100
            typer.echo(
                f"{target}: {status} shrink={s} MAE={m['share_MAE']:.5f} "
                f"(baseline={m['share_MAE_baseline']:.5f}, {improvement:+.1f}%) "
                f"KL={m['KL']:.4f}"
            )


if __name__ == "__main__":
    app()
