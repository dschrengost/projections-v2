"""
Train Neural Network models for usage shares prediction.

Architecture:
- Embedding layers for categorical features
- Shared trunk (MLP) producing shared latent
- 3 heads (one per target) producing logits per player
- Per-group softmax normalization
- KL divergence loss with per-target validity masking

Usage:
    uv run python -m scripts.usage_shares_v1.train_nn \\
        --data-root /home/daniel/projections-data \\
        --targets fga,tov \\
        --start-date 2024-11-01 \\
        --end-date 2025-02-01 \\
        --epochs 30
"""

from __future__ import annotations

import json
import subprocess
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
from torch.utils.data import DataLoader, Dataset

from projections.paths import data_path
from projections.usage_shares_v1.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    GROUP_COLS,
    NUMERIC_COLS,
    add_derived_features,
    build_category_maps,
)
from projections.usage_shares_v1.metrics import (
    TargetMetrics,
    check_odds_leakage,
    compute_metrics,
)

app = typer.Typer(add_completion=False, help=__doc__)


# =============================================================================
# Data Loading
# =============================================================================


def load_training_data(
    data_root: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load usage_shares_training_base partitions for the date range."""
    root = data_root / "gold" / "usage_shares_training_base"
    frames: list[pd.DataFrame] = []
    
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
    
    if not frames:
        raise FileNotFoundError(
            f"No usage_shares_training_base partitions found for {start_date.date()}..{end_date.date()}"
        )
    
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df


def get_git_sha() -> str | None:
    """Get current git SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


# =============================================================================
# Dataset and Model
# =============================================================================


@dataclass
class GroupedBatch:
    """A batch of grouped data for NN training."""
    X_num: torch.Tensor       # [batch, max_players, n_numeric]
    X_cat: torch.Tensor       # [batch, max_players, n_cat]
    mask: torch.Tensor        # [batch, max_players] - 1 for real, 0 for pad
    shares: torch.Tensor      # [batch, max_players, n_targets]
    valid: torch.Tensor       # [batch, n_targets] - 1 if target valid for this group
    group_ids: list[tuple[int, int]]  # (game_id, team_id) for each group
    row_indices: list[list[int]]  # Original df row indices per group


class GroupedDataset(Dataset):
    """Dataset that groups players by (game_id, team_id)."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        category_maps: dict[str, dict[Any, int]],
        categorical_cols: list[str],
        targets: list[str],
        scaler_params: dict[str, tuple[float, float]] | None = None,
    ):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.category_maps = category_maps
        self.targets = targets
        self.df = df.reset_index(drop=True)  # Keep original for metrics
        
        # Compute or use provided scaler params
        if scaler_params is None:
            self.scaler_params = {}
            for col in numeric_cols:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                    self.scaler_params[col] = (float(vals.mean()), float(vals.std() + 1e-6))
                else:
                    self.scaler_params[col] = (0.0, 1.0)
        else:
            self.scaler_params = scaler_params
        
        # Group by (game_id, team_id)
        self.groups: list[pd.DataFrame] = []
        self.group_ids: list[tuple[int, int]] = []
        self.row_indices: list[list[int]] = []
        
        for (game_id, team_id), group_df in df.groupby(GROUP_COLS):
            self.groups.append(group_df)
            self.group_ids.append((int(game_id), int(team_id)))
            self.row_indices.append(group_df.index.tolist())
    
    def __len__(self) -> int:
        return len(self.groups)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int], list[int]]:
        group_df = self.groups[idx]
        n_players = len(group_df)
        
        # Numeric features - scaled
        X_num = np.zeros((n_players, len(self.numeric_cols)), dtype=np.float32)
        for i, col in enumerate(self.numeric_cols):
            if col in group_df.columns:
                vals = pd.to_numeric(group_df[col], errors="coerce").fillna(0.0).values
            else:
                vals = np.zeros(n_players)
            mean, std = self.scaler_params.get(col, (0.0, 1.0))
            X_num[:, i] = (vals - mean) / std
        
        # Categorical features - encoded with UNK=0
        X_cat = np.zeros((n_players, len(self.categorical_cols)), dtype=np.int64)
        for i, col in enumerate(self.categorical_cols):
            col_map = self.category_maps.get(col, {})
            if col in group_df.columns:
                X_cat[:, i] = group_df[col].map(lambda x: col_map.get(x, 0)).fillna(0).astype(int).values
        
        # Shares per target
        shares = np.zeros((n_players, len(self.targets)), dtype=np.float32)
        valid = np.zeros(len(self.targets), dtype=np.float32)
        
        for t, target in enumerate(self.targets):
            share_col = f"share_{target}"
            valid_col = f"share_{target}_valid"
            
            if share_col in group_df.columns:
                shares[:, t] = group_df[share_col].fillna(0.0).values
            
            # Check validity - NaN validity treated as valid if shares are finite
            if valid_col in group_df.columns:
                # If explicit valid column exists, use it (NaN = assume valid)
                explicit_flags = group_df[valid_col].fillna(True)
                valid[t] = 1.0 if explicit_flags.all() else 0.0
            else:
                # No validity column - valid if share is finite
                valid[t] = 1.0 if np.isfinite(shares[:, t]).all() else 0.0
        
        return X_num, X_cat, shares, valid, self.group_ids[idx], self.row_indices[idx]


def collate_groups(batch: list) -> GroupedBatch:
    """Collate grouped data with padding."""
    X_nums, X_cats, shares_list, valids, group_ids, row_indices = zip(*batch)
    
    batch_size = len(batch)
    max_players = max(x.shape[0] for x in X_nums)
    n_numeric = X_nums[0].shape[1]
    n_cat = X_cats[0].shape[1]
    n_targets = shares_list[0].shape[1]
    
    # Pad to max_players
    X_num_padded = np.zeros((batch_size, max_players, n_numeric), dtype=np.float32)
    X_cat_padded = np.zeros((batch_size, max_players, n_cat), dtype=np.int64)
    shares_padded = np.zeros((batch_size, max_players, n_targets), dtype=np.float32)
    mask = np.zeros((batch_size, max_players), dtype=np.float32)
    valid_batch = np.stack(valids, axis=0)
    
    for i, (xn, xc, s) in enumerate(zip(X_nums, X_cats, shares_list)):
        n = xn.shape[0]
        X_num_padded[i, :n] = xn
        X_cat_padded[i, :n] = xc
        shares_padded[i, :n] = s
        mask[i, :n] = 1.0
    
    return GroupedBatch(
        X_num=torch.from_numpy(X_num_padded),
        X_cat=torch.from_numpy(X_cat_padded),
        mask=torch.from_numpy(mask),
        shares=torch.from_numpy(shares_padded),
        valid=torch.from_numpy(valid_batch),
        group_ids=list(group_ids),
        row_indices=list(row_indices),
    )


class UsageSharesNN(nn.Module):
    """Neural network for usage shares prediction."""
    
    def __init__(
        self,
        n_numeric: int,
        n_cat: int,
        vocab_sizes: list[int],
        embed_dim: int = 8,
        hidden_sizes: list[int] | None = None,
        n_targets: int = 3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        
        # Embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, min(embed_dim, (vocab_size + 1) // 2 + 1))
            for vocab_size in vocab_sizes
        ])
        embed_total = sum(e.embedding_dim for e in self.embeddings)
        
        input_dim = n_numeric + embed_total
        
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        
        self.heads = nn.ModuleList([
            nn.Linear(prev_dim, 1) for _ in range(n_targets)
        ])
        
        self.n_targets = n_targets
    
    def forward(self, X_num: torch.Tensor, X_cat: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits [batch, max_players, n_targets]."""
        embeds = []
        for i, embed_layer in enumerate(self.embeddings):
            cat_col = X_cat[:, :, i]
            embeds.append(embed_layer(cat_col))
        
        if embeds:
            x = torch.cat([X_num] + embeds, dim=-1)
        else:
            x = X_num
        
        x = self.trunk(x)
        logits = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=-1)
        return logits


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply softmax within groups, masking padded positions."""
    masked_logits = logits + (1 - mask) * (-1e9)
    return F.softmax(masked_logits, dim=-1) * mask


def kl_loss(share_true: torch.Tensor, share_pred: torch.Tensor, mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """KL divergence loss per group."""
    true_safe = share_true + eps
    pred_safe = share_pred + eps
    kl_per_player = mask * true_safe * torch.log(true_safe / pred_safe)
    return kl_per_player.sum(dim=-1)


# =============================================================================
# Evaluation using shared metrics
# =============================================================================


def evaluate_with_shared_metrics(
    model: UsageSharesNN,
    dataloader: DataLoader,
    df: pd.DataFrame,
    targets: list[str],
    device: torch.device,
    alpha: float,
) -> dict[str, TargetMetrics]:
    """
    Evaluate using the SAME metrics code as LGBM.
    
    Converts NN logits to log-weights and uses compute_metrics() from shared module.
    """
    model.eval()
    
    # Collect predictions by row index
    all_log_weights: dict[str, dict[int, float]] = {t: {} for t in targets}
    
    with torch.no_grad():
        for batch in dataloader:
            X_num = batch.X_num.to(device)
            X_cat = batch.X_cat.to(device)
            row_indices = batch.row_indices
            
            logits = model(X_num, X_cat)  # [batch, max_players, n_targets]
            
            for b, indices in enumerate(row_indices):
                n_players = len(indices)
                for t, target in enumerate(targets):
                    player_logits = logits[b, :n_players, t].cpu().numpy()
                    for player_idx, row_idx in enumerate(indices):
                        all_log_weights[target][row_idx] = float(player_logits[player_idx])
    
    # Compute metrics per target using shared function
    metrics = {}
    for target in targets:
        valid_col = f"share_{target}_valid"
        if valid_col in df.columns:
            valid_mask = df[valid_col].fillna(False).astype(bool)
            target_df = df[valid_mask].copy()
        else:
            target_df = df.copy()
        
        if len(target_df) == 0:
            continue
        
        # Get log-weights for valid rows
        log_weights = np.array([all_log_weights[target].get(i, 0.0) for i in target_df.index])
        
        # Use shared metrics function (same as LGBM)
        metrics[target] = compute_metrics(target_df, log_weights, target, alpha)
    
    return metrics


# =============================================================================
# Training Loop
# =============================================================================


def train_epoch(
    model: UsageSharesNN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    targets: list[str],
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {t: 0.0 for t in targets}
    n_batches = {t: 0 for t in targets}
    
    for batch in dataloader:
        X_num = batch.X_num.to(device)
        X_cat = batch.X_cat.to(device)
        mask = batch.mask.to(device)
        shares = batch.shares.to(device)
        valid = batch.valid.to(device)
        
        optimizer.zero_grad()
        logits = model(X_num, X_cat)
        
        total_loss = torch.tensor(0.0, device=device)
        
        for t, target in enumerate(targets):
            share_pred = masked_softmax(logits[:, :, t], mask)
            share_true = shares[:, :, t]
            kl = kl_loss(share_true, share_pred, mask)
            target_valid = valid[:, t]
            if target_valid.sum() > 0:
                loss_t = (kl * target_valid).sum() / target_valid.sum()
                total_loss = total_loss + loss_t
                total_losses[target] += loss_t.item()
                n_batches[target] += 1
        
        total_loss.backward()
        optimizer.step()
    
    avg_losses = {}
    for t in targets:
        avg_losses[t] = total_losses[t] / n_batches[t] if n_batches[t] > 0 else 0.0
    
    return avg_losses


# =============================================================================
# Main Command
# =============================================================================


@app.command()
def main(
    data_root: Path = typer.Option(None),
    run_id: str = typer.Option(None),
    start_date: str = typer.Option(...),
    end_date: str = typer.Option(...),
    targets: str = typer.Option("fga,tov"),
    alpha: float = typer.Option(0.5),
    min_minutes_actual: float = typer.Option(4.0),
    val_days: int = typer.Option(30),
    seed: int = typer.Option(1337),
    epochs: int = typer.Option(30),
    batch_groups: int = typer.Option(64),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(1e-4),
    hidden_sizes: str = typer.Option("128,64"),
    embed_dim: int = typer.Option(8),
) -> None:
    """Train neural network models for usage shares prediction."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    root = data_root or data_path()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    target_list = [t.strip() for t in targets.split(",")]
    hidden_list = [int(h.strip()) for h in hidden_sizes.split(",")]
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"[nn] Using device: {device}")
    
    typer.echo(f"[nn] Loading data from {start.date()} to {end.date()}...")
    df = load_training_data(root, start, end)
    typer.echo(f"[nn] Loaded {len(df):,} rows")
    
    df = add_derived_features(df)
    
    # Leakage check
    n_leaky, n_checked, missing_frac = check_odds_leakage(df)
    typer.echo(f"[nn] Odds leakage check: {n_leaky}/{n_checked} rows have odds_as_of_ts > tip_ts")
    typer.echo(f"[nn] Odds timestamp missing for {missing_frac:.1%} of rows")
    if n_leaky > 0:
        typer.echo("[nn] ERROR: Detected leaky odds! Aborting.")
        raise typer.Exit(1)
    
    if "minutes_actual" in df.columns:
        df = df[df["minutes_actual"] >= min_minutes_actual].copy()
        typer.echo(f"[nn] Filtered to {len(df):,} rows with minutes_actual >= {min_minutes_actual}")
    
    available_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    available_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        typer.echo(f"[nn] Warning: missing features: {missing_features}")
        for col in missing_features:
            df[col] = 0.0 if col in NUMERIC_COLS else -1
        available_numeric = list(NUMERIC_COLS)
        available_cat = list(CATEGORICAL_COLS)
    
    typer.echo(f"[nn] Using {len(available_numeric)} numeric, {len(available_cat)} categorical features")
    
    category_maps = build_category_maps(df, available_cat)
    vocab_sizes = [len(category_maps.get(col, {})) for col in available_cat]
    typer.echo(f"[nn] Category vocab sizes: {dict(zip(available_cat, vocab_sizes))}")
    
    unique_dates = sorted(df["game_date"].unique())
    if len(unique_dates) <= val_days:
        raise ValueError(f"Not enough dates: {len(unique_dates)} <= {val_days}")
    
    val_start_date = unique_dates[-val_days]
    train_df = df[df["game_date"] < val_start_date].copy()
    val_df = df[df["game_date"] >= val_start_date].copy()
    
    val_dates = sorted(val_df["game_date"].unique())
    typer.echo(f"[nn] Train: {len(train_df):,} rows, Val: {len(val_df):,} rows")
    typer.echo(f"[nn] Val dates: {val_dates[0].date()} to {val_dates[-1].date()} ({len(val_dates)} days)")
    
    train_dataset = GroupedDataset(train_df, available_numeric, category_maps, available_cat, target_list)
    val_dataset = GroupedDataset(val_df, available_numeric, category_maps, available_cat, target_list, scaler_params=train_dataset.scaler_params)
    
    typer.echo(f"[nn] Train groups: {len(train_dataset)}, Val groups: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_groups, shuffle=True, collate_fn=collate_groups)
    val_loader = DataLoader(val_dataset, batch_size=batch_groups, shuffle=False, collate_fn=collate_groups)
    
    model = UsageSharesNN(
        n_numeric=len(available_numeric),
        n_cat=len(available_cat),
        vocab_sizes=vocab_sizes,
        embed_dim=embed_dim,
        hidden_sizes=hidden_list,
        n_targets=len(target_list),
    ).to(device)
    
    typer.echo(f"[nn] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_kl = float("inf")
    best_epoch = 0
    
    for epoch in range(epochs):
        train_losses = train_epoch(model, train_loader, optimizer, target_list, device)
        val_metrics = evaluate_with_shared_metrics(model, val_loader, val_df, target_list, device, alpha)
        
        avg_val_kl = np.mean([m.KL for m in val_metrics.values()]) if val_metrics else 0
        
        if avg_val_kl < best_val_kl:
            best_val_kl = avg_val_kl
            best_epoch = epoch
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            loss_str = " ".join([f"{t}={loss:.4f}" for t, loss in train_losses.items()])
            typer.echo(f"[nn] Epoch {epoch+1}/{epochs}: train_loss=[{loss_str}] val_KL={avg_val_kl:.4f}")
    
    typer.echo(f"\n[nn] Best epoch: {best_epoch + 1} with val_KL={best_val_kl:.4f}")
    
    final_metrics = evaluate_with_shared_metrics(model, val_loader, val_df, target_list, device, alpha)
    all_metrics: dict[str, dict[str, Any]] = {}
    
    typer.echo("\n=== SUMMARY (same baseline as LGBM) ===")
    for target in target_list:
        if target in final_metrics:
            val_m = final_metrics[target]
            status = "✓" if val_m.share_MAE < val_m.share_MAE_baseline else "✗"
            improvement = (1 - val_m.share_MAE / val_m.share_MAE_baseline) * 100
            typer.echo(
                f"{target}: {status} MAE={val_m.share_MAE:.4f} "
                f"(baseline={val_m.share_MAE_baseline:.4f}, {improvement:+.1f}%) "
                f"KL={val_m.KL:.4f} top1={val_m.top1_acc:.2%}"
            )
            all_metrics[target] = {"val": val_m.to_dict()}
    
    # Save artifacts
    artifacts_dir = root / "artifacts" / "usage_shares_v1" / "runs" / run_id
    nn_dir = artifacts_dir / "nn"
    nn_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = nn_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    typer.echo(f"[nn] Saved model to {model_path}")
    
    nn_config = {
        "n_numeric": len(available_numeric),
        "n_cat": len(available_cat),
        "numeric_cols": available_numeric,
        "categorical_cols": available_cat,
        "vocab_sizes": vocab_sizes,
        "category_maps": {col: {str(k): v for k, v in m.items()} for col, m in category_maps.items()},
        "embed_dim": embed_dim,
        "hidden_sizes": hidden_list,
        "n_targets": len(target_list),
        "targets": target_list,
    }
    (nn_dir / "nn_config.json").write_text(json.dumps(nn_config, indent=2))
    (nn_dir / "scaler.json").write_text(json.dumps(train_dataset.scaler_params, indent=2))
    
    feature_cols_path = artifacts_dir / "feature_columns.json"
    if not feature_cols_path.exists():
        feature_cols_path.write_text(json.dumps({
            "feature_cols": available_numeric + available_cat,
            "numeric_cols": available_numeric,
            "categorical_cols": available_cat,
        }, indent=2))
    
    metrics_path = artifacts_dir / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        for t, m in all_metrics.items():
            existing.setdefault(t, {})["nn_val"] = m["val"]
        metrics_path.write_text(json.dumps(existing, indent=2))
    else:
        metrics_path.write_text(json.dumps(all_metrics, indent=2))
    
    meta = {
        "run_id": run_id,
        "backend": "nn",
        "git_sha": get_git_sha(),
        "date_range": [start.date().isoformat(), end.date().isoformat()],
        "val_split": {"method": "tail_days", "n_days": val_days, "val_dates": [d.date().isoformat() for d in val_dates]},
        "alpha": alpha,
        "min_minutes_actual": min_minutes_actual,
        "seed": seed,
        "epochs": epochs,
        "targets": target_list,
        "n_train_groups": len(train_dataset),
        "n_val_groups": len(val_dataset),
        "leakage_check": {"n_leaky": n_leaky, "n_checked": n_checked, "missing_frac": round(missing_frac, 4)},
        "created_at": datetime.now().isoformat(),
    }
    (nn_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    
    typer.echo(f"[nn] Artifacts saved to {artifacts_dir}")


if __name__ == "__main__":
    app()
