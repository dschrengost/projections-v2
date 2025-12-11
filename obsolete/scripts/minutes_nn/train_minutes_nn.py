"""Train the experimental multi-quantile minutes NN on the minutes_v1 dataset."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from projections import paths
from projections.labels import derive_starter_flag_labels
from projections.minutes_v1.artifacts import compute_feature_hash, ensure_run_directory, write_json
from projections.minutes_v1.datasets import load_feature_frame
from projections.models.minutes_features import MINUTES_TARGET_COL, build_feature_spec, infer_feature_columns
from projections.models.minutes_lgbm import (
    DEFAULT_CAL_END,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VAL_END,
    DateWindow,
    _filter_out_players,
    _window_defaults,
)
from projections.models.minutes_nn import (
    MinutesFeatureSpec,
    MinutesNNConfig,
    MinutesPreprocessorState,
    MinutesQuantileMLP,
    TabularMinutesDataset,
    fit_preprocessor,
    multi_quantile_loss,
    transform_frame,
)

app = typer.Typer(help=__doc__)


def _resolve_cal_start(cal_days: int | None, train_end: datetime, val_start: datetime | None, cal_start: datetime | None) -> datetime | None:
    if cal_days is None:
        return cal_start
    if val_start is None:
        raise typer.BadParameter("--cal-days requires --val-start so the calibration window can be derived.")
    if cal_days <= 0:
        raise typer.BadParameter("--cal-days must be positive.")
    candidate = val_start - timedelta(days=cal_days)
    min_cal_start = train_end + timedelta(days=1)
    if candidate < min_cal_start:
        raise typer.BadParameter("Calibration window would overlap training; reduce --cal-days or shorten training.")
    return candidate


def _prepare_window(
    *,
    train_start: datetime,
    train_end: datetime,
    cal_start: datetime | None,
    cal_end: datetime,
    val_start: datetime | None,
    val_end: datetime,
) -> tuple[DateWindow, DateWindow, DateWindow]:
    return _window_defaults(
        train_start=train_start,
        train_end=train_end,
        cal_start=cal_start,
        cal_end=cal_end,
        val_start=val_start,
        val_end=val_end,
    )


def _prepare_loader(
    df: pd.DataFrame,
    *,
    spec: MinutesFeatureSpec,
    state: MinutesPreprocessorState,
    target_col: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_cont, x_cat = transform_frame(df, spec, state)
    y = df[target_col].to_numpy(dtype=np.float32)
    dataset = TabularMinutesDataset(x_cont, x_cat, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_epoch(
    model: MinutesQuantileMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alphas: Sequence[float],
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for x_cont, x_cat, y in loader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)
        preds = model(x_cont, x_cat)
        loss = multi_quantile_loss(preds, y, alphas)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def _evaluate_epoch(
    model: MinutesQuantileMLP,
    loader: DataLoader,
    device: torch.device,
    alphas: Sequence[float],
) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    for x_cont, x_cat, y in loader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)
        preds = model(x_cont, x_cat)
        loss = multi_quantile_loss(preds, y, alphas)
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


@app.command()
def main(
    *,
    run_id: str = typer.Option(..., "--run-id", help="Unique identifier for this NN experiment."),
    artifact_root: Path = typer.Option(Path("artifacts/minutes_nn"), help="Root directory for NN artifacts."),
    data_root: Path = typer.Option(paths.get_data_root(), help="Base data directory."),
    season: int | None = typer.Option(None, help="Season partition for canonical feature path."),
    month: int | None = typer.Option(None, help="Month partition (1-12) for canonical feature path."),
    features: Path | None = typer.Option(None, help="Explicit feature parquet path."),
    target_col: str = typer.Option(MINUTES_TARGET_COL, help="Target column to predict."),
    train_start: datetime = typer.Option(DEFAULT_TRAIN_START, help="Inclusive train window start (UTC)."),
    train_end: datetime = typer.Option(DEFAULT_TRAIN_END, help="Inclusive train window end (UTC)."),
    cal_start: datetime | None = typer.Option(None, help="Inclusive calibration window start (UTC)."),
    cal_end: datetime = typer.Option(DEFAULT_CAL_END, help="Inclusive calibration window end (UTC)."),
    val_start: datetime | None = typer.Option(None, help="Inclusive validation window start (UTC)."),
    val_end: datetime = typer.Option(DEFAULT_VAL_END, help="Inclusive validation window end (UTC)."),
    cal_days: int | None = typer.Option(None, help="If provided, derive calibration window as N days ending before --val-start."),
    categorical_col: list[str] | None = typer.Option(
        None,
        "--categorical-col",
        help="Feature column to embed (repeat flag for multiple columns). Defaults to none.",
    ),
    emb_dim: int = typer.Option(16, help="Embedding width for each categorical column."),
    hidden_dim: list[int] | None = typer.Option(
        None,
        "--hidden-dim",
        help="Hidden layer width; repeat to specify architecture (defaults to 128,128).",
    ),
    dropout: float = typer.Option(0.1, help="Dropout rate between hidden layers."),
    lr: float = typer.Option(1e-3, help="Adam learning rate."),
    batch_size: int = typer.Option(1024, help="Mini-batch size."),
    max_epochs: int = typer.Option(50, help="Maximum training epochs."),
    early_stop_patience: int = typer.Option(5, help="Early stopping patience on calibration loss."),
    weight_decay: float = typer.Option(1e-5, help="Adam weight decay (L2)."),
    device: str | None = typer.Option(None, help="Torch device override (defaults to CUDA if available)."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Train a multi-quantile MLP on the minutes_v1 dataset."""

    if not run_id:
        raise typer.BadParameter("--run-id must be provided.")
    resolved_cal_start = _resolve_cal_start(cal_days, train_end, val_start, cal_start)
    train_window, calibration_window, val_window = _prepare_window(
        train_start=train_start,
        train_end=train_end,
        cal_start=resolved_cal_start,
        cal_end=cal_end,
        val_start=val_start,
        val_end=val_end,
    )

    feature_df = load_feature_frame(
        features_path=features,
        data_root=data_root,
        season=season,
        month=month,
    )
    feature_df = derive_starter_flag_labels(feature_df, output_col="starter_flag")
    train_df = _filter_out_players(train_window.slice(feature_df))
    cal_df = _filter_out_players(calibration_window.slice(feature_df))
    val_df = _filter_out_players(val_window.slice(feature_df))
    feature_columns = infer_feature_columns(feature_df, target_col=target_col)
    cat_columns = list(categorical_col or [])
    feature_spec = build_feature_spec(feature_columns, categorical_columns=cat_columns or None)

    train_cond_df = train_df[train_df[target_col] > 0]
    cal_cond_df = cal_df[cal_df[target_col] > 0]
    if train_cond_df.empty or cal_cond_df.empty:
        raise RuntimeError("Positive-minute rows required in both train and calibration windows.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    custom_hidden_dims = list(hidden_dim or [])
    config = MinutesNNConfig(
        emb_dim=emb_dim,
        hidden_dims=custom_hidden_dims or [128, 128],
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stop_patience=early_stop_patience,
        weight_decay=weight_decay,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    preproc_state = fit_preprocessor(train_cond_df, feature_spec)
    train_loader = _prepare_loader(
        train_cond_df,
        spec=feature_spec,
        state=preproc_state,
        target_col=target_col,
        batch_size=config.batch_size,
        shuffle=True,
    )
    cal_loader = _prepare_loader(
        cal_cond_df,
        spec=feature_spec,
        state=preproc_state,
        target_col=target_col,
        batch_size=config.batch_size,
        shuffle=False,
    )

    cat_cardinalities = [len(preproc_state.categorical[col]) + 1 for col in feature_spec.categorical]
    device_obj = torch.device(config.device)
    model = MinutesQuantileMLP(
        n_continuous=len(feature_spec.continuous),
        cat_cardinalities=cat_cardinalities,
        alphas=config.alphas,
        hidden_dims=config.hidden_dims,
        emb_dim=config.emb_dim,
        dropout=config.dropout,
    ).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_cal_loss = float("inf")
    epochs_without_improvement = 0
    train_log: list[dict[str, float | int]] = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device_obj, config.alphas)
        cal_loss = _evaluate_epoch(model, cal_loader, device_obj, config.alphas)
        train_log.append({"epoch": epoch, "train_loss": train_loss, "cal_loss": cal_loss})
        typer.echo(f"[epoch {epoch:02d}] train_loss={train_loss:.5f} cal_loss={cal_loss:.5f}")
        if cal_loss + 1e-6 < best_cal_loss:
            best_cal_loss = cal_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.early_stop_patience:
            typer.echo(f"Early stopping after {epoch} epochs (no cal improvement for {config.early_stop_patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    run_dir = ensure_run_directory(run_id, root=artifact_root)
    torch.save(model.state_dict(), run_dir / "model.pt")
    write_json(run_dir / "config.json", config.to_metadata())
    write_json(run_dir / "preproc_state.json", preproc_state.to_json_dict())
    write_json(run_dir / "feature_columns.json", {"columns": feature_columns})
    pd.DataFrame(train_log).to_csv(run_dir / "train_log.csv", index=False)

    metadata = {
        "run_id": run_id,
        "feature_columns": feature_columns,
        "categorical_columns": feature_spec.categorical,
        "continuous_columns": feature_spec.continuous,
        "target_col": target_col,
        "train_rows": len(train_df),
        "cal_rows": len(cal_df),
        "val_rows": len(val_df),
        "train_positive_rows": len(train_cond_df),
        "cal_positive_rows": len(cal_cond_df),
        "alphas": config.alphas,
        "windows": {
            "train": train_window.to_metadata(),
            "cal": calibration_window.to_metadata(),
            "val": val_window.to_metadata(),
        },
        "data_params": {
            "features_path": str(features.resolve()) if features else None,
            "data_root": str(data_root),
            "season": season,
            "month": month,
        },
        "feature_hash": compute_feature_hash(feature_columns),
    }
    write_json(run_dir / "metadata.json", metadata)
    typer.echo(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    app()
