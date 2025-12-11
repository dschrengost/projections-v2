"""Evaluate a trained minutes NN against the canonical minutes_v1 splits."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from projections import paths
from projections.labels import derive_starter_flag_labels
from projections.metrics.minutes import compute_mae_by_actual_minutes_bucket
from projections.minutes_v1.artifacts import write_json
from projections.minutes_v1.datasets import KEY_COLUMNS, load_feature_frame
from projections.models.minutes_features import MINUTES_TARGET_COL, build_feature_spec, infer_feature_columns
from projections.models.minutes_lgbm import (
    DEFAULT_CAL_END,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VAL_END,
    _filter_out_players,
    _window_defaults,
)
from projections.models.minutes_nn import (
    MinutesFeatureSpec,
    MinutesPreprocessorState,
    MinutesQuantileMLP,
    TabularMinutesDataset,
    transform_frame,
)

app = typer.Typer(help=__doc__)


def _alpha_key(alpha: float) -> int:
    return int(round(alpha * 100))


def _quantile_column(alpha: float) -> str:
    return f"minutes_q{int(round(alpha * 100)):02d}_nn"


def _prepare_dataset(
    df: pd.DataFrame,
    *,
    spec: MinutesFeatureSpec,
    state: MinutesPreprocessorState,
    target_col: str,
) -> TabularMinutesDataset:
    x_cont, x_cat = transform_frame(df, spec, state)
    y = df[target_col].to_numpy(dtype=np.float32)
    return TabularMinutesDataset(x_cont, x_cat, y)


@torch.no_grad()
def _predict(model: MinutesQuantileMLP, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for x_cont, x_cat, _ in loader:
        preds = model(x_cont.to(device), x_cat.to(device))
        outputs.append(preds.cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, len(model.alphas)), dtype=np.float32)


def _coverage(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    if not len(y_true):
        return float("nan")
    inside = (y_true >= low) & (y_true <= high)
    return float(np.mean(inside))


def _conditional_coverage(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    mask = y_true > 0
    if not np.any(mask):
        return float("nan")
    return _coverage(y_true[mask], low[mask], high[mask])


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@app.command()
def main(
    *,
    model_dir: Path = typer.Option(..., help="Path to the trained NN artifact directory."),
    output_dir: Path | None = typer.Option(None, help="Directory for evaluation artifacts (defaults to <model_dir>/eval)."),
    data_root: Path | None = typer.Option(None, help="Override data root when loading features."),
    season: int | None = typer.Option(None, help="Override season partition if reloading features."),
    month: int | None = typer.Option(None, help="Override month partition if reloading features."),
    features: Path | None = typer.Option(None, help="Explicit feature parquet path override."),
    target_col: str | None = typer.Option(None, help="Target column override (defaults to training metadata)."),
    eval_batch_size: int = typer.Option(2048, help="Batch size for NN inference."),
    device: str | None = typer.Option(None, help="Torch device override for evaluation."),
) -> None:
    """Evaluate a trained minutes NN model on train/cal/val splits."""

    config_path = model_dir / "config.json"
    metadata_path = model_dir / "metadata.json"
    preproc_path = model_dir / "preproc_state.json"
    model_path = model_dir / "model.pt"
    if not config_path.exists() or not metadata_path.exists():
        raise typer.BadParameter(f"{model_dir} is missing config/metadata.json.")
    config = _load_json(config_path)
    metadata = _load_json(metadata_path)
    preproc_state = MinutesPreprocessorState.from_json_dict(_load_json(preproc_path))
    alphas: list[float] = list(config["alphas"])
    alpha_lookup = {_alpha_key(alpha): idx for idx, alpha in enumerate(alphas)}
    q10_idx = alpha_lookup.get(_alpha_key(0.10))
    q25_idx = alpha_lookup.get(_alpha_key(0.25))
    q50_idx = alpha_lookup.get(_alpha_key(0.50))
    q75_idx = alpha_lookup.get(_alpha_key(0.75))
    q90_idx = alpha_lookup.get(_alpha_key(0.90))
    if q50_idx is None:
        raise RuntimeError("Median quantile missing from trained model; cannot evaluate.")

    target = target_col or metadata.get("target_col", MINUTES_TARGET_COL)
    data_params = metadata.get("data_params", {})
    features_path = features or (Path(data_params["features_path"]) if data_params.get("features_path") else None)
    resolved_data_root = data_root or Path(data_params.get("data_root", paths.get_data_root()))
    resolved_season = season if season is not None else data_params.get("season")
    resolved_month = month if month is not None else data_params.get("month")
    feature_df = load_feature_frame(
         features_path=features_path,
         data_root=resolved_data_root,
         season=resolved_season,
         month=resolved_month,
    )
    feature_df = derive_starter_flag_labels(feature_df, output_col="starter_flag")

    windows_payload = metadata.get("windows")
    if not windows_payload:
        train_start = DEFAULT_TRAIN_START
        train_end = DEFAULT_TRAIN_END
        cal_start = None
        cal_end = DEFAULT_CAL_END
        val_start = None
        val_end = DEFAULT_VAL_END
    else:
        train_start = pd.Timestamp(windows_payload["train"]["start"]).to_pydatetime()
        train_end = pd.Timestamp(windows_payload["train"]["end"]).to_pydatetime()
        cal_start = pd.Timestamp(windows_payload["cal"]["start"]).to_pydatetime()
        cal_end = pd.Timestamp(windows_payload["cal"]["end"]).to_pydatetime()
        val_start = pd.Timestamp(windows_payload["val"]["start"]).to_pydatetime()
        val_end = pd.Timestamp(windows_payload["val"]["end"]).to_pydatetime()

    train_window, cal_window, val_window = _window_defaults(
        train_start=train_start,
        train_end=train_end,
        cal_start=cal_start,
        cal_end=cal_end,
        val_start=val_start,
        val_end=val_end,
    )

    split_frames = {
        "train": _filter_out_players(train_window.slice(feature_df)),
        "cal": _filter_out_players(cal_window.slice(feature_df)),
        "val": _filter_out_players(val_window.slice(feature_df)),
    }
    feature_columns = metadata.get("feature_columns") or infer_feature_columns(feature_df, target_col=target)
    spec = MinutesFeatureSpec(
        continuous=list(metadata.get("continuous_columns", [])),
        categorical=list(metadata.get("categorical_columns", [])),
    )
    if not spec.continuous and not spec.categorical:
        spec = build_feature_spec(feature_columns, categorical_columns=spec.categorical)

    dataset_loaders = {
        split: DataLoader(
            _prepare_dataset(df, spec=spec, state=preproc_state, target_col=target),
            batch_size=eval_batch_size,
            shuffle=False,
        )
        for split, df in split_frames.items()
    }

    cat_cardinalities = [len(preproc_state.categorical[col]) + 1 for col in spec.categorical]
    device_obj = torch.device(device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MinutesQuantileMLP(
        n_continuous=len(spec.continuous),
        cat_cardinalities=cat_cardinalities,
        alphas=alphas,
        hidden_dims=config["hidden_dims"],
        emb_dim=config["emb_dim"],
        dropout=config["dropout"],
    ).to(device_obj)
    state_dict = torch.load(model_path, map_location=device_obj)
    model.load_state_dict(state_dict)

    split_predictions: dict[str, np.ndarray] = {}
    prediction_frames: list[pd.DataFrame] = []
    id_columns = [col for col in (*KEY_COLUMNS, "game_date", "feature_as_of_ts") if col in feature_df.columns]

    for split, loader in dataset_loaders.items():
        preds = _predict(model, loader, device_obj)
        split_predictions[split] = preds
        df = split_frames[split].reset_index(drop=True).copy()
        output = df[id_columns + [target]].copy()
        for idx, alpha in enumerate(alphas):
            output[_quantile_column(alpha)] = preds[:, idx]
        output["split"] = split
        prediction_frames.append(output)

    metrics: dict[str, dict[str, float | dict[str, float]]] = {}
    for split, df in split_frames.items():
        preds = split_predictions[split]
        y = df[target].to_numpy(dtype=np.float32)
        split_metrics: dict[str, float | dict[str, float]] = {}
        q50 = preds[:, q50_idx]
        split_metrics["mae_q50"] = float(np.mean(np.abs(q50 - y)))
        if q10_idx is not None and q90_idx is not None:
            q10 = preds[:, q10_idx]
            q90 = preds[:, q90_idx]
            split_metrics["coverage_q10_q90"] = _coverage(y, q10, q90)
            split_metrics["cond_coverage_q10_q90"] = _conditional_coverage(y, q10, q90)
            split_metrics["mean_pi_width_q10_q90"] = float(np.mean(q90 - q10))
        if q25_idx is not None and q75_idx is not None:
            q25 = preds[:, q25_idx]
            q75 = preds[:, q75_idx]
            split_metrics["coverage_q25_q75"] = _coverage(y, q25, q75)
            split_metrics["cond_coverage_q25_q75"] = _conditional_coverage(y, q25, q75)
        bucket_metrics = compute_mae_by_actual_minutes_bucket(y, q50)
        split_metrics["mae_buckets"] = bucket_metrics
        metrics[split] = split_metrics

    output_dir = output_dir or (model_dir / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.parquet"
    pd.concat(prediction_frames, ignore_index=True).to_parquet(predictions_path, index=False)
    metrics_payload = {
        "run_id": metadata.get("run_id"),
        "alphas": alphas,
        "target_col": target,
        "metrics": metrics,
    }
    write_json(output_dir / "metrics.json", metrics_payload)

    summary_lines = [
        "# Minutes NN Evaluation",
        "",
        "| split | MAE q50 | cov q10-90 | cond cov q10-90 |",
        "| --- | --- | --- | --- |",
    ]
    for split in ("train", "cal", "val"):
        split_metrics = metrics.get(split, {})
        mae_val = split_metrics.get("mae_q50", float("nan"))
        cov_val = split_metrics.get("coverage_q10_q90", float("nan"))
        cond_val = split_metrics.get("cond_coverage_q10_q90", float("nan"))
        summary_lines.append(
            f"| {split} | {mae_val:.3f} | {cov_val:.3f} | {cond_val:.3f} |"
        )
    (output_dir / "metrics.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    typer.echo(f"Predictions saved to {predictions_path}")
    typer.echo(f"Metrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    app()
