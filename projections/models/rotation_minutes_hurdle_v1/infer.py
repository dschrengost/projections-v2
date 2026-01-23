from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
import torch

from projections.models.minutes_nn import transform_frame

from .bundle import RMHBundle, load_bundle
from .data import add_has_injury_row
from .mixture import mixture_quantiles_q10_q50_q90


app = typer.Typer(add_completion=False, help="Run RMH_v1 inference from a saved artifact bundle.")


def _ensure_feature_columns(df: pd.DataFrame, bundle: RMHBundle) -> pd.DataFrame:
    out = df.copy()
    for col in bundle.feature_spec.continuous:
        if col not in out.columns:
            out[col] = 0.0
    for col in bundle.feature_spec.categorical:
        if col not in out.columns:
            out[col] = "__nan__"
    return out


@torch.no_grad()
def predict_frame(
    df: pd.DataFrame,
    *,
    bundle: RMHBundle,
    batch_size: int = 8192,
) -> pd.DataFrame:
    """Attach RMH_v1 predictions to a frame containing model features."""

    work = add_has_injury_row(df)
    work = _ensure_feature_columns(work, bundle)

    x_cont, x_cat = transform_frame(work, bundle.feature_spec, bundle.preprocessor)

    device = str(bundle.config.get("device_resolved") or "cpu")
    model = bundle.model.to(device)
    model.eval()

    n = len(work)
    p_play = np.empty(n, dtype=np.float64)
    q10_cond = np.empty(n, dtype=np.float64)
    q50_cond = np.empty(n, dtype=np.float64)
    q90_cond = np.empty(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        xb_cont = torch.from_numpy(x_cont[start:end]).to(device)
        xb_cat = torch.from_numpy(x_cat[start:end]).to(device)
        out = model(xb_cont, xb_cat)
        p_play[start:end] = torch.sigmoid(out.logits_play).detach().cpu().numpy()
        q10_cond[start:end] = out.q10_cond.detach().cpu().numpy()
        q50_cond[start:end] = out.q50_cond.detach().cpu().numpy()
        q90_cond[start:end] = out.q90_cond.detach().cpu().numpy()

    play_threshold = float(bundle.config.get("play_threshold", 1.0))
    taus = [float(x) for x in bundle.config.get("quantiles", [0.10, 0.50, 0.90])]

    q_uncond = mixture_quantiles_q10_q50_q90(
        p_play=p_play,
        q10_cond=q10_cond,
        q50_cond=q50_cond,
        q90_cond=q90_cond,
        taus=taus,
        play_threshold=play_threshold,
    )

    out_df = work.copy()
    out_df["p_play"] = p_play
    out_df["minutes_q10_cond"] = q10_cond
    out_df["minutes_q50_cond"] = q50_cond
    out_df["minutes_q90_cond"] = q90_cond
    out_df["minutes_mean_uncond"] = p_play * q50_cond
    out_df["minutes_q10_uncond"] = q_uncond.get(0.10, q_uncond[taus[0]])
    out_df["minutes_q50_uncond"] = q_uncond.get(0.50, q_uncond[taus[1]])
    out_df["minutes_q90_uncond"] = q_uncond.get(0.90, q_uncond[taus[2]])

    return out_df


def _print_no_clamp_example() -> None:
    # A minimal log example to show the adjusted tau mapping without clamping to 0.10.
    example = {"p_play": 0.11, "tau": 0.90, "tau_pos": (0.90 - (1.0 - 0.11)) / 0.11}
    typer.echo(f"[rmh] mixture mapping example: {json.dumps(example, sort_keys=True)}")


@app.command()
def main(
    artifact_dir: Path = typer.Option(..., "--artifact-dir", exists=True, help="RMH bundle run directory."),
    dataset_dir: Path = typer.Option(
        ..., "--dataset-dir", exists=True, help="Dataset dir containing features.parquet (for batch inference)."
    ),
    output_path: Path | None = typer.Option(None, "--output-path", help="Where to write predictions parquet."),
    limit_rows: int | None = typer.Option(None, "--limit-rows", help="Optional row limit."),
) -> None:
    bundle = load_bundle(artifact_dir)
    features_path = dataset_dir / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features.parquet: {features_path}")
    df = pd.read_parquet(features_path)
    if limit_rows is not None and limit_rows < len(df):
        df = df.head(int(limit_rows)).copy()

    _print_no_clamp_example()
    preds = predict_frame(df, bundle=bundle)

    out_path = output_path or (dataset_dir / "rmh_v1_preds.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_path, index=False)
    typer.echo(f"[rmh] wrote: {out_path} rows={len(preds)} cols={len(preds.columns)}")


if __name__ == "__main__":
    app()

