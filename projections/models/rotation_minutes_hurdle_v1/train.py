from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from projections.models.minutes_features import build_feature_spec, infer_feature_columns
from projections.models.minutes_nn import fit_preprocessor, transform_frame

from .bundle import save_bundle
from .config import RMHConfig, load_config
from .data import add_has_injury_row, build_y_play, compute_recency_weights, dataset_summary, load_labeled_frame
from .mixture import mixture_quantile_q10_q50_q90
from .model import RotationMinutesHurdleMLP, conditional_minutes_pinball_loss_q10_q50_q90, weighted_bce_with_logits


app = typer.Typer(add_completion=False, help="Train RMH_v1 (rotation minutes hurdle model).")

# Explicitly exclude post-game / realized rotation columns that can leak minutes.
_EXTRA_EXCLUDED_FEATURES: set[str] = {
    # Current-game stint-derived values (leaky in a pre-tip model).
    "minutes_from_stints",
    "team_total_minutes_from_stints",
    "num_stints",
    "max_stint_len_real",
    "first_in_time_real",
    "last_out_time_real",
}

# Small-cardinality categorical features available in the rotation training datasets.
_CATEGORICAL_COLS: list[str] = [
    # Stable identifiers that exist in both training and live inference.
    "player_id",
    "team_id",
]


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ece_table(y_true: np.ndarray, p: np.ndarray, *, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    y = np.asarray(y_true, dtype=float)
    probs = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    total = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        frac = float(mask.mean())
        mean_p = float(probs[mask].mean())
        mean_y = float(y[mask].mean())
        gap = abs(mean_p - mean_y)
        ece += frac * gap
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(mask.sum()),
                "mean_pred": mean_p,
                "mean_true": mean_y,
                "abs_gap": gap,
            }
        )
    if total == 0:
        return float("nan"), rows
    return float(ece), rows


@torch.no_grad()
def _predict(
    model: RotationMinutesHurdleMLP,
    *,
    x_cont: np.ndarray,
    x_cat: np.ndarray,
    device: str,
    batch_size: int = 8192,
) -> dict[str, np.ndarray]:
    model.eval()
    model.to(device)
    n = x_cont.shape[0]
    p_play = np.empty(n, dtype=np.float64)
    q10 = np.empty(n, dtype=np.float64)
    q50 = np.empty(n, dtype=np.float64)
    q90 = np.empty(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        xb_cont = torch.from_numpy(x_cont[start:end]).to(device)
        xb_cat = torch.from_numpy(x_cat[start:end]).to(device)
        out = model(xb_cont, xb_cat)
        p_play[start:end] = torch.sigmoid(out.logits_play).detach().cpu().numpy()
        q10[start:end] = out.q10_cond.detach().cpu().numpy()
        q50[start:end] = out.q50_cond.detach().cpu().numpy()
        q90[start:end] = out.q90_cond.detach().cpu().numpy()

    return {"p_play": p_play, "q10_cond": q10, "q50_cond": q50, "q90_cond": q90}


def train_rmh(cfg: RMHConfig) -> Path:
    _seed_everything(cfg.seed)
    run_id = cfg.run_id or _timestamp_run_id()
    run_dir = (cfg.artifact_dir / run_id).expanduser().resolve()

    df = load_labeled_frame(cfg.train_dataset_dir)
    df = add_has_injury_row(df)
    if cfg.limit_rows is not None and cfg.limit_rows < len(df):
        df = df.sample(n=int(cfg.limit_rows), random_state=cfg.seed).reset_index(drop=True)

    y_minutes = df["minutes"].astype(float).to_numpy(dtype=np.float32, copy=False)
    y_play = build_y_play(df["minutes"], play_threshold=cfg.play_threshold)
    w_play, w_play_summary = compute_recency_weights(df["tip_ts"], half_life_days=cfg.half_life_play_days)
    w_minutes, w_minutes_summary = compute_recency_weights(
        df["tip_ts"], half_life_days=cfg.half_life_minutes_days
    )

    ds_summary = dataset_summary(df, y_play=y_play)

    cont_cols = infer_feature_columns(df, target_col="minutes", excluded=_EXTRA_EXCLUDED_FEATURES)
    cat_cols = [col for col in _CATEGORICAL_COLS if col in df.columns]
    feature_spec = build_feature_spec([*cont_cols, *cat_cols], categorical_columns=cat_cols)
    preprocessor = fit_preprocessor(df, feature_spec)
    x_cont, x_cat = transform_frame(df, feature_spec, preprocessor)

    device = cfg.resolved_device()
    model = RotationMinutesHurdleMLP(
        n_continuous=x_cont.shape[1],
        cat_cardinalities=[len(preprocessor.categorical[col]) + 1 for col in feature_spec.categorical],
        emb_dim=cfg.emb_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    ds = TensorDataset(
        torch.from_numpy(x_cont.astype(np.float32, copy=False)),
        torch.from_numpy(x_cat.astype(np.int64, copy=False)),
        torch.from_numpy(y_play.astype(np.float32, copy=False)),
        torch.from_numpy(y_minutes.astype(np.float32, copy=False)),
        torch.from_numpy(w_play.astype(np.float32, copy=False)),
        torch.from_numpy(w_minutes.astype(np.float32, copy=False)),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    typer.echo(
        f"[rmh] train rows={ds_summary.n_rows} played_rate={ds_summary.played_rate:.3f} "
        f"status_counts={json.dumps(ds_summary.status_counts, sort_keys=True)}"
    )
    typer.echo(f"[rmh] half_life_play_days={cfg.half_life_play_days} summary={json.dumps(w_play_summary)}")
    typer.echo(f"[rmh] half_life_minutes_days={cfg.half_life_minutes_days} summary={json.dumps(w_minutes_summary)}")
    typer.echo(f"[rmh] feature_cols={len(feature_spec.continuous)} device={device} run_dir={run_dir}")

    start_time = time.time()
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_play = 0.0
        epoch_min = 0.0
        epoch_total = 0.0
        n_batches = 0

        for xb_cont, xb_cat, yb_play, yb_minutes, wb_play, wb_minutes in loader:
            xb_cont = xb_cont.to(device)
            xb_cat = xb_cat.to(device)
            yb_play = yb_play.to(device)
            yb_minutes = yb_minutes.to(device)
            wb_play = wb_play.to(device)
            wb_minutes = wb_minutes.to(device)

            out = model(xb_cont, xb_cat)
            loss_play = weighted_bce_with_logits(out.logits_play, yb_play, wb_play)
            loss_minutes = conditional_minutes_pinball_loss_q10_q50_q90(
                q10=out.q10_cond,
                q50=out.q50_cond,
                q90=out.q90_cond,
                y_minutes=yb_minutes,
                y_play=yb_play,
                sample_weight=wb_minutes,
                quantile_weights=cfg.quantile_weights,
            )
            loss = cfg.play_loss_weight * loss_play + cfg.minutes_loss_weight * loss_minutes

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_play += float(loss_play.detach().cpu())
            epoch_min += float(loss_minutes.detach().cpu())
            epoch_total += float(loss.detach().cpu())
            n_batches += 1

        row = {
            "epoch": float(epoch),
            "loss_play": epoch_play / max(n_batches, 1),
            "loss_minutes": epoch_min / max(n_batches, 1),
            "loss_total": epoch_total / max(n_batches, 1),
        }
        history.append(row)
        typer.echo(
            f"[rmh] epoch {epoch:03d}/{cfg.epochs} "
            f"loss_play={row['loss_play']:.4f} loss_minutes={row['loss_minutes']:.4f} loss_total={row['loss_total']:.4f}"
        )

    train_seconds = float(time.time() - start_time)

    # --- Training-set metrics (quick sanity checks; full eval uses eval.py) ---
    preds = _predict(model, x_cont=x_cont, x_cat=x_cat, device=device)
    p_play = preds["p_play"]
    y_play_int = y_play.astype(int)

    play_auc = float("nan")
    play_pr_auc = float("nan")
    try:
        if len(np.unique(y_play_int)) > 1:
            play_auc = float(roc_auc_score(y_play_int, p_play))
            play_pr_auc = float(average_precision_score(y_play_int, p_play))
    except Exception:
        pass
    ece, ece_bins = _ece_table(y_play_int, p_play, bins=10)

    # Conditional minutes coverage on played-only slice
    played = y_play_int == 1
    cov10 = float(np.mean(y_minutes[played] <= preds["q10_cond"][played])) if played.any() else float("nan")
    cov50 = float(np.mean(y_minutes[played] <= preds["q50_cond"][played])) if played.any() else float("nan")
    cov90 = float(np.mean(y_minutes[played] <= preds["q90_cond"][played])) if played.any() else float("nan")

    # Mixture quantile example (ensures we are not clamping adjusted taus to 0.10).
    example = {
        "p_play": 0.11,
        "tau": 0.90,
        "tau_pos": (0.90 - (1.0 - 0.11)) / 0.11,
        "q10_cond": 15.0,
        "q50_cond": 25.0,
        "q90_cond": 35.0,
    }
    ex_q90 = mixture_quantile_q10_q50_q90(
        p_play=np.array([example["p_play"]]),
        q10_cond=np.array([example["q10_cond"]]),
        q50_cond=np.array([example["q50_cond"]]),
        q90_cond=np.array([example["q90_cond"]]),
        tau=example["tau"],
        play_threshold=cfg.play_threshold,
    )[0]
    example["q_tau_uncond"] = float(ex_q90)
    typer.echo(f"[rmh] mixture example (no clamping): {json.dumps(example, sort_keys=True)}")

    metrics: dict[str, Any] = {
        "train": {
            "rows": ds_summary.n_rows,
            "played_rate": ds_summary.played_rate,
            "status_counts": ds_summary.status_counts,
            "play_auc": play_auc,
            "play_pr_auc": play_pr_auc,
            "play_ece": float(ece),
            "play_ece_bins": ece_bins,
            "minutes_cov_played": {"p(y<=q10)": cov10, "p(y<=q50)": cov50, "p(y<=q90)": cov90},
            "history": history,
            "train_seconds": train_seconds,
        },
        "recency": {
            "half_life_play_days": cfg.half_life_play_days,
            "half_life_minutes_days": cfg.half_life_minutes_days,
            "play_weight_summary": w_play_summary,
            "minutes_weight_summary": w_minutes_summary,
        },
        "mixture_example": example,
    }

    save_bundle(
        run_dir,
        model=model.to("cpu"),
        feature_spec=feature_spec,
        preprocessor=preprocessor,
        config=cfg.to_json_dict(),
        metrics=metrics,
    )
    typer.echo(f"[rmh] saved bundle: {run_dir}")
    return run_dir


@app.command()
def main(
    config_path: Path = typer.Option(..., "--config", exists=True, help="Path to RMH_v1 YAML/JSON config."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id override."),
    limit_rows: int | None = typer.Option(None, "--limit-rows", help="Optional row limit (dev/smoke)."),
) -> None:
    cfg = load_config(config_path)
    if run_id is not None:
        cfg = cfg.model_copy(update={"run_id": run_id})
    if limit_rows is not None:
        cfg = cfg.model_copy(update={"limit_rows": limit_rows})
    train_rmh(cfg)


if __name__ == "__main__":
    app()
