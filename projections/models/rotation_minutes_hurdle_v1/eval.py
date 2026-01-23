from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import average_precision_score, roc_auc_score

from .bundle import load_bundle
from .data import build_y_play, load_labeled_frame
from .infer import predict_frame
from .model import pinball_loss


app = typer.Typer(add_completion=False, help="Evaluate RMH_v1 on a labeled dataset.")


STATUSES = ["Ava", "OUT", "AVAIL", "UNK", "Q", "PROB"]


def _safe_auc(y_true: np.ndarray, p: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, p))
    except Exception:
        return None


def _safe_pr_auc(y_true: np.ndarray, p: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(average_precision_score(y_true, p))
    except Exception:
        return None


def _ece_table(y_true: np.ndarray, p: np.ndarray, *, bins: int = 10) -> tuple[float | None, list[dict[str, Any]]]:
    y = np.asarray(y_true, dtype=float)
    probs = np.asarray(p, dtype=float)
    if len(y) == 0:
        return None, []
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
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
    return float(ece), rows


def _brier(y_true: np.ndarray, p: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    y = np.asarray(y_true, dtype=float)
    probs = np.asarray(p, dtype=float)
    return float(np.mean((y - probs) ** 2))


def _minutes_metrics_played(df: pd.DataFrame, *, y_play: np.ndarray) -> dict[str, Any]:
    played_mask = y_play.astype(bool)
    if not np.any(played_mask):
        return {
            "n_played": 0,
            "pinball": None,
            "coverage": None,
        }

    y = df.loc[played_mask, "minutes"].astype(float).to_numpy()
    q10 = df.loc[played_mask, "minutes_q10_cond"].astype(float).to_numpy()
    q50 = df.loc[played_mask, "minutes_q50_cond"].astype(float).to_numpy()
    q90 = df.loc[played_mask, "minutes_q90_cond"].astype(float).to_numpy()

    # Pinball losses (mean)
    y_t = torch_from_numpy(y)
    p10_t = torch_from_numpy(q10)
    p50_t = torch_from_numpy(q50)
    p90_t = torch_from_numpy(q90)
    pin10 = float(pinball_loss(p10_t, y_t, 0.10).mean().item())
    pin50 = float(pinball_loss(p50_t, y_t, 0.50).mean().item())
    pin90 = float(pinball_loss(p90_t, y_t, 0.90).mean().item())
    pin_joint = float(np.mean([pin10, pin50, pin90]))

    cov10 = float(np.mean(y <= q10))
    cov50 = float(np.mean(y <= q50))
    cov90 = float(np.mean(y <= q90))

    return {
        "n_played": int(played_mask.sum()),
        "pinball": {"q10": pin10, "q50": pin50, "q90": pin90, "joint_mean": pin_joint},
        "coverage": {"p(y<=q10)": cov10, "p(y<=q50)": cov50, "p(y<=q90)": cov90},
    }


def torch_from_numpy(x: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _print_status_table(rows: list[dict[str, Any]], *, title: str) -> None:
    typer.echo(f"\n[{title}]")
    for row in rows:
        typer.echo(json.dumps(row, sort_keys=True))


@app.command()
def main(
    artifact_dir: Path = typer.Option(..., "--artifact-dir", exists=True, help="RMH bundle run directory."),
    dataset_dir: Path = typer.Option(..., "--dataset-dir", exists=True, help="Dataset dir with features/labels parquet."),
    output_json: Path | None = typer.Option(None, "--output-json", help="Optional path to write metrics JSON."),
    limit_rows: int | None = typer.Option(None, "--limit-rows", help="Optional row limit (dev)."),
) -> None:
    bundle = load_bundle(artifact_dir)
    cfg = bundle.config
    play_threshold = float(cfg.get("play_threshold", 1.0))

    df = load_labeled_frame(dataset_dir)
    if limit_rows is not None and limit_rows < len(df):
        df = df.head(int(limit_rows)).copy()

    scored = predict_frame(df, bundle=bundle)
    y_play = build_y_play(scored["minutes"], play_threshold=play_threshold).astype(int)
    p_play = scored["p_play"].astype(float).to_numpy()

    overall: dict[str, Any] = {
        "rows": int(len(scored)),
        "played_rate": float(y_play.mean()) if len(y_play) else None,
        "play": {},
        "minutes_cond": {},
    }

    overall["play"]["auc"] = _safe_auc(y_play, p_play)
    overall["play"]["pr_auc"] = _safe_pr_auc(y_play, p_play)
    overall["play"]["brier"] = _brier(y_play, p_play)
    ece, bins = _ece_table(y_play, p_play, bins=10)
    overall["play"]["ece"] = ece
    overall["play"]["reliability"] = bins

    overall["minutes_cond"] = _minutes_metrics_played(scored, y_play=y_play)

    # --- Status slices ---
    play_slices: list[dict[str, Any]] = []
    minutes_slices: list[dict[str, Any]] = []
    stability_slices: list[dict[str, Any]] = []
    for status in STATUSES:
        mask = scored["status"].astype(str) == status
        if not mask.any():
            continue
        ys = y_play[mask.to_numpy()]
        ps = p_play[mask.to_numpy()]
        play_slices.append(
            {
                "status": status,
                "n": int(mask.sum()),
                "played_rate": float(ys.mean()) if len(ys) else None,
                "p_play_mean": float(ps.mean()) if len(ps) else None,
                "auc": _safe_auc(ys, ps),
                "pr_auc": _safe_pr_auc(ys, ps),
                "ece": _ece_table(ys, ps, bins=10)[0],
            }
        )

        # Conditional minutes slice on played-only rows
        played_mask = mask & (scored["minutes"].astype(float) >= play_threshold)
        if played_mask.any():
            sub = scored.loc[played_mask].copy()
            y_sub = sub["minutes"].astype(float).to_numpy()
            q10 = sub["minutes_q10_cond"].astype(float).to_numpy()
            q50 = sub["minutes_q50_cond"].astype(float).to_numpy()
            q90 = sub["minutes_q90_cond"].astype(float).to_numpy()

            y_t = torch_from_numpy(y_sub)
            minutes_slices.append(
                {
                    "status": status,
                    "n_played": int(len(sub)),
                    "pinball_q10": float(pinball_loss(torch_from_numpy(q10), y_t, 0.10).mean().item()),
                    "pinball_q50": float(pinball_loss(torch_from_numpy(q50), y_t, 0.50).mean().item()),
                    "pinball_q90": float(pinball_loss(torch_from_numpy(q90), y_t, 0.90).mean().item()),
                    "cov_q10": float(np.mean(y_sub <= q10)),
                    "cov_q50": float(np.mean(y_sub <= q50)),
                    "cov_q90": float(np.mean(y_sub <= q90)),
                }
            )

            stability_slices.append(
                {
                    "status": status,
                    "n_played": int(len(sub)),
                    "actual_minutes_mean": float(np.mean(y_sub)),
                    "pred_q50_cond_mean": float(np.mean(q50)),
                    "pred_q50_cond_p50": float(np.percentile(q50, 50)),
                }
            )

    metrics = {"overall": overall, "by_status": {"play": play_slices, "minutes_cond": minutes_slices}}

    typer.echo("\n[rmh] overall")
    typer.echo(json.dumps(overall, indent=2, sort_keys=True))
    _print_status_table(play_slices, title="rmh play slices (status)")
    _print_status_table(minutes_slices, title="rmh conditional minutes slices (played-only, by status)")
    _print_status_table(stability_slices, title="rmh conditional minutes stability (played-only, by status)")

    # Demonstrate no-clamp behavior on an actual row if possible.
    demo = None
    cand = scored[(scored["p_play"] > 0.101) & (scored["p_play"] < 0.111)]
    if not cand.empty:
        r = cand.iloc[0]
        p = float(r["p_play"])
        tau = 0.90
        tau_pos = (tau - (1.0 - p)) / p
        demo = {
            "p_play": p,
            "tau": tau,
            "tau_pos": float(tau_pos),
            "q10_cond": float(r["minutes_q10_cond"]),
            "q90_uncond": float(r["minutes_q90_uncond"]),
        }
    else:
        demo = {
            "note": "No row with p_play in (0.101, 0.111); see train-time synthetic example in bundle metrics.",
        }
    typer.echo(f"\n[rmh] mixture no-clamp demo: {json.dumps(demo, sort_keys=True)}")
    metrics["mixture_demo"] = demo

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(f"[rmh] wrote metrics: {output_json}")


if __name__ == "__main__":
    app()

