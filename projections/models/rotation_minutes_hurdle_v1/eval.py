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
    """Compute conditional minutes metrics on played-only slice (v1.1: all 7 quantiles)."""
    played_mask = y_play.astype(bool)
    if not np.any(played_mask):
        return {
            "n_played": 0,
            "pinball": None,
            "coverage": None,
        }

    y = df.loc[played_mask, "minutes"].astype(float).to_numpy()
    y_t = torch_from_numpy(y)

    # v1.1: Compute pinball and coverage for all 7 quantiles
    quantile_cols = [
        ("q05", 0.05, "minutes_q05_cond"),
        ("q10", 0.10, "minutes_q10_cond"),
        ("q25", 0.25, "minutes_q25_cond"),
        ("q50", 0.50, "minutes_q50_cond"),
        ("q75", 0.75, "minutes_q75_cond"),
        ("q90", 0.90, "minutes_q90_cond"),
        ("q95", 0.95, "minutes_q95_cond"),
    ]

    pinball_dict = {}
    coverage_dict = {}
    for name, tau, col in quantile_cols:
        q_vals = df.loc[played_mask, col].astype(float).to_numpy()
        q_t = torch_from_numpy(q_vals)
        pinball_dict[name] = float(pinball_loss(q_t, y_t, tau).mean().item())
        coverage_dict[f"p(y<={name})"] = float(np.mean(y <= q_vals))

    pinball_dict["joint_mean"] = float(np.mean(list(pinball_dict.values())))

    return {
        "n_played": int(played_mask.sum()),
        "pinball": pinball_dict,
        "coverage": coverage_dict,
    }


def torch_from_numpy(x: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _print_status_table(rows: list[dict[str, Any]], *, title: str) -> None:
    typer.echo(f"\n[{title}]")
    for row in rows:
        typer.echo(json.dumps(row, sort_keys=True))


def _non_out_play_metrics(
    scored: pd.DataFrame,
    y_play: np.ndarray,
    p_play: np.ndarray,
) -> dict[str, Any]:
    """v1.1 Spec 4.1: Non-OUT slice metrics for play head.

    Global AUC is dominated by OUT vs non-OUT. This provides explicit metrics
    on status != 'OUT' to better evaluate prediction quality on uncertain cases.
    """
    # Non-OUT mask: status != 'OUT'
    non_out_mask = scored["status"].astype(str) != "OUT"
    if not non_out_mask.any():
        return {"n": 0, "auc": None, "pr_auc": None, "ece": None}

    ys = y_play[non_out_mask.to_numpy()]
    ps = p_play[non_out_mask.to_numpy()]

    return {
        "n": int(non_out_mask.sum()),
        "played_rate": float(ys.mean()) if len(ys) else None,
        "p_play_mean": float(ps.mean()) if len(ps) else None,
        "auc": _safe_auc(ys, ps),
        "pr_auc": _safe_pr_auc(ys, ps),
        "ece": _ece_table(ys, ps, bins=10)[0],
    }


def _unk_q_calibration_table(
    scored: pd.DataFrame,
    y_play: np.ndarray,
    p_play: np.ndarray,
    *,
    play_threshold: float,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """v1.1 Spec 4.2: UNK/Q calibration table.

    For status in {UNK, Q}, bucket p_play into deciles and report
    empirical P(minutes >= play_threshold) per bucket.

    This is the primary guardrail against early-day pessimism.
    """
    # Filter to UNK/Q rows only
    status = scored["status"].astype(str)
    unk_q_mask = status.isin(["UNK", "Q"])
    if not unk_q_mask.any():
        return []

    ys = y_play[unk_q_mask.to_numpy()]
    ps = p_play[unk_q_mask.to_numpy()]

    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (ps >= lo) & (ps < hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(mask.sum()),
                "mean_p_play": float(ps[mask].mean()),
                "empirical_play_rate": float(ys[mask].mean()),
                "abs_gap": abs(float(ps[mask].mean()) - float(ys[mask].mean())),
            }
        )
    return rows


def _coverage_by_minutes_bucket(
    scored: pd.DataFrame,
    *,
    play_threshold: float,
) -> list[dict[str, Any]]:
    """v1.1 Spec 4.3: Conditional minutes coverage by realized minutes bucket.

    On played-only rows (minutes >= threshold), bucket by realized minutes:
    - threshold-10, 10-20, 20-30, 30+

    Report coverage for all 7 quantiles.
    """
    # Played-only rows
    played_mask = scored["minutes"].astype(float) >= play_threshold
    if not played_mask.any():
        return []

    sub = scored.loc[played_mask].copy()
    y = sub["minutes"].astype(float).to_numpy()

    # Define buckets: [threshold, 10), [10, 20), [20, 30), [30, inf)
    buckets = [
        (play_threshold, 10.0, f"{play_threshold:.0f}-10"),
        (10.0, 20.0, "10-20"),
        (20.0, 30.0, "20-30"),
        (30.0, float("inf"), "30+"),
    ]

    quantile_cols = [
        ("q05", "minutes_q05_cond"),
        ("q10", "minutes_q10_cond"),
        ("q25", "minutes_q25_cond"),
        ("q50", "minutes_q50_cond"),
        ("q75", "minutes_q75_cond"),
        ("q90", "minutes_q90_cond"),
        ("q95", "minutes_q95_cond"),
    ]

    rows: list[dict[str, Any]] = []
    for lo, hi, label in buckets:
        bucket_mask = (y >= lo) & (y < hi)
        if not np.any(bucket_mask):
            continue

        y_bucket = y[bucket_mask]
        row: dict[str, Any] = {
            "bucket": label,
            "n": int(bucket_mask.sum()),
            "minutes_mean": float(np.mean(y_bucket)),
        }

        for q_name, col in quantile_cols:
            q_vals = sub.iloc[bucket_mask.nonzero()[0]][col].astype(float).to_numpy()
            row[f"cov_{q_name}"] = float(np.mean(y_bucket <= q_vals))

        rows.append(row)

    return rows


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

    # v1.1 Spec 4.1: Non-OUT play head metrics
    overall["play"]["non_out_slice"] = _non_out_play_metrics(scored, y_play, p_play)

    overall["minutes_cond"] = _minutes_metrics_played(scored, y_play=y_play)

    # v1.1 Spec 4.2: UNK/Q calibration table
    unk_q_calibration = _unk_q_calibration_table(
        scored, y_play, p_play, play_threshold=play_threshold
    )

    # v1.1 Spec 4.3: Coverage by minutes bucket
    coverage_by_bucket = _coverage_by_minutes_bucket(scored, play_threshold=play_threshold)

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

        # Conditional minutes slice on played-only rows (v1.1: all 7 quantiles)
        played_mask = mask & (scored["minutes"].astype(float) >= play_threshold)
        if played_mask.any():
            sub = scored.loc[played_mask].copy()
            y_sub = sub["minutes"].astype(float).to_numpy()
            y_t = torch_from_numpy(y_sub)

            quantile_cols = [
                ("q05", 0.05, "minutes_q05_cond"),
                ("q10", 0.10, "minutes_q10_cond"),
                ("q25", 0.25, "minutes_q25_cond"),
                ("q50", 0.50, "minutes_q50_cond"),
                ("q75", 0.75, "minutes_q75_cond"),
                ("q90", 0.90, "minutes_q90_cond"),
                ("q95", 0.95, "minutes_q95_cond"),
            ]

            slice_row: dict[str, Any] = {
                "status": status,
                "n_played": int(len(sub)),
            }
            for name, tau, col in quantile_cols:
                q_vals = sub[col].astype(float).to_numpy()
                q_t = torch_from_numpy(q_vals)
                slice_row[f"pinball_{name}"] = float(pinball_loss(q_t, y_t, tau).mean().item())
                slice_row[f"cov_{name}"] = float(np.mean(y_sub <= q_vals))

            minutes_slices.append(slice_row)

            q50 = sub["minutes_q50_cond"].astype(float).to_numpy()
            stability_slices.append(
                {
                    "status": status,
                    "n_played": int(len(sub)),
                    "actual_minutes_mean": float(np.mean(y_sub)),
                    "pred_q50_cond_mean": float(np.mean(q50)),
                    "pred_q50_cond_p50": float(np.percentile(q50, 50)),
                }
            )

    metrics = {
        "overall": overall,
        "by_status": {"play": play_slices, "minutes_cond": minutes_slices},
        "unk_q_calibration": unk_q_calibration,  # v1.1 Spec 4.2
        "coverage_by_bucket": coverage_by_bucket,  # v1.1 Spec 4.3
    }

    typer.echo("\n[rmh] overall")
    typer.echo(json.dumps(overall, indent=2, sort_keys=True))
    _print_status_table(play_slices, title="rmh play slices (status)")
    _print_status_table(minutes_slices, title="rmh conditional minutes slices (played-only, by status)")
    _print_status_table(stability_slices, title="rmh conditional minutes stability (played-only, by status)")

    # v1.1: Print new eval sections
    _print_status_table(unk_q_calibration, title="rmh UNK/Q calibration (v1.1 Spec 4.2)")
    _print_status_table(coverage_by_bucket, title="rmh coverage by minutes bucket (v1.1 Spec 4.3)")

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

