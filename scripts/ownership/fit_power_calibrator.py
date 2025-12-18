"""Fit a power-scaling calibrator on a joined production-path dataset.

Workflow:
1) Build a joined dataset (preds + actuals) via:
     uv run python scripts/ownership/evaluate_ownership_production_path.py \
       --start-date 2025-12-05 --end-date 2025-12-16 \
       --out-parquet /tmp/ownership_joined.parquet

2) Fit a gamma for (s + eps)^gamma slate allocation:
     uv run python scripts/ownership/fit_power_calibrator.py \
       --in-parquet /tmp/ownership_joined.parquet \
       --out-json artifacts/ownership_v1/power_calibrator.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from projections.ownership_v1.calibration import PowerCalibrationParams, PowerCalibrator
from projections.ownership_v1.evaluation import evaluate_predictions
from projections.ownership_v1.score import normalize_ownership_to_target_sum


@dataclass(frozen=True)
class FitResult:
    gamma: float
    eps: float
    R: float
    cap_pct: float
    objective: float
    mae_all: float
    mae_top20: float
    top10_bias: float
    max_pred_mean: float
    max_pred_p95: float

    def to_dict(self) -> dict:
        return asdict(self)


def _apply_gamma(
    df: pd.DataFrame,
    *,
    gamma: float,
    eps: float,
    R: float,
    cap_pct: float,
    score_col: str,
    slate_id_col: str,
    min_proj_fpts: float,
) -> pd.Series:
    out = pd.Series(0.0, index=df.index, name="pred_own_pct_power")
    calibrator = PowerCalibrator(params=PowerCalibrationParams(gamma=gamma, eps=eps, R=R))
    target_sum_pct = float(R) * 100.0

    for _, g in df.groupby(slate_id_col, sort=False):
        scores = pd.to_numeric(g[score_col], errors="coerce").fillna(0.0).astype(float).to_numpy()
        playable = pd.to_numeric(g["proj_fpts"], errors="coerce").fillna(0.0).astype(float).to_numpy() >= float(
            min_proj_fpts
        )
        frac = calibrator.apply(scores, mask=playable)
        pct = frac * 100.0

        pct = normalize_ownership_to_target_sum(
            pd.Series(pct, index=g.index),
            target_sum_pct=target_sum_pct,
            cap_pct=cap_pct,
        ).to_numpy()
        out.loc[g.index] = pct

    return out


def _compute_fit_metrics(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str,
    slate_id_col: str,
) -> tuple[float, float, float, float, float]:
    work = df[[slate_id_col, pred_col, actual_col]].copy()
    work = work.dropna(subset=[pred_col, actual_col]).copy()
    work[pred_col] = work[pred_col].astype(float)
    work[actual_col] = work[actual_col].astype(float)

    err = (work[pred_col] - work[actual_col]).to_numpy()
    mae_all = float(np.mean(np.abs(err))) if len(err) else float("nan")

    top20_mae = []
    top10_bias = []
    max_by_slate = []
    for _, g in work.groupby(slate_id_col, sort=False):
        if len(g) >= 20:
            top20 = g.nlargest(20, actual_col)
            top20_mae.append(float(np.mean(np.abs((top20[pred_col] - top20[actual_col]).to_numpy()))))
        if len(g) >= 10:
            top10 = g.nlargest(10, actual_col)
            top10_bias.append(float(np.mean((top10[pred_col] - top10[actual_col]).to_numpy())))
        max_by_slate.append(float(g[pred_col].max()))

    mae_top20 = float(np.mean(top20_mae)) if top20_mae else float("nan")
    bias_top10 = float(np.mean(top10_bias)) if top10_bias else float("nan")
    max_pred_mean = float(np.mean(max_by_slate)) if max_by_slate else float("nan")
    max_pred_p95 = float(np.percentile(np.array(max_by_slate), 95)) if max_by_slate else float("nan")

    return mae_all, mae_top20, bias_top10, max_pred_mean, max_pred_p95


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a power ownership calibrator (gamma) on joined preds+actuals")
    parser.add_argument("--in-parquet", type=Path, required=True, help="Joined parquet from evaluate_ownership_production_path.py")
    parser.add_argument("--out-json", type=Path, required=True, help="Write calibrator JSON here (relative or absolute)")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional: write a markdown summary here")
    parser.add_argument("--score-col", default="pred_own_pct_raw")
    parser.add_argument("--actual-col", default="actual_own_pct")
    parser.add_argument("--slate-id-col", default="slate_id")
    parser.add_argument("--min-proj-fpts", type=float, default=8.0)
    parser.add_argument("--R", type=float, default=8.0, help="Slots per lineup (DK classic=8)")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--cap-pct", type=float, default=100.0)
    parser.add_argument("--gamma-min", type=float, default=0.6)
    parser.add_argument("--gamma-max", type=float, default=2.6)
    parser.add_argument("--gamma-step", type=float, default=0.05)
    parser.add_argument(
        "--objective",
        choices=["mae_top20", "mae_top20_plus_all"],
        default="mae_top20_plus_all",
        help="Selection metric (prioritizes top-chalk accuracy).",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.in_parquet)
    required = {args.score_col, args.actual_col, args.slate_id_col, "proj_fpts"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {args.in_parquet}: {missing}")

    gammas = np.arange(args.gamma_min, args.gamma_max + 1e-9, args.gamma_step, dtype=float)
    best: FitResult | None = None

    for gamma in gammas:
        pred = _apply_gamma(
            df,
            gamma=float(gamma),
            eps=float(args.eps),
            R=float(args.R),
            cap_pct=float(args.cap_pct),
            score_col=str(args.score_col),
            slate_id_col=str(args.slate_id_col),
            min_proj_fpts=float(args.min_proj_fpts),
        )
        work = df.copy()
        work["_pred_power"] = pred
        mae_all, mae_top20, top10_bias, max_mean, max_p95 = _compute_fit_metrics(
            work,
            pred_col="_pred_power",
            actual_col=str(args.actual_col),
            slate_id_col=str(args.slate_id_col),
        )

        if args.objective == "mae_top20":
            obj = mae_top20
        else:
            obj = mae_top20 + 0.25 * mae_all

        cand = FitResult(
            gamma=float(gamma),
            eps=float(args.eps),
            R=float(args.R),
            cap_pct=float(args.cap_pct),
            objective=float(obj),
            mae_all=float(mae_all),
            mae_top20=float(mae_top20),
            top10_bias=float(top10_bias),
            max_pred_mean=float(max_mean),
            max_pred_p95=float(max_p95),
        )
        if best is None or cand.objective < best.objective:
            best = cand

    if best is None:
        raise SystemExit("No candidates evaluated.")

    # Recompute predictions for the best gamma so we can report full eval metrics.
    best_pred = _apply_gamma(
        df,
        gamma=float(best.gamma),
        eps=float(best.eps),
        R=float(best.R),
        cap_pct=float(best.cap_pct),
        score_col=str(args.score_col),
        slate_id_col=str(args.slate_id_col),
        min_proj_fpts=float(args.min_proj_fpts),
    )
    eval_df = df.copy()
    eval_df["pred_own_pct_power"] = best_pred
    eval_df = eval_df.dropna(subset=[str(args.actual_col), "pred_own_pct_power"]).copy()
    eval_res = evaluate_predictions(
        eval_df,
        slice_name=f"power_fit_{args.in_parquet.name}",
        pred_col="pred_own_pct_power",
        actual_col=str(args.actual_col),
        slate_id_col=str(args.slate_id_col),
        target_sum_pct=float(best.R) * 100.0,
        normalization="none",
    )

    calibrator = PowerCalibrator(params=PowerCalibrationParams(gamma=best.gamma, eps=best.eps, R=best.R))
    payload = {
        "type": "power",
        "params": calibrator.params.to_dict(),
        "fit": best.to_dict(),
        "eval": eval_res.to_dict(),
        "notes": {
            "objective": args.objective,
            "min_proj_fpts": float(args.min_proj_fpts),
            "cap_pct": float(args.cap_pct),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.out_md is not None:
        def fmt(x: float) -> str:
            return f"{x:.4f}" if x == x else "NaN"

        md_lines = [
            "# Power Calibrator Fit",
            "",
            f"- gamma: {best.gamma:.4f} (eps={best.eps:g}, R={best.R:.1f}, cap_pct={best.cap_pct:.1f})",
            f"- objective ({args.objective}): {fmt(best.objective)}",
            f"- MAE all / top20: {fmt(best.mae_all)} / {fmt(best.mae_top20)}",
            f"- Top10 mean bias: {fmt(best.top10_bias)}",
            f"- Max pred mean / p95: {fmt(best.max_pred_mean)} / {fmt(best.max_pred_p95)}",
            "",
            "## Eval (pred_own_pct_power)",
            f"- MAE/RMSE: {fmt(payload['eval']['regression']['mae_pct'])} / {fmt(payload['eval']['regression']['rmse_pct'])}",
            f"- Spearman pooled: {fmt(payload['eval']['ranking']['spearman_pooled'])}",
            f"- Spearman top10/top20: {fmt(payload['eval']['ranking']['spearman_top10_mean'])} / {fmt(payload['eval']['ranking']['spearman_top20_mean'])}",
            f"- Recall@10/20: {fmt(payload['eval']['ranking']['recall_at_10'])} / {fmt(payload['eval']['ranking']['recall_at_20'])}",
            f"- ECE: {fmt(payload['eval']['calibration']['ece_pct'])}",
        ]
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
