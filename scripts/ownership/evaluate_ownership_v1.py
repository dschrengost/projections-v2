"""CLI to evaluate ownership_v1 predictions on a fixed validation slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from projections.ownership_v1.evaluation import (
    OwnershipEvalSlice,
    default_eval_slice_path,
    evaluate_predictions,
    load_eval_slice,
    load_run_val_predictions,
)


def _render_markdown(run_id: str, slice_def: OwnershipEvalSlice, raw: dict, scaled: dict) -> str:
    def fmt(x: float) -> str:
        if x != x:  # NaN
            return "NaN"
        return f"{x:.4f}"

    def get(d: dict, path: str) -> float:
        cur = d
        for part in path.split("."):
            cur = cur[part]
        return float(cur)

    lines: list[str] = []
    lines.append(f"## {run_id}")
    lines.append("")
    lines.append(f"- Slice: `{slice_def.name}` (n_slates={raw['n_slates']}, n_rows={raw['n_rows']}, dates={raw['date_min']}..{raw['date_max']})")
    lines.append(f"- Target sum: {slice_def.target_sum_pct:.1f}%")
    lines.append("")

    lines.append("### Error")
    lines.append(f"- Raw MAE/RMSE: {fmt(get(raw, 'regression.mae_pct'))} / {fmt(get(raw, 'regression.rmse_pct'))} (pct points)")
    lines.append(f"- Raw MAE/RMSE (logit): {fmt(get(raw, 'regression.mae_logit'))} / {fmt(get(raw, 'regression.rmse_logit'))}")
    lines.append(f"- Scaled-to-sum MAE/RMSE: {fmt(get(scaled, 'regression.mae_pct'))} / {fmt(get(scaled, 'regression.rmse_pct'))} (pct points)")
    lines.append("")

    lines.append("### Ranking")
    lines.append(f"- Spearman pooled raw/scaled: {fmt(get(raw, 'ranking.spearman_pooled'))} / {fmt(get(scaled, 'ranking.spearman_pooled'))}")
    lines.append(f"- Spearman per-slate mean±std raw/scaled: {fmt(get(raw, 'ranking.spearman_per_slate_mean'))} ± {fmt(get(raw, 'ranking.spearman_per_slate_std'))} / {fmt(get(scaled, 'ranking.spearman_per_slate_mean'))} ± {fmt(get(scaled, 'ranking.spearman_per_slate_std'))}")
    lines.append(f"- Spearman top10/top20 actual raw/scaled: {fmt(get(raw, 'ranking.spearman_top10_mean'))} / {fmt(get(raw, 'ranking.spearman_top20_mean'))} / {fmt(get(scaled, 'ranking.spearman_top10_mean'))} / {fmt(get(scaled, 'ranking.spearman_top20_mean'))}")
    lines.append(f"- Recall@10/20 raw/scaled: {fmt(get(raw, 'ranking.recall_at_10'))} / {fmt(get(raw, 'ranking.recall_at_20'))} / {fmt(get(scaled, 'ranking.recall_at_10'))} / {fmt(get(scaled, 'ranking.recall_at_20'))}")
    lines.append("")

    lines.append("### Calibration")
    lines.append(f"- ECE raw/scaled: {fmt(get(raw, 'calibration.ece_pct'))} / {fmt(get(scaled, 'calibration.ece_pct'))} (pct points)")
    lines.append(f"- Top10/top20 mean bias raw/scaled: {fmt(get(raw, 'segment_bias.top10_mean_bias_pct'))} / {fmt(get(raw, 'segment_bias.top20_mean_bias_pct'))} / {fmt(get(scaled, 'segment_bias.top10_mean_bias_pct'))} / {fmt(get(scaled, 'segment_bias.top20_mean_bias_pct'))} (pct points)")
    lines.append(f"- Tail <=5% / <=1% mean bias raw/scaled: {fmt(get(raw, 'segment_bias.tail_le_5_mean_bias_pct'))} / {fmt(get(raw, 'segment_bias.tail_le_1_mean_bias_pct'))} / {fmt(get(scaled, 'segment_bias.tail_le_5_mean_bias_pct'))} / {fmt(get(scaled, 'segment_bias.tail_le_1_mean_bias_pct'))} (pct points)")
    lines.append("")

    lines.append("### Sum Constraint")
    lines.append(f"- Actual sum mean±std: {fmt(get(raw, 'sums.sum_actual_mean'))} ± {fmt(get(raw, 'sums.sum_actual_std'))} (min={fmt(get(raw, 'sums.sum_actual_min'))}, max={fmt(get(raw, 'sums.sum_actual_max'))})")
    lines.append(f"- Actual mean |sum - target|: {fmt(get(raw, 'sums.mean_abs_actual_sum_error_to_target'))}")
    lines.append(f"- Raw sum(pred) mean±std: {fmt(get(raw, 'sums.sum_pred_mean'))} ± {fmt(get(raw, 'sums.sum_pred_std'))} (min={fmt(get(raw, 'sums.sum_pred_min'))}, max={fmt(get(raw, 'sums.sum_pred_max'))})")
    lines.append(f"- Raw mean |sum - target|: {fmt(get(raw, 'sums.mean_abs_sum_error_to_target'))}")
    lines.append(f"- Raw max(pred) mean/p95: {fmt(get(raw, 'sums.max_pred_mean'))} / {fmt(get(raw, 'sums.max_pred_p95'))}")
    lines.append(f"- Raw count pred>60 / pred>70 / pred>100: {int(get(raw, 'sums.n_pred_over_60'))} / {int(get(raw, 'sums.n_pred_over_70'))} / {int(get(raw, 'sums.n_pred_over_100'))}")
    lines.append(f"- Scaled sum(pred) mean±std: {fmt(get(scaled, 'sums.sum_pred_mean'))} ± {fmt(get(scaled, 'sums.sum_pred_std'))} (min={fmt(get(scaled, 'sums.sum_pred_min'))}, max={fmt(get(scaled, 'sums.sum_pred_max'))})")
    lines.append(f"- Scaled mean |sum - target|: {fmt(get(scaled, 'sums.mean_abs_sum_error_to_target'))}")
    lines.append(f"- Scaled max(pred) mean/p95: {fmt(get(scaled, 'sums.max_pred_mean'))} / {fmt(get(scaled, 'sums.max_pred_p95'))}")
    lines.append(f"- Scaled count pred>60 / pred>70 / pred>100: {int(get(scaled, 'sums.n_pred_over_60'))} / {int(get(scaled, 'sums.n_pred_over_70'))} / {int(get(scaled, 'sums.n_pred_over_100'))}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ownership_v1 run predictions on a fixed slice")
    parser.add_argument("--run-id", required=True, help="ownership_v1 run_id to evaluate (must have val_predictions.csv)")
    parser.add_argument(
        "--slice-config",
        type=Path,
        default=default_eval_slice_path(),
        help="Path to fixed validation slice JSON (default: config/ownership_eval_slice.json)",
    )
    parser.add_argument("--out-md", type=Path, default=None, help="Write a markdown summary to this path")
    parser.add_argument("--out-json", type=Path, default=None, help="Write metrics JSON (raw + scaled) to this path")
    args = parser.parse_args()

    slice_def = load_eval_slice(args.slice_config)
    df = load_run_val_predictions(args.run_id)
    df = slice_def.filter_df(df)
    if df.empty:
        raise SystemExit(f"Slice produced 0 rows for run_id={args.run_id} using {args.slice_config}")

    raw_res = evaluate_predictions(
        df,
        slice_name=slice_def.name,
        target_sum_pct=slice_def.target_sum_pct,
        normalization="none",
    )
    scaled_res = evaluate_predictions(
        df,
        slice_name=slice_def.name,
        target_sum_pct=slice_def.target_sum_pct,
        normalization="scale_to_sum",
    )

    payload = {
        "run_id": args.run_id,
        "slice": slice_def.to_dict(),
        "raw": raw_res.to_dict(),
        "scaled_to_sum": scaled_res.to_dict(),
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        md = _render_markdown(args.run_id, slice_def, raw_res.to_dict(), scaled_res.to_dict())
        args.out_md.write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
