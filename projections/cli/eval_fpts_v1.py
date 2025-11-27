"""Evaluate an FPTS v1 run over an arbitrary calendar window."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections import paths
from projections.fpts_v1.datasets import FptsDatasetBuilder, MinutesSource
from projections.fpts_v1.eval import baseline_per_minute, evaluate_fpts_run
from projections.fpts_v1.production import load_fpts_model, predict_fpts_per_min

app = typer.Typer(help=__doc__)
DEFAULT_ARTIFACT_ROOT = Path("artifacts/fpts_lgbm")
DEFAULT_EVAL_ROOT = Path("artifacts/fpts_eval")


def _normalize_date(value: datetime | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.tz_localize(None).normalize()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _format_metric(value: float) -> str:
    return f"{value:.2f}"


def _format_bucket_table(title: str, buckets: dict[str, dict[str, Any]]) -> str:
    if not buckets:
        return f"### {title}\n\n_No rows in this bucket._\n"
    lines = [
        f"### {title}",
        "",
        "| bucket | rows | model MAE (FPTS) | baseline MAE (FPTS) | ΔMAE | model SMAPE (FPTS) | baseline SMAPE (FPTS) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bucket, payload in buckets.items():
        rows = payload.get("rows", 0)
        model_metrics = payload.get("model", {})
        baseline_metrics = payload.get("baseline", {})
        delta = payload.get("delta", {})
        lines.append(
            "| {bucket} | {rows} | {mae_model} | {mae_base} | {delta_mae} | {smape_model} | {smape_base} |".format(
                bucket=bucket,
                rows=rows,
                mae_model=_format_metric(model_metrics.get("mae_fpts", 0.0)),
                mae_base=_format_metric(baseline_metrics.get("mae_fpts", 0.0)),
                delta_mae=_format_metric(delta.get("mae_fpts", 0.0)),
                smape_model=_format_metric(model_metrics.get("smape_fpts", 0.0)),
                smape_base=_format_metric(baseline_metrics.get("smape_fpts", 0.0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_report(
    *,
    run_id: str,
    minutes_source: str,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    rows: int,
    metrics: dict[str, Any],
    artifact_root: Path,
    data_root: Path,
) -> str:
    overall = metrics.get("overall", {})
    model_overall = overall.get("model", {})
    baseline_overall = overall.get("baseline", {})
    delta_overall = overall.get("delta", {})
    lines = [
        f"# FPTS Evaluation – {run_id}",
        "",
        f"- Date window: {start_day.date().isoformat()} → {end_day.date().isoformat()}",
        f"- Minutes source: {minutes_source}",
        f"- Rows evaluated: {rows}",
        f"- Artifact root: {artifact_root}",
        f"- Data root: {data_root}",
        "",
        "## Overall Metrics",
        "",
        "| metric | model | baseline | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("mae_fpts", "rmse_fpts", "smape_fpts"):
        lines.append(
            f"| {key} | {_format_metric(model_overall.get(key, 0.0))} | "
            f"{_format_metric(baseline_overall.get(key, 0.0))} | {_format_metric(delta_overall.get(key, 0.0))} |"
        )
    lines.append("")

    for title, bucket_key in [
        ("Projected FPTS tiers", "by_proj_fpts_bucket"),
        ("Minutes buckets", "by_minutes_bucket"),
        ("Injury context (teammates out)", "by_injury_context"),
        ("Minutes volatility buckets", "by_volatility_bucket"),
        ("Salary buckets", "by_salary_bucket"),
        ("Role/legacy buckets", "by_role_context"),
    ]:
        lines.append(_format_bucket_table(title, metrics.get(bucket_key, {})))
    return "\n".join(lines)


@app.command()
def main(
    run_id: str = typer.Option(..., "--run-id", help="FPTS model run identifier."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Root directory containing bronze/silver/gold projections data.",
    ),
    artifact_root: Path = typer.Option(
        DEFAULT_ARTIFACT_ROOT, "--artifact-root", help="Directory containing FPTS LightGBM runs."
    ),
    start_date: datetime = typer.Option(..., "--start-date", help="Inclusive evaluation start date (YYYY-MM-DD)."),
    end_date: datetime = typer.Option(..., "--end-date", help="Inclusive evaluation end date (YYYY-MM-DD)."),
    minutes_source: str = typer.Option(
        "predicted",
        "--minutes-source",
        help="Minutes source used when building the dataset (predicted or actual).",
    ),
    minutes_run_id: str | None = typer.Option(
        None,
        "--minutes-run-id",
        help="Minutes prediction log run id (defaults to production pointer when omitted).",
    ),
    out_root: Path = typer.Option(
        DEFAULT_EVAL_ROOT,
        "--out-root",
        help="Directory where evaluation artifacts should be written.",
    ),
) -> None:
    start_day = _normalize_date(start_date)
    end_day = _normalize_date(end_date)
    if end_day < start_day:
        raise typer.BadParameter("--end-date must be on or after --start-date.")

    minutes_key = minutes_source.strip().lower()
    if minutes_key not in {"predicted", "actual"}:
        raise typer.BadParameter("--minutes-source must be 'predicted' or 'actual'.")
    minutes_enum: MinutesSource = "predicted" if minutes_key == "predicted" else "actual"
    if minutes_enum == "actual":
        typer.echo(
            "[fpts-eval] WARNING: minutes_source=actual is optimistic and should only be used for experiments.",
            err=True,
        )

    data_root = data_root.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    run_dir = (artifact_root / run_id).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory missing at {run_dir}")
    config = _load_config(run_dir / "config.json")

    builder = FptsDatasetBuilder(
        data_root=data_root,
        minutes_source=minutes_enum,
        minutes_run_id=minutes_run_id,
    )
    frame = builder.build(start_day.to_pydatetime(), end_day.to_pydatetime())
    if frame.empty:
        raise RuntimeError("Evaluation dataset is empty for the requested window.")
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    eval_mask = (frame["game_date"] >= start_day) & (frame["game_date"] <= end_day)
    frame = frame.loc[eval_mask].copy()
    if frame.empty:
        raise RuntimeError("No rows remain after filtering to the requested dates.")

    bundle = load_fpts_model(run_id, artifact_root=artifact_root)
    preds_series = predict_fpts_per_min(bundle, frame)
    baseline_preds = baseline_per_minute(frame)
    minutes = pd.to_numeric(frame["actual_minutes"], errors="coerce").fillna(0.0)
    frame["fpts_pred"] = preds_series.to_numpy(dtype=float) * minutes
    frame["fpts_baseline"] = baseline_preds * minutes
    minutes_pred = pd.to_numeric(frame.get("minutes_p50_pred"), errors="coerce").fillna(minutes)
    frame["proj_fpts"] = preds_series.to_numpy(dtype=float) * minutes_pred

    eval_payload = evaluate_fpts_run(
        frame,
        model_preds=preds_series.to_numpy(dtype=float),
        baseline_preds=baseline_preds,
    )

    out_dir = (out_root / run_id).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    window_tag = f"{start_day.date().isoformat()}_{end_day.date().isoformat()}"
    base_filename = f"eval_{window_tag}_{minutes_enum}"
    json_path = out_dir / f"{base_filename}.json"
    md_path = out_dir / f"{base_filename}.md"

    artifact = {
        "run_id": run_id,
        "minutes_source": minutes_enum,
        "date_window": {"start": start_day.date().isoformat(), "end": end_day.date().isoformat()},
        "rows": int(len(frame)),
        "data_root": str(data_root),
        "artifact_root": str(run_dir),
        "minutes_run_id": minutes_run_id,
        "config": config,
        "metrics": eval_payload,
    }
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        _markdown_report(
            run_id=run_id,
            minutes_source=minutes_enum,
            start_day=start_day,
            end_day=end_day,
            rows=len(frame),
            metrics=eval_payload,
            artifact_root=run_dir,
            data_root=data_root,
        ),
        encoding="utf-8",
    )
    typer.echo(
        "[fpts-eval] Saved evaluation for run "
        f"{run_id} ({start_day.date().isoformat()} to {end_day.date().isoformat()}):\n"
        f"  - {json_path}\n"
        f"  - {md_path}"
    )


if __name__ == "__main__":
    app()
