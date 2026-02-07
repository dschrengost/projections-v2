"""Build and train a recency-weighted Minutes V1 retrain run."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import typer

from projections.minutes_v1.retrain_dataset import build_retrain_dataset

app = typer.Typer(help=__doc__)


def _default_run_id() -> str:
    return datetime.now(tz=UTC).strftime("minutes_v1_recency_h35_%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@app.command("build-dataset")
def build_dataset(
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="Dataset run id under $PROJECTIONS_DATA_ROOT/artifacts/minutes_retrain_runs/.",
    ),
    data_root: Path = typer.Option(
        Path("/home/daniel/projections-data"),
        "--data-root",
        help="Projections data root.",
    ),
    season: int = typer.Option(2025, "--season", help="Season partition in labels/features paths."),
    train_start_date: str = typer.Option("2025-02-01", "--train-start-date"),
    train_end_date: str = typer.Option("2026-01-31", "--train-end-date"),
    cal_start_date: str = typer.Option("2026-02-01", "--cal-start-date"),
    cal_end_date: str = typer.Option("2026-02-05", "--cal-end-date"),
    half_life_days: float = typer.Option(35.0, "--half-life-days"),
) -> None:
    """Build the leakage-safe retrain parquet with recency weights."""

    effective_run_id = run_id.strip() or _default_run_id()
    result = build_retrain_dataset(
        data_root=data_root,
        run_id=effective_run_id,
        season=season,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        cal_start_date=cal_start_date,
        cal_end_date=cal_end_date,
        half_life_days=half_life_days,
    )
    meta = _read_json(result.meta_path)
    summary = meta.get("summary", {})
    typer.echo(
        "[retrain-dataset] "
        f"run_id={result.run_id} rows={summary.get('dataset_rows')} "
        f"split_counts={summary.get('split_counts')} "
        f"path={result.dataset_path}"
    )
    typer.echo(f"[retrain-dataset] meta={result.meta_path}")


@app.command("run")
def run(
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="Run id used for both dataset artifact and trained model bundle.",
    ),
    data_root: Path = typer.Option(
        Path("/home/daniel/projections-data"),
        "--data-root",
        help="Projections data root.",
    ),
    season: int = typer.Option(2025, "--season", help="Season partition in labels/features paths."),
    train_start_date: str = typer.Option("2025-02-01", "--train-start-date"),
    train_end_date: str = typer.Option("2026-01-31", "--train-end-date"),
    cal_start_date: str = typer.Option("2026-02-01", "--cal-start-date"),
    cal_end_date: str = typer.Option("2026-02-05", "--cal-end-date"),
    half_life_days: float = typer.Option(35.0, "--half-life-days"),
    train_random_state: int = typer.Option(42, "--train-random-state"),
    allow_guard_failure: bool = typer.Option(
        False,
        "--allow-guard-failure",
        help="Forwarded to minutes_lgbm to emit guardrail violations as warnings instead of aborting.",
        is_flag=True,
    ),
) -> None:
    """Build dataset, train minutes_lgbm with recency weights, and print validation gates."""

    effective_run_id = run_id.strip() or _default_run_id()
    dataset = build_retrain_dataset(
        data_root=data_root,
        run_id=effective_run_id,
        season=season,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        cal_start_date=cal_start_date,
        cal_end_date=cal_end_date,
        half_life_days=half_life_days,
    )
    dataset_meta = _read_json(dataset.meta_path)

    model_root = data_root / "artifacts" / "minutes_lgbm"
    model_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "projections.models.minutes_lgbm",
        "--features",
        str(dataset.dataset_path),
        "--run-id",
        effective_run_id,
        "--artifact-root",
        str(model_root),
        "--sample-weight-col",
        "weight_recency",
        "--target-col",
        "minutes",
        "--train-start",
        train_start_date,
        "--train-end",
        train_end_date,
        "--cal-start",
        cal_start_date,
        "--cal-end",
        cal_end_date,
        "--val-start",
        cal_start_date,
        "--val-end",
        cal_end_date,
        "--allow-val-cal-overlap",
        "--random-state",
        str(train_random_state),
    ]
    if allow_guard_failure:
        cmd.append("--allow-guard-failure")
    typer.echo("[retrain-train] running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    model_run_dir = model_root / effective_run_id
    metrics_path = model_run_dir / "metrics.json"
    model_meta_path = model_run_dir / "meta.json"
    if not metrics_path.exists() or not model_meta_path.exists():
        raise RuntimeError(f"Training completed but expected artifacts are missing under {model_run_dir}")

    metrics = _read_json(metrics_path)
    model_meta = _read_json(model_meta_path)
    model_meta["retrain_dataset"] = {
        "run_id": dataset.run_id,
        "dataset_path": str(dataset.dataset_path),
        "meta_path": str(dataset.meta_path),
    }
    model_meta["recency_decay"] = dataset_meta.get("recency_decay", {})
    model_meta["train_cal_windows"] = dataset_meta.get("effective_windows", {})
    _write_json(model_meta_path, model_meta)

    typer.echo(
        "[retrain-metrics] "
        f"false_active={metrics.get('val_false_active_rate_p_ge_0_5')} "
        f"false_inactive={metrics.get('val_false_inactive_rate_p_le_0_2')} "
        f"brier={metrics.get('val_play_prob_brier')} "
        f"mae_p50_cond={metrics.get('val_mae_p50_conditional')} "
        f"bench_smear={metrics.get('val_bench_smear_proxy_p50_gt_10_actual_lt_1')}"
    )
    typer.echo(f"[retrain-output] dataset={dataset.dataset_path}")
    typer.echo(f"[retrain-output] model_bundle={model_run_dir}")


if __name__ == "__main__":  # pragma: no cover
    app()
