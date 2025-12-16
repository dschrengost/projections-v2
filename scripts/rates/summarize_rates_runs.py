"""
Summarize recent rates_v1 runs for quick comparison.

Reads artifacts/rates_v1/runs/<run_id>/metrics.json and meta.json and prints mean
MAE for cal/val splits.

Example:
    uv run python -m scripts.rates.summarize_rates_runs \\
        --runs rates_v1_stage0_20251202_101500 rates_v1_stage2_tracking_20251202_110000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import typer

from projections.paths import data_path

app = typer.Typer(add_completion=False)


def _load_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    def _mean_mae(key: str) -> float | None:
        vals = [v.get(key) for v in metrics.values() if v.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    return {
        "run_id": meta.get("run_id", run_dir.name),
        "feature_set": meta.get("feature_set") or meta.get("run_tag"),
        "train_rows": meta.get("train_rows"),
        "cal_rows": meta.get("cal_rows"),
        "val_rows": meta.get("val_rows"),
        "date_window": meta.get("date_window"),
        "cal_mean_mae": _mean_mae("cal_mae"),
        "val_mean_mae": _mean_mae("val_mae"),
    }


def _iter_run_dirs(root: Path, runs: Optional[Iterable[str]]) -> list[Path]:
    runs_root = root / "artifacts" / "rates_v1" / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")
    if runs:
        return [runs_root / r for r in runs]
    return sorted([p for p in runs_root.iterdir() if p.is_dir()])


@app.command()
def main(
    runs: Optional[list[str]] = typer.Argument(
        None, help="Optional run_ids to summarize. When omitted, summarizes all."
    ),
    artifacts_root: Optional[Path] = typer.Option(
        None, help="Override artifacts root (defaults to PROJECTIONS_DATA_ROOT)."
    ),
) -> None:
    root = artifacts_root or data_path()
    run_dirs = _iter_run_dirs(root, runs)
    summaries: list[dict] = []
    for rd in run_dirs:
        try:
            summaries.append(_load_run(rd))
        except FileNotFoundError:
            typer.echo(f"[summary] skipping {rd} (missing files)")
        except json.JSONDecodeError:
            typer.echo(f"[summary] skipping {rd} (invalid JSON)")

    if not summaries:
        typer.echo("[summary] no runs summarized.")
        raise typer.Exit(code=1)

    headers = ["run_id", "feature_set", "cal_mean_mae", "val_mean_mae", "train_rows", "cal_rows", "val_rows"]
    typer.echo(" | ".join(headers))
    for row in summaries:
        typer.echo(
            " | ".join(
                str(row.get(h, ""))
                for h in headers
            )
        )


if __name__ == "__main__":
    app()
