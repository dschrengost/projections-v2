"""DFS slate-day readiness checklist.

Designed for "play tonight" workflows: run once before building lineups (and
again near lock) to catch stale inputs, run_id mismatches, and broken output
invariants.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console

from projections import paths
from projections.pipeline.dfs_readiness import run_dfs_readiness


app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@app.command()
def main(
    date: str = typer.Option(None, "--date", help="Slate date (YYYY-MM-DD). Defaults to today (UTC)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run_id to validate (defaults to latest pointers)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Root directory containing artifacts (defaults to PROJECTIONS_DATA_ROOT).",
    ),
    as_of_ts: datetime | None = typer.Option(
        None,
        "--as-of-ts",
        help="Reference timestamp for freshness checks (defaults to now UTC).",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="In strict mode, treat near-lock freshness failures as errors (exit 1).",
    ),
) -> None:
    if date is None:
        date = datetime.utcnow().date().isoformat()

    report = run_dfs_readiness(
        game_date=date,
        data_root=data_root.resolve(),
        run_id=run_id,
        as_of_ts=as_of_ts,
        strict=strict,
    )

    status = "PASS" if report.passed else "FAIL"
    console.print(f"[dfs-readiness] {status} game_date={report.game_date} run_id={report.run_id} as_of={report.as_of_ts}")

    if report.errors:
        console.print("\n[dfs-readiness] errors:")
        for msg in report.errors:
            console.print(f"  - {msg}")

    if report.warnings:
        console.print("\n[dfs-readiness] warnings:")
        for msg in report.warnings:
            console.print(f"  - {msg}")

    # Minimal metrics dump (paths + key numbers).
    metrics = report.metrics or {}
    if metrics:
        console.print("\n[dfs-readiness] metrics:")
        for key in (
            "schedule_path",
            "first_tip_ts",
            "minutes_to_first_tip",
            "minutes_team_sum_max_dev",
            "pred_own_pct_sum",
        ):
            if key in metrics:
                console.print(f"  - {key}={metrics.get(key)}")

    raise typer.Exit(code=0 if report.passed else 1)


if __name__ == "__main__":
    app()

