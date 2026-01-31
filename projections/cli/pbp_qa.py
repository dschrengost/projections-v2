"""Run QA gates for a PBP v1 bundle (Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from projections.pbp.constants import PBP_V1_SCHEMA_VERSION
from projections.pbp.qa import run_qa_gates

console = Console()
app = typer.Typer(help="Run QA gates on stints and write qa_report + qa_failures.")


def _atomic_write_json(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out_path)


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


@app.command()
def run(
    bundle_dir: Path = typer.Argument(..., help="Run bundle dir (e.g. /.../artifacts/pbp_v1/<run_id>/)."),
    tolerance_sec: int = typer.Option(1, "--tolerance-sec", help="Allowed duration sum mismatch tolerance."),
    allow_failures: bool = typer.Option(
        False,
        "--allow-failures",
        help="Write QA outputs but do not exit non-zero on failures.",
    ),
) -> None:
    stints_path = bundle_dir / "stints.parquet"
    qa_report_path = bundle_dir / "qa_report.json"
    qa_failures_path = bundle_dir / "qa_failures.parquet"

    if not stints_path.exists():
        console.print(f"[red]Missing input[/red] {stints_path}")
        raise typer.Exit(1)

    stints = pd.read_parquet(stints_path)
    if len(stints) == 0:
        console.print("[red]stints is empty[/red]")
        raise typer.Exit(1)

    season_id = str(stints["season_id"].iloc[0]) if "season_id" in stints.columns else "unknown"
    run_id = bundle_dir.name

    outputs = run_qa_gates(
        stints,
        season_id=season_id,
        run_id=run_id,
        schema_version=PBP_V1_SCHEMA_VERSION,
        tolerance_sec=tolerance_sec,
    )

    _atomic_write_json(outputs.report, qa_report_path)
    _atomic_write_parquet(outputs.failures, qa_failures_path)

    totals = outputs.report.get("totals", {})
    console.print(f"qa_report: {qa_report_path}")
    console.print(f"qa_failures: {qa_failures_path} ({len(outputs.failures):,} rows)")
    console.print(
        f"games_total={totals.get('games_total')} passed={totals.get('games_passed')} "
        f"failed={totals.get('games_failed')} pass_rate={totals.get('pass_rate')}"
    )

    if totals.get("games_failed", 0) and not allow_failures:
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

