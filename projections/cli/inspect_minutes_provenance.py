"""CLI to inspect minutes artifact provenance.

Usage:
    uv run python -m projections.cli.inspect_minutes_provenance --date 2026-01-24 --run-id <run_id>
    uv run python -m projections.cli.inspect_minutes_provenance --artifact-dir /path/to/run/dir

This tool answers "what ran?" for any minutes artifact by reading and displaying
the provenance stamp from summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from projections import paths
from projections.minutes import (
    MINUTES_CONTRACT_VERSION,
    PLAY_THRESHOLD_MINUTES,
    ROTATION_THRESHOLD_MINUTES,
    find_minutes_run_dirs,
    get_latest_run_dir,
    load_provenance_from_summary,
    minutes_contract_hash,
)

app = typer.Typer(help=__doc__, add_completion=False)


def _print_contract_info() -> None:
    """Print current contract definitions."""
    typer.echo("Current Minutes Contract:")
    typer.echo(f"  version:                    {MINUTES_CONTRACT_VERSION}")
    typer.echo(f"  hash:                       {minutes_contract_hash()[:16]}...")
    typer.echo(f"  play_threshold_minutes:     {PLAY_THRESHOLD_MINUTES}")
    typer.echo(f"  rotation_threshold_minutes: {ROTATION_THRESHOLD_MINUTES}")
    typer.echo("")


def _print_provenance_from_dir(run_dir: Path) -> bool:
    """Print provenance from a run directory. Returns True if found."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        typer.echo(f"No summary.json found at {summary_path}", err=True)
        return False

    provenance = load_provenance_from_summary(summary_path)
    if provenance is None:
        # Try to extract basic info from summary even without full provenance
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            typer.echo(f"Invalid JSON in {summary_path}", err=True)
            return False

        typer.echo(f"Artifact: {run_dir}")
        typer.echo("  (No structured provenance stamp found)")
        typer.echo("")

        # Print available fields
        for key in ["date", "run_id", "model", "bundle_dir", "generated_at"]:
            if key in summary:
                typer.echo(f"  {key}: {summary[key]}")

        if "rotation_set_minutes" in summary:
            rot = summary["rotation_set_minutes"]
            typer.echo(f"  rotation_set_enabled: {rot.get('enabled', 'N/A')}")
            if rot.get("enabled"):
                typer.echo(f"  rotation_set_model_dir: {rot.get('model_dir', 'N/A')}")

        return True

    # Print full provenance
    typer.echo(f"Artifact: {run_dir}")
    typer.echo("=" * 60)
    typer.echo("Provenance Stamp:")
    typer.echo(f"  minutes_alloc_mode:         {provenance.minutes_alloc_mode}")
    typer.echo(f"  minutes_model_dir:          {provenance.minutes_model_dir or 'N/A'}")
    typer.echo(f"  minutes_run_id:             {provenance.minutes_run_id}")
    typer.echo(f"  game_date:                  {provenance.game_date}")
    typer.echo("")
    typer.echo("Code Version:")
    typer.echo(f"  git_sha:                    {provenance.git_sha}")
    typer.echo(f"  git_branch:                 {provenance.git_branch}")
    typer.echo(f"  git_dirty:                  {provenance.git_dirty}")
    typer.echo("")
    typer.echo("Contract:")
    typer.echo(f"  contract_version:           {provenance.contract_version}")
    typer.echo(f"  contract_hash:              {provenance.contract_hash[:16]}...")
    typer.echo(f"  play_threshold_minutes:     {provenance.play_threshold_minutes}")
    typer.echo(f"  rotation_threshold_minutes: {provenance.rotation_threshold_minutes}")
    typer.echo("")

    if provenance.degraded:
        typer.echo("Degradation:")
        typer.echo(f"  degraded:                   {provenance.degraded}")
        typer.echo(f"  degraded_reason:            {provenance.degraded_reason}")
        typer.echo("")

    typer.echo(f"Generated at: {provenance.generated_at}")

    if provenance.extras:
        typer.echo("")
        typer.echo("Extras:")
        for key, val in provenance.extras.items():
            typer.echo(f"  {key}: {val}")

    typer.echo("=" * 60)
    return True


@app.command()
def main(
    *,
    date: str | None = typer.Option(
        None,
        "--date",
        help="Slate date (YYYY-MM-DD). Use with --run-id or finds latest.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Run ID to inspect (required if --date is used without --artifact-dir).",
    ),
    artifact_dir: Path | None = typer.Option(
        None,
        "--artifact-dir",
        help="Direct path to run directory (overrides --date/--run-id).",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="PROJECTIONS_DATA_ROOT override.",
    ),
    list_runs: bool = typer.Option(
        False,
        "--list",
        help="List all runs for the given date instead of inspecting.",
    ),
    show_contract: bool = typer.Option(
        False,
        "--show-contract",
        help="Show current contract definitions and exit.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output provenance as JSON (for scripting).",
    ),
) -> None:
    """Inspect minutes artifact provenance to answer 'what ran?'."""

    if show_contract:
        _print_contract_info()
        return

    artifact_root = data_root / "artifacts" / "minutes_v1" / "daily"

    # Determine which directory to inspect
    if artifact_dir is not None:
        run_dir = Path(artifact_dir).expanduser().resolve()
    elif date is not None:
        if list_runs:
            runs = find_minutes_run_dirs(artifact_root, date)
            if not runs:
                typer.echo(f"No runs found for date={date}", err=True)
                raise typer.Exit(code=1)
            typer.echo(f"Runs for {date}:")
            for r in runs:
                typer.echo(f"  {r.name}")
            return

        if run_id is not None:
            run_dir = artifact_root / date / f"run={run_id}"
        else:
            # Find latest
            run_dir = get_latest_run_dir(artifact_root, date)
            if run_dir is None:
                typer.echo(f"No runs found for date={date}", err=True)
                raise typer.Exit(code=1)
            typer.echo(f"(Using latest run: {run_dir.name})", err=True)
    else:
        typer.echo(
            "Must specify either --date or --artifact-dir",
            err=True,
        )
        raise typer.Exit(code=1)

    if not run_dir.exists():
        typer.echo(f"Run directory not found: {run_dir}", err=True)
        raise typer.Exit(code=1)

    if json_output:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            typer.echo(f"No summary.json found at {summary_path}", err=True)
            raise typer.Exit(code=1)

        provenance = load_provenance_from_summary(summary_path)
        if provenance is not None:
            print(provenance.to_json())
        else:
            # Return whatever is in summary.json
            print(summary_path.read_text(encoding="utf-8"))
        return

    if not _print_provenance_from_dir(run_dir):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
