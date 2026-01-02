"""Diagnostic CLI to check registry resolution status.

Usage:
    uv run python -m projections.cli.check_registry
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import typer

app = typer.Typer(help="Check MLflow registry resolution status")


@app.command()
def main(
    tracking_uri: Optional[str] = typer.Option(
        None, "--tracking-uri", help="MLflow tracking URI override"
    ),
    show_env: bool = typer.Option(False, "--show-env", help="Show environment variables"),
) -> None:
    """Print resolver metadata for minutes_v1 and rates_v1."""
    from projections.registry.bundle_resolver import (
        resolve_minutes_bundle,
        resolve_rates_bundle,
        clear_bundle_caches,
    )

    if show_env:
        typer.echo("Environment:")
        typer.echo(f"  MLFLOW_TRACKING_URI={os.environ.get('MLFLOW_TRACKING_URI', '(not set)')}")
        typer.echo(f"  MINUTES_USE_FILESYSTEM_BUNDLE={os.environ.get('MINUTES_USE_FILESYSTEM_BUNDLE', '(not set)')}")
        typer.echo(f"  RATES_USE_FILESYSTEM_BUNDLE={os.environ.get('RATES_USE_FILESYSTEM_BUNDLE', '(not set)')}")
        typer.echo("")

    # Clear caches to get fresh resolution
    clear_bundle_caches()

    typer.echo("Registry Resolution Status:")
    typer.echo("-" * 50)

    # Check minutes_v1
    typer.echo("\nminutes_v1:")
    try:
        _, meta = resolve_minutes_bundle("minutes_v1", "production", tracking_uri=tracking_uri)
        typer.echo(f"  source:            {meta.get('source')}")
        typer.echo(f"  run_id:            {meta.get('run_id')}")
        if meta.get("source") == "mlflow_registry":
            typer.echo(f"  version:           {meta.get('version')}")
            typer.echo(f"  alias:             {meta.get('alias')}")
            typer.echo(f"  resolution_method: {meta.get('resolution_method')}")
        typer.echo(f"  ✅ Resolution successful")
    except Exception as e:
        typer.echo(f"  ❌ Resolution failed: {e}", err=True)

    # Check rates_v1
    typer.echo("\nrates_v1:")
    try:
        _, meta = resolve_rates_bundle("rates_v1", "production", tracking_uri=tracking_uri)
        typer.echo(f"  source:            {meta.get('source')}")
        typer.echo(f"  run_id:            {meta.get('run_id')}")
        if meta.get("source") == "mlflow_registry":
            typer.echo(f"  version:           {meta.get('version')}")
            typer.echo(f"  alias:             {meta.get('alias')}")
            typer.echo(f"  resolution_method: {meta.get('resolution_method')}")
        typer.echo(f"  ✅ Resolution successful")
    except Exception as e:
        typer.echo(f"  ❌ Resolution failed: {e}", err=True)


if __name__ == "__main__":
    app()
