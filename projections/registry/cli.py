"""CLI for model registry operations.

Provides commands to:
- List registered models and versions
- Promote models between stages
- Show production model info
- Register new models from training runs
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from projections.registry.manifest import (
    Stage,
    load_manifest,
    save_manifest,
    promote_model,
    get_production_model,
    list_versions,
    DEFAULT_MANIFEST_PATH,
)

console = Console()
app = typer.Typer(help="Model registry operations.")


@app.command("list")
def list_models(
    model_name: Optional[str] = typer.Argument(None, help="Model name to list versions for."),
    stage: Optional[str] = typer.Option(None, "--stage", "-s", help="Filter by stage (dev/staging/prod)."),
    manifest_path: Path = typer.Option(
        DEFAULT_MANIFEST_PATH, "--manifest", "-m", help="Path to manifest file."
    ),
) -> None:
    """List registered models and their versions."""
    manifest = load_manifest(manifest_path)

    if not manifest.models:
        console.print("[yellow]No models registered.[/yellow]")
        return

    if model_name:
        # List versions for specific model
        stage_filter: Stage | None = stage if stage in ("dev", "staging", "prod") else None  # type: ignore
        versions = list_versions(manifest, model_name, stage=stage_filter)

        if not versions:
            console.print(f"[yellow]No versions found for {model_name}[/yellow]")
            return

        entry = manifest.get_model(model_name)
        table = Table(title=f"Model: {model_name}")
        table.add_column("Version", style="cyan")
        table.add_column("Run ID", style="dim")
        table.add_column("Stage", style="green")
        table.add_column("Created", style="dim")
        table.add_column("Notes")

        for v in versions:
            is_prod = entry and entry.current_prod == v.version
            is_staging = entry and entry.current_staging == v.version
            marker = ""
            if is_prod:
                marker = " ★ PROD"
            elif is_staging:
                marker = " → STAGING"

            table.add_row(
                v.version + marker,
                v.run_id[:16] + "...",
                v.stage,
                v.created_at[:10],
                v.description[:30] if v.description else "",
            )

        console.print(table)
    else:
        # List all models
        table = Table(title="Registered Models")
        table.add_column("Model", style="cyan")
        table.add_column("Versions", style="dim")
        table.add_column("Production", style="green")
        table.add_column("Staging", style="yellow")

        for name, entry in manifest.models.items():
            table.add_row(
                name,
                str(len(entry.versions)),
                entry.current_prod or "-",
                entry.current_staging or "-",
            )

        console.print(table)


@app.command("promote")
def promote(
    model_name: str = typer.Argument(..., help="Model name."),
    version: str = typer.Argument(..., help="Version to promote."),
    stage: str = typer.Option(..., "--stage", "-s", help="Target stage (dev/staging/prod)."),
    manifest_path: Path = typer.Option(
        DEFAULT_MANIFEST_PATH, "--manifest", "-m", help="Path to manifest file."
    ),
) -> None:
    """Promote a model version to a new stage."""
    if stage not in ("dev", "staging", "prod"):
        console.print(f"[red]Invalid stage: {stage}. Must be dev/staging/prod.[/red]")
        raise typer.Exit(code=1)

    manifest = load_manifest(manifest_path)
    try:
        model_version = promote_model(manifest, model_name=model_name, version=version, stage=stage)  # type: ignore
        save_manifest(manifest, manifest_path)
        console.print(
            f"[green]✓ Promoted {model_name} v{version} to {stage}[/green]"
        )
        if stage == "prod":
            console.print(f"  Artifact path: {model_version.artifact_path}")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("show")
def show(
    model_name: str = typer.Argument(..., help="Model name."),
    version: Optional[str] = typer.Argument(None, help="Version (defaults to production)."),
    manifest_path: Path = typer.Option(
        DEFAULT_MANIFEST_PATH, "--manifest", "-m", help="Path to manifest file."
    ),
) -> None:
    """Show details for a specific model version."""
    manifest = load_manifest(manifest_path)
    entry = manifest.get_model(model_name)

    if entry is None:
        console.print(f"[red]Model {model_name} not found.[/red]")
        raise typer.Exit(code=1)

    if version is None:
        # Default to production
        version = entry.current_prod
        if version is None:
            console.print(f"[yellow]No production version set for {model_name}.[/yellow]")
            raise typer.Exit(code=1)

    model_version = entry.get_version(version)
    if model_version is None:
        console.print(f"[red]Version {version} not found.[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]{model_name}[/bold cyan] v{model_version.version}")
    console.print(f"  Stage: [green]{model_version.stage}[/green]")
    console.print(f"  Run ID: {model_version.run_id}")
    console.print(f"  Artifact: {model_version.artifact_path}")
    console.print(f"  Created: {model_version.created_at}")
    if model_version.promoted_at:
        console.print(f"  Promoted: {model_version.promoted_at}")
    if model_version.training_start:
        console.print(f"  Training window: {model_version.training_start} → {model_version.training_end}")
    if model_version.metrics:
        console.print("  Metrics:")
        for k, v in model_version.metrics.items():
            console.print(f"    {k}: {v:.4f}")
    if model_version.description:
        console.print(f"  Description: {model_version.description}")


@app.command("production")
def production(
    model_name: str = typer.Argument(..., help="Model name."),
    manifest_path: Path = typer.Option(
        DEFAULT_MANIFEST_PATH, "--manifest", "-m", help="Path to manifest file."
    ),
) -> None:
    """Get the production model path (for use in scripts)."""
    manifest = load_manifest(manifest_path)
    prod = get_production_model(manifest, model_name)

    if prod is None:
        console.print(f"[red]No production version for {model_name}[/red]", err=True)
        raise typer.Exit(code=1)

    # Output just the path for scripting
    print(prod.artifact_path)


def main() -> None:
    """Entry point for `python -m projections.registry.cli`."""
    app()


if __name__ == "__main__":
    main()
