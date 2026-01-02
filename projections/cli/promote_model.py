"""CLI for promoting trained models to production aliases.

Usage:
    uv run python -m projections.cli.promote_model \\
        --model minutes_v1 \\
        --run-id abc123 \\
        --alias production \\
        --reason "Improved MAE on validation set"
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import typer

try:
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MlflowClient = None  # type: ignore
    MLFLOW_AVAILABLE = False

from projections.mlflow_utils import get_tracking_uri, log_artifact_text
from projections.registry.promotion_gates import (
    check_promotion_readiness,
    get_run_metadata,
)

app = typer.Typer(help="Promote trained models to production aliases in MLflow registry")


@app.command()
def promote(
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g., minutes_v1)"),
    run_id: str = typer.Option(..., "--run-id", "-r", help="MLflow run ID to promote"),
    alias: str = typer.Option("production", "--alias", "-a", help="Alias to set (e.g., production, staging)"),
    reason: str = typer.Option(..., "--reason", help="Reason for promotion (recorded in manifest)"),
    skip_gates: bool = typer.Option(False, "--skip-gates", help="Skip promotion readiness checks"),
    tracking_uri: Optional[str] = typer.Option(None, "--tracking-uri", help="MLflow tracking URI override"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without making changes"),
) -> None:
    """Promote a trained model run to a registry alias.

    This command:
    1. Validates the run has required artifacts (model, manifest, schema, metrics)
    2. Registers the model to MLflow Model Registry if not already registered
    3. Sets the specified alias on the model version
    4. Records promotion metadata (who, when, why)
    """
    if not MLFLOW_AVAILABLE:
        typer.echo("Error: MLflow is not installed. Install with: pip install mlflow", err=True)
        raise typer.Exit(code=1)

    uri = tracking_uri or get_tracking_uri()
    if not uri:
        typer.echo("Error: MLFLOW_TRACKING_URI not set and --tracking-uri not provided", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Promoting model '{model}' from run '{run_id}' to alias '{alias}'")
    typer.echo(f"Tracking URI: {uri}")

    # Run promotion gates
    if not skip_gates:
        typer.echo("\nRunning promotion readiness checks...")
        passed, issues = check_promotion_readiness(run_id, tracking_uri=uri)

        if not passed:
            typer.echo("\n❌ Promotion readiness check FAILED:", err=True)
            for issue in issues:
                typer.echo(f"  • {issue}", err=True)
            typer.echo("\nUse --skip-gates to bypass these checks (not recommended)", err=True)
            raise typer.Exit(code=1)

        typer.echo("✅ All readiness checks passed")

    if dry_run:
        typer.echo("\n[DRY RUN] Would perform the following actions:")
        typer.echo(f"  1. Register model '{model}' from run {run_id}")
        typer.echo(f"  2. Set alias '{alias}' on the new version")
        typer.echo(f"  3. Record promotion reason: {reason}")
        return

    # Initialize client
    client = MlflowClient(tracking_uri=uri)

    # Get run metadata for the promotion record
    run_metadata = get_run_metadata(run_id, tracking_uri=uri)

    # Check if model exists in registry, create if not
    try:
        client.get_registered_model(model)
        typer.echo(f"\nModel '{model}' already exists in registry")
    except Exception:
        typer.echo(f"\nCreating new registered model '{model}'")
        client.create_registered_model(
            model,
            description=f"Minutes prediction model. Created during promotion of run {run_id}",
        )

    # Find model artifacts in the run to register
    artifacts = client.list_artifacts(run_id)
    artifact_names = [a.path for a in artifacts]

    # Determine artifact path - look for common model patterns
    model_artifact_path = None
    for candidate in ["model", "lgbm_quantiles.joblib", "minute_share_model.joblib", "rotation_share_model.joblib"]:
        if candidate in artifact_names:
            model_artifact_path = candidate
            break

    if not model_artifact_path:
        # Look for any .joblib file
        joblib_files = [a for a in artifact_names if a.endswith(".joblib")]
        if joblib_files:
            model_artifact_path = joblib_files[0]

    if not model_artifact_path and not skip_gates:
        typer.echo("Error: Could not find model artifact in run", err=True)
        raise typer.Exit(code=1)

    # Register model version
    typer.echo(f"\nRegistering model from artifact: {model_artifact_path or 'model'}")
    try:
        model_uri = f"runs:/{run_id}/{model_artifact_path or 'model'}"
        result = client.create_model_version(
            name=model,
            source=model_uri,
            run_id=run_id,
            description=f"Promoted to {alias}: {reason}",
        )
        version = result.version
        typer.echo(f"Created model version: {version}")
    except Exception as e:
        typer.echo(f"Error registering model version: {e}", err=True)
        raise typer.Exit(code=1)

    # Set the alias
    try:
        client.set_registered_model_alias(model, alias, version)
        typer.echo(f"Set alias '{alias}' on version {version}")
    except Exception as e:
        typer.echo(f"Error setting alias: {e}", err=True)
        raise typer.Exit(code=1)

    # Record promotion metadata as tags on the version
    promotion_time = datetime.now(timezone.utc).isoformat()
    user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))

    tags_to_set = {
        "promotion_alias": alias,
        "promotion_time": promotion_time,
        "promotion_reason": reason[:250],  # Truncate for tag limits
        "promotion_user": user,
        "source_run_id": run_id,
    }

    for key, value in tags_to_set.items():
        try:
            client.set_model_version_tag(model, version, key, value)
        except Exception as e:
            typer.echo(f"Warning: Failed to set tag {key}: {e}", err=True)

    # Create promotion manifest artifact in the original run
    promotion_manifest = {
        "model_name": model,
        "version": version,
        "alias": alias,
        "run_id": run_id,
        "reason": reason,
        "promoted_by": user,
        "promoted_at": promotion_time,
        "run_metadata": run_metadata,
    }

    try:
        # Log to the original run
        import mlflow

        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_id=run_id):
            log_artifact_text("promotion_manifest.json", json.dumps(promotion_manifest, indent=2, default=str))
        typer.echo("Recorded promotion manifest in run")
    except Exception as e:
        typer.echo(f"Warning: Failed to record promotion manifest: {e}", err=True)

    typer.echo(f"\n✅ Successfully promoted {model} v{version} to '{alias}'")
    typer.echo(f"   Run ID: {run_id}")
    typer.echo(f"   Reason: {reason}")


@app.command()
def list_versions(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    tracking_uri: Optional[str] = typer.Option(None, "--tracking-uri", help="MLflow tracking URI override"),
) -> None:
    """List all versions of a registered model."""
    if not MLFLOW_AVAILABLE:
        typer.echo("Error: MLflow not installed", err=True)
        raise typer.Exit(code=1)

    uri = tracking_uri or get_tracking_uri()
    client = MlflowClient(tracking_uri=uri)

    try:
        versions = client.search_model_versions(f"name='{model}'")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    if not versions:
        typer.echo(f"No versions found for model '{model}'")
        return

    typer.echo(f"\nVersions of '{model}':\n")
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        aliases = v.aliases if hasattr(v, "aliases") else []
        alias_str = f" [{', '.join(aliases)}]" if aliases else ""
        typer.echo(f"  v{v.version}{alias_str} - Run: {v.run_id[:12]} - {v.status}")


@app.command()
def get_production(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    alias: str = typer.Option("production", "--alias", "-a", help="Alias to look up"),
    tracking_uri: Optional[str] = typer.Option(None, "--tracking-uri", help="MLflow tracking URI override"),
) -> None:
    """Get info about the current production model version."""
    if not MLFLOW_AVAILABLE:
        typer.echo("Error: MLflow not installed", err=True)
        raise typer.Exit(code=1)

    uri = tracking_uri or get_tracking_uri()
    client = MlflowClient(tracking_uri=uri)

    try:
        version = client.get_model_version_by_alias(model, alias)
    except Exception as e:
        typer.echo(f"Error: No model found with alias '{alias}': {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\n'{model}' @ {alias}:")
    typer.echo(f"  Version: {version.version}")
    typer.echo(f"  Run ID: {version.run_id}")
    typer.echo(f"  Created: {version.creation_timestamp}")
    typer.echo(f"  Description: {version.description or '(none)'}")


if __name__ == "__main__":
    app()
