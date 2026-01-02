"""CLI to check Prefect deployment status.

Usage:
    uv run python -m projections.cli.prefect_status --deployment nba-live-pipeline/nba-live-pipeline
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import typer

app = typer.Typer(help="Check Prefect deployment status")


@app.command()
def main(
    deployment: str = typer.Option(
        "nba-live-pipeline/nba-live-pipeline",
        "--deployment", "-d",
        help="Deployment name in format 'flow-name/deployment-name'",
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of runs to show"),
) -> None:
    """Print last N runs and most recent Completed run."""
    try:
        from prefect.client import get_client
        import asyncio
    except ImportError:
        typer.echo("Error: prefect not installed", err=True)
        raise typer.Exit(code=1)

    async def _get_runs():
        async with get_client() as client:
            # Get deployment by name
            try:
                flow_name, deploy_name = deployment.split("/", 1)
            except ValueError:
                typer.echo(f"Error: deployment must be 'flow-name/deployment-name', got '{deployment}'", err=True)
                raise typer.Exit(code=1)

            # Find deployment
            deployments = await client.read_deployments()
            target_deploy = None
            for d in deployments:
                if d.name == deploy_name:
                    # Verify flow name matches
                    flow = await client.read_flow(d.flow_id)
                    if flow.name == flow_name:
                        target_deploy = d
                        break

            if not target_deploy:
                typer.echo(f"Error: deployment '{deployment}' not found", err=True)
                raise typer.Exit(code=1)

            # Get flow runs for this deployment
            from prefect.client.schemas.filters import (
                FlowRunFilter,
                FlowRunFilterDeploymentId,
            )
            from prefect.client.schemas.sorting import FlowRunSort

            flow_runs = await client.read_flow_runs(
                flow_run_filter=FlowRunFilter(
                    deployment_id=FlowRunFilterDeploymentId(any_=[target_deploy.id])
                ),
                sort=FlowRunSort.START_TIME_DESC,
                limit=limit,
            )

            return flow_runs

    runs = asyncio.run(_get_runs())

    if not runs:
        typer.echo(f"No runs found for deployment '{deployment}'")
        return

    typer.echo(f"\nLast {min(len(runs), limit)} runs for '{deployment}':")
    typer.echo("-" * 80)
    typer.echo(f"{'Start Time':<25} {'State':<12} {'Duration':<12} {'Run ID'}")
    typer.echo("-" * 80)

    last_completed = None

    for run in runs:
        # Format start time
        start = run.start_time or run.expected_start_time or run.created
        if start:
            start_str = start.strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_str = "(not started)"

        # Format state
        state_name = run.state.name if run.state else "UNKNOWN"

        # Track last completed
        if state_name == "Completed" and last_completed is None:
            last_completed = run

        # Format duration
        if run.start_time and run.end_time:
            duration = run.end_time - run.start_time
            mins = int(duration.total_seconds() // 60)
            secs = int(duration.total_seconds() % 60)
            duration_str = f"{mins}m {secs}s"
        else:
            duration_str = "-"

        # Color state
        if state_name == "Completed":
            state_display = f"✅ {state_name}"
        elif state_name == "Failed":
            state_display = f"❌ {state_name}"
        elif state_name == "Running":
            state_display = f"🔄 {state_name}"
        else:
            state_display = f"   {state_name}"

        typer.echo(f"{start_str:<25} {state_display:<12} {duration_str:<12} {str(run.id)[:8]}")

    typer.echo("-" * 80)

    # Summary
    if last_completed:
        start = last_completed.start_time or last_completed.created
        typer.echo(f"\n✅ Last Completed: {start.strftime('%Y-%m-%d %H:%M:%S')} | run_id={str(last_completed.id)[:12]}")
    else:
        typer.echo(f"\n❌ No Completed runs in last {limit} runs")


if __name__ == "__main__":
    app()
