#!/usr/bin/env python3
"""Apply deployment settings that `prefect deploy` does not yet persist.

Prefect's deployment YAML support lags a few server-side fields (notably
`concurrency_options`). When we redeploy from `prefect.yaml`, these settings can
get dropped and the UI will show many flow runs "Awaiting concurrency slot".

Run this after `uv run prefect deploy --all`.
"""

from __future__ import annotations

import typer

from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import DeploymentUpdate
from prefect.client.schemas.objects import ConcurrencyOptions

app = typer.Typer(add_completion=False)


@app.command()
def main(
    deployment: str = typer.Option(
        "nba-live-pipeline-v3/nba-live-pipeline",
        "--deployment",
        help="Deployment name in <flow>/<deployment> form.",
    ),
    collision_strategy: str = typer.Option(
        "CANCEL_NEW",
        "--collision-strategy",
        help="What to do when concurrency is exhausted (e.g., CANCEL_NEW).",
    ),
    concurrency_limit: int = typer.Option(
        1,
        "--concurrency-limit",
        min=1,
        help="Hard cap for concurrent runs on the deployment.",
    ),
) -> None:
    with get_client(sync_client=True) as client:
        dep = client.read_deployment_by_name(deployment)
        update = DeploymentUpdate(
            concurrency_limit=concurrency_limit,
            concurrency_options=ConcurrencyOptions(collision_strategy=collision_strategy),
        )
        client.update_deployment(deployment_id=dep.id, deployment=update)
        typer.echo(
            "[prefect] updated "
            f"{deployment} "
            f"concurrency_limit={concurrency_limit} "
            f"concurrency_options.collision_strategy={collision_strategy}"
        )


if __name__ == "__main__":
    app()
