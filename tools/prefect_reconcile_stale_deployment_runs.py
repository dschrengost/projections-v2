#!/usr/bin/env python3
"""Cancel stale Prefect deployment runs that reference dead infrastructure PIDs.

This is intended for deployment-level concurrency cleanup. If the worker is
restarted while a deployment run is marked RUNNING, Prefect can retain the
deployment lease until the stale run is cancelled manually. That causes new
triggers to be cancelled immediately when the deployment uses
`collision_strategy=CANCEL_NEW`.

Run this script before starting the worker. It only cancels runs that are old
enough to be considered stale and either:
1. reference a dead `infrastructure_pid`, or
2. have no `infrastructure_pid` after the grace window.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib import error, request

import typer

app = typer.Typer(add_completion=False)


def _api_url() -> str:
    api_url = os.environ.get("PREFECT_API_URL", "").strip()
    if not api_url:
        raise typer.BadParameter("PREFECT_API_URL is required")
    return api_url.rstrip("/")


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
) -> Any:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    if not data:
        return None
    return json.loads(data.decode("utf-8"))


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    norm = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(norm)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _pid_alive(pid_value: str | int | None) -> bool:
    if pid_value in (None, "", 0, "0"):
        return False
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _cancel_flow_run(api_url: str, flow_run_id: str, message: str, dry_run: bool) -> None:
    if dry_run:
        typer.echo(f"[prefect-janitor] dry-run would cancel {flow_run_id}: {message}")
        return
    payload = {
        "state": {
            "type": "CANCELLED",
            "name": "Cancelled",
            "message": message,
        },
        "force": True,
    }
    _request_json(
        "POST",
        f"{api_url}/flow_runs/{flow_run_id}/set_state",
        payload=payload,
    )
    typer.echo(f"[prefect-janitor] cancelled stale run {flow_run_id}: {message}")


@app.command()
def main(
    deployment: str = typer.Option(
        "nba-live-pipeline-v3/nba-live-pipeline",
        "--deployment",
        help="Deployment in <flow>/<deployment> form.",
    ),
    max_age_minutes: int = typer.Option(
        10,
        "--max-age-minutes",
        min=1,
        help="Only cancel runs older than this when infrastructure_pid is missing.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report stale runs without cancelling them.",
    ),
) -> None:
    api_url = _api_url()
    if "/" not in deployment:
        raise typer.BadParameter("deployment must be in <flow>/<deployment> form")
    _flow_name, deployment_name = deployment.split("/", 1)

    deployments = _request_json(
        "POST",
        f"{api_url}/deployments/filter",
        payload={"deployments": {"name": {"any_": [deployment_name]}}, "limit": 200},
    )
    deployment_payload = next(
        (item for item in deployments if item.get("name") == deployment_name),
        None,
    )
    if deployment_payload is None:
        raise typer.BadParameter(f"deployment not found: {deployment}")
    deployment_id = deployment_payload["id"]

    runs = _request_json(
        "POST",
        f"{api_url}/flow_runs/filter",
        payload={
            "flow_runs": {
                "deployment_id": {"any_": [deployment_id]},
                "state": {"type": {"any_": ["RUNNING", "CANCELLING"]}},
            },
            "limit": 200,
            "sort": "START_TIME_ASC",
        },
    )

    now = datetime.now(tz=UTC)
    missing_pid_cutoff = now - timedelta(minutes=max_age_minutes)
    cancelled = 0

    for run in runs:
        flow_run_id = run["id"]
        pid_value = run.get("infrastructure_pid")
        start_time = _parse_ts(run.get("start_time"))
        updated_time = _parse_ts(run.get("updated"))
        age_anchor = start_time or updated_time
        if pid_value not in (None, "", 0, "0"):
            if _pid_alive(pid_value):
                continue
            message = (
                f"Cancelled stale deployment run after worker restart; "
                f"infrastructure PID {pid_value} no longer exists."
            )
            _cancel_flow_run(api_url, flow_run_id, message, dry_run)
            cancelled += 1
            continue

        if age_anchor is None or age_anchor > missing_pid_cutoff:
            continue
        message = (
            "Cancelled stale deployment run with missing infrastructure PID "
            f"older than {max_age_minutes} minutes."
        )
        _cancel_flow_run(api_url, flow_run_id, message, dry_run)
        cancelled += 1

    typer.echo(
        f"[prefect-janitor] deployment={deployment_name} stale_runs_cancelled={cancelled}"
    )


if __name__ == "__main__":
    try:
        app()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"[prefect-janitor] HTTP {exc.code}: {detail}", file=sys.stderr)
        raise SystemExit(1) from exc
