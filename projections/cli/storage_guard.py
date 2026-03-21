"""Run free-space guard check for hot/root filesystems."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from projections.storage_retention.config import load_storage_retention_policy
from projections.storage_retention.guard import evaluate_storage_guard
from projections.storage_retention.paths import resolve_storage_roots

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data_root: Path | None = typer.Option(None, "--data-root"),
    hot_root: Path | None = typer.Option(None, "--hot-root"),
    config_path: Path | None = typer.Option(None, "--config-path"),
    root_path: Path = typer.Option(Path("/"), "--root-path"),
    report_path: Path | None = typer.Option(None, "--report-path"),
) -> None:
    roots = resolve_storage_roots(data_root=data_root, hot_root=hot_root)
    policy = load_storage_retention_policy(config_path=config_path)
    result = evaluate_storage_guard(
        hot_root=roots.hot_root,
        guard_policy=policy.guard,
        root_path=root_path,
    )

    payload = result.payload
    payload["ok"] = result.ok
    payload["hard_stop"] = result.hard_stop

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    typer.echo(
        "[storage_guard] "
        f"ok={result.ok} hard_stop={result.hard_stop} "
        f"hot_free_gb={float(payload['hot']['free_gb']):.1f} "
        f"hot_free_pct={float(payload['hot']['free_pct']):.1f} "
        f"root_free_gb={float(payload['root']['free_gb']):.1f}"
    )

    if result.hard_stop:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
