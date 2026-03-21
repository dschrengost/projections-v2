"""Select canonical runs and optionally persist decision files."""

from __future__ import annotations

from pathlib import Path

import typer

from projections.storage_retention.canonical import (
    classify_inventory_runs,
    write_decision_reports,
)
from projections.storage_retention.config import load_storage_retention_policy
from projections.storage_retention.inventory import (
    InventoryParams,
    build_inventory,
)
from projections.storage_retention.paths import FAMILY_ROOTS, resolve_storage_roots

app = typer.Typer(add_completion=False)


def _parse_families(raw: str | None) -> tuple[str, ...]:
    if raw is None or not str(raw).strip():
        return tuple(FAMILY_ROOTS.keys())
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


@app.command()
def main(
    data_root: Path | None = typer.Option(None, "--data-root"),
    hot_root: Path | None = typer.Option(None, "--hot-root"),
    config_path: Path | None = typer.Option(None, "--config-path"),
    families: str | None = typer.Option(None, "--families"),
    start_date: str | None = typer.Option(None, "--start-date", help="YYYY-MM-DD"),
    end_date: str | None = typer.Option(None, "--end-date", help="YYYY-MM-DD"),
    skip_errors: bool = typer.Option(False, "--skip-errors/--no-skip-errors"),
    write_decisions: bool = typer.Option(False, "--write-decisions/--no-write-decisions"),
) -> None:
    selected = _parse_families(families)
    roots = resolve_storage_roots(data_root=data_root, hot_root=hot_root)
    policy = load_storage_retention_policy(config_path=config_path)

    inventory = build_inventory(
        InventoryParams(
            data_root=roots.data_root,
            hot_root=roots.hot_root,
            families=selected,
            start_date=start_date,
            end_date=end_date,
            skip_errors=bool(skip_errors),
        )
    )
    canonical = classify_inventory_runs(
        inventory=inventory,
        retention_policy=policy.retention,
    )
    reports = write_decision_reports(
        canonical_output=canonical,
        hot_root=roots.hot_root,
        write_per_run_decisions=bool(write_decisions),
    )

    decisions = list(canonical.get("decisions") or [])
    protected = [row for row in decisions if bool(row.get("protected"))]
    pruneable = [
        row
        for row in decisions
        if (not bool(row.get("protected"))) and str(row.get("classification")) == "noncanonical"
    ]

    typer.echo(
        "[storage_select_canonical] "
        f"decisions={len(decisions)} protected={len(protected)} "
        f"prune_candidates={len(pruneable)} map={reports['canonical_map_json']}"
    )


if __name__ == "__main__":
    app()
