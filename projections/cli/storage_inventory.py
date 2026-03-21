"""Build read-only storage inventory reports."""

from __future__ import annotations

from pathlib import Path

import typer

from projections.storage_retention.inventory import (
    InventoryParams,
    build_inventory,
    write_inventory_reports,
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
    families: str | None = typer.Option(
        None,
        "--families",
        help="Comma-separated families. Defaults: gtv2_worlds,sim_v2_worlds_fpts_v2",
    ),
    start_date: str | None = typer.Option(None, "--start-date", help="YYYY-MM-DD"),
    end_date: str | None = typer.Option(None, "--end-date", help="YYYY-MM-DD"),
    skip_errors: bool = typer.Option(False, "--skip-errors/--no-skip-errors"),
) -> None:
    selected = _parse_families(families)
    roots = resolve_storage_roots(data_root=data_root, hot_root=hot_root)

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
    reports = write_inventory_reports(inventory, hot_root=roots.hot_root)

    typer.echo(
        "[storage_inventory] "
        f"runs={len(list(inventory.get('runs') or []))} "
        f"errors={len(list(inventory.get('errors') or []))} "
        f"json={reports['json']} csv={reports['csv']}"
    )


if __name__ == "__main__":
    app()
