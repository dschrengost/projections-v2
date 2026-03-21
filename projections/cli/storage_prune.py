"""Plan/execute storage prune using canonical retention decisions."""

from __future__ import annotations

from pathlib import Path

import typer

from projections.storage_retention.canonical import classify_inventory_runs
from projections.storage_retention.config import load_storage_retention_policy
from projections.storage_retention.inventory import InventoryParams, build_inventory
from projections.storage_retention.paths import FAMILY_ROOTS, resolve_storage_roots
from projections.storage_retention.prune import (
    PruneParams,
    assert_no_active_writer,
    build_prune_plan,
    execute_prune_plan,
    load_json,
    write_prune_reports,
)

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
    canonical_json: Path | None = typer.Option(
        None,
        "--canonical-json",
        help="Optional canonical map JSON from storage_select_canonical output.",
    ),
    families: str | None = typer.Option(None, "--families"),
    start_date: str | None = typer.Option(None, "--start-date", help="YYYY-MM-DD"),
    end_date: str | None = typer.Option(None, "--end-date", help="YYYY-MM-DD"),
    min_age_hours: float = typer.Option(0.0, "--min-age-hours"),
    max_delete_files: int | None = typer.Option(None, "--max-delete-files"),
    max_delete_bytes: int | None = typer.Option(None, "--max-delete-bytes"),
    execute: bool = typer.Option(False, "--execute/--dry-run"),
    skip_errors: bool = typer.Option(False, "--skip-errors/--no-skip-errors"),
) -> None:
    roots = resolve_storage_roots(data_root=data_root, hot_root=hot_root)
    policy = load_storage_retention_policy(config_path=config_path)

    if canonical_json is not None:
        canonical = load_json(Path(canonical_json))
    else:
        selected = _parse_families(families)
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

    params = PruneParams(
        execute=bool(execute),
        max_delete_files=max_delete_files,
        max_delete_bytes=max_delete_bytes,
        min_age_hours=float(min_age_hours),
    )
    plan = build_prune_plan(canonical_output=canonical, params=params)

    if execute:
        assert_no_active_writer(data_root=roots.data_root)
        ledger = execute_prune_plan(plan=plan)
    else:
        ledger = None

    reports = write_prune_reports(hot_root=roots.hot_root, plan=plan, ledger=ledger)

    summary = plan.get("summary") or {}
    msg = (
        "[storage_prune] "
        f"mode={'execute' if execute else 'dry-run'} "
        f"candidates={summary.get('candidate_count', 0)} "
        f"bytes={summary.get('candidate_bytes', 0)} "
        f"files={summary.get('candidate_files', 0)} "
        f"plan={reports['plan']}"
    )
    if ledger is not None:
        msg += f" ledger={reports.get('ledger', '')}"
    typer.echo(msg)


if __name__ == "__main__":
    app()
