"""Plan/execute storage archive using canonical retention decisions."""

from __future__ import annotations

from pathlib import Path

import typer

from projections.storage_retention.archive import (
    ArchiveParams,
    build_archive_plan,
    execute_archive_plan,
    load_json,
    write_archive_reports,
)
from projections.storage_retention.canonical import classify_inventory_runs
from projections.storage_retention.config import load_storage_retention_policy
from projections.storage_retention.inventory import InventoryParams, build_inventory
from projections.storage_retention.paths import FAMILY_ROOTS, resolve_storage_roots

app = typer.Typer(add_completion=False)


def _parse_families(raw: str | None) -> tuple[str, ...]:
    if raw is None or not str(raw).strip():
        return tuple(FAMILY_ROOTS.keys())
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _parse_classes(raw: str | None) -> tuple[str, ...]:
    if raw is None or not str(raw).strip():
        return ("noncanonical",)
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


@app.command()
def main(
    data_root: Path | None = typer.Option(None, "--data-root"),
    hot_root: Path | None = typer.Option(None, "--hot-root"),
    archive_root: Path | None = typer.Option(None, "--archive-root"),
    config_path: Path | None = typer.Option(None, "--config-path"),
    canonical_json: Path | None = typer.Option(
        None,
        "--canonical-json",
        help="Optional canonical map JSON from storage_select_canonical output.",
    ),
    families: str | None = typer.Option(None, "--families"),
    include_classifications: str | None = typer.Option(
        "noncanonical",
        "--include-classifications",
        help="Comma-separated classifications eligible for archive (default: noncanonical).",
    ),
    include_protected: bool = typer.Option(False, "--include-protected/--no-include-protected"),
    start_date: str | None = typer.Option(None, "--start-date", help="YYYY-MM-DD"),
    end_date: str | None = typer.Option(None, "--end-date", help="YYYY-MM-DD"),
    max_archive_files: int | None = typer.Option(None, "--max-archive-files"),
    max_archive_bytes: int | None = typer.Option(None, "--max-archive-bytes"),
    execute: bool = typer.Option(False, "--execute/--dry-run"),
    skip_errors: bool = typer.Option(False, "--skip-errors/--no-skip-errors"),
) -> None:
    roots = resolve_storage_roots(
        data_root=data_root,
        hot_root=hot_root,
        archive_root=archive_root,
    )
    if roots.archive_root is None:
        raise typer.BadParameter(
            "archive root is required; pass --archive-root or set PROJECTIONS_ARCHIVE_ROOT"
        )

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

    params = ArchiveParams(
        execute=bool(execute),
        max_archive_files=max_archive_files,
        max_archive_bytes=max_archive_bytes,
        include_protected=bool(include_protected),
        include_classifications=_parse_classes(include_classifications),
    )
    plan = build_archive_plan(
        canonical_output=canonical,
        hot_root=roots.hot_root,
        archive_root=roots.archive_root,
        params=params,
    )

    if execute:
        ledger = execute_archive_plan(plan=plan)
    else:
        ledger = None

    reports = write_archive_reports(hot_root=roots.hot_root, plan=plan, ledger=ledger)

    summary = dict(plan.get("summary") or {})
    msg = (
        "[storage_archive] "
        f"mode={'execute' if execute else 'dry-run'} "
        f"candidates={summary.get('candidate_count', 0)} "
        f"bytes={summary.get('candidate_bytes', 0)} "
        f"files={summary.get('candidate_files', 0)} "
        f"already={summary.get('already_archived_count', 0)} "
        f"plan={reports['plan']}"
    )
    if ledger is not None:
        msg += f" ledger={reports.get('ledger', '')}"
    typer.echo(msg)


if __name__ == "__main__":
    app()
