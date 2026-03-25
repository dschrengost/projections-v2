"""Run weekly storage retention orchestration (dry-run by default)."""

from __future__ import annotations

from pathlib import Path

import typer

from projections.storage_retention.paths import FAMILY_ROOTS
from projections.storage_retention.scheduler import (
    WeeklyRetentionParams,
    run_weekly_retention,
)

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
    families: str | None = typer.Option(None, "--families"),
    include_classifications: str | None = typer.Option(
        "noncanonical",
        "--include-classifications",
        help="Comma-separated classifications to archive.",
    ),
    start_date: str | None = typer.Option(None, "--start-date", help="YYYY-MM-DD"),
    end_date: str | None = typer.Option(None, "--end-date", help="YYYY-MM-DD"),
    skip_errors: bool = typer.Option(False, "--skip-errors/--no-skip-errors"),
    write_decisions: bool = typer.Option(True, "--write-decisions/--no-write-decisions"),
    include_protected_archive: bool = typer.Option(False, "--include-protected-archive/--no-include-protected-archive"),
    max_archive_files: int | None = typer.Option(None, "--max-archive-files"),
    max_archive_bytes: int | None = typer.Option(None, "--max-archive-bytes"),
    max_delete_files: int | None = typer.Option(None, "--max-delete-files"),
    max_delete_bytes: int | None = typer.Option(None, "--max-delete-bytes"),
    min_prune_age_hours: float = typer.Option(0.0, "--min-prune-age-hours"),
    require_archive_receipt_for_prune: bool = typer.Option(
        True,
        "--require-archive-receipt-for-prune/--allow-prune-without-archive-receipt",
    ),
    execute: bool = typer.Option(False, "--execute/--dry-run"),
) -> None:
    payload = run_weekly_retention(
        WeeklyRetentionParams(
            data_root=data_root,
            hot_root=hot_root,
            archive_root=archive_root,
            config_path=config_path,
            families=_parse_families(families),
            start_date=start_date,
            end_date=end_date,
            skip_errors=bool(skip_errors),
            execute=bool(execute),
            write_decisions=bool(write_decisions),
            include_classifications=_parse_classes(include_classifications),
            include_protected_archive=bool(include_protected_archive),
            max_archive_files=max_archive_files,
            max_archive_bytes=max_archive_bytes,
            max_delete_files=max_delete_files,
            max_delete_bytes=max_delete_bytes,
            min_prune_age_hours=float(min_prune_age_hours),
            require_archive_receipt_for_prune=bool(require_archive_receipt_for_prune),
        )
    )

    summary = dict(payload.get("summary") or {})
    reports = dict(payload.get("reports") or {})
    typer.echo(
        "[storage_retention_weekly] "
        f"mode={'execute' if execute else 'dry-run'} "
        f"runs={summary.get('inventory_runs', 0)} "
        f"archive_candidates={summary.get('archive_candidates', 0)} "
        f"prune_candidates={summary.get('prune_candidates', 0)} "
        f"report={reports.get('weekly', '')}"
    )


if __name__ == "__main__":
    app()
