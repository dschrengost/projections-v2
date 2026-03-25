"""Weekly retention orchestration for archive + prune workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from projections.storage_retention.archive import (
    ArchiveParams,
    build_archive_plan,
    execute_archive_plan,
    write_archive_reports,
)
from projections.storage_retention.canonical import (
    classify_inventory_runs,
    write_decision_reports,
)
from projections.storage_retention.common import utc_now_compact, write_json
from projections.storage_retention.config import load_storage_retention_policy
from projections.storage_retention.inventory import (
    InventoryParams,
    build_inventory,
    write_inventory_reports,
)
from projections.storage_retention.paths import (
    FAMILY_ROOTS,
    resolve_storage_roots,
    retention_reports_dir,
)
from projections.storage_retention.prune import (
    PruneParams,
    assert_no_active_writer,
    build_prune_plan,
    execute_prune_plan,
    write_prune_reports,
)


@dataclass(frozen=True)
class WeeklyRetentionParams:
    data_root: Path | None = None
    hot_root: Path | None = None
    archive_root: Path | None = None
    config_path: Path | None = None
    families: tuple[str, ...] = tuple(FAMILY_ROOTS.keys())
    start_date: str | None = None
    end_date: str | None = None
    skip_errors: bool = False
    execute: bool = False
    write_decisions: bool = True
    include_classifications: tuple[str, ...] = ("noncanonical",)
    include_protected_archive: bool = False
    max_archive_files: int | None = None
    max_archive_bytes: int | None = None
    max_delete_files: int | None = None
    max_delete_bytes: int | None = None
    min_prune_age_hours: float = 0.0
    require_archive_receipt_for_prune: bool = True


def run_weekly_retention(params: WeeklyRetentionParams) -> dict[str, Any]:
    roots = resolve_storage_roots(
        data_root=params.data_root,
        hot_root=params.hot_root,
        archive_root=params.archive_root,
    )

    if params.execute and roots.archive_root is None:
        raise ValueError(
            "archive root required for execute mode; pass archive_root or set PROJECTIONS_ARCHIVE_ROOT"
        )

    policy = load_storage_retention_policy(config_path=params.config_path)

    inventory = build_inventory(
        InventoryParams(
            data_root=roots.data_root,
            hot_root=roots.hot_root,
            families=params.families,
            start_date=params.start_date,
            end_date=params.end_date,
            skip_errors=bool(params.skip_errors),
        )
    )
    inventory_reports = write_inventory_reports(inventory, hot_root=roots.hot_root)

    canonical = classify_inventory_runs(
        inventory=inventory,
        retention_policy=policy.retention,
    )
    canonical_reports = write_decision_reports(
        canonical_output=canonical,
        hot_root=roots.hot_root,
        write_per_run_decisions=bool(params.write_decisions),
    )

    if roots.archive_root is None:
        archive_plan = {
            "generated_at": None,
            "execute": bool(params.execute),
            "summary": {
                "candidate_count": 0,
                "candidate_files": 0,
                "candidate_bytes": 0,
                "already_archived_count": 0,
                "skipped_count": 0,
            },
            "candidates": [],
            "already_archived": [],
            "skipped": [{"reason": "archive_root_not_configured"}],
        }
        archive_reports = {"plan": ""}
        archive_ledger = None
    else:
        archive_plan = build_archive_plan(
            canonical_output=canonical,
            hot_root=roots.hot_root,
            archive_root=roots.archive_root,
            params=ArchiveParams(
                execute=bool(params.execute),
                max_archive_files=params.max_archive_files,
                max_archive_bytes=params.max_archive_bytes,
                include_protected=bool(params.include_protected_archive),
                include_classifications=params.include_classifications,
            ),
        )
        if params.execute:
            archive_ledger = execute_archive_plan(plan=archive_plan)
        else:
            archive_ledger = None
        archive_reports = write_archive_reports(
            hot_root=roots.hot_root,
            plan=archive_plan,
            ledger=archive_ledger,
        )

    prune_plan = build_prune_plan(
        canonical_output=canonical,
        params=PruneParams(
            execute=bool(params.execute),
            max_delete_files=params.max_delete_files,
            max_delete_bytes=params.max_delete_bytes,
            min_age_hours=float(params.min_prune_age_hours),
            require_archive_receipt=bool(params.require_archive_receipt_for_prune),
        ),
        hot_root=roots.hot_root,
    )
    if params.execute:
        assert_no_active_writer(data_root=roots.data_root)
        prune_ledger = execute_prune_plan(plan=prune_plan)
    else:
        prune_ledger = None
    prune_reports = write_prune_reports(
        hot_root=roots.hot_root,
        plan=prune_plan,
        ledger=prune_ledger,
    )

    payload: dict[str, Any] = {
        "generated_at": utc_now_compact(),
        "execute": bool(params.execute),
        "roots": {
            "data_root": str(roots.data_root),
            "hot_root": str(roots.hot_root),
            "archive_root": None if roots.archive_root is None else str(roots.archive_root),
        },
        "inputs": {
            "families": list(params.families),
            "start_date": params.start_date,
            "end_date": params.end_date,
            "skip_errors": bool(params.skip_errors),
        },
        "reports": {
            "inventory": inventory_reports,
            "canonical": canonical_reports,
            "archive": archive_reports,
            "prune": prune_reports,
        },
        "summary": {
            "inventory_runs": int(len(list(inventory.get("runs") or []))),
            "inventory_errors": int(len(list(inventory.get("errors") or []))),
            "canonical_decisions": int(len(list(canonical.get("decisions") or []))),
            "archive_candidates": int((archive_plan.get("summary") or {}).get("candidate_count") or 0),
            "archive_errors": int(0 if archive_ledger is None else (archive_ledger.get("summary") or {}).get("error_count") or 0),
            "prune_candidates": int((prune_plan.get("summary") or {}).get("candidate_count") or 0),
            "prune_errors": int(0 if prune_ledger is None else (prune_ledger.get("summary") or {}).get("error_count") or 0),
        },
    }

    out_path = retention_reports_dir(hot_root=roots.hot_root) / f"{utc_now_compact()}_storage_retention_weekly.json"
    write_json(out_path, payload)
    payload["reports"]["weekly"] = str(out_path)
    return payload
