"""Inventory scanner for storage retention families."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from projections.storage_retention.common import (
    dir_size_bytes,
    parse_iso_date,
    parse_run_id_ts,
    read_json_file,
    utc_now_compact,
    write_json,
)
from projections.storage_retention.paths import (
    FAMILY_ROOTS,
    iter_game_date_partitions,
    parse_run_id_from_partition,
    retention_reports_dir,
    resolve_storage_roots,
)


@dataclass(frozen=True)
class InventoryParams:
    data_root: Path | None = None
    hot_root: Path | None = None
    families: tuple[str, ...] = tuple(FAMILY_ROOTS.keys())
    start_date: str | None = None
    end_date: str | None = None
    skip_errors: bool = False


def _collect_pointer_run_ids(day_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    pointer_files = {
        "latest_current": day_dir / "LATEST" / "current.json",
        "latest_run": day_dir / "latest_run.json",
        "pinned_run": day_dir / "pinned_run.json",
    }
    for key, path in pointer_files.items():
        payload = read_json_file(path)
        if not payload:
            continue
        run_id = payload.get("run_id")
        if run_id:
            out[key] = str(run_id)
    return out


def _scan_day(family: str, game_date: str, day_dir: Path) -> dict[str, Any]:
    pointers = _collect_pointer_run_ids(day_dir)
    run_entries: list[dict[str, Any]] = []

    for run_dir in sorted(day_dir.glob("run=*")):
        if not run_dir.is_dir():
            continue
        run_id = parse_run_id_from_partition(run_dir.name)
        if not run_id:
            continue
        size_bytes, file_count = dir_size_bytes(run_dir)
        ts = parse_run_id_ts(run_id)
        run_entries.append(
            {
                "family": family,
                "game_date": game_date,
                "run_id": run_id,
                "run_path": str(run_dir),
                "size_bytes": int(size_bytes),
                "file_count": int(file_count),
                "run_ts": None if ts is None else ts.isoformat(),
                "is_pointer_latest_current": pointers.get("latest_current") == run_id,
                "is_pointer_latest_run": pointers.get("latest_run") == run_id,
                "is_pointer_pinned": pointers.get("pinned_run") == run_id,
            }
        )

    total_bytes = sum(int(row["size_bytes"]) for row in run_entries)
    return {
        "family": family,
        "game_date": game_date,
        "day_dir": str(day_dir),
        "pointer_run_ids": pointers,
        "run_count": int(len(run_entries)),
        "total_bytes": int(total_bytes),
        "runs": run_entries,
    }


def build_inventory(params: InventoryParams) -> dict[str, Any]:
    roots = resolve_storage_roots(data_root=params.data_root, hot_root=params.hot_root)
    start = parse_iso_date(params.start_date)
    end = parse_iso_date(params.end_date)

    families_out: list[dict[str, Any]] = []
    runs_out: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for family in params.families:
        if family not in FAMILY_ROOTS:
            raise ValueError(f"Unsupported family: {family}")
        family_days: list[dict[str, Any]] = []
        for game_date, day_dir in iter_game_date_partitions(
            family=family,
            hot_root=roots.hot_root,
            start_date=start,
            end_date=end,
        ):
            try:
                day_payload = _scan_day(family, game_date, day_dir)
            except Exception as exc:  # noqa: BLE001
                if not params.skip_errors:
                    raise
                errors.append(
                    {
                        "family": family,
                        "game_date": game_date,
                        "path": str(day_dir),
                        "error": str(exc),
                    }
                )
                continue
            family_days.append(day_payload)
            runs_out.extend(day_payload["runs"])

        families_out.append(
            {
                "family": family,
                "day_count": int(len(family_days)),
                "run_count": int(sum(int(day["run_count"]) for day in family_days)),
                "total_bytes": int(sum(int(day["total_bytes"]) for day in family_days)),
                "days": family_days,
            }
        )

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "hot_root": str(roots.hot_root),
        "data_root": str(roots.data_root),
        "families": families_out,
        "runs": runs_out,
        "errors": errors,
    }


def write_inventory_reports(inventory: dict[str, Any], *, hot_root: Path) -> dict[str, str]:
    ts = utc_now_compact()
    reports_dir = retention_reports_dir(hot_root=hot_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / f"{ts}_storage_inventory.json"
    csv_path = reports_dir / f"{ts}_storage_inventory_runs.csv"
    write_json(json_path, inventory)

    rows = list(inventory.get("runs") or [])
    fieldnames = [
        "family",
        "game_date",
        "run_id",
        "run_path",
        "size_bytes",
        "file_count",
        "run_ts",
        "is_pointer_latest_current",
        "is_pointer_latest_run",
        "is_pointer_pinned",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    return {
        "json": str(json_path),
        "csv": str(csv_path),
    }
