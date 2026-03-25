"""Archive planner/executor for retention payload directories."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from projections.storage_retention.common import (
    read_json_file,
    utc_now_compact,
    write_json,
)
from projections.storage_retention.paths import retention_decision_dir, retention_reports_dir


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ArchiveParams:
    execute: bool = False
    max_archive_files: int | None = None
    max_archive_bytes: int | None = None
    include_protected: bool = False
    include_classifications: tuple[str, ...] = ("noncanonical",)


def _build_file_inventory(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(stat.st_size),
            }
        )
    total_bytes = int(sum(int(row["size_bytes"]) for row in files))
    digest = _stable_digest(files)
    return {
        "file_count": int(len(files)),
        "total_bytes": total_bytes,
        "digest": digest,
        "files": files,
    }


def _inventories_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        int(left.get("file_count") or 0) == int(right.get("file_count") or 0)
        and int(left.get("total_bytes") or 0) == int(right.get("total_bytes") or 0)
        and str(left.get("digest") or "") == str(right.get("digest") or "")
    )


def _receipt_path_for(*, hot_root: Path, row: dict[str, Any]) -> Path:
    return (
        retention_decision_dir(
            hot_root=hot_root,
            family=str(row.get("family") or ""),
            game_date=str(row.get("game_date") or ""),
            run_id=str(row.get("run_id") or ""),
        )
        / "archive_receipt.json"
    )


def _target_archive_path(*, source_run_path: Path, hot_root: Path, archive_root: Path) -> Path:
    rel = source_run_path.resolve().relative_to(hot_root.resolve())
    return archive_root.resolve() / rel


def build_archive_plan(
    *,
    canonical_output: dict[str, Any],
    hot_root: Path,
    archive_root: Path,
    params: ArchiveParams,
) -> dict[str, Any]:
    now = datetime.now(tz=UTC)
    decisions = list(canonical_output.get("decisions") or [])

    include_classes = set(str(v).strip() for v in params.include_classifications if str(v).strip())
    if not include_classes:
        include_classes = {"noncanonical"}

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    already_archived: list[dict[str, Any]] = []

    archive_files = 0
    archive_bytes = 0

    for row in decisions:
        classification = str(row.get("classification") or "")
        protected = bool(row.get("protected"))
        run_id = str(row.get("run_id") or "")
        run_path = Path(str(row.get("run_path") or "")).expanduser()

        if classification not in include_classes:
            skipped.append({"run_id": run_id, "run_path": str(run_path), "reason": "classification_filtered"})
            continue
        if protected and not bool(params.include_protected):
            skipped.append({"run_id": run_id, "run_path": str(run_path), "reason": "protected"})
            continue
        if not run_path.exists() or not run_path.is_dir():
            skipped.append({"run_id": run_id, "run_path": str(run_path), "reason": "missing_run_path"})
            continue

        try:
            archive_path = _target_archive_path(
                source_run_path=run_path,
                hot_root=hot_root,
                archive_root=archive_root,
            )
        except ValueError:
            skipped.append({"run_id": run_id, "run_path": str(run_path), "reason": "run_not_under_hot_root"})
            continue

        receipt_path = _receipt_path_for(hot_root=hot_root, row=row)
        receipt = read_json_file(receipt_path)
        if receipt and str(receipt.get("status") or "").lower() == "verified" and archive_path.exists():
            already_archived.append(
                {
                    "run_id": run_id,
                    "run_path": str(run_path),
                    "archive_run_path": str(archive_path),
                    "receipt_path": str(receipt_path),
                }
            )
            continue

        source_inventory = _build_file_inventory(run_path)
        file_count = int(source_inventory["file_count"])
        size_bytes = int(source_inventory["total_bytes"])

        if params.max_archive_files is not None and (archive_files + file_count) > int(params.max_archive_files):
            skipped.append({"run_id": run_id, "run_path": str(run_path), "reason": "max_archive_files_cap"})
            continue
        if params.max_archive_bytes is not None and (archive_bytes + size_bytes) > int(params.max_archive_bytes):
            skipped.append({"run_id": run_id, "run_path": str(run_path), "reason": "max_archive_bytes_cap"})
            continue

        archive_files += file_count
        archive_bytes += size_bytes

        candidates.append(
            {
                "family": str(row.get("family") or ""),
                "game_date": str(row.get("game_date") or ""),
                "run_id": run_id,
                "classification": classification,
                "protected": protected,
                "canonical_key": str(row.get("canonical_key") or ""),
                "pointer_refs": dict(row.get("pointer_refs") or {}),
                "protection_reasons": dict(row.get("protection_reasons") or {}),
                "run_path": str(run_path),
                "archive_run_path": str(archive_path),
                "receipt_path": str(receipt_path),
                "file_count": file_count,
                "size_bytes": size_bytes,
            }
        )

    return {
        "generated_at": now.isoformat(),
        "execute": bool(params.execute),
        "hot_root": str(Path(hot_root).expanduser().resolve()),
        "archive_root": str(Path(archive_root).expanduser().resolve()),
        "summary": {
            "candidate_count": int(len(candidates)),
            "candidate_files": int(archive_files),
            "candidate_bytes": int(archive_bytes),
            "already_archived_count": int(len(already_archived)),
            "skipped_count": int(len(skipped)),
        },
        "candidates": candidates,
        "already_archived": already_archived,
        "skipped": skipped,
    }


def _write_verified_receipt(
    *,
    row: dict[str, Any],
    source_inventory: dict[str, Any],
    archive_inventory: dict[str, Any],
) -> None:
    receipt_path = Path(str(row.get("receipt_path") or "")).expanduser()
    payload = {
        "version": 1,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "status": "verified",
        "family": str(row.get("family") or ""),
        "game_date": str(row.get("game_date") or ""),
        "run_id": str(row.get("run_id") or ""),
        "classification": str(row.get("classification") or ""),
        "protected": bool(row.get("protected")),
        "canonical_key": str(row.get("canonical_key") or ""),
        "pointer_refs": dict(row.get("pointer_refs") or {}),
        "protection_reasons": dict(row.get("protection_reasons") or {}),
        "source_run_path": str(row.get("run_path") or ""),
        "archive_run_path": str(row.get("archive_run_path") or ""),
        "verification": {
            "method": "file_manifest_digest",
            "matches": bool(_inventories_match(source_inventory, archive_inventory)),
            "source_digest": str(source_inventory.get("digest") or ""),
            "archive_digest": str(archive_inventory.get("digest") or ""),
        },
        "source_inventory": source_inventory,
        "archive_inventory": archive_inventory,
    }
    write_json(receipt_path, payload)


def execute_archive_plan(*, plan: dict[str, Any]) -> dict[str, Any]:
    archived: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for row in list(plan.get("candidates") or []):
        source_path = Path(str(row.get("run_path") or "")).expanduser()
        archive_path = Path(str(row.get("archive_run_path") or "")).expanduser()
        temp_path = archive_path.parent / f"{archive_path.name}.tmp.{utc_now_compact()}"

        try:
            if not source_path.exists() or not source_path.is_dir():
                raise FileNotFoundError(f"source missing: {source_path}")

            source_inventory = _build_file_inventory(source_path)

            if archive_path.exists():
                archive_inventory = _build_file_inventory(archive_path)
                if not _inventories_match(source_inventory, archive_inventory):
                    raise RuntimeError(
                        "existing archive payload does not match source inventory "
                        f"(source={source_path}, archive={archive_path})"
                    )
                _write_verified_receipt(
                    row=row,
                    source_inventory=source_inventory,
                    archive_inventory=archive_inventory,
                )
                archived.append(
                    {
                        "run_id": str(row.get("run_id") or ""),
                        "run_path": str(source_path),
                        "archive_run_path": str(archive_path),
                        "status": "already_archived_verified",
                        "file_count": int(source_inventory["file_count"]),
                        "size_bytes": int(source_inventory["total_bytes"]),
                    }
                )
                continue

            if temp_path.exists():
                shutil.rmtree(temp_path)

            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_path, temp_path, dirs_exist_ok=False)

            temp_inventory = _build_file_inventory(temp_path)
            if not _inventories_match(source_inventory, temp_inventory):
                raise RuntimeError(
                    "archive temp payload failed verification "
                    f"(source={source_path}, temp={temp_path})"
                )

            temp_path.rename(archive_path)
            archive_inventory = _build_file_inventory(archive_path)
            _write_verified_receipt(
                row=row,
                source_inventory=source_inventory,
                archive_inventory=archive_inventory,
            )
            archived.append(
                {
                    "run_id": str(row.get("run_id") or ""),
                    "run_path": str(source_path),
                    "archive_run_path": str(archive_path),
                    "status": "archived_verified",
                    "file_count": int(source_inventory["file_count"]),
                    "size_bytes": int(source_inventory["total_bytes"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
            errors.append(
                {
                    "run_id": str(row.get("run_id") or ""),
                    "run_path": str(source_path),
                    "archive_run_path": str(archive_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "executed_at": datetime.now(tz=UTC).isoformat(),
        "archived": archived,
        "errors": errors,
        "summary": {
            "archived_count": int(len([x for x in archived if x.get("status") == "archived_verified"])),
            "already_archived_count": int(
                len([x for x in archived if x.get("status") == "already_archived_verified"])
            ),
            "archived_files": int(sum(int(x.get("file_count") or 0) for x in archived)),
            "archived_bytes": int(sum(int(x.get("size_bytes") or 0) for x in archived)),
            "error_count": int(len(errors)),
        },
    }


def write_archive_reports(
    *,
    hot_root: Path,
    plan: dict[str, Any],
    ledger: dict[str, Any] | None = None,
) -> dict[str, str]:
    ts = utc_now_compact()
    reports_dir = retention_reports_dir(hot_root=hot_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    plan_path = reports_dir / f"{ts}_storage_archive_plan.json"
    write_json(plan_path, plan)

    out = {"plan": str(plan_path)}
    if ledger is not None:
        ledger_path = reports_dir / f"{ts}_storage_archive_ledger.json"
        write_json(ledger_path, ledger)
        out["ledger"] = str(ledger_path)
    return out


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload
