from __future__ import annotations

import json
from pathlib import Path

from projections.storage_retention.archive import (
    ArchiveParams,
    build_archive_plan,
    execute_archive_plan,
)
from projections.storage_retention.prune import PruneParams, build_prune_plan
from projections.storage_retention.scheduler import (
    WeeklyRetentionParams,
    run_weekly_retention,
)
from projections.storage_retention.paths import retention_decision_dir


def _write_manifest(path: Path, *, as_of_ts: str, tip_ts: str) -> None:
    payload = {
        "as_of_ts": as_of_ts,
        "source_freshness": {
            "per_game": {
                "123": {
                    "game_id": 123,
                    "tip_ts": tip_ts,
                }
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prune_requires_verified_archive_receipt_by_default(tmp_path: Path) -> None:
    run_path = tmp_path / "artifacts" / "gtv2_worlds" / "game_date=2026-03-20" / "run=20260320T180000Z"
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "worlds.parquet").write_text("x", encoding="utf-8")

    canonical = {
        "hot_root": str(tmp_path),
        "decisions": [
            {
                "family": "gtv2_worlds",
                "game_date": "2026-03-20",
                "run_id": "20260320T180000Z",
                "run_path": str(run_path),
                "size_bytes": 1,
                "file_count": 1,
                "classification": "noncanonical",
                "protected": False,
            }
        ],
    }

    plan = build_prune_plan(
        canonical_output=canonical,
        params=PruneParams(require_archive_receipt=True),
        hot_root=tmp_path,
    )
    assert int(plan["summary"]["candidate_count"]) == 0
    assert any(str(row.get("reason")) == "missing_archive_receipt" for row in plan["skipped"])

    archive_path = tmp_path / "archive" / "artifacts" / "gtv2_worlds" / "game_date=2026-03-20" / "run=20260320T180000Z"
    archive_path.mkdir(parents=True, exist_ok=True)
    receipt_path = (
        retention_decision_dir(
            hot_root=tmp_path,
            family="gtv2_worlds",
            game_date="2026-03-20",
            run_id="20260320T180000Z",
        )
        / "archive_receipt.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps({"status": "verified", "archive_run_path": str(archive_path)}),
        encoding="utf-8",
    )

    plan_after = build_prune_plan(
        canonical_output=canonical,
        params=PruneParams(require_archive_receipt=True),
        hot_root=tmp_path,
    )
    assert int(plan_after["summary"]["candidate_count"]) == 1


def test_archive_execute_writes_receipt_and_is_idempotent(tmp_path: Path) -> None:
    hot_root = tmp_path / "hot"
    archive_root = tmp_path / "archive"
    run_path = hot_root / "artifacts" / "gtv2_worlds" / "game_date=2026-03-20" / "run=20260320T180000Z"
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "worlds.parquet").write_text("abc", encoding="utf-8")
    (run_path / "manifest.json").write_text("{}", encoding="utf-8")

    canonical = {
        "hot_root": str(hot_root),
        "decisions": [
            {
                "family": "gtv2_worlds",
                "game_date": "2026-03-20",
                "run_id": "20260320T180000Z",
                "run_path": str(run_path),
                "classification": "noncanonical",
                "protected": False,
                "canonical_key": "k",
                "pointer_refs": {},
                "protection_reasons": {},
            }
        ],
    }

    plan = build_archive_plan(
        canonical_output=canonical,
        hot_root=hot_root,
        archive_root=archive_root,
        params=ArchiveParams(),
    )
    assert int(plan["summary"]["candidate_count"]) == 1

    ledger = execute_archive_plan(plan=plan)
    assert int(ledger["summary"]["archived_count"]) == 1
    assert int(ledger["summary"]["error_count"]) == 0

    receipt = (
        retention_decision_dir(
            hot_root=hot_root,
            family="gtv2_worlds",
            game_date="2026-03-20",
            run_id="20260320T180000Z",
        )
        / "archive_receipt.json"
    )
    assert receipt.exists()
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["status"] == "verified"

    plan_2 = build_archive_plan(
        canonical_output=canonical,
        hot_root=hot_root,
        archive_root=archive_root,
        params=ArchiveParams(),
    )
    assert int(plan_2["summary"]["candidate_count"]) == 0
    assert int(plan_2["summary"]["already_archived_count"]) == 1


def test_weekly_execute_archives_noncanonical_then_prunes(tmp_path: Path) -> None:
    hot_root = tmp_path / "hot"
    archive_root = tmp_path / "archive"
    game_date = "2026-03-20"

    base = hot_root / "artifacts" / "gtv2_worlds" / f"game_date={game_date}"
    run_old = base / "run=20260320T180000Z"
    run_new = base / "run=20260320T185500Z"
    for path in (run_old, run_new):
        path.mkdir(parents=True, exist_ok=True)
        (path / "worlds.parquet").write_text("payload", encoding="utf-8")

    _write_manifest(run_old / "manifest.json", as_of_ts="2026-03-20T18:00:00Z", tip_ts="2026-03-20T19:00:00Z")
    _write_manifest(run_new / "manifest.json", as_of_ts="2026-03-20T18:55:00Z", tip_ts="2026-03-20T19:00:00Z")

    payload = run_weekly_retention(
        WeeklyRetentionParams(
            data_root=hot_root,
            hot_root=hot_root,
            archive_root=archive_root,
            families=("gtv2_worlds",),
            start_date=game_date,
            end_date=game_date,
            execute=True,
            include_classifications=("noncanonical",),
            require_archive_receipt_for_prune=True,
        )
    )

    assert not run_old.exists()
    assert run_new.exists()

    archived_old = archive_root / "artifacts" / "gtv2_worlds" / f"game_date={game_date}" / "run=20260320T180000Z"
    assert archived_old.exists()

    receipt_old = (
        retention_decision_dir(
            hot_root=hot_root,
            family="gtv2_worlds",
            game_date=game_date,
            run_id="20260320T180000Z",
        )
        / "archive_receipt.json"
    )
    assert receipt_old.exists()

    summary = payload["summary"]
    assert int(summary["archive_candidates"]) >= 1
    assert int(summary["prune_candidates"]) >= 1
