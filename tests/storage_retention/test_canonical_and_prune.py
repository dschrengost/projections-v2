from __future__ import annotations

import json
from pathlib import Path

from projections.storage_retention.canonical import classify_inventory_runs
from projections.storage_retention.config import RetentionPolicy
from projections.storage_retention.prune import PruneParams, build_prune_plan


def _write_manifest(path: Path, as_of_ts: str, tip_ts: str) -> None:
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


def test_canonical_selection_and_prune_plan(tmp_path: Path) -> None:
    day = "2026-03-20"
    family = "gtv2_worlds"
    base = tmp_path / "artifacts" / "gtv2_worlds" / f"game_date={day}"
    run1 = base / "run=20260320T180000Z"
    run2 = base / "run=20260320T185500Z"
    run3 = base / "run=20260320T190500Z"
    for path in (run1, run2, run3):
        path.mkdir(parents=True, exist_ok=True)

    _write_manifest(run1 / "manifest.json", "2026-03-20T18:00:00Z", "2026-03-20T19:00:00Z")
    _write_manifest(run2 / "manifest.json", "2026-03-20T18:55:00Z", "2026-03-20T19:00:00Z")
    _write_manifest(run3 / "manifest.json", "2026-03-20T19:05:00Z", "2026-03-20T19:00:00Z")

    inventory = {
        "hot_root": str(tmp_path),
        "runs": [
            {
                "family": family,
                "game_date": day,
                "run_id": "20260320T180000Z",
                "run_path": str(run1),
                "size_bytes": 10,
                "file_count": 1,
                "is_pointer_latest_current": False,
                "is_pointer_latest_run": False,
                "is_pointer_pinned": False,
            },
            {
                "family": family,
                "game_date": day,
                "run_id": "20260320T185500Z",
                "run_path": str(run2),
                "size_bytes": 20,
                "file_count": 1,
                "is_pointer_latest_current": False,
                "is_pointer_latest_run": False,
                "is_pointer_pinned": False,
            },
            {
                "family": family,
                "game_date": day,
                "run_id": "20260320T190500Z",
                "run_path": str(run3),
                "size_bytes": 30,
                "file_count": 1,
                "is_pointer_latest_current": False,
                "is_pointer_latest_run": False,
                "is_pointer_pinned": False,
            },
        ],
    }

    canonical = classify_inventory_runs(
        inventory=inventory,
        retention_policy=RetentionPolicy(
            lead_time_minutes=2,
            start_time_bucket_minutes=30,
            keep_latest_debug_runs=1,
            protect_current_day=False,
        ),
    )

    decisions = {str(row["run_id"]): row for row in canonical["decisions"]}
    assert decisions["20260320T185500Z"]["classification"] == "canonical"
    assert decisions["20260320T185500Z"]["protected"] is True

    assert decisions["20260320T190500Z"]["classification"] == "debug_keep"
    assert decisions["20260320T190500Z"]["protected"] is True

    assert decisions["20260320T180000Z"]["classification"] == "noncanonical"
    assert decisions["20260320T180000Z"]["protected"] is False

    plan = build_prune_plan(
        canonical_output=canonical,
        params=PruneParams(require_archive_receipt=False),
    )
    candidates = list(plan["candidates"])
    assert len(candidates) == 1
    assert candidates[0]["run_id"] == "20260320T180000Z"
