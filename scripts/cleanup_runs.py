#!/usr/bin/env python
"""Cleanup old projection runs according to retention policy.

Retention policy:
- Keep 1 run per hour (the first run in each hour)
- Keep the final run before each game lock (within 10 min of first tip)
- Delete remaining runs older than the retention window

Usage:
    python cleanup_runs.py --game-date 2025-12-23 --retention-hours 72
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Run directories follow pattern: run=YYYYMMDDTHHMMSSz
RUN_DIR_PATTERN = re.compile(r"^run=(\d{8}T\d{6}Z)$")


def parse_run_id(run_dir: Path) -> datetime | None:
    """Parse run directory name to datetime."""
    match = RUN_DIR_PATTERN.match(run_dir.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def get_runs_for_date(projections_root: Path, game_date: str) -> list[tuple[Path, datetime]]:
    """Get all run directories for a given date with their timestamps."""
    day_dir = projections_root / game_date
    if not day_dir.exists():
        return []
    
    runs = []
    for subdir in day_dir.iterdir():
        if not subdir.is_dir():
            continue
        ts = parse_run_id(subdir)
        if ts is not None:
            runs.append((subdir, ts))
    
    return sorted(runs, key=lambda x: x[1])


def identify_runs_to_keep(
    runs: list[tuple[Path, datetime]],
    first_tip_ts: datetime | None = None,
) -> set[Path]:
    """Identify which runs to keep according to retention policy."""
    keep = set()
    
    # Group runs by hour
    hourly: dict[str, list[tuple[Path, datetime]]] = {}
    for path, ts in runs:
        hour_key = ts.strftime("%Y-%m-%dT%H")
        hourly.setdefault(hour_key, []).append((path, ts))
    
    # Keep first run of each hour
    for hour_runs in hourly.values():
        if hour_runs:
            keep.add(hour_runs[0][0])
    
    # Keep the last run before first tip (pre-lock snapshot)
    if first_tip_ts is not None:
        pre_lock_runs = [(p, ts) for p, ts in runs if ts < first_tip_ts]
        if pre_lock_runs:
            keep.add(pre_lock_runs[-1][0])
    
    # Always keep the most recent run
    if runs:
        keep.add(runs[-1][0])
    
    # Keep pinned run if specified
    for path, ts in runs:
        day_dir = path.parent
        pinned_ptr = day_dir / "pinned_run.json"
        if pinned_ptr.exists():
            try:
                with open(pinned_ptr, encoding="utf-8") as f:
                    pinned_data = json.load(f)
                pinned_id = pinned_data.get("run_id")
                if pinned_id and path.name == f"run={pinned_id}":
                    keep.add(path)
            except Exception:
                pass
    
    return keep


def cleanup_runs(
    projections_root: Path,
    game_date: str,
    first_tip_ts: datetime | None = None,
    dry_run: bool = True,
    retention_hours: int = 72,
) -> dict[str, int]:
    """
    Clean up old runs according to retention policy.
    
    Returns dict with counts of kept/deleted runs.
    """
    runs = get_runs_for_date(projections_root, game_date)
    
    if not runs:
        return {"total": 0, "kept": 0, "deleted": 0, "skipped_recent": 0}
    
    keep = identify_runs_to_keep(runs, first_tip_ts)
    
    # Don't delete runs newer than retention window
    cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
    
    deleted = 0
    skipped = 0
    for path, ts in runs:
        if path in keep:
            continue
        if ts > cutoff:
            skipped += 1
            continue
        
        if dry_run:
            print(f"[dry-run] Would delete: {path}")
        else:
            shutil.rmtree(path)
            print(f"[cleanup] Deleted: {path}")
        deleted += 1
    
    return {
        "total": len(runs),
        "kept": len(keep),
        "deleted": deleted,
        "skipped_recent": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="Cleanup old projection runs")
    parser.add_argument("--game-date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--data-root", default="/home/daniel/projections-data",
                        help="Data root directory")
    parser.add_argument("--retention-hours", type=int, default=72,
                        help="Retention window in hours (default: 72)")
    parser.add_argument("--first-tip-ts", type=str, default=None,
                        help="First tip timestamp (ISO format) for pre-lock detection")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete (default is dry-run)")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    projections_root = data_root / "artifacts" / "projections"
    
    first_tip = None
    if args.first_tip_ts:
        first_tip = datetime.fromisoformat(args.first_tip_ts.replace("Z", "+00:00"))
    
    result = cleanup_runs(
        projections_root,
        args.game_date,
        first_tip_ts=first_tip,
        dry_run=not args.execute,
        retention_hours=args.retention_hours,
    )
    
    print(f"\n[cleanup] Summary for {args.game_date}:")
    print(f"  Total runs: {result['total']}")
    print(f"  Kept: {result['kept']}")
    print(f"  Deleted: {result['deleted']}")
    print(f"  Skipped (< {args.retention_hours}h old): {result['skipped_recent']}")


if __name__ == "__main__":
    main()
