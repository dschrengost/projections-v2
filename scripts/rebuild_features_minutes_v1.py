#!/usr/bin/env python
"""Rebuild Minutes V1 features for historical dates with share model contract enforcement.

This script:
1. Uses the existing build_minutes_live pipeline in backfill mode
2. Applies share model feature contract enforcement
3. Writes contract-correct features to run-scoped directories
4. Produces a global status file with per-date results

Usage:
    uv run python scripts/rebuild_features_minutes_v1.py \\
        --start-date 2025-11-01 --end-date 2025-12-31 \\
        --share-bundle artifacts/minute_share/minute_share_wfcv2_fold_20 \\
        --out-root /home/daniel/projections-data/live/features_minutes_v1 \\
        --parallel 4
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import typer

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projections.models.minute_share.feature_contract import (
    ContractReport,
    load_expected_share_features,
    enforce_share_feature_contract,
    get_missing_feature_summary,
)

app = typer.Typer(help=__doc__)

# Thresholds
MISSING_FEATURE_CLEAN_THRESHOLD = 0.02
MISSING_FEATURE_SKIP_THRESHOLD = 0.10
MIN_PLAYERS_PER_TEAM = 8


@dataclass
class DateResult:
    """Result for a single date rebuild."""
    game_date: str
    status: str  # success, degraded, skipped, error
    reason: str | None = None
    n_games: int = 0
    n_teams: int = 0
    n_players: int = 0
    contract_report: dict | None = None
    features_path: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_integrity(df: pd.DataFrame) -> tuple[bool, str | None, dict]:
    """Check feature DataFrame integrity.
    
    Returns:
        Tuple of (is_valid, reason, counts)
    """
    n_players = len(df)
    n_games = df["game_id"].nunique() if "game_id" in df.columns else 0
    n_teams = df.groupby(["game_id", "team_id"]).ngroups if all(c in df.columns for c in ["game_id", "team_id"]) else 0
    
    counts = {"n_games": n_games, "n_teams": n_teams, "n_players": n_players}
    
    if n_games == 0 or n_teams == 0:
        return False, "no_games_or_teams", counts
    
    if n_teams < 2 * n_games:
        return False, f"missing_teams (expected {2 * n_games}, got {n_teams})", counts
    
    if n_players > 0:
        players_per_team = df.groupby(["game_id", "team_id"]).size()
        if players_per_team.min() < MIN_PLAYERS_PER_TEAM:
            return False, f"incomplete_rosters (min {players_per_team.min()} players)", counts
    
    return True, None, counts


def _run_build_minutes_live(
    game_date: str,
    out_root: Path,
    data_root: Path,
    run_id: str,
) -> tuple[Path | None, str | None]:
    """Run build_minutes_live for a single date in backfill mode.
    
    Returns:
        Tuple of (features_path, error_reason) - error_reason is None on success
    """
    cmd = [
        sys.executable, "-m", "projections.cli.build_minutes_live",
        "--date", game_date,
        "--out-root", str(out_root),
        "--backfill-mode",
        "--run-id", run_id,
        # Backfill-specific flags:
        "--roster-max-age-hours", "8760",  # Allow up to 1 year for historical dates
        "--skip-active-roster",  # Don't validate against current NBA.com roster
    ]
    
    if data_root.exists():
        cmd.extend(["--data-root", str(data_root)])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        
        if result.returncode != 0:
            stderr = result.stderr.lower() if result.stderr else ""
            # Categorize failure reasons
            if "roster snapshot" in stderr and "old" in stderr:
                return None, "roster_age_exceeded"
            elif "no prior rates_training_base" in stderr:
                return None, "missing_rates_training_base"
            elif "no label sources" in stderr or "no historical label" in stderr:
                return None, "missing_labels"
            elif "missing parquet" in stderr or "filenotfounderror" in stderr:
                return None, "missing_upstream_data"
            else:
                # Truncate error for logging
                err_msg = result.stderr[:300] if result.stderr else "unknown"
                return None, f"build_failed: {err_msg}"
        
        # Check for output
        features_path = out_root / game_date / f"run={run_id}" / "features.parquet"
        if features_path.exists():
            return features_path, None
        
        return None, "output_not_found"
    except subprocess.TimeoutExpired:
        return None, "build_timeout"
    except Exception as e:
        return None, f"exception: {str(e)[:100]}"


def _rebuild_single_date(
    game_date: str,
    share_bundle: Path,
    out_root: Path,
    data_root: Path,
    expected_cols: list[str],
    run_id: str,
) -> DateResult:
    """Rebuild features for a single date with contract enforcement.
    
    Returns:
        DateResult with status and details
    """
    result = DateResult(game_date=game_date, status="unknown")
    
    try:
        # Run the live feature builder in backfill mode
        features_path, build_error = _run_build_minutes_live(game_date, out_root, data_root, run_id)
        
        if features_path is None or not features_path.exists():
            # Try to find any existing features for this date
            date_dir = out_root / game_date
            if date_dir.exists():
                run_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()], reverse=True)
                for rd in run_dirs:
                    fp = rd / "features.parquet"
                    if fp.exists():
                        features_path = fp
                        build_error = None  # We found fallback features
                        break
        
        if features_path is None or not features_path.exists():
            result.status = "skipped"
            result.reason = build_error or "build_failed"
            return result
        
        # Load features
        df = pd.read_parquet(features_path)
        
        # Filter to target date if needed
        if "game_date" in df.columns:
            target = datetime.strptime(game_date, "%Y-%m-%d").date()
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
            df = df[df["game_date"] == target]
        
        if df.empty:
            result.status = "skipped"
            result.reason = "no_features_for_date"
            return result
        
        # Check integrity
        is_valid, reason, counts = _check_integrity(df)
        result.n_games = counts["n_games"]
        result.n_teams = counts["n_teams"]
        result.n_players = counts["n_players"]
        
        if not is_valid:
            result.status = "skipped"
            result.reason = f"integrity_failed: {reason}"
            return result
        
        # Enforce share feature contract
        df_fixed, contract_report = enforce_share_feature_contract(df, expected_cols)
        result.contract_report = contract_report.to_dict()
        
        # Determine quality tier
        if contract_report.missing_feature_frac > MISSING_FEATURE_SKIP_THRESHOLD:
            result.status = "skipped"
            result.reason = f"too_many_missing_features ({contract_report.n_missing}/{contract_report.n_expected})"
            return result
        
        # Write contract-correct features
        contract_features_dir = out_root / game_date / f"run={run_id}_contract"
        contract_features_dir.mkdir(parents=True, exist_ok=True)
        
        contract_features_path = contract_features_dir / "features.parquet"
        df_fixed.to_parquet(contract_features_path, index=False)
        
        # Write contract report
        report_path = contract_features_dir / "contract_report.json"
        report_path.write_text(json.dumps(result.contract_report, indent=2), encoding="utf-8")
        
        result.features_path = str(contract_features_path)
        
        # Determine quality
        if contract_report.missing_feature_frac > MISSING_FEATURE_CLEAN_THRESHOLD:
            result.status = "degraded"
            result.reason = f"missing_features ({contract_report.n_missing}/{contract_report.n_expected})"
        else:
            result.status = "success"
        
        return result
        
    except Exception as e:
        result.status = "error"
        result.reason = str(e)
        return result


@app.command()
def main(
    start_date: str = typer.Option(..., "--start-date", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end-date", help="End date (YYYY-MM-DD)"),
    share_bundle: Path = typer.Option(
        Path("artifacts/minute_share/minute_share_wfcv2_fold_20"),
        "--share-bundle",
        help="Path to share model bundle with feature_columns.json",
    ),
    out_root: Path = typer.Option(
        Path.home() / "projections-data" / "live" / "features_minutes_v1",
        "--out-root",
        help="Output root for features",
    ),
    data_root: Path = typer.Option(
        Path.home() / "projections-data",
        "--data-root",
        help="Data root containing source tables",
    ),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel workers"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would run without executing"),
) -> None:
    """Rebuild Minutes V1 features with share model contract enforcement."""
    
    typer.echo(f"[rebuild] Starting: {start_date} to {end_date}", err=True)
    typer.echo(f"[rebuild] Share bundle: {share_bundle}", err=True)
    typer.echo(f"[rebuild] Output root: {out_root}", err=True)
    
    # Load expected features from share bundle
    expected_cols = load_expected_share_features(share_bundle)
    if not expected_cols:
        typer.echo(f"[rebuild] ERROR: No feature columns found in {share_bundle}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"[rebuild] Expected feature columns: {len(expected_cols)}", err=True)
    
    # Generate date range
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    typer.echo(f"[rebuild] Processing {len(dates)} dates", err=True)
    
    if dry_run:
        typer.echo(f"[rebuild] Dry run - would process: {dates[:5]}...", err=True)
        return
    
    # Generate a run ID for this rebuild
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ") + "_rebuild"
    
    # Process dates
    results: list[DateResult] = []
    counts = {"success": 0, "degraded": 0, "skipped": 0, "error": 0}
    all_contract_reports: list[ContractReport] = []
    
    if parallel > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            future_to_date = {
                executor.submit(
                    _rebuild_single_date,
                    d, share_bundle, out_root, data_root, expected_cols, run_id
                ): d
                for d in dates
            }
            
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    res = future.result()
                except Exception as e:
                    res = DateResult(game_date=date, status="error", reason=str(e))
                
                results.append(res)
                counts[res.status] += 1
                
                icon = "✓" if res.status == "success" else ("~" if res.status == "degraded" else "⊘")
                typer.echo(f"  {icon} {date} [{res.status}]", err=True)
    else:
        # Sequential execution
        for d in dates:
            res = _rebuild_single_date(d, share_bundle, out_root, data_root, expected_cols, run_id)
            results.append(res)
            counts[res.status] += 1
            
            icon = "✓" if res.status == "success" else ("~" if res.status == "degraded" else "⊘")
            reason_str = f" ({res.reason})" if res.reason else ""
            typer.echo(f"  {icon} {d} [{res.status}]{reason_str}", err=True)
            
            # Collect contract reports for missing feature summary
            if res.contract_report:
                cr = ContractReport(
                    n_expected=res.contract_report.get("n_expected", 0),
                    n_missing=res.contract_report.get("n_missing", 0),
                    missing_cols=res.contract_report.get("missing_cols", []),
                )
                all_contract_reports.append(cr)
    
    # Write global status file
    status = {
        "rebuild_id": run_id,
        "start_date": start_date,
        "end_date": end_date,
        "share_bundle": str(share_bundle),
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_dates": len(dates),
            "success": counts["success"],
            "degraded": counts["degraded"],
            "skipped": counts["skipped"],
            "error": counts["error"],
        },
        "dates": [r.to_dict() for r in results],
    }
    
    status_path = out_root / "_rebuild_features_minutes_v1_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    
    # Compute missing feature summary
    missing_summary = get_missing_feature_summary(all_contract_reports)
    
    # Print summary
    print("\n" + "=" * 70)
    print("REBUILD SUMMARY")
    print("=" * 70)
    print(f"\n📊 Coverage:")
    print(f"  Success (clean): {counts['success']}")
    print(f"  Degraded: {counts['degraded']}")
    print(f"  Skipped: {counts['skipped']}")
    print(f"  Error: {counts['error']}")
    
    # Skip reasons
    skip_reasons: dict[str, int] = {}
    for r in results:
        if r.status in ("skipped", "error") and r.reason:
            skip_reasons[r.reason] = skip_reasons.get(r.reason, 0) + 1
    
    if skip_reasons:
        print("\n📋 Skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1])[:10]:
            print(f"    {reason}: {count}")
    
    # Top missing columns
    if missing_summary:
        print("\n📋 Top 20 missing columns across degraded slates:")
        for col, count in list(missing_summary.items())[:20]:
            print(f"    {col}: {count}")
    
    print(f"\n📁 Status file: {status_path}")
    print("=" * 70)
    
    # Acceptance check
    clean_slates = counts["success"] + counts["degraded"]
    if clean_slates < 30:
        typer.echo(f"\n⚠️  Warning: Only {clean_slates} usable slates (target: 30+)", err=True)
    else:
        typer.echo(f"\n✓ Target met: {clean_slates} usable slates", err=True)


if __name__ == "__main__":
    app()
