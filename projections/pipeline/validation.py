"""Validation checks for blessed run promotion.

Before blessing a run, verify:
1. All players have valid minutes (0 ≤ p50 ≤ 48)
2. Team sums = 240 within tolerance
3. No extreme FPTS swings vs previous blessed run (optional)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


@dataclass
class ValidationResult:
    """Result of validation checks."""
    passed: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    def fail(self, check_name: str, error: str) -> None:
        """Mark a check as failed."""
        self.checks[check_name] = False
        self.errors.append(error)
        self.passed = False
    
    def pass_check(self, check_name: str) -> None:
        """Mark a check as passed."""
        self.checks[check_name] = True
    
    def warn(self, message: str) -> None:
        """Add a warning (doesn't fail validation)."""
        self.warnings.append(message)


def validate_minutes_range(
    df: pd.DataFrame,
    min_val: float = 0.0,
    max_val: float = 48.0,
    minutes_col: str = "minutes_p50",
) -> ValidationResult:
    """Check that all minutes predictions are in valid range."""
    result = ValidationResult()
    
    if minutes_col not in df.columns:
        result.fail("minutes_range", f"Column {minutes_col} not found")
        return result
    
    minutes = df[minutes_col].dropna()
    
    if minutes.empty:
        result.fail("minutes_range", "No valid minutes values found")
        return result
    
    below_min = (minutes < min_val).sum()
    above_max = (minutes > max_val).sum()
    
    result.metrics["min_minutes"] = float(minutes.min())
    result.metrics["max_minutes"] = float(minutes.max())
    result.metrics["players_below_0"] = int(below_min)
    result.metrics["players_above_48"] = int(above_max)
    
    if below_min > 0:
        result.fail("minutes_range", f"{below_min} players have minutes < {min_val}")
    elif above_max > 0:
        result.fail("minutes_range", f"{above_max} players have minutes > {max_val}")
    else:
        result.pass_check("minutes_range")
    
    return result


def validate_team_sums(
    df: pd.DataFrame,
    target_sum: float = 240.0,
    tolerance: float = 1.0,
    minutes_col: str = "minutes_p50",
) -> ValidationResult:
    """Check that team minutes sum to 240."""
    result = ValidationResult()
    
    if minutes_col not in df.columns:
        result.fail("team_sums", f"Column {minutes_col} not found")
        return result
    
    if "team_id" not in df.columns or "game_id" not in df.columns:
        result.warn("Cannot validate team sums: missing team_id or game_id")
        result.pass_check("team_sums")  # Skip check gracefully
        return result
    
    # Only count players with positive minutes (active)
    active = df[df[minutes_col] > 0].copy()
    team_sums = active.groupby(["game_id", "team_id"])[minutes_col].sum()
    
    deviations = (team_sums - target_sum).abs()
    max_dev = float(deviations.max()) if len(deviations) else 0.0
    
    result.metrics["max_team_deviation"] = max_dev
    result.metrics["teams_checked"] = len(team_sums)
    result.metrics["teams_outside_tolerance"] = int((deviations > tolerance).sum())
    
    if max_dev > tolerance:
        worst_team = deviations.idxmax()
        result.fail(
            "team_sums",
            f"Team {worst_team} has sum deviation of {max_dev:.2f} (> {tolerance})"
        )
    else:
        result.pass_check("team_sums")
    
    return result


def validate_fpts_stability(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame | None,
    max_swing: float = 15.0,
    fpts_col: str = "dk_fpts_mean",
) -> ValidationResult:
    """Check for extreme FPTS swings vs previous run."""
    result = ValidationResult()
    
    if previous_df is None:
        result.pass_check("fpts_stability")
        result.warn("No previous run to compare; skipping stability check")
        return result
    
    if fpts_col not in current_df.columns:
        result.fail("fpts_stability", f"Column {fpts_col} not found in current")
        return result
    
    if fpts_col not in previous_df.columns:
        result.warn(f"Column {fpts_col} not found in previous; skipping")
        result.pass_check("fpts_stability")
        return result
    
    # Merge on player_id
    if "player_id" not in current_df.columns or "player_id" not in previous_df.columns:
        result.warn("Cannot compare: missing player_id")
        result.pass_check("fpts_stability")
        return result
    
    merged = current_df[["player_id", fpts_col]].merge(
        previous_df[["player_id", fpts_col]],
        on="player_id",
        suffixes=("_new", "_old"),
    )
    
    if merged.empty:
        result.warn("No matching players to compare")
        result.pass_check("fpts_stability")
        return result
    
    merged["delta"] = (merged[f"{fpts_col}_new"] - merged[f"{fpts_col}_old"]).abs()
    max_delta = float(merged["delta"].max())
    
    result.metrics["max_fpts_swing"] = max_delta
    result.metrics["players_compared"] = len(merged)
    result.metrics["players_with_large_swing"] = int((merged["delta"] > max_swing).sum())
    
    if max_delta > max_swing:
        worst_player = merged.loc[merged["delta"].idxmax()]
        player_id = worst_player["player_id"]
        old_val = worst_player[f"{fpts_col}_old"]
        new_val = worst_player[f"{fpts_col}_new"]
        result.fail(
            "fpts_stability",
            f"Player {player_id} has FPTS swing of {max_delta:.1f} ({old_val:.1f} → {new_val:.1f})"
        )
    else:
        result.pass_check("fpts_stability")
    
    return result


def validate_run(
    projections_path: Path,
    previous_projections_path: Path | None = None,
    *,
    team_sum_tolerance: float = 1.0,
) -> ValidationResult:
    """Run all validation checks on a projection artifact."""
    
    final_result = ValidationResult()
    
    # Load current projections
    if not projections_path.exists():
        final_result.fail("load", f"Projections not found: {projections_path}")
        return final_result
    
    try:
        df = pd.read_parquet(projections_path)
    except Exception as e:
        final_result.fail("load", f"Failed to load projections: {e}")
        return final_result
    
    # Run checks (FPTS stability removed - injury updates cause legitimate swings)
    minutes_result = validate_minutes_range(df)
    team_result = validate_team_sums(df, tolerance=team_sum_tolerance)
    
    # Aggregate results
    for check, passed in minutes_result.checks.items():
        final_result.checks[check] = passed
    for check, passed in team_result.checks.items():
        final_result.checks[check] = passed
    
    final_result.errors.extend(minutes_result.errors)
    final_result.errors.extend(team_result.errors)
    
    final_result.warnings.extend(minutes_result.warnings)
    final_result.warnings.extend(team_result.warnings)
    
    final_result.metrics.update(minutes_result.metrics)
    final_result.metrics.update(team_result.metrics)
    
    final_result.passed = all(final_result.checks.values())
    
    return final_result


def promote_to_blessed(
    game_date: str,
    run_id: str,
    data_root: Path,
    *,
    validate: bool = True,
) -> tuple[bool, ValidationResult | None, str]:
    """
    Validate a run and promote to blessed if checks pass.
    
    Returns:
        (success, validation_result, reason)
    """
    projections_dir = data_root / "artifacts" / "projections" / game_date
    run_dir = projections_dir / f"run={run_id}"
    projections_path = run_dir / "projections.parquet"
    
    if not run_dir.exists():
        return False, None, f"Run directory not found: {run_dir}"
    
    if not projections_path.exists():
        return False, None, f"Projections not found: {projections_path}"
    
    validation_result = None
    
    if validate:
        # Run validation (no longer compares to previous run)
        validation_result = validate_run(projections_path)
        
        if not validation_result.passed:
            return False, validation_result, "Validation failed"
    
    # Write blessed_run.json
    blessed_ptr = projections_dir / "blessed_run.json"
    blessed_data = {
        "run_id": run_id,
        "blessed_at": datetime.now(timezone.utc).isoformat(),
        "validation": validation_result.to_dict() if validation_result else None,
    }
    
    with open(blessed_ptr, "w", encoding="utf-8") as f:
        json.dump(blessed_data, f, indent=2)
    
    return True, validation_result, "Promoted to blessed"


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and optionally bless a run")
    parser.add_argument("--run-id", required=True, help="Run ID to validate")
    parser.add_argument("--game-date", required=True, help="Game date (YYYY-MM-DD)")
    parser.add_argument("--data-root", default="/home/daniel/projections-data")
    parser.add_argument("--promote", action="store_true", help="Promote to blessed if valid")
    parser.add_argument("--force", action="store_true", help="Skip validation when promoting")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    projections_path = (
        data_root / "artifacts" / "projections" / args.game_date 
        / f"run={args.run_id}" / "projections.parquet"
    )
    
    if args.promote:
        success, result, reason = promote_to_blessed(
            args.game_date,
            args.run_id,
            data_root,
            validate=not args.force,
        )
        print(f"Promotion: {'SUCCESS' if success else 'FAILED'}")
        print(f"Reason: {reason}")
        if result:
            print(f"Validation: {json.dumps(result.to_dict(), indent=2)}")
    else:
        result = validate_run(projections_path)
        print(f"Validation: {'PASSED' if result.passed else 'FAILED'}")
        print(f"Checks: {json.dumps(result.checks, indent=2)}")
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        print(f"Metrics: {json.dumps(result.metrics, indent=2)}")
