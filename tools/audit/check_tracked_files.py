#!/usr/bin/env python3
"""Check for forbidden files tracked at repository root.

This guardrail script ensures repo hygiene by detecting files that should
never be committed, such as crash dumps, log files, and data artifacts.

Exit codes:
    0 - No violations found
    1 - Violations detected
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path


# Patterns for files/directories that should NOT be tracked at repo root
FORBIDDEN_ROOT_PATTERNS = [
    # Crash dumps and logs
    "hs_err_pid*",
    "replay_pid*",
    "nohup.out",
    "*.log",
    # Data files
    "*.csv",
    "*.parquet",
    # Runtime directories
    "mlruns",
    "mlruns/",
    # Command-paste accidents
    "udo *",
    "sudo *",
]

# Directories that should never be tracked
FORBIDDEN_DIRECTORIES = [
    "mlruns",
    "artifacts",
    "runs",
    "data",
]


def get_tracked_files() -> list[str]:
    """Get list of all tracked files from git."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split("\n") if result.stdout.strip() else []


def get_root_files(tracked_files: list[str]) -> list[str]:
    """Filter to only files at repo root (no directory separator)."""
    return [f for f in tracked_files if "/" not in f]


def check_pattern(filename: str, pattern: str) -> bool:
    """Check if a filename matches a forbidden pattern."""
    return fnmatch.fnmatch(filename, pattern)


def check_forbidden_directories(tracked_files: list[str]) -> list[str]:
    """Check if any tracked files are in forbidden directories."""
    violations = []
    for filepath in tracked_files:
        for forbidden_dir in FORBIDDEN_DIRECTORIES:
            if filepath.startswith(f"{forbidden_dir}/"):
                violations.append(filepath)
                break
    return violations


def main() -> int:
    """Run the audit checks and report violations."""
    print("🔍 Checking for forbidden tracked files...")
    
    try:
        tracked_files = get_tracked_files()
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get tracked files: {e}")
        return 1
    
    root_files = get_root_files(tracked_files)
    violations: list[str] = []
    
    # Check root files against forbidden patterns
    for filepath in root_files:
        for pattern in FORBIDDEN_ROOT_PATTERNS:
            if check_pattern(filepath, pattern):
                violations.append(f"Root file matches '{pattern}': {filepath}")
                break
    
    # Check for tracked files in forbidden directories
    dir_violations = check_forbidden_directories(tracked_files)
    for filepath in dir_violations:
        violations.append(f"File in forbidden directory: {filepath}")
    
    if violations:
        print(f"\n❌ Found {len(violations)} violation(s):\n")
        for v in violations:
            print(f"  • {v}")
        print("\nThese files should be untracked and added to .gitignore.")
        return 1
    
    print("✅ No violations found. Repository root is clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
