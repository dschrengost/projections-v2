#!/usr/bin/env python3
"""Quick sanity checks for the live pipeline environment.

Run this script inside the virtual environment (e.g., via ``uv run``) to verify
that the core dependencies for the injuries, daily lineups, odds, and roster
ETLs are available before scheduling them via cron/systemd.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from projections import paths


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def check_java() -> None:
    java = shutil.which("java")
    if not java:
        raise SystemExit("java executable not found on PATH (install OpenJDK 11+).")
    result = _run([java, "-version"])
    if result.returncode != 0:
        raise SystemExit(f"java -version failed:\n{result.stderr}")
    sys.stdout.write(f"[ok] Java detected: {result.stderr.splitlines()[0]}\n")


def check_python_packages() -> None:
    try:
        import jpype  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment specific
        raise SystemExit("jpype1 is not installed; run `uv sync` to install dependencies.") from exc

    try:
        from tabula.io import read_pdf  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment specific
        raise SystemExit(
            "tabula-py is not installed or misconfigured. Ensure `uv sync` completed successfully."
        ) from exc
    sys.stdout.write("[ok] Python packages (jpype1, tabula-py) are importable.\n")


def check_data_root() -> None:
    data_root_env = os.environ.get("PROJECTIONS_DATA_ROOT")
    data_root = paths.get_data_root()
    sys.stdout.write(f"[info] Using data root: {data_root}\n")
    if data_root_env:
        sys.stdout.write(f"[info] PROJECTIONS_DATA_ROOT={data_root_env}\n")
    data_root.mkdir(parents=True, exist_ok=True)
    expected = [
        data_root / "bronze",
        data_root / "silver",
        data_root / "labels",
    ]
    for path in expected:
        path.mkdir(parents=True, exist_ok=True)
    sys.stdout.write("[ok] Data root directories exist or were created.\n")


def main() -> None:
    check_java()
    check_python_packages()
    check_data_root()
    sys.stdout.write("Environment checks completed successfully.\n")


if __name__ == "__main__":
    main()
