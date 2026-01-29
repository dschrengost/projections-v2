"""
Run a real-slate check that inactive players never receive minutes.

Usage:
    uv run python scripts/diagnostics/check_inactive_zero_minutes.py --date 2026-01-28 --profile baseline
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import subprocess
import sys

import typer

app = typer.Typer(add_completion=False)


@app.command()
def run(
    date: str = typer.Option(..., "--date", help="Game date (YYYY-MM-DD)."),
    profile: str = typer.Option("baseline", "--profile", help="Sim profile name."),
    n_worlds: int = typer.Option(200, "--n-worlds", help="Number of worlds to sample."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Override data root."),
) -> None:
    # Enable dev assertions so generate_worlds_fpts_v2 raises on inactive minutes.
    os.environ["PROJECTIONS_SIM_DEV_ASSERTS"] = "1"

    with tempfile.TemporaryDirectory(prefix="sim_v2_inactive_check_") as tmpdir:
        cmd = [
            "uv",
            "run",
            sys.executable,
            "-m",
            "scripts.sim_v2.generate_worlds_fpts_v2",
            "--start-date",
            date,
            "--end-date",
            date,
            "--n-worlds",
            str(n_worlds),
            "--profile",
            profile,
            "--output-root",
            str(Path(tmpdir)),
            "--run-id",
            "inactive_zero_minutes_check",
        ]
        if data_root is not None:
            cmd.extend(["--data-root", str(data_root)])
        subprocess.run(cmd, check=True)

    typer.echo(
        f"[inactive-zero-minutes] ok date={date} profile={profile} worlds={n_worlds}"
    )


if __name__ == "__main__":
    app()
