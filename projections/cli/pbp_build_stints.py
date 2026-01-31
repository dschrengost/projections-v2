"""Build on-court stints from canonical `pbp_events.parquet` (Phase 1).

Algorithm:
- Requires on-court lineup fields already present per event (resolved to player_id).
- Compress consecutive rows with identical (home_lineup, away_lineup) within a period.
- Compute duration_sec using period elapsed deltas (no possession reconstruction).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from projections.pbp.constants import PBP_V1_SCHEMA_VERSION
from projections.pbp.stints import build_stints_from_pbp_events

console = Console()
app = typer.Typer(help="Build stints + player stints from pbp_events.")


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


@app.command()
def run(
    bundle_dir: Path = typer.Argument(..., help="Run bundle dir (e.g. /.../artifacts/pbp_v1/<run_id>/)."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite outputs if present."),
) -> None:
    pbp_events_path = bundle_dir / "pbp_events.parquet"
    stints_path = bundle_dir / "stints.parquet"
    player_stints_path = bundle_dir / "player_stints.parquet"

    if not pbp_events_path.exists():
        console.print(f"[red]Missing input[/red] {pbp_events_path}")
        raise typer.Exit(1)

    if (stints_path.exists() or player_stints_path.exists()) and not overwrite:
        console.print("[red]Refusing to overwrite existing outputs; use --overwrite[/red]")
        raise typer.Exit(2)

    pbp_events = pd.read_parquet(pbp_events_path)
    result = build_stints_from_pbp_events(pbp_events, schema_version=PBP_V1_SCHEMA_VERSION)

    _atomic_write_parquet(result.stints, stints_path)
    _atomic_write_parquet(result.player_stints, player_stints_path)

    console.print(f"stints: {stints_path} ({len(result.stints):,} rows)")
    console.print(f"player_stints: {player_stints_path} ({len(result.player_stints):,} rows)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
