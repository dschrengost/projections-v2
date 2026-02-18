"""Ingest vendor NBA PBP CSVs and canonicalize to `pbp_events.parquet`.

Phase 1 assumptions:
- Vendor CSV includes on-court lineup columns (a1..a5, h1..h5). If missing, fail fast.
- Vendor does not provide player IDs; we resolve stable internal `player_id` values by name normalization.
"""

from __future__ import annotations

import json
import re
import glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import typer
from rich.console import Console
from rich.progress import track

from projections.pbp.constants import PBP_V1_SCHEMA_VERSION
from projections.pbp.identity import load_players_dim
from projections.pbp.publish import read_latest_published_run_id
from projections.pbp.vendor_ingest import (
    IngestResult,
    ingest_vendor_game_csv,
    save_input_hashes,
)
from projections.pbp.vendor_ingest import VendorLineupMissingError

console = Console()
app = typer.Typer(help="Ingest paid vendor PBP CSVs into canonical pbp_events + players_dim.")


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


def _combine_part_parquets(parts: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not parts:
        raise ValueError("No part parquets found to combine.")

    # Deterministic combine order (game_id then filename).
    parts = sorted(parts, key=lambda p: (p.stem, p.name))
    first = pq.read_table(parts[0])
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    writer = pq.ParquetWriter(tmp, schema=first.schema)
    try:
        writer.write_table(first)
        for p in parts[1:]:
            writer.write_table(pq.read_table(p))
    finally:
        writer.close()
    tmp.replace(out_path)


def _find_game_id_in_filename(path: Path) -> Optional[str]:
    m = re.search(r"-(\d{10})-", path.name)
    if m:
        return m.group(1)
    return None


@app.command()
def run(
    input_glob: str = typer.Option(
        "/home/daniel/projections-data/bronze/pbp_vendor/season_2024_25/2024-2025_NBA_PbP_Logs/*.csv",
        "--input-glob",
        help="Glob for vendor CSV files (one CSV per game).",
    ),
    artifacts_root: Path = typer.Option(
        Path("/home/daniel/projections-data/artifacts/pbp_v1"),
        "--artifacts-root",
        help="Artifacts root. Run bundle is written to <artifacts_root>/<run_id>/",
    ),
    season_id: str = typer.Option("2024-25", "--season-id", help="Season identifier (e.g. '2024-25')."),
    run_id: str = typer.Option(None, "--run-id", help="Run ID (default: UTC timestamp)."),
    prev_bundle: Optional[Path] = typer.Option(
        None,
        "--prev-bundle",
        help="Optional previous bundle directory for identity reuse (defaults to LATEST_PUBLISHED).",
    ),
    game_id: Optional[str] = typer.Option(
        None,
        "--game-id",
        help="Optional single game_id (10-digit) to ingest.",
    ),
    limit_games: Optional[int] = typer.Option(
        None,
        "--limit-games",
        help="Optional limit for number of games to ingest (after filtering).",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume an existing run_id (skips existing per-game parts and loads players_dim if present).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite outputs (pbp_events/parquets) in the run bundle.",
    ),
    skip_bad_games: bool = typer.Option(
        False,
        "--skip-bad-games",
        help="Skip games that fail ingest and write ingest_failures.parquet (useful for full-season runs).",
    ),
) -> None:
    """Read vendor CSVs, resolve player IDs, and write canonical `pbp_events.parquet` + `players_dim.parquet`."""
    run_id = run_id or _default_run_id()
    run_dir = artifacts_root / run_id
    parts_dir = run_dir / "_parts" / "pbp_events"
    parts_dir.mkdir(parents=True, exist_ok=True)

    pbp_events_path = run_dir / "pbp_events.parquet"
    players_dim_path = run_dir / "players_dim.parquet"
    unmapped_players_path = run_dir / "unmapped_players.parquet"
    ingest_failures_path = run_dir / "ingest_failures.parquet"
    input_hashes_path = run_dir / "input_hashes.json"

    if pbp_events_path.exists() and not overwrite and not resume:
        console.print(f"[red]Refusing to overwrite existing file[/red] {pbp_events_path}")
        raise typer.Exit(2)

    # Determine previous bundle for identity seeding.
    prev_players_dim = None
    if resume and players_dim_path.exists():
        console.print(f"Resuming: loading existing players_dim from {players_dim_path}")
        prev_players_dim = load_players_dim(players_dim_path)
    else:
        if prev_bundle is None:
            latest = read_latest_published_run_id(artifacts_root)
            prev_bundle = (artifacts_root / latest) if latest else None
        if prev_bundle:
            p = prev_bundle / "players_dim.parquet"
            if p.exists():
                console.print(f"Seeding identity from previous bundle: {p}")
                prev_players_dim = load_players_dim(p)

    # Collect input files.
    files = [Path(p) for p in sorted(glob.glob(input_glob))]
    if not files:
        console.print(f"[red]No files matched[/red] {input_glob}")
        raise typer.Exit(1)
    combined_files = [p for p in files if re.search(r"combined-stats\.csv$", p.name)]
    if combined_files:
        console.print(f"Skipping {len(combined_files)} combined season CSV file(s).")
        files = [p for p in files if not re.search(r"combined-stats\.csv$", p.name)]

    if game_id:
        filtered: list[Path] = []
        for p in files:
            if _find_game_id_in_filename(p) == game_id:
                filtered.append(p)
        if not filtered:
            console.print(f"[red]No file found for game_id[/red] {game_id}")
            raise typer.Exit(1)
        if len(filtered) > 1:
            console.print(f"[red]Multiple files matched game_id[/red] {game_id}: {filtered}")
            raise typer.Exit(1)
        files = filtered

    if limit_games is not None:
        files = files[:limit_games]

    input_hashes: dict[str, str] = {}
    if resume and input_hashes_path.exists():
        input_hashes = json.loads(input_hashes_path.read_text(encoding="utf-8")).get("files", {})

    unmapped_accum: list[pd.DataFrame] = []
    ingest_failures: list[dict] = []

    for csv_path in track(files, description="Ingesting vendor PBP CSVs..."):
        part_path = parts_dir / f"{csv_path.stem}.parquet"
        if resume and part_path.exists() and not overwrite:
            continue
        if overwrite and part_path.exists():
            part_path.unlink()

        try:
            result: IngestResult = ingest_vendor_game_csv(
                csv_path,
                season_id=season_id,
                schema_version=PBP_V1_SCHEMA_VERSION,
                prev_players_dim=prev_players_dim,
            )
        except Exception as exc:
            msg = str(exc)
            console.print(f"[red]Failed ingest[/red] {csv_path.name}: {msg}")
            if not skip_bad_games:
                raise typer.Exit(1)
            before = None
            after = None
            if isinstance(exc, VendorLineupMissingError):
                before = exc.num_rows_with_lineup_na_before_fill
                after = exc.num_rows_with_lineup_na_after_fill
            ingest_failures.append(
                {
                    "season_id": season_id,
                    "run_id": run_id,
                    "file_name": csv_path.name,
                    "error": msg,
                    "num_rows_with_lineup_na_before_fill": before,
                    "num_rows_with_lineup_na_after_fill": after,
                }
            )
            continue

        prev_players_dim = result.identity.players_dim
        input_hashes[csv_path.name] = result.input_sha256
        if len(result.identity.unmapped_players):
            unmapped_accum.append(result.identity.unmapped_players)

        _atomic_write_parquet(result.pbp_events, part_path)
        if prev_players_dim is not None:
            _atomic_write_parquet(prev_players_dim, players_dim_path)
        save_input_hashes(input_hashes, input_hashes_path)

    # Combine parts -> pbp_events.parquet (single canonical output).
    if overwrite and pbp_events_path.exists():
        pbp_events_path.unlink()
    part_files = sorted(parts_dir.glob("*.parquet"))
    if not part_files:
        console.print("[red]No pbp_events parts were produced; aborting combine.[/red]")
        raise typer.Exit(1)
    _combine_part_parquets(part_files, pbp_events_path)

    if prev_players_dim is not None and (overwrite or not players_dim_path.exists()):
        _atomic_write_parquet(prev_players_dim, players_dim_path)

    if unmapped_accum:
        unmapped = pd.concat(unmapped_accum, ignore_index=True).drop_duplicates().sort_values(["player_id"])
        _atomic_write_parquet(unmapped, unmapped_players_path)
        console.print(f"Wrote {len(unmapped):,} unmapped/new players to {unmapped_players_path}")
    else:
        if unmapped_players_path.exists():
            unmapped_players_path.unlink()

    console.print(f"pbp_events: {pbp_events_path}")
    console.print(f"players_dim: {players_dim_path}")
    console.print(f"input_hashes: {input_hashes_path}")
    if ingest_failures:
        failures_df = pd.DataFrame(ingest_failures)
        _atomic_write_parquet(failures_df, ingest_failures_path)
        console.print(f"ingest_failures: {ingest_failures_path} ({len(failures_df):,} rows)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
