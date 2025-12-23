"""Health checks for the live projections pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

from projections import paths
from projections.minutes_v1.constants import AvailabilityStatus

app = typer.Typer(help=__doc__)
console = Console()


def _load_run_id(pointer: Path) -> str | None:
    if not pointer.exists():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run_id = payload.get("run_id")
    return str(run_id) if run_id else None


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _newest_run_dir(day_dir: Path) -> Path | None:
    if not day_dir.exists():
        return None
    run_dirs = [p for p in day_dir.iterdir() if p.is_dir() and p.name.startswith("run=")]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _require_file(path: Path, *, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing {label}: {path}")


@app.command()
def check_latest_projections(
    lookback_hours: int = typer.Option(4, help="How far back to look for a projection file."),
    min_minutes_per_team: float = typer.Option(230.0, help="Minimum total minutes per team."),
    max_minutes_per_team: float = typer.Option(250.0, help="Maximum total minutes per team."),
) -> None:
    """Verify the integrity of the latest projections."""
    
    # 1. Find latest projection file
    proj_dir = paths.data_path("live", "projections_minutes_v1")
    if not proj_dir.exists():
        console.print(f"[yellow]Projections directory not found: {proj_dir} (skipping health check)[/yellow]")
        raise typer.Exit(code=0)
        
    # Find latest csv
    files = sorted(proj_dir.glob("*.csv"))
    if not files:
        console.print(f"[yellow]No projection files found in {proj_dir}; skipping health check[/yellow]")
        raise typer.Exit(code=0)
        
    latest_file = files[-1]
    file_age = pd.Timestamp.now() - pd.Timestamp.fromtimestamp(latest_file.stat().st_mtime)
    
    console.print(f"Checking latest file: [bold]{latest_file.name}[/bold] (Age: {file_age})")
    
    if file_age > pd.Timedelta(hours=lookback_hours):
        console.print(f"[red]Latest projection file is too old! > {lookback_hours} hours.[/red]")
        # We might not want to fail hard if it's just morning and no games yet, but for now warn.
        # raise typer.Exit(code=1) 
    
    df = pd.read_csv(latest_file)
    
    # 2. Check for NaNs in predictions
    if df["minutes"].isna().any():
        console.print("[red]Found NaN predictions![/red]")
        print(df[df["minutes"].isna()])
        raise typer.Exit(code=1)
        
    # 3. Check Team Totals
    team_totals = df.groupby("team_tricode")["minutes"].sum()
    console.print("\nTeam Totals:")
    print(team_totals)
    
    failed_teams = team_totals[
        (team_totals < min_minutes_per_team) | (team_totals > max_minutes_per_team)
    ]
    
    if not failed_teams.empty:
        console.print(f"\n[red]Teams with invalid minute totals (Range {min_minutes_per_team}-{max_minutes_per_team}):[/red]")
        print(failed_teams)
        raise typer.Exit(code=1)
        
    console.print("\n[green]All checks passed![/green]")


@app.command()
def check_rates_sanity(
    date_str: Optional[str] = typer.Option(None, "--date", help="Date to check (YYYY-MM-DD), defaults to today."),
    rates_root: Path = typer.Option(
        paths.data_path("gold", "rates_v1_live"),
        "--rates-root",
        help="Root directory for rates predictions.",
    ),
    min_fga2_median: float = typer.Option(0.15, help="Minimum pred_fga2_per_min median."),
    min_top_fpts: float = typer.Option(45.0, help="Minimum expected top FPTS on a slate."),
) -> None:
    """Check that rates predictions are plausible."""
    from projections.pipeline.guardrails import check_rates_output_sanity

    if date_str is None:
        import datetime
        date_str = datetime.date.today().isoformat()

    day_dir = rates_root / date_str
    if not day_dir.exists():
        console.print(f"[yellow]No rates predictions for {date_str}[/yellow]")
        raise typer.Exit(code=0)

    # Find latest run
    runs = sorted(day_dir.glob("run=*"))
    if not runs:
        console.print(f"[yellow]No runs found in {day_dir}[/yellow]")
        raise typer.Exit(code=0)

    latest_run = runs[-1]
    rates_file = latest_run / "rates.parquet"
    if not rates_file.exists():
        console.print(f"[red]Missing rates.parquet in {latest_run}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Checking rates: [bold]{rates_file}[/bold]")
    df = pd.read_parquet(rates_file)

    result = check_rates_output_sanity(
        df,
        min_fga2_median=min_fga2_median,
        min_top_fpts=min_top_fpts,
    )

    console.print(f"\nMetrics: {result.metrics}")

    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warn in result.warnings:
            console.print(f"  - {warn}")

    if result.passed:
        console.print("\n[green]Rates sanity check passed![/green]")
    else:
        console.print("\n[red]Rates sanity check FAILED[/red]")
        raise typer.Exit(code=1)


@app.command()
def check_feature_coverage(
    date_str: Optional[str] = typer.Option(None, "--date", help="Date to check (YYYY-MM-DD), defaults to today."),
    features_root: Path = typer.Option(
        paths.data_path("live", "features_rates_v1"),
        "--features-root",
        help="Root directory for rates features.",
    ),
    expected_rows: Optional[int] = typer.Option(None, help="Expected number of rows (if known)."),
) -> None:
    """Check that rates features have adequate coverage."""
    from projections.pipeline.guardrails import check_feature_coverage as _check_feature_coverage

    if date_str is None:
        import datetime
        date_str = datetime.date.today().isoformat()

    day_dir = features_root / date_str
    if not day_dir.exists():
        console.print(f"[yellow]No features for {date_str}[/yellow]")
        raise typer.Exit(code=0)

    runs = sorted(day_dir.glob("run=*"))
    if not runs:
        console.print(f"[yellow]No runs found in {day_dir}[/yellow]")
        raise typer.Exit(code=0)

    latest_run = runs[-1]
    features_file = latest_run / "features.parquet"
    if not features_file.exists():
        console.print(f"[red]Missing features.parquet in {latest_run}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Checking features: [bold]{features_file}[/bold]")
    df = pd.read_parquet(features_file)

    critical_cols = ["game_id", "player_id", "team_id", "minutes_pred_p50", "season_fga_per_min"]
    result = _check_feature_coverage(
        df,
        expected_rows=expected_rows,
        critical_cols=critical_cols,
    )

    console.print(f"\nMetrics: {result.metrics}")

    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warn in result.warnings:
            console.print(f"  - {warn}")

    if result.passed:
        console.print("\n[green]Feature coverage check passed![/green]")
    else:
        console.print("\n[red]Feature coverage check FAILED[/red]")
        raise typer.Exit(code=1)


@app.command()
def check_artifact_pointers(
    date_str: Optional[str] = typer.Option(None, "--date", help="Date to check (YYYY-MM-DD), defaults to today."),
    data_root: Path = typer.Option(
        paths.data_path(),
        "--data-root",
        help="Root directory containing artifacts and gold outputs (defaults to PROJECTIONS_DATA_ROOT).",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Fail (strict) or warn (no-strict) on pointer inconsistencies.",
    ),
) -> None:
    """Validate latest_run.json pointers + unified projections summary.json for a slate date."""

    if date_str is None:
        import datetime

        date_str = datetime.date.today().isoformat()

    errors: list[str] = []
    warnings: list[str] = []

    def _warn_or_error(message: str) -> None:
        (errors if strict else warnings).append(message)

    raw_minutes_root = os.environ.get("MINUTES_DAILY_ROOT")
    if raw_minutes_root:
        minutes_daily_root = Path(raw_minutes_root).expanduser().resolve()
    else:
        minutes_daily_root = data_root / "artifacts" / "minutes_v1" / "daily"
    minutes_day = minutes_daily_root / date_str
    minutes_run_id_daily = _load_run_id(minutes_day / "latest_run.json")

    # Legacy minutes pointer (gold layer) for backfills / transition.
    minutes_gold_day = data_root / "gold" / "projections_minutes_v1" / f"game_date={date_str}"
    minutes_run_id_gold = _load_run_id(minutes_gold_day / "latest_run.json")

    minutes_run_id = minutes_run_id_daily or minutes_run_id_gold
    minutes_run_dir: Path | None = None
    if minutes_run_id_daily:
        minutes_run_dir = minutes_day / f"run={minutes_run_id_daily}"
    elif minutes_run_id_gold:
        minutes_run_dir = minutes_gold_day / f"run={minutes_run_id_gold}"

    if minutes_run_id_daily is None:
        _warn_or_error(
            f"Minutes daily latest_run.json missing under {minutes_day}"
            + (f" (gold fallback run_id={minutes_run_id_gold})" if minutes_run_id_gold else "")
        )

    if minutes_run_id is None:
        _warn_or_error(f"Minutes latest_run.json missing under both {minutes_day} and {minutes_gold_day}")
    elif minutes_run_dir is None or not minutes_run_dir.exists():
        errors.append(f"Minutes run dir missing for run_id={minutes_run_id}: {minutes_run_dir or minutes_day}")
    else:
        _require_file(minutes_run_dir / "minutes.parquet", label="minutes.parquet", errors=errors)
        if minutes_run_dir.parent == minutes_day:
            _require_file(minutes_run_dir / "summary.json", label="minutes summary.json", errors=errors)
            newest = _newest_run_dir(minutes_day)
            if newest is not None and newest != minutes_run_dir:
                _warn_or_error(
                    f"Minutes pointer run != newest dir: pointer={minutes_run_dir.name} newest={newest.name}"
                )

    rates_day = data_root / "gold" / "rates_v1_live" / date_str
    rates_run_id = _load_run_id(rates_day / "latest_run.json")
    rates_run_dir = rates_day / f"run={rates_run_id}" if rates_run_id else None
    if rates_run_id is None:
        _warn_or_error(f"Rates latest_run.json missing under {rates_day}")
    elif rates_run_dir is None or not rates_run_dir.exists():
        errors.append(f"Rates run dir missing for run_id={rates_run_id}: {rates_day}")
    else:
        _require_file(rates_run_dir / "rates.parquet", label="rates.parquet", errors=errors)
        _require_file(rates_run_dir / "summary.json", label="rates summary.json", errors=errors)
        newest = _newest_run_dir(rates_day)
        if newest is not None and newest != rates_run_dir:
            _warn_or_error(f"Rates pointer run != newest dir: pointer={rates_run_dir.name} newest={newest.name}")

    sim_day = data_root / "artifacts" / "sim_v2" / "worlds_fpts_v2" / f"game_date={date_str}"
    sim_run_id = _load_run_id(sim_day / "latest_run.json")
    sim_run_dir = sim_day / f"run={sim_run_id}" if sim_run_id else None
    if sim_run_id is None:
        _warn_or_error(f"Sim worlds latest_run.json missing under {sim_day}")
    elif sim_run_dir is None or not sim_run_dir.exists():
        errors.append(f"Sim worlds run dir missing for run_id={sim_run_id}: {sim_day}")
    else:
        _require_file(sim_run_dir / "projections.parquet", label="sim projections.parquet", errors=errors)
        newest = _newest_run_dir(sim_day)
        if newest is not None and newest != sim_run_dir:
            _warn_or_error(f"Sim pointer run != newest dir: pointer={sim_run_dir.name} newest={newest.name}")

    projections_day = data_root / "artifacts" / "projections" / date_str
    projections_run_id = _load_run_id(projections_day / "latest_run.json")
    projections_run_dir = projections_day / f"run={projections_run_id}" if projections_run_id else None
    if projections_run_id is None:
        _warn_or_error(f"Unified projections latest_run.json missing under {projections_day}")
    elif projections_run_dir is None or not projections_run_dir.exists():
        errors.append(f"Unified projections run dir missing for run_id={projections_run_id}: {projections_day}")
    else:
        _require_file(projections_run_dir / "projections.parquet", label="unified projections.parquet", errors=errors)
        _require_file(projections_run_dir / "summary.json", label="unified summary.json", errors=errors)
        newest = _newest_run_dir(projections_day)
        if newest is not None and newest != projections_run_dir:
            _warn_or_error(
                f"Unified projections pointer run != newest dir: pointer={projections_run_dir.name} newest={newest.name}"
            )

        summary_path = projections_run_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                errors.append(f"Unified summary.json unreadable: {summary_path}")
            else:
                if summary.get("projections_run_id") != projections_run_id:
                    errors.append(
                        "Unified summary.projections_run_id mismatch: "
                        f"summary={summary.get('projections_run_id')} pointer={projections_run_id}"
                    )
                required_keys = (
                    "minutes_run_id",
                    "rates_run_id",
                    "sim_run_id",
                    "sim_profile",
                    "n_worlds",
                )
                for key in required_keys:
                    if summary.get(key) in (None, ""):
                        errors.append(f"Unified summary.{key} missing/empty: {summary_path}")

                if minutes_run_id and summary.get("minutes_run_id") != minutes_run_id:
                    errors.append(
                        "Unified summary.minutes_run_id mismatch: "
                        f"summary={summary.get('minutes_run_id')} minutes_pointer={minutes_run_id}"
                    )
                if rates_run_id and summary.get("rates_run_id") != rates_run_id:
                    errors.append(
                        "Unified summary.rates_run_id mismatch: "
                        f"summary={summary.get('rates_run_id')} rates_pointer={rates_run_id}"
                    )
                if sim_run_id and summary.get("sim_run_id") != sim_run_id:
                    errors.append(
                        "Unified summary.sim_run_id mismatch: "
                        f"summary={summary.get('sim_run_id')} sim_pointer={sim_run_id}"
                    )

    if warnings:
        console.print("\n[yellow]Pointer warnings:[/yellow]")
        for warn in warnings:
            console.print(f"  - {warn}")

    if errors:
        console.print("\n[red]Pointer errors:[/red]")
        for err in errors:
            console.print(f"  - {err}")
        if strict:
            raise typer.Exit(code=1)

    console.print("\n[green]Artifact pointer check complete.[/green]")


@app.command()
def pin_projections_run(
    date_str: Optional[str] = typer.Option(None, "--date", help="Date to pin (YYYY-MM-DD), defaults to today."),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Projections run_id to pin (omit to clear pin).",
    ),
    data_root: Path = typer.Option(
        paths.data_path(),
        "--data-root",
        help="Root directory containing artifacts (defaults to PROJECTIONS_DATA_ROOT).",
    ),
) -> None:
    """Pin the projections API/dashboard to a specific unified projections run.

    Writes (or removes) `$DATA_ROOT/artifacts/projections/<DATE>/pinned_run.json`.
    The API prefers this pinned pointer over latest_run.json so rescore runs can
    be inspected without being immediately replaced by the live pipeline.
    """
    if date_str is None:
        import datetime

        date_str = datetime.date.today().isoformat()

    projections_day = data_root / "artifacts" / "projections" / date_str
    pinned_path = projections_day / "pinned_run.json"

    if run_id is None:
        if pinned_path.exists():
            pinned_path.unlink()
            console.print(f"[green]Removed pinned pointer:[/green] {pinned_path}")
        else:
            console.print(f"[yellow]No pinned pointer present:[/yellow] {pinned_path}")
        raise typer.Exit(code=0)

    run_dir = projections_day / f"run={run_id}"
    parquet_path = run_dir / "projections.parquet"
    if not parquet_path.exists():
        console.print(f"[red]Run not found or missing projections.parquet:[/red] {parquet_path}")
        raise typer.Exit(code=1)

    payload = {
        "run_id": run_id,
    }
    try:
        from datetime import datetime as dt, timezone

        payload["updated_at"] = dt.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        payload["updated_at"] = None

    _atomic_write_json(pinned_path, payload)
    console.print(f"[green]Pinned projections run:[/green] {date_str} -> run={run_id}")


if __name__ == "__main__":
    app()
