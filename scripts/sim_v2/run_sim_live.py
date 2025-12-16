"""Run sim_v2 worlds + aggregation for a live slate."""

from __future__ import annotations

import shutil
from datetime import date as date_cls
from pathlib import Path

import typer

from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main

app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_date: str | None = typer.Option(
        None, "--run-date", help="Date to run sim for (YYYY-MM-DD). Defaults to today."
    ),
    profile_name: str = typer.Option(
        "baseline",
        "--profile-name",
        "--profile",
        help="Sim profile name (defaults to baseline for rates path).",
    ),
    num_worlds: int = typer.Option(1000, "--num-worlds", "--n-worlds", help="Number of worlds per slate to simulate."),
    data_root: Path | None = typer.Option(None, "--data-root", help="Optional override for data root."),
    profiles_path: Path | None = typer.Option(None, "--profiles-path", help="Optional override for sim profile config."),
    worlds_root: Path | None = typer.Option(None, "--worlds-root", help="Optional output root for worlds parquet files."),
    projections_root: Path | None = typer.Option(
        None, "--projections-root", help="Optional output root for aggregated projections."
    ),
    include_std: bool = typer.Option(True, "--std/--no-std", help="Compute std when aggregating worlds."),
    minutes_run_id: str | None = typer.Option(None, "--minutes-run-id", help="Explicit minutes run_id to load."),
    rates_run_id: str | None = typer.Option(None, "--rates-run-id", help="Explicit rates run_id to load."),
) -> None:
    target_date = run_date or date_cls.today().isoformat()
    typer.echo(f"[sim_v2] live sim run date={target_date} profile={profile_name} worlds={num_worlds}")

    worlds_output = worlds_root
    projections_output = projections_root
    if data_root is not None:
        base_root = Path(data_root)
        worlds_output = worlds_output or base_root / "artifacts" / "sim_v2" / "worlds_fpts_v2"
        projections_output = projections_output or base_root / "artifacts" / "sim_v2" / "projections"

    # Generate worlds - now outputs projections.parquet directly with in-memory aggregation
    generate_worlds_main(
        start_date=target_date,
        end_date=target_date,
        n_worlds=num_worlds,
        profile=profile_name,
        data_root=data_root,
        profiles_path=profiles_path,
        output_root=worlds_output,
        fpts_run_id=None,
        use_rates_noise=None,
        rates_noise_split=None,
        team_sigma_scale=None,
        player_sigma_scale=None,
        rates_run_id=rates_run_id,
        minutes_run_id=minutes_run_id,
        use_minutes_noise=None,
        minutes_noise_run_id=None,
        minutes_sigma_min=None,
        seed=None,
        min_play_prob=None,
        team_factor_sigma=None,
        team_factor_gamma=None,
    )

    # Copy projections to output location if different from worlds root
    # (generate_worlds now outputs projections.parquet directly)
    if projections_output is not None and worlds_output is not None:
        src_proj = (worlds_output or Path()) / f"game_date={target_date}" / "projections.parquet"
        dst_dir = projections_output / f"game_date={target_date}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_proj = dst_dir / "projections.parquet"
        if src_proj.exists():
            shutil.copy2(src_proj, dst_proj)
            typer.echo(f"[sim_v2] copied projections to {dst_proj}")


if __name__ == "__main__":
    app()

