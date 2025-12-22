"""Run sim_v2 worlds + aggregation for a live slate."""

from __future__ import annotations

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
    include_std: bool = typer.Option(True, "--std/--no-std", help="Compute std when aggregating worlds."),
    sim_run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run id to partition outputs under game_date=.../run=...",
    ),
    run_as_of_ts: str | None = typer.Option(
        None,
        "--run-as-of-ts",
        help="Optional as-of timestamp to record in latest_run.json (UTC string).",
    ),
    minutes_run_id: str | None = typer.Option(None, "--minutes-run-id", help="Explicit minutes run_id to load."),
    rates_run_id: str | None = typer.Option(None, "--rates-run-id", help="Explicit rates run_id to load."),
) -> None:
    target_date = run_date or date_cls.today().isoformat()
    typer.echo(f"[sim_v2] live sim run date={target_date} profile={profile_name} worlds={num_worlds}")

    worlds_output = worlds_root
    if data_root is not None:
        base_root = Path(data_root)
        worlds_output = worlds_output or base_root / "artifacts" / "sim_v2" / "worlds_fpts_v2"

    # Generate worlds - outputs projections.parquet directly with in-memory aggregation
    generate_worlds_main(
        start_date=target_date,
        end_date=target_date,
        n_worlds=num_worlds,
        profile=profile_name,
        data_root=data_root,
        profiles_path=profiles_path,
        output_root=worlds_output,
        sim_run_id=sim_run_id,
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

    # Write latest_run.json pointer if run_id specified
    if sim_run_id and worlds_output is not None:
        import json
        from datetime import datetime, timezone

        payload = {
            "run_id": sim_run_id,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if run_as_of_ts:
            payload["run_as_of_ts"] = run_as_of_ts

        day_dir = worlds_output / f"game_date={target_date}"
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "latest_run.json").write_text(json.dumps(payload), encoding="utf-8")



if __name__ == "__main__":
    app()
