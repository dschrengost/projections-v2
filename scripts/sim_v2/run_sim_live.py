"""Run sim_v2 worlds + aggregation for a live slate."""

from __future__ import annotations

# ruff: noqa: E402

from projections.runtime_safety import configure_runtime_safety

configure_runtime_safety()

from datetime import date as date_cls
from pathlib import Path

import typer

from projections.sim_v2.config import load_sim_v2_profile
from scripts.sim_v2.generate_worlds_fpts_v2 import main as generate_worlds_main

app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_date: str | None = typer.Option(
        None, "--run-date", help="Date to run sim for (YYYY-MM-DD). Defaults to today."
    ),
    profile_name: str = typer.Option(
        "sim_v3",
        "--profile-name",
        "--profile",
        help="Sim profile name (defaults to sim_v3).",
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
    minutes_override_mode: str = typer.Option(
        "v2",
        "--minutes-override-mode",
        help="Minutes override mode: v2 (default) or legacy compatibility mode.",
    ),
    override_infeasible: str = typer.Option(
        "error",
        "--override-infeasible",
        help="Behavior when v2 override constraints are infeasible: error|relax|ignore.",
    ),
) -> None:
    target_date = run_date or date_cls.today().isoformat()
    typer.echo(f"[sim_v2] live sim run date={target_date} profile={profile_name} worlds={num_worlds}")

    try:
        profile_cfg = load_sim_v2_profile(profile=profile_name, profiles_path=profiles_path)
        typer.echo(
            f"[sim_v2] profile_cfg.use_play_prob_masking={profile_cfg.use_play_prob_masking} "
            f"(min_play_prob={profile_cfg.min_play_prob})"
        )
    except Exception as exc:
        typer.echo(f"[sim_v2] warning: failed to resolve profile config ({exc})", err=True)

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
        use_efficiency_scoring=None,
        export_attempt_means=False,
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
        minutes_override_mode=minutes_override_mode,
        override_infeasible=override_infeasible,
    )

    # Write latest_run.json pointer if run_id specified
    if sim_run_id and worlds_output is not None:
        import os

        if os.environ.get("PROJECTIONS_SKIP_POINTER_WRITES", "").strip().lower() in {"1", "true", "yes"}:
            return

        from projections.pipeline import writer_guard

        import json
        from datetime import datetime, timezone

        writer_guard.assert_can_write_pointers(purpose=f"run_sim_live promote {worlds_output}/{target_date}")
        
        # Build provenance for model tracking
        try:
            from projections.registry.bundle_resolver import (
                resolve_minutes_bundle,
                resolve_rates_bundle,
                build_provenance,
            )
            
            # Get metadata about which models produced predictions
            _, minutes_meta = resolve_minutes_bundle("minutes_v1", "production")
            _, rates_meta = resolve_rates_bundle("rates_v1", "production")
            
            provenance = build_provenance(
                minutes_meta=minutes_meta,
                rates_meta=rates_meta,
                sim_profile=profile_name,
                seed=None,  # Could be passed via generate_worlds_main if needed
                n_worlds=num_worlds,
            )
        except Exception as e:
            typer.echo(f"[sim_v2] warning: could not build provenance: {e}", err=True)
            provenance = {
                "sim_profile": profile_name,
                "n_worlds": num_worlds,
            }
        
        payload = {
            "run_id": sim_run_id,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "provenance": provenance,
        }
        if run_as_of_ts:
            payload["run_as_of_ts"] = run_as_of_ts

        day_dir = worlds_output / f"game_date={target_date}"
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "latest_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")



if __name__ == "__main__":
    app()
