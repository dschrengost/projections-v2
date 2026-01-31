from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from projections.rotations.eval import run_rotation_generator_eval

console = Console()
app = typer.Typer(help="Evaluate TemplateRotationGenerator realism against rot_v1 rotation_labels truth.")


def _default_data_root() -> Path:
    return Path(os.getenv("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data"))


@app.command()
def main(
    *,
    rot_bundle: Path = typer.Option(
        None,
        "--rot-bundle",
        help="Path to rot_v1 bundle directory or LATEST_PUBLISHED pointer file.",
    ),
    run_id: str = typer.Option(..., "--run-id", help="Output run_id (directory name under artifacts/rot_eval_v1/)."),
    n_worlds: int = typer.Option(2000, "--n-worlds", help="Number of generator worlds to sample per team-game."),
    seed: int = typer.Option(0, "--seed", help="RNG seed for deterministic team-game sampling."),
    limit_team_games: int = typer.Option(200, "--limit-team-games", help="Limit sampled team-games (<=0 means all)."),
    sample_mode: str = typer.Option("random", "--sample-mode", help="Sampling mode: random|first"),
    use_truth_minutes_prior: bool = typer.Option(
        True,
        "--use-truth-minutes-prior/--no-use-truth-minutes-prior",
        help="Use truth minutes_actual as minutes_prior (mapping stabilizer only).",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing output directory."),
) -> None:
    data_root = _default_data_root()
    rot_bundle = rot_bundle or (data_root / "artifacts" / "rot_v1" / "LATEST_PUBLISHED")
    artifacts_root = data_root / "artifacts" / "rot_eval_v1"
    out_dir = artifacts_root / run_id

    result = run_rotation_generator_eval(
        rot_bundle_path=rot_bundle,
        run_id=run_id,
        n_worlds=n_worlds,
        seed=seed,
        limit_team_games=limit_team_games,
        sample_mode=sample_mode,  # validated in eval runner
        out_dir=out_dir,
        overwrite=overwrite,
        use_truth_minutes_prior=use_truth_minutes_prior,
    )

    metrics = result.get("metrics", {})
    brier_ge1 = float(metrics.get("brier_ge1", float("nan")))
    brier_ge5 = float(metrics.get("brier_ge5", float("nan")))
    minutes_mae = float(metrics.get("minutes_mae", float("nan")))
    console.print(f"out_dir: {result.get('out_dir')}")
    console.print(f"rows: players={metrics.get('n_players')} team_games={metrics.get('n_team_games')}")
    console.print(
        "headline: "
        f"brier_ge1={brier_ge1:.6f} "
        f"brier_ge5={brier_ge5:.6f} "
        f"minutes_mae={minutes_mae:.3f}"
    )
    console.print(f"report: {result.get('report_path')}")


if __name__ == "__main__":
    app()
