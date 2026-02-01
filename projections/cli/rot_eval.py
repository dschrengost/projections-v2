from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console

from projections.rotations.eval import run_rotation_generator_eval
from projections.rotations.priors_humility import HumilityConfig, load_humility_config_json
from projections.rotations.rotation_gate import GateConfig, load_gate_config_json

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
    minutes_prior_parquet: Path | None = typer.Option(
        None,
        "--minutes-prior-parquet",
        help=(
            "Optional minutes prior parquet keyed by (game_id, team_id, player_id) with columns "
            "minutes_prior, minutes_p10, minutes_p90, and play_prob (p10/p90 are optional). When provided, "
            "it is used as the mapping-stabilizer prior "
            "(instead of truth minutes_actual)."
        ),
    ),
    restrict_to_prior_games: bool | None = typer.Option(
        None,
        "--restrict-to-prior-games/--no-restrict-to-prior-games",
        help="When --minutes-prior-parquet is provided, restrict evaluation to games present in the prior.",
    ),
    candidate_pool: str = typer.Option(
        "truth",
        "--candidate-pool",
        help="Candidate pool mode: truth|prior_topn|prior_threshold|predictor_threshold|roster (default: truth).",
    ),
    candidate_top_n: int = typer.Option(
        12,
        "--candidate-top-n",
        help="For prior_topn: include top N by minutes_prior (ties by player_id). Also used for backfill in other pools.",
    ),
    candidate_min_minutes_prior: float = typer.Option(
        0.0,
        "--candidate-min-minutes-prior",
        help="For prior pools: include anyone with minutes_prior >= this threshold.",
    ),
    candidate_min_play_prob: float = typer.Option(
        0.8,
        "--candidate-min-play-prob",
        help="For prior pools: include anyone with play_prob >= this threshold.",
    ),
    candidate_min_candidates: int = typer.Option(
        8,
        "--candidate-min-candidates",
        help="Ensure each team-game candidate pool has at least this many players (deterministic backfill).",
    ),
    pool_max_size: int = typer.Option(
        11,
        "--pool-max-size",
        help="For predictor_threshold: maximum candidate pool size per team-game.",
    ),
    t_ge15: float = typer.Option(
        0.35,
        "--t-ge15",
        help="For predictor_threshold: include players with p_ge15_pred >= this threshold.",
    ),
    t_ge5: float = typer.Option(
        0.35,
        "--t-ge5",
        help="For predictor_threshold: include players with p_ge5_pred >= this threshold (only when p_ge15_pred is below t_ge15).",
    ),
    always_include_top_n: int = typer.Option(
        8,
        "--always-include-top-n",
        help="For predictor_threshold: always include top N by minutes_prior (ties by p_ge15 then p_ge5).",
    ),
    humility: bool = typer.Option(
        True,
        "--humility/--no-humility",
        help="Enable PriorHumilityLayer guardrails when constructing generator priors (default: on).",
    ),
    humility_config: Path | None = typer.Option(
        None,
        "--humility-config",
        help="Optional JSON file of HumilityConfig overrides.",
    ),
    gate: bool = typer.Option(
        False,
        "--gate/--no-gate",
        help="Enable RotationGateLayer hard gating (default: off).",
    ),
    gate_config: Path | None = typer.Option(
        None,
        "--gate-config",
        help="Optional JSON file of GateConfig overrides.",
    ),
    rotation_predictor_bundle: Path | None = typer.Option(
        None,
        "--rotation-predictor-bundle",
        help="Path to artifacts/rotation_predictor_v1/<run_id> (or a pointer file).",
    ),
    gate_feature_source: str = typer.Option(
        "cached_preds",
        "--gate-feature-source",
        help="Gate feature source: cached_all|cached_preds|cached_train|none (recommend cached_all; default: cached_preds).",
    ),
    gate_max_train_rows: int | None = typer.Option(
        None,
        "--gate-max-train-rows",
        help="Optional max rows to read from cached training dataset (cached_train mode).",
    ),
    baseline_out_dir: Path | None = typer.Option(
        None,
        "--baseline-out-dir",
        help="Optional prior rot_eval_v1 output directory to compare against in report.md.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing output directory."),
) -> None:
    data_root = _default_data_root()
    rot_bundle = rot_bundle or (data_root / "artifacts" / "rot_v1" / "LATEST_PUBLISHED")
    artifacts_root = data_root / "artifacts" / "rot_eval_v1"
    out_dir = artifacts_root / run_id

    minutes_prior_parquet_path = Path(minutes_prior_parquet) if minutes_prior_parquet is not None else None
    if restrict_to_prior_games is None:
        restrict_to_prior_games = minutes_prior_parquet_path is not None

    cfg = HumilityConfig()
    if humility_config is not None:
        cfg = load_humility_config_json(Path(humility_config), base=cfg)
    cfg = replace(cfg, enabled=bool(humility))

    gate_cfg = GateConfig()
    if gate_config is not None:
        gate_cfg = load_gate_config_json(Path(gate_config), base=gate_cfg)
    gate_cfg = replace(gate_cfg, enabled=bool(gate))

    pool_mode = str(candidate_pool).strip().lower()
    if pool_mode == "predictor_threshold":
        if minutes_prior_parquet_path is None:
            raise typer.BadParameter(
                "--minutes-prior-parquet is required when --candidate-pool predictor_threshold"
            )
        if rotation_predictor_bundle is None:
            raise typer.BadParameter(
                "--rotation-predictor-bundle is required when --candidate-pool predictor_threshold"
            )

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
        minutes_prior_parquet=minutes_prior_parquet_path,
        restrict_to_prior_games=bool(restrict_to_prior_games),
        candidate_pool=str(candidate_pool),
        candidate_top_n=int(candidate_top_n),
        candidate_min_minutes_prior=float(candidate_min_minutes_prior),
        candidate_min_play_prob=float(candidate_min_play_prob),
        candidate_min_candidates=int(candidate_min_candidates),
        pool_max_size=int(pool_max_size),
        pool_t_ge15=float(t_ge15),
        pool_t_ge5=float(t_ge5),
        pool_always_include_top_n=int(always_include_top_n),
        humility_config=cfg,
        gate_config=gate_cfg,
        rotation_predictor_bundle=Path(rotation_predictor_bundle) if rotation_predictor_bundle is not None else None,
        gate_feature_source=str(gate_feature_source),
        gate_max_train_rows=int(gate_max_train_rows) if gate_max_train_rows is not None else None,
        baseline_out_dir=Path(baseline_out_dir) if baseline_out_dir is not None else None,
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
