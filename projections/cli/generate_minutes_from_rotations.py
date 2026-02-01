"""Dev CLI: sample minutes worlds from rot_v1 rotation templates."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from projections.rotations.generator import TeamContext
from projections.rotations.priors_humility import HumilityConfig, load_humility_config_json
from projections.rotations.rotation_gate import GateConfig, load_gate_config_json
from projections.rotations.rotation_predictor import (
    canonicalize_game_id,
    load_cached_all_predictions,
    load_cached_predictions,
    load_cached_train_predictions,
    load_rotation_predictor_bundle,
    season_start_year_from_game_id,
)
from projections.rotations.template_generator import TemplateRotationGenerator
from projections.rotations.player_map import build_person_id_to_internal_id_map

console = Console()
app = typer.Typer(help="Generate minutes samples using TemplateRotationGenerator (dev tool).")


def _parse_csv_ints(value: str) -> list[int]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return [int(p) for p in parts]


def _load_minutes_prior(path: Path) -> dict[int, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("minutes_prior JSON must be an object mapping player_id -> minutes")
    out: dict[int, float] = {}
    for k, v in raw.items():
        out[int(k)] = float(v)
    return out


@app.command()
def run(
    game_id: str = typer.Option("unknown", "--game-id", help="Optional NBA game_id (prefer 10-digit, e.g. 0022400001)."),
    season_id: str = typer.Option("unknown", "--season-id", help="Optional season_id string (e.g. 2024-25)."),
    team_id: int = typer.Option(..., "--team-id", help="NBA team_id (integer)."),
    n_worlds: int = typer.Option(2000, "--n-worlds", help="Number of sampled worlds."),
    seed: int = typer.Option(0, "--seed", help="RNG seed (deterministic)."),
    rot_bundle: Path = typer.Option(
        Path("/home/daniel/projections-data/artifacts/rot_v1/LATEST_PUBLISHED"),
        "--rot-bundle",
        help="rot_v1 bundle dir or LATEST_PUBLISHED pointer file.",
    ),
    candidates: str | None = typer.Option(
        None,
        "--candidates",
        help="Optional CSV list of candidate player_ids (e.g. '123,456,789').",
    ),
    minutes_prior_json: Path | None = typer.Option(
        None,
        "--minutes-prior",
        help="Optional JSON file mapping player_id -> prior minutes.",
    ),
    starter_candidates: str | None = typer.Option(
        None,
        "--starter-candidates",
        help="Optional CSV list of starter candidates (e.g. '123,456,789,101,102').",
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
) -> None:
    minutes_prior = _load_minutes_prior(minutes_prior_json) if minutes_prior_json else None
    candidate_ids = _parse_csv_ints(candidates) if candidates else None
    starter_ids = _parse_csv_ints(starter_candidates) if starter_candidates else None

    cfg = HumilityConfig()
    if humility_config is not None:
        cfg = load_humility_config_json(Path(humility_config), base=cfg)
    cfg = replace(cfg, enabled=bool(humility))

    gate_cfg = GateConfig()
    if gate_config is not None:
        gate_cfg = load_gate_config_json(Path(gate_config), base=gate_cfg)
    gate_cfg = replace(gate_cfg, enabled=bool(gate))

    gate_preds = None
    gate_feature_source = str(gate_feature_source).strip().lower()
    if bool(gate_cfg.enabled) and rotation_predictor_bundle is not None and gate_feature_source != "none":
        bundle = load_rotation_predictor_bundle(Path(rotation_predictor_bundle))
        allow_game = canonicalize_game_id(game_id)
        game_allow = {allow_game} if allow_game else None
        team_allow = {int(team_id)}

        id_map = None
        season_start_year = season_start_year_from_game_id(allow_game) if allow_game else None
        if season_start_year is not None:
            try:
                id_map = build_person_id_to_internal_id_map(season_start_year=int(season_start_year)).person_id_to_internal_id
            except Exception:
                id_map = None

        if gate_feature_source == "cached_preds":
            gate_preds = load_cached_predictions(
                bundle,
                person_id_to_internal_id=id_map,
                game_id_allow=game_allow,
                team_id_allow=team_allow,
            )
        elif gate_feature_source == "cached_all":
            gate_preds = load_cached_all_predictions(
                bundle,
                person_id_to_internal_id=id_map,
                game_id_allow=game_allow,
                team_id_allow=team_allow,
            )
        elif gate_feature_source == "cached_train":
            gate_preds = load_cached_train_predictions(
                bundle,
                person_id_to_internal_id=id_map,
                game_id_allow=game_allow,
                team_id_allow=team_allow,
            )
        else:
            raise ValueError(f"Unknown --gate-feature-source: {gate_feature_source}")

    gen = TemplateRotationGenerator(rot_bundle=rot_bundle, humility_config=cfg, gate_config=gate_cfg, gate_preds=gate_preds)
    ctx = TeamContext(
        season_id=str(season_id),
        game_id=canonicalize_game_id(game_id) or str(game_id),
        team_id=int(team_id),
        opponent_team_id=-1,
        is_home=False,
        candidate_player_ids=candidate_ids,
        starter_candidates=starter_ids,
        minutes_prior=minutes_prior,
        n_worlds=int(n_worlds),
        rng_seed=int(seed),
    )
    worlds = gen.generate(ctx)

    # Summaries
    rows: list[dict] = []
    for pid, arr in sorted(worlds.minutes_by_player.items(), key=lambda kv: kv[0]):
        a = np.asarray(arr, dtype=float)
        rows.append(
            {
                "player_id": int(pid),
                "mean": float(a.mean()),
                "p10": float(np.quantile(a, 0.10)),
                "p50": float(np.quantile(a, 0.50)),
                "p90": float(np.quantile(a, 0.90)),
                "play_prob_ge_1": float((a >= 1.0).mean()),
                "rotation_prob_ge_5": float((a >= 5.0).mean()),
            }
        )

    if not rows:
        console.print("[red]No minutes generated.[/red]")
        raise typer.Exit(1)

    summary = sorted(rows, key=lambda r: (-r["mean"], r["player_id"]))
    for r in summary:
        console.print(
            f"pid={r['player_id']} mean={r['mean']:.2f} p10={r['p10']:.2f} p50={r['p50']:.2f} p90={r['p90']:.2f} "
            f"play_prob={r['play_prob_ge_1']:.3f} rot_prob={r['rotation_prob_ge_5']:.3f}"
        )

    console.print(f"diagnostics: {json.dumps(worlds.diagnostics or {}, sort_keys=True)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
