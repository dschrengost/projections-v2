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
from projections.rotations.template_generator import TemplateRotationGenerator

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
) -> None:
    minutes_prior = _load_minutes_prior(minutes_prior_json) if minutes_prior_json else None
    candidate_ids = _parse_csv_ints(candidates) if candidates else None
    starter_ids = _parse_csv_ints(starter_candidates) if starter_candidates else None

    cfg = HumilityConfig()
    if humility_config is not None:
        cfg = load_humility_config_json(Path(humility_config), base=cfg)
    cfg = replace(cfg, enabled=bool(humility))

    gen = TemplateRotationGenerator(rot_bundle=rot_bundle, humility_config=cfg)
    ctx = TeamContext(
        season_id="unknown",
        game_id="unknown",
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
