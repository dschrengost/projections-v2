#!/usr/bin/env python
"""Build lineups from gold projections + salaries (CSV or gold dk_salaries).

Output CSV columns:
  lineup_id, site, game_date, contest_type,
  p1_id..p8_id (int player_id), p1_name..p8_name (string), mean_fpts, total_salary
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from projections.optimizer.player_pool_loader import build_player_pool_from_gold
from projections.optimizer.adapter import build_lineups_from_player_pool


app = typer.Typer(add_completion=False, help="Build lineups from local gold projections + salaries.")


@app.command()
def main(
    game_date: str = typer.Option(..., help="Slate date in YYYY-MM-DD format."),
    site: str = typer.Option("dk", help="Site to optimize for (dk or fd)."),
    draft_group_id: Optional[int] = typer.Option(
        None,
        "--draft-group-id",
        help="DraftKings draft_group_id for the slate. Required if no --salaries-csv is provided.",
    ),
    num_lineups: int = typer.Option(150, help="Number of lineups to generate."),
    salaries_csv: Optional[Path] = typer.Option(
        None,
        "--salaries-csv",
        exists=True,
        readable=True,
        help="Optional path to salaries CSV. If omitted, gold dk_salaries parquet is used (requires --draft-group-id).",
    ),
    out: Optional[Path] = typer.Option(
        None,
        help=(
            "Output CSV path. Defaults to artifacts/lineups/{site}/game_date={game_date}/"
            "[draft_group_id=<id>/]lineups_pointproj.csv"
        ),
    ),
) -> None:
    if salaries_csv is None and draft_group_id is None:
        raise typer.BadParameter(
            "Either --draft-group-id (for gold dk_salaries) or --salaries-csv must be provided."
        )

    if out is None:
        base = Path("artifacts") / "lineups" / site / f"game_date={game_date}"
        if draft_group_id is not None:
            base = base / f"draft_group_id={draft_group_id}"
        base.mkdir(parents=True, exist_ok=True)
        target_out = base / "lineups_pointproj.csv"
    else:
        target_out = out
        target_out.parent.mkdir(parents=True, exist_ok=True)

    player_pool_df = build_player_pool_from_gold(
        game_date=game_date,
        site=site,
        draft_group_id=draft_group_id,
        root=None,
        salaries_csv=str(salaries_csv) if salaries_csv is not None else None,
    )
    lineups_df = build_lineups_from_player_pool(
        player_pool_df=player_pool_df,
        num_lineups=num_lineups,
        site=site,
        game_date=game_date,
    )

    lineups_df.to_csv(target_out, index=False)
    typer.echo(f"Wrote {len(lineups_df)} lineup rows to {target_out}")


if __name__ == "__main__":
    app()

# Smoke test:
# UV_PROJECT_ENVIRONMENT=.venv-user uv run python scripts/optimizer/build_lineups_from_gold.py \
#   --game-date YYYY-MM-DD --site dk --draft-group-id <ID> --num-lineups 20
