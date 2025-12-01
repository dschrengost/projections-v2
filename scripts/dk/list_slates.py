from __future__ import annotations

"""List NBA DraftKings draft groups (slates) for a given date."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import typer

from projections import paths
from projections.dk import api
from projections.dk.slates import SlateType, list_draft_groups_for_date

app = typer.Typer(help="List NBA DraftKings draft groups for a date and slate type.")


def _resolve_data_root(value: Path | None) -> Path:
    return (value or paths.get_data_root()).resolve()


def _format_table(df: pd.DataFrame) -> str:
    display = df.copy()
    display["game_date"] = display["game_date"].astype(str)
    for col in ("earliest_start", "latest_start"):
        if col in display.columns:
            display[col] = display[col].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    return display.to_string(index=False)


@app.command()
def main(
    game_date: str = typer.Option(..., "--game-date", help="Target date (YYYY-MM-DD)."),
    slate_type: SlateType = typer.Option(
        "all",
        "--slate-type",
        help="Slate type filter (e.g. main, night, turbo, early, showdown, all).",
    ),
    data_root: Path | None = typer.Option(
        None,
        "--data-root",
        help="Override PROJECTIONS_DATA_ROOT (defaults to env or ./data).",
    ),
    out_json: Path | None = typer.Option(
        None,
        "--out-json",
        help=(
            "Optional path for raw contests payload. Defaults to "
            "<data_root>/bronze/dk/contests/contests_raw_<game_date>.json"
        ),
    ),
):
    try:
        parsed_date = date.fromisoformat(game_date)
    except ValueError:
        typer.secho(
            f"[list_slates] invalid game-date supplied: {game_date}", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    resolved_root = _resolve_data_root(data_root)

    try:
        contests_payload = api.fetch_nba_contests()
    except Exception as exc:  # pragma: no cover - network guard
        typer.secho(f"[list_slates] failed to fetch contests: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    contests = contests_payload.get("Contests", []) if isinstance(contests_payload, dict) else []
    typer.echo(f"[dk] fetched {len(contests)} contests (NBA) from lobby")

    default_out = (
        resolved_root
        / "bronze"
        / "dk"
        / "contests"
        / f"contests_raw_{parsed_date.isoformat()}.json"
    )
    output_path = (out_json or default_out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(contests_payload, f, indent=2)
    typer.echo(f"[dk] wrote contests payload -> {output_path}")

    try:
        df = list_draft_groups_for_date(
            game_date=parsed_date.isoformat(),
            slate_type=slate_type,
            contests_payload=contests_payload,
        )
    except Exception as exc:
        typer.secho(f"[list_slates] failed to parse contests: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(
        f"[dk] found {len(df)} draft groups for date={parsed_date.isoformat()} slate_type={slate_type}"
    )

    if df.empty:
        typer.echo("[list_slates] no NBA contests found for that date/slate_type")
        raise typer.Exit(code=0)

    typer.echo(_format_table(df))


if __name__ == "__main__":
    app()
