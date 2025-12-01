from __future__ import annotations

"""Fetch DraftKings draftables for a draft group."""

import json
from datetime import date
from pathlib import Path
from typing import Dict

import typer

from projections import paths
from projections.dk import api
from projections.dk.normalize import draftables_json_to_df
from projections.dk.slates import SlateType, list_draft_groups_for_date

app = typer.Typer(help="Fetch NBA draftables for a DraftKings draft group.")


def _resolve_data_root(value: Path | None) -> Path:
    return (value or paths.get_data_root()).resolve()


def _parse_game_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid game-date (expected YYYY-MM-DD): {value}") from exc


def _resolve_draft_group_id(
    *,
    explicit_id: int | None,
    game_date: date | None,
    slate_type: SlateType,
) -> tuple[int, pd.DataFrame | None]:
    if explicit_id is not None:
        return explicit_id, None

    if game_date is None:
        raise typer.BadParameter("--game-date is required when --draft-group-id is not provided")

    contests_payload = api.fetch_nba_contests()
    contests = contests_payload.get("Contests", []) if isinstance(contests_payload, dict) else []
    typer.echo(f"[dk] fetched {len(contests)} contests (NBA) from lobby")

    slates_df = list_draft_groups_for_date(
        game_date=game_date.isoformat(),
        slate_type=slate_type,
        contests_payload=contests_payload,
    )
    typer.echo(
        f"[dk] found {len(slates_df)} draft groups for date={game_date} slate_type={slate_type}"
    )
    if slates_df.empty:
        typer.secho(
            "[dk] no matching draft groups for that date/slate_type; cannot fetch draftables",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    chosen = slates_df
    if len(slates_df) > 1:
        typer.secho(
            "[dk] multiple draft groups matched; selecting the one with the most contests",
            fg=typer.colors.YELLOW,
        )
        chosen = slates_df.sort_values("n_contests", ascending=False).head(1)

    row = chosen.iloc[0]
    return int(row["draft_group_id"]), slates_df


@app.command()
def main(
    draft_group_id: int | None = typer.Option(
        None, "--draft-group-id", help="Draft Group ID (if known)."
    ),
    game_date: str | None = typer.Option(
        None, "--game-date", help="Game date (YYYY-MM-DD) to resolve draft group."
    ),
    slate_type: SlateType = typer.Option(
        "main",
        "--slate-type",
        help="Slate type filter when resolving a date (main, night, turbo, early, showdown, all).",
    ),
    out_json: Path | None = typer.Option(
        None,
        "--out-json",
        help=(
            "Output path for raw draftables JSON. Defaults to "
            "<data_root>/bronze/dk/draftables/draftables_raw_<draft_group_id>.json"
        ),
    ),
    out_csv: Path | None = typer.Option(
        None,
        "--out-csv",
        help="Optional CSV output path for flattened draftables.",
    ),
    data_root: Path | None = typer.Option(
        None,
        "--data-root",
        help="Override PROJECTIONS_DATA_ROOT (defaults to env or ./data).",
    ),
):
    parsed_date = _parse_game_date(game_date)
    resolved_root = _resolve_data_root(data_root)

    try:
        resolved_id, slates_df = _resolve_draft_group_id(
            explicit_id=draft_group_id, game_date=parsed_date, slate_type=slate_type
        )
    except typer.BadParameter:
        raise
    except typer.Exit:
        raise
    except Exception as exc:
        typer.secho(f"[dk] failed to resolve draft group id: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"[dk] fetching draftables for draft_group_id={resolved_id}")
    payload = api.fetch_draftables(resolved_id)
    draftables_df = draftables_json_to_df(payload, resolved_id)
    typer.echo(f"[dk] got {len(draftables_df)} draftables")

    default_json = (
        resolved_root
        / "bronze"
        / "dk"
        / "draftables"
        / f"draftables_raw_{resolved_id}.json"
    )
    json_path = (out_json or default_json).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    typer.echo(f"[dk] wrote raw draftables -> {json_path}")

    if out_csv:
        out_csv = out_csv.resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        draftables_df.to_csv(out_csv, index=False)
        typer.echo(f"[dk] wrote CSV draftables -> {out_csv}")


if __name__ == "__main__":
    app()
