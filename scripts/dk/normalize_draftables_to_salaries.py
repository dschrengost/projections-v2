from __future__ import annotations

"""Normalize DraftKings draftables bronze payloads into gold dk_salaries."""

import json
from datetime import date
from pathlib import Path
from typing import Dict

import pandas as pd
import typer

from projections import paths
from projections.dk import api
from projections.dk.normalize import (
    draftables_json_to_df,
    normalize_draftables_to_salaries,
    write_salaries_gold,
)
from projections.dk.slates import SlateType, list_draft_groups_for_date

app = typer.Typer(help="Normalize DK draftables into gold dk_salaries.")


def _resolve_data_root(root: Path | None) -> Path:
    return (root or paths.get_data_root()).resolve()


def _parse_game_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid game-date (expected YYYY-MM-DD): {value}") from exc


def _load_draftables_df(
    *,
    draft_group_id: int | str,
    draftables_csv: Path | None,
    draftables_json: Path | None,
    data_root: Path,
    fetch_if_missing: bool,
) -> pd.DataFrame:
    if draftables_csv:
        return pd.read_csv(draftables_csv)

    default_json = data_root / "bronze" / "dk" / "draftables" / f"draftables_raw_{draft_group_id}.json"
    json_path = (draftables_json or default_json).resolve()

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return draftables_json_to_df(payload, draft_group_id)

    if not fetch_if_missing:
        raise FileNotFoundError(f"Draftables JSON not found: {json_path}")

    print(f"[dk-salaries] draftables JSON missing; fetching live for draft_group_id={draft_group_id}")
    payload = api.fetch_draftables(draft_group_id)
    return draftables_json_to_df(payload, draft_group_id)


def _pick_draft_group(
    *, game_date: date, slate_type: SlateType, contests_payload: Dict[str, Any]
) -> tuple[int, pd.DataFrame]:
    slates_df = list_draft_groups_for_date(
        game_date=game_date.isoformat(), slate_type=slate_type, contests_payload=contests_payload
    )
    if slates_df.empty:
        raise RuntimeError("No matching draft groups for that date/slate_type")
    chosen = slates_df
    if len(slates_df) > 1:
        print(
            "[dk-salaries] multiple draft groups matched; selecting the one with the most contests"
        )
        chosen = slates_df.sort_values("n_contests", ascending=False).head(1)
    draft_group_id = int(chosen.iloc[0]["draft_group_id"])
    return draft_group_id, slates_df


@app.command()
def main(
    game_date: str = typer.Option(..., "--game-date", help="Target game date (YYYY-MM-DD)."),
    site: str = typer.Option("dk", "--site", help="Site identifier (default: dk)."),
    draft_group_id: int | None = typer.Option(
        None, "--draft-group-id", help="Draft Group ID (if known)."
    ),
    slate_type: SlateType = typer.Option(
        "main", "--slate-type", help="Slate type used to resolve draft group when id not provided."
    ),
    draftables_csv: Path | None = typer.Option(
        None, "--draftables-csv", help="Optional CSV source of draftables (pre-flattened)."
    ),
    draftables_json: Path | None = typer.Option(
        None, "--draftables-json", help="Optional raw draftables JSON payload to use instead of default bronze path."
    ),
    data_root: Path | None = typer.Option(
        None, "--data-root", help="Override PROJECTIONS_DATA_ROOT (defaults to env or ./data)."
    ),
    fetch_live_if_missing: bool = typer.Option(
        True,
        "--fetch-live-if-missing/--no-fetch-live",
        help="Fetch draftables from DK API when bronze JSON is missing.",
    ),
):
    parsed_date = _parse_game_date(game_date)
    resolved_root = _resolve_data_root(data_root)

    resolved_dg = draft_group_id
    contests_payload: Dict[str, Any] | None = None
    if resolved_dg is None:
        contests_payload = api.fetch_nba_contests()
        contests = contests_payload.get("Contests", []) if isinstance(contests_payload, dict) else []
        print(f"[dk-salaries] fetched {len(contests)} contests (NBA) from lobby")
        resolved_dg, _ = _pick_draft_group(
            game_date=parsed_date, slate_type=slate_type, contests_payload=contests_payload
        )

    if resolved_dg is None:
        raise typer.BadParameter("Unable to resolve draft_group_id; provide --draft-group-id or slate details")

    draftables_df = _load_draftables_df(
        draft_group_id=resolved_dg,
        draftables_csv=draftables_csv,
        draftables_json=draftables_json,
        data_root=resolved_root,
        fetch_if_missing=fetch_live_if_missing,
    )

    print(
        f"[dk-salaries] loaded draftables rows={len(draftables_df)} for draft_group_id={resolved_dg}"
    )

    salaries_df = normalize_draftables_to_salaries(
        root=resolved_root,
        site=site,
        game_date=parsed_date.isoformat(),
        draft_group_id=resolved_dg,
        df=draftables_df,
    )

    path = write_salaries_gold(
        root=resolved_root,
        site=site,
        game_date=parsed_date.isoformat(),
        draft_group_id=resolved_dg,
        salaries_df=salaries_df,
    )

    print(
        f"[dk-salaries] game_date={parsed_date} site={site} draft_group_id={resolved_dg} "
        f"rows={len(salaries_df)}"
    )
    print(f"[dk-salaries] wrote {path}")


if __name__ == "__main__":
    app()
