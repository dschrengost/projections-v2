from __future__ import annotations

"""Daily DK salaries job: resolve slates, ensure bronze draftables, write gold salaries."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import typer

from projections import paths
from projections.dk import api
from projections.dk.normalize import (
    draftables_json_to_df,
    normalize_draftables_to_salaries,
    write_salaries_gold,
)
from projections.dk.salaries_schema import dk_salaries_gold_path
from projections.dk.slates import list_draft_groups_for_date

app = typer.Typer(help="Resolve DK slates, fetch draftables, and write gold dk_salaries.")


def _resolve_game_date(value: str | None) -> date:
    if value:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid game_date; expected YYYY-MM-DD: {value}") from exc
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    return today_et


def _ensure_bronze_draftables(
    *, draft_group_id: int, data_root: Path, force_refresh: bool
) -> dict:
    bronze_path = (
        data_root / "bronze" / "dk" / "draftables" / f"draftables_raw_{draft_group_id}.json"
    )
    bronze_path.parent.mkdir(parents=True, exist_ok=True)

    if bronze_path.exists() and not force_refresh:
        with bronze_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    payload = api.fetch_draftables(draft_group_id)
    with bronze_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


@app.command()
def main(
    game_date: str | None = typer.Option(
        None,
        help="Game date as YYYY-MM-DD. Defaults to today in America/New_York.",
    ),
    site: str = typer.Option("dk", help="Site identifier (currently only dk)."),
    slate_types: List[str] = typer.Option(
        ["main"], "--slate-type", help="Slate types to process (repeatable)."
    ),
    force_refresh: bool = typer.Option(
        False,
        help="Refetch draftables from API even if bronze JSON exists.",
    ),
):
    resolved_date = _resolve_game_date(game_date)
    data_root = paths.get_data_root().resolve()

    try:
        contests_payload = api.fetch_nba_contests()
        contests = contests_payload.get("Contests", []) if isinstance(contests_payload, dict) else []
        print(f"[dk-salaries] fetched {len(contests)} contests (NBA) from lobby")
    except Exception as exc:  # pragma: no cover - network guard
        print(f"[dk-salaries] failed to fetch contests: {exc}")
        raise typer.Exit(code=1)

    failures = False
    any_written = False

    for slate_type in slate_types:
        slate_type_norm = slate_type.lower()
        print(
            f"[dk-salaries] game_date={resolved_date} site={site} slate_type={slate_type_norm}"
        )

        try:
            slates_df = list_draft_groups_for_date(
                game_date=resolved_date.isoformat(),
                slate_type=slate_type_norm,
                contests_payload=contests_payload,
            )
        except Exception as exc:
            print(
                f"[dk-salaries] failed to list draft groups for slate_type={slate_type_norm}: {exc}"
            )
            failures = True
            continue

        if slates_df.empty:
            print(
                f"[dk-salaries] no draft groups found for game_date={resolved_date} slate_type={slate_type_norm}"
            )
            failures = True
            continue

        if "n_contests" in slates_df.columns:
            chosen = slates_df.sort_values("n_contests", ascending=False).iloc[0]
        else:
            chosen = slates_df.iloc[0]
        draft_group_id = int(chosen["draft_group_id"])
        print(
            f"  draft_group_id={draft_group_id} n_contests={chosen.get('n_contests', 'n/a')}"
        )

        try:
            payload = _ensure_bronze_draftables(
                draft_group_id=draft_group_id, data_root=data_root, force_refresh=force_refresh
            )
        except Exception as exc:
            print(
                f"[dk-salaries] failed to load/fetch draftables for draft_group_id={draft_group_id}: {exc}"
            )
            failures = True
            continue

        try:
            draftables_df = draftables_json_to_df(payload, draft_group_id)
        except Exception as exc:
            print(
                f"[dk-salaries] failed to parse draftables for draft_group_id={draft_group_id}: {exc}"
            )
            failures = True
            continue

        if draftables_df.empty:
            print(
                f"[dk-salaries] no draftables rows for draft_group_id={draft_group_id}; skipping"
            )
            failures = True
            continue

        try:
            salaries_df = normalize_draftables_to_salaries(
                root=data_root,
                site=site,
                game_date=resolved_date.isoformat(),
                draft_group_id=draft_group_id,
                df=draftables_df,
            )
        except Exception as exc:
            print(
                f"[dk-salaries] failed to normalize draftables for draft_group_id={draft_group_id}: {exc}"
            )
            failures = True
            continue

        if salaries_df.empty:
            print(
                f"[dk-salaries] normalization produced zero rows for draft_group_id={draft_group_id}"
            )
            failures = True
            continue

        try:
            path = write_salaries_gold(
                root=data_root,
                site=site,
                game_date=resolved_date.isoformat(),
                draft_group_id=draft_group_id,
                salaries_df=salaries_df,
            )
        except Exception as exc:
            print(
                f"[dk-salaries] failed to write gold salaries for draft_group_id={draft_group_id}: {exc}"
            )
            failures = True
            continue

        any_written = True
        print(
            f"  draft_group_id={draft_group_id} n_raw_rows={len(draftables_df)} n_players={len(salaries_df)}"
        )
        print(f"  gold_path={path}")

    if failures or not any_written:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
