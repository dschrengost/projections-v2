"""Daily FD salaries job: resolve slates, fetch players payloads, write gold salaries."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import typer

from projections import paths
from projections.dk.normalize import write_salaries_gold
from projections.fd import api
from projections.fd.normalize import normalize_fd_players_to_salaries, players_json_to_df
from projections.fd.slates import list_fixture_lists_for_date

app = typer.Typer(help="Resolve FanDuel slates, fetch players payloads, and write gold salaries.")


def _resolve_game_date(value: str | None) -> date:
    if value:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid game_date; expected YYYY-MM-DD: {value}") from exc
    return datetime.now(ZoneInfo("America/New_York")).date()


def _bronze_slate_dir(*, data_root: Path, game_date: str, draft_group_id: int | str) -> Path:
    return (
        data_root
        / "bronze"
        / "fd"
        / "fixture_lists"
        / f"game_date={game_date}"
        / f"draft_group_id={draft_group_id}"
    )


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _ensure_bronze_payloads(
    *,
    game_date: str,
    draft_group_id: int | str,
    data_root: Path,
    client_id: str,
    force_refresh: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    slate_dir = _bronze_slate_dir(data_root=data_root, game_date=game_date, draft_group_id=draft_group_id)
    slate_dir.mkdir(parents=True, exist_ok=True)

    detail_path = slate_dir / "detail.json"
    players_path = slate_dir / "players.json"
    contests_path = slate_dir / "contests.json"

    detail_payload = None if force_refresh else _load_json_if_exists(detail_path)
    players_payload = None if force_refresh else _load_json_if_exists(players_path)
    contests_payload = None if force_refresh else _load_json_if_exists(contests_path)

    if detail_payload is None:
        detail_payload = api.fetch_fixture_list_detail(draft_group_id, client_id=client_id)
        detail_path.write_text(json.dumps(detail_payload, indent=2), encoding="utf-8")

    if players_payload is None:
        players_payload = api.fetch_fixture_list_players(draft_group_id, client_id=client_id)
        players_path.write_text(json.dumps(players_payload, indent=2), encoding="utf-8")

    if contests_payload is None:
        contests_payload = api.fetch_contests(fixture_list_id=draft_group_id, client_id=client_id)
        contests_path.write_text(json.dumps(contests_payload, indent=2), encoding="utf-8")

    return detail_payload, players_payload, contests_payload


@app.command()
def main(
    game_date: str | None = typer.Option(
        None,
        help="Game date as YYYY-MM-DD. Defaults to today in America/New_York.",
    ),
    site: str = typer.Option("fd", help="Site identifier (currently only fd)."),
    slate_types: list[str] = typer.Option(
        ["all"],
        "--slate-type",
        help="Slate types to process (repeatable). Use 'all' for all slates.",
    ),
    force_refresh: bool = typer.Option(
        False,
        help="Refetch fixture payloads from API even if bronze JSON exists.",
    ),
    single_slate: bool = typer.Option(
        False,
        "--single-slate",
        help="Only process one slate per type (highest contest count). Default: process ALL slates.",
    ),
    client_id: str | None = typer.Option(
        None,
        "--client-id",
        help="FanDuel API client id (optional; defaults to env discovery).",
    ),
) -> None:
    if site.lower() != "fd":
        raise typer.BadParameter(f"Unsupported site={site!r}; expected 'fd'")

    resolved_date = _resolve_game_date(game_date)
    game_date_str = resolved_date.isoformat()
    data_root = paths.get_data_root().resolve()

    try:
        resolved_client_id = api.resolve_client_id(client_id)
    except Exception as exc:
        print(f"[fd-salaries] failed to resolve client id: {exc}")
        raise typer.Exit(code=1) from exc

    try:
        fixture_lists_payload = api.fetch_fixture_lists(client_id=resolved_client_id)
        all_fixture_lists = fixture_lists_payload.get("fixture_lists", []) if isinstance(fixture_lists_payload, dict) else []
        nba_count = sum(1 for row in all_fixture_lists if str((row or {}).get("sport", "")).lower() == "nba")
        print(f"[fd-salaries] fetched fixture lists (nba={nba_count})")
    except Exception as exc:
        print(f"[fd-salaries] failed to fetch fixture lists: {exc}")
        raise typer.Exit(code=1) from exc

    failures = False
    any_written = False
    processed_draft_groups: set[str] = set()

    for slate_type in slate_types:
        slate_type_norm = str(slate_type).strip().lower()
        print(f"[fd-salaries] game_date={game_date_str} site=fd slate_type={slate_type_norm}")

        try:
            slates_df = list_fixture_lists_for_date(
                game_date=game_date_str,
                slate_type=slate_type_norm,  # type: ignore[arg-type]
                fixture_lists_payload=fixture_lists_payload,
                client_id=resolved_client_id,
            )
        except Exception as exc:
            print(f"[fd-salaries] failed to list fixture lists for slate_type={slate_type_norm}: {exc}")
            failures = True
            continue

        if slates_df.empty:
            print(f"[fd-salaries] no fixture lists found for game_date={game_date_str} slate_type={slate_type_norm}")
            failures = True
            continue

        slates_to_process = [slates_df.iloc[i] for i in range(len(slates_df))]
        if single_slate:
            if "n_contests" in slates_df.columns:
                slates_to_process = [slates_df.sort_values("n_contests", ascending=False).iloc[0]]
            else:
                slates_to_process = [slates_df.iloc[0]]

        for chosen in slates_to_process:
            draft_group_id = chosen.get("draft_group_id")
            if draft_group_id is None:
                continue
            draft_group_key = str(draft_group_id)
            if draft_group_key in processed_draft_groups:
                continue
            processed_draft_groups.add(draft_group_key)

            print(
                "  draft_group_id=%s n_contests=%s slate_type=%s"
                % (
                    draft_group_key,
                    chosen.get("n_contests", "n/a"),
                    chosen.get("slate_type", slate_type_norm),
                )
            )

            try:
                detail_payload, players_payload, contests_payload = _ensure_bronze_payloads(
                    game_date=game_date_str,
                    draft_group_id=draft_group_id,
                    data_root=data_root,
                    client_id=resolved_client_id,
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                print(f"[fd-salaries] failed to fetch/load bronze payloads for draft_group_id={draft_group_key}: {exc}")
                failures = True
                continue

            try:
                players_df = players_json_to_df(
                    players_payload,
                    fixture_list_id=draft_group_id,
                    fixture_detail=detail_payload,
                    contests_payload=contests_payload,
                )
            except Exception as exc:
                print(f"[fd-salaries] failed to parse players for draft_group_id={draft_group_key}: {exc}")
                failures = True
                continue

            if players_df.empty:
                print(f"[fd-salaries] no player rows for draft_group_id={draft_group_key}")
                failures = True
                continue

            try:
                salaries_df = normalize_fd_players_to_salaries(
                    root=data_root,
                    site="fd",
                    game_date=game_date_str,
                    draft_group_id=draft_group_id,
                    df=players_df,
                )
            except Exception as exc:
                print(f"[fd-salaries] failed to normalize players for draft_group_id={draft_group_key}: {exc}")
                failures = True
                continue

            if salaries_df.empty:
                print(f"[fd-salaries] normalization produced zero rows for draft_group_id={draft_group_key}")
                failures = True
                continue

            try:
                path = write_salaries_gold(
                    root=data_root,
                    site="fd",
                    game_date=game_date_str,
                    draft_group_id=draft_group_id,
                    salaries_df=salaries_df,
                )
            except Exception as exc:
                print(f"[fd-salaries] failed to write gold salaries for draft_group_id={draft_group_key}: {exc}")
                failures = True
                continue

            any_written = True
            print(f"    n_raw_rows={len(players_df)} n_players={len(salaries_df)} gold_path={path}")

    print(f"[fd-salaries] processed {len(processed_draft_groups)} fixture lists")
    if failures or not any_written:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
