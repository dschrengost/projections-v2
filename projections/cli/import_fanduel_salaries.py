"""Import FanDuel salary CSV into gold salaries parquet partitions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from projections.dk.salaries_schema import dk_salaries_gold_path, normalize_positions

app = typer.Typer(add_completion=False, help=__doc__)


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_status(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NAN", "NONE", "NULL"}:
        return None
    return text


@app.command("run")
def run(
    csv_path: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="FanDuel salaries CSV path."),
    game_date: str = typer.Option(..., help="Slate date (YYYY-MM-DD)."),
    draft_group_id: int = typer.Option(..., help="Internal slate key used by optimizer APIs."),
    data_root: Path = typer.Option(
        Path(os.environ.get("PROJECTIONS_DATA_ROOT", "/home/daniel/projections-data")),
        help="Data root (defaults to PROJECTIONS_DATA_ROOT).",
    ),
) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise typer.BadParameter(f"CSV has no rows: {csv_path}")

    name_col = _first_present(df, ["Nickname", "display_name", "Name", "name"])
    first_col = _first_present(df, ["First Name", "first_name"])
    last_col = _first_present(df, ["Last Name", "last_name"])
    id_col = _first_present(df, ["Id", "ID", "fd_player_id", "player_id"])
    pos_col = _first_present(df, ["Position", "position", "positions"])
    salary_col = _first_present(df, ["Salary", "salary"])
    team_col = _first_present(df, ["Team", "team_abbrev", "team"])
    status_col = _first_present(df, ["Injury Indicator", "Injury Status", "status"])
    game_col = _first_present(df, ["Game", "game", "matchup"])

    missing = [label for label, col in [("id", id_col), ("position", pos_col), ("salary", salary_col), ("team", team_col)] if col is None]
    if missing:
        raise typer.BadParameter(f"CSV missing required columns: {missing}")

    work = df.copy()
    if name_col is None:
        if first_col and last_col:
            work["_display_name"] = (work[first_col].astype(str).str.strip() + " " + work[last_col].astype(str).str.strip()).str.strip()
            name_col = "_display_name"
        else:
            raise typer.BadParameter("CSV missing player name columns (Nickname/Name or First Name + Last Name).")

    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        raw_id = row.get(id_col) if id_col else None
        if pd.isna(raw_id):
            continue
        fd_player_id = str(raw_id).strip()
        if not fd_player_id or fd_player_id.upper() in {"NAN", "NONE", "NULL"}:
            continue

        raw_salary = row.get(salary_col) if salary_col else None
        salary = pd.to_numeric(raw_salary, errors="coerce")
        if pd.isna(salary):
            continue

        positions = normalize_positions(str(row.get(pos_col, "")))
        if not positions:
            continue

        team = str(row.get(team_col, "")).strip().upper()
        if not team:
            continue

        display_name = str(row.get(name_col, "")).strip()
        if not display_name:
            continue

        matchup = str(row.get(game_col, "")).strip() if game_col else ""
        status = _normalize_status(row.get(status_col) if status_col else None)

        raw_payload = {str(k): (None if pd.isna(v) else v) for k, v in row.to_dict().items()}

        rows.append(
            {
                "site": "fd",
                "game_date": game_date,
                "draft_group_id": int(draft_group_id),
                "site_player_id": fd_player_id,
                "fd_player_id": fd_player_id,
                "display_name": display_name,
                "positions": positions,
                "salary": int(salary),
                "team_abbrev": team,
                "status": status,
                "is_swappable": True,
                "is_disabled": False,
                "raw_competition_ids": [],
                "game_matchup": matchup or None,
                "raw_data": json.dumps(raw_payload, default=str),
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise typer.BadParameter("No valid rows after normalization.")

    out_path = dk_salaries_gold_path(
        root=data_root,
        site="fd",
        game_date=game_date,
        draft_group_id=draft_group_id,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    typer.echo(f"[fd-salaries] wrote {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    app()
