#!/usr/bin/env python3
"""Backfill boxscore bronze and labels from archived JSON dumps."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import typer

from projections import paths

app = typer.Typer(help=__doc__)


def _load_games(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _tip_timestamp(game: Dict) -> pd.Timestamp:
    tip = game.get("game_time_utc") or game.get("game_time_local")
    ts = pd.Timestamp(tip)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _game_date(game: Dict) -> pd.Timestamp:
    ts = _tip_timestamp(game)
    return ts.tz_convert("America/New_York").tz_localize(None).normalize()


def _labels_from_games(games: List[Dict], start: pd.Timestamp, end: pd.Timestamp, season_label: str) -> pd.DataFrame:
    records: list[dict] = []
    for game in games:
        game_date = _game_date(game)
        if not (start <= game_date <= end):
            continue
        game_id = game.get("game_id")
        for side in ("home", "away"):
            team = game.get(side) or {}
            team_id = team.get("team_id")
            for player in team.get("players", []):
                stats = player.get("statistics", {}) or {}
                minutes_raw = stats.get("minutes")
                records.append(
                    {
                        "game_id": int(game_id),
                        "player_id": int(player.get("person_id", 0) or 0),
                        "player_name": player.get("name"),
                        "team_id": int(team_id),
                        "season": season_label,
                        "game_date": game_date,
                        "minutes": stats.get("minutes", "0"),
                        "starter_flag": int(bool(player.get("starter"))),
                        "listed_pos": player.get("position"),
                        "source": "nba.com/boxscore",
                    }
                )
    return pd.DataFrame(records)


def _write_bronze(games: List[Dict], data_root: Path, season: int) -> None:
    base = data_root / "bronze" / "boxscores_raw" / f"season={season}"
    for game in games:
        day_dir = base / f"date={_game_date(game).date().isoformat()}"
        day_dir.mkdir(parents=True, exist_ok=True)
        payload = pd.DataFrame([
            {
                "game_id": game.get("game_id"),
                "payload": json.dumps(game),
                "tip_ts": _tip_timestamp(game),
            }
        ])
        payload.to_parquet(day_dir / "boxscores_raw.parquet", index=False)


def _write_labels(df: pd.DataFrame, data_root: Path, season: int) -> None:
    labels_dir = data_root / "labels" / f"season={season}"
    labels_dir.mkdir(parents=True, exist_ok=True)
    target = labels_dir / "boxscore_labels.parquet"
    if df.empty:
        raise typer.BadParameter("No label rows to write.")
    df.to_parquet(target, index=False)
    typer.echo(f"[boxscores-json] wrote {len(df)} label rows -> {target}")


@app.command()
def main(
    season: int = typer.Option(..., help="Season label (e.g., 2022)."),
    start: datetime = typer.Option(..., help="Start date inclusive."),
    end: datetime = typer.Option(..., help="End date inclusive."),
    boxscores_json: Path = typer.Option(..., exists=True, help="Path to archived boxscores JSON."),
    data_root: Path = typer.Option(paths.get_data_root(), help="Data root for outputs."),
) -> None:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    games = _load_games(boxscores_json)
    selected = [game for game in games if start_day <= _game_date(game) <= end_day]
    if not selected:
        raise typer.BadParameter("No games found in the requested window.")
    _write_bronze(selected, data_root, season)
    labels_df = _labels_from_games(selected, start_day, end_day, str(season))
    _write_labels(labels_df, data_root, season)


if __name__ == "__main__":
    app()
