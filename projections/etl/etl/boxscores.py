"""Freeze NBA.com boxscore labels into the immutable labels tier."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import typer

from projections import paths
from projections.minutes_v1.labels import freeze_boxscore_labels
from projections.minutes_v1.schemas import BOX_SCORE_LABELS_SCHEMA, enforce_schema, validate_with_pandera
from projections.minutes_v1.season_dataset import _parse_minutes_iso
from scrapers.nba_boxscore import (
    NbaComBoxScoreScraper,
    NbaComGameBoxScore,
    NbaComPlayerStatLine,
    NbaComTeamBoxScore,
)

app = typer.Typer(help=__doc__)


def _iter_dates(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    current = start
    while current <= end:
        yield current
        current += pd.Timedelta(days=1)


def _team_players(team: NbaComTeamBoxScore | None) -> List[NbaComPlayerStatLine]:
    if team is None:
        return []
    return team.players or []


def _games_to_labels(games: List[NbaComGameBoxScore], *, season_label: str) -> pd.DataFrame:
    records: list[dict] = []
    for game in games:
        if not game.game_id:
            continue
        try:
            game_id = int(game.game_id)
        except (TypeError, ValueError):
            continue
        tip_ts = pd.Timestamp(game.game_time_utc) if game.game_time_utc else pd.NaT
        if pd.isna(tip_ts) and game.game_time_local:
            tip_ts = pd.Timestamp(game.game_time_local)
        if pd.isna(tip_ts):
            continue
        tip_ts = tip_ts.tz_localize("UTC") if tip_ts.tzinfo is None else tip_ts.tz_convert("UTC")
        game_date = tip_ts.tz_convert(None).normalize()
        for team in (game.home, game.away):
            if team is None or team.team_id is None:
                continue
            team_id = int(team.team_id)
            for player in _team_players(team):
                try:
                    player_id = int(player.person_id)
                except (TypeError, ValueError):
                    continue
                minutes_raw = (player.statistics or {}).get("minutes")
                minutes = _parse_minutes_iso(minutes_raw)
                records.append(
                    {
                        "game_id": game_id,
                        "player_id": player_id,
                        "player_name": player.name,
                        "team_id": team_id,
                        "season": season_label,
                        "game_date": game_date,
                        "minutes": minutes,
                        "starter_flag": bool(player.starter),
                        "listed_pos": player.position,
                        "source": "nba.com/boxscore",
                    }
                )
    columns = list(BOX_SCORE_LABELS_SCHEMA.columns)
    return pd.DataFrame(records, columns=columns)


@app.command()
def main(
    start: datetime = typer.Option(..., help="Inclusive start date (YYYY-MM-DD)."),
    end: datetime = typer.Option(..., help="Inclusive end date (YYYY-MM-DD)."),
    season: int = typer.Option(..., help="Season label used for output partitions (e.g., 2025)."),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        help="Base data directory containing labels/ (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
    timeout: float = typer.Option(10.0, "--timeout", help="HTTP timeout for NBA.com requests."),
) -> None:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    if end_day < start_day:
        raise typer.BadParameter("--end must be on/after --start.")

    scraper = NbaComBoxScoreScraper(timeout=timeout)
    games: list[NbaComGameBoxScore] = []
    for cursor in _iter_dates(start_day, end_day):
        games.extend(scraper.fetch_daily_box_scores(cursor.date()))
    if not games:
        raise RuntimeError("NBA.com box score scraper returned zero games for requested window.")

    season_label = str(season)
    labels_df = _games_to_labels(games, season_label=season_label)
    labels_df = labels_df[
        (labels_df["game_date"] >= start_day) & (labels_df["game_date"] <= end_day)
    ].copy()
    if labels_df.empty:
        raise RuntimeError("No label rows matched the requested date window.")

    labels_df = enforce_schema(labels_df, BOX_SCORE_LABELS_SCHEMA)
    validate_with_pandera(labels_df, BOX_SCORE_LABELS_SCHEMA)

    labels_root = data_root / "labels"
    season_dir = labels_root / f"season={season_label}"
    existing_path = season_dir / "boxscore_labels.parquet"
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        labels_df = pd.concat([existing, labels_df], ignore_index=True, sort=False)
        labels_df.sort_values(["game_date", "game_id", "team_id", "player_id"], inplace=True)
        labels_df = labels_df.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")

    written = freeze_boxscore_labels(labels_df, labels_root, overwrite=True)
    target = written.get(season_label)
    if not target:
        raise RuntimeError("freeze_boxscore_labels did not yield an output path.")
    typer.echo(f"[boxscores] wrote {len(labels_df):,} label rows -> {target}")


if __name__ == "__main__":  # pragma: no cover
    app()
