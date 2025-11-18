"""Freeze NBA.com boxscore labels into the immutable labels tier."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import typer

from projections import paths
from projections.etl.common import load_schedule_data
from projections.minutes_v1.labels import freeze_boxscore_labels
from projections.minutes_v1.schemas import BOX_SCORE_LABELS_SCHEMA, enforce_schema, validate_with_pandera
from projections.minutes_v1.smoke_dataset import _parse_minutes_iso
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
    schedule: List[str] = typer.Option(
        [],
        "--schedule",
        help="Optional schedule parquet glob(s). Defaults to the silver schedule for the requested season.",
    ),
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

    scraper = NbaComBoxScoreScraper(timeout=timeout, request_delay=0.0)
    default_schedule = schedule or [
        str(
            data_root
            / "silver"
            / "schedule"
            / f"season={season}"
            / "month=*"
            / "schedule.parquet"
        )
    ]
    schedule_df = load_schedule_data(default_schedule, start_day, end_day, timeout)
    schedule_df["game_date"] = pd.to_datetime(schedule_df["game_date"]).dt.normalize()

    games: list[NbaComGameBoxScore] = []
    skipped_days: list[pd.Timestamp] = []
    season_dir = data_root / "bronze" / "boxscores_raw" / f"season={season}"
    season_dir.mkdir(parents=True, exist_ok=True)
    for cursor in _iter_dates(start_day, end_day):
        typer.echo(f"[boxscores] fetching {cursor.date()}...")
        day_games = schedule_df.loc[schedule_df["game_date"] == cursor]
        if day_games.empty:
            skipped_days.append(cursor)
            continue
        for gid in day_games["game_id"].dropna().unique():
            typer.echo(f"[boxscores] fetching boxscore for game_id={gid}")
            box_score = None
            for attempt in range(3):
                try:
                    box_score = scraper.fetch_box_score(str(int(gid)))
                    break
                except RuntimeError as exc:
                    if attempt == 2:
                        typer.echo(f"[boxscores] warning: {exc}; skipping {gid}")
                    else:
                        time.sleep(0.25)
                        continue
            if box_score:
                games.append(box_score)
    if not games:
        typer.echo(
            "[boxscores] warning: no boxscores were fetched for this window; skipping write."
        )
        return

    season_label = str(season)
    labels_df = _games_to_labels(games, season_label=season_label)
    labels_df = labels_df[
        (labels_df["game_date"] >= start_day) & (labels_df["game_date"] <= end_day)
    ].copy()
    if labels_df.empty:
        raise RuntimeError("No label rows matched the requested date window.")

    bronze_df = pd.DataFrame([game.__dict__ for game in games])
    bronze_df.to_parquet(
        season_dir / f"date={start_day.date()}" / "boxscores_raw.parquet",
        index=False,
    )

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
