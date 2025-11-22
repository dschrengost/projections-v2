"""Build minutes_v1 labels from bronze box score payloads."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import typer

from projections import paths
from projections.etl import storage
from projections.labels import derive_starter_flag_labels
from projections.minutes_v1.labels import REQUIRED_LABEL_COLUMNS
from projections.minutes_v1.schemas import BOX_SCORE_LABELS_SCHEMA, enforce_schema, validate_with_pandera
from projections.minutes_v1.smoke_dataset import _parse_minutes_iso

SOURCE = "nba.com/boxscore"
LABEL_FILENAME = "labels.parquet"
UTC = timezone.utc

app = typer.Typer(help=__doc__)


def _season_start_from_day(day: pd.Timestamp) -> int:
    """Return the season start year for a given day."""

    return day.year if day.month >= 8 else day.year - 1


def _iter_days(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    current = start.normalize()
    end_norm = end.normalize()
    while current <= end_norm:
        yield current
        current += pd.Timedelta(days=1)


def _coerce_timestamp(value: object) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _normalize_team_payload(raw: object) -> dict:
    if raw is None or pd.isna(raw):
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}
    if is_dataclass(raw):
        return asdict(raw)
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "__dict__"):
        try:
            return dict(raw.__dict__)
        except Exception:
            return {}
    return {}


def _players_from_team(team_payload: dict) -> list[dict]:
    players = team_payload.get("players") or []
    normalized: list[dict] = []
    for player in players:
        if is_dataclass(player):
            normalized.append(asdict(player))
        elif isinstance(player, dict):
            normalized.append(player)
        else:
            try:
                normalized.append(dict(player.__dict__))
            except Exception:
                continue
    return normalized


def _tip_timestamp(game_payload: dict) -> pd.Timestamp:
    for key in (
        "game_time_local",
        "gameTimeLocal",
        "game_time_home",
        "gameTimeHome",
        "game_time_away",
        "gameTimeAway",
        "game_time_utc",
        "gameTimeUTC",
    ):
        if key in game_payload:
            ts = _coerce_timestamp(game_payload[key])
            if not pd.isna(ts):
                return ts
    return pd.NaT


def _unwrap_game(raw: object) -> dict | None:
    """Return the inner game payload regardless of wrapper type."""

    payload: object = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    return payload.get("game") or payload


def _boxscore_rows(game_payload: dict, *, season_label: str) -> list[dict]:
    """Convert a single game payload into boxscore label rows."""

    rows: list[dict] = []
    game_id_value = game_payload.get("game_id") or game_payload.get("gameId")
    try:
        game_id = int(game_id_value)
    except (TypeError, ValueError):
        return rows

    tip_ts = _tip_timestamp(game_payload)
    if pd.isna(tip_ts):
        return rows
    game_date = tip_ts.tz_localize(None).normalize() if tip_ts.tzinfo is not None else tip_ts.normalize()

    team_candidates = [
        game_payload.get("homeTeam") or game_payload.get("home"),
        game_payload.get("awayTeam") or game_payload.get("away"),
    ]
    for raw_team in team_candidates:
        team_payload = _normalize_team_payload(raw_team)
        if not team_payload:
            continue
        team_id_value = team_payload.get("team_id") or team_payload.get("teamId")
        try:
            team_id = int(team_id_value)
        except (TypeError, ValueError):
            continue
        for player in _players_from_team(team_payload):
            player_id_value = player.get("personId") or player.get("person_id") or player.get("player_id")
            try:
                player_id = int(player_id_value)
            except (TypeError, ValueError):
                continue
            stats = player.get("statistics") or {}
            minutes_raw = stats.get("minutes")
            rows.append(
                {
                    "game_id": game_id,
                    "player_id": player_id,
                    "player_name": player.get("name") or player.get("player_name"),
                    "team_id": team_id,
                    "season": season_label,
                    "game_date": game_date,
                    "minutes": _parse_minutes_iso(minutes_raw) if isinstance(minutes_raw, str) else minutes_raw,
                    "starter_flag": bool(player.get("starter") or player.get("starter_flag")),
                    "listed_pos": player.get("position"),
                    "source": SOURCE,
                    "label_frozen_ts": pd.Timestamp(datetime.now(tz=UTC)),
                }
            )
    return rows


def _prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Derive starter labels, coerce types, and validate the schema."""

    missing = REQUIRED_LABEL_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Label dataframe missing required columns: {', '.join(sorted(missing))}")

    working = derive_starter_flag_labels(df)
    numeric_columns = ("game_id", "player_id", "team_id", "starter_flag", "starter_flag_label")
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    if "label_frozen_ts" in working:
        working["label_frozen_ts"] = pd.to_datetime(
            working["label_frozen_ts"], utc=True, errors="coerce"
        ).fillna(pd.Timestamp(datetime.now(tz=UTC)))
    prepared = enforce_schema(working, BOX_SCORE_LABELS_SCHEMA)
    validate_with_pandera(prepared, BOX_SCORE_LABELS_SCHEMA)
    prepared.sort_values(list(BOX_SCORE_LABELS_SCHEMA.primary_key) + ["label_frozen_ts"], inplace=True)
    prepared = prepared.drop_duplicates(subset=list(BOX_SCORE_LABELS_SCHEMA.primary_key), keep="last")
    return prepared


def _load_raw_partitions(
    data_root: Path, *, start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, Path]]:
    """Return available raw parquet partitions for the requested window."""

    partitions: list[tuple[pd.Timestamp, Path]] = []
    for day in _iter_days(start, end):
        season_value = _season_start_from_day(day)
        candidate = storage.bronze_partition_path(
            "boxscores_raw",
            data_root=data_root,
            season=season_value,
            target_date=day.date(),
        )
        partitions.append((day, candidate))
    return partitions


@app.command()
def main(
    start_date: datetime = typer.Option(
        ...,
        "--start-date",
        "--start",
        help="Start date (inclusive) in YYYY-MM-DD.",
    ),
    end_date: datetime = typer.Option(
        ...,
        "--end-date",
        "--end",
        help="End date (inclusive) in YYYY-MM-DD.",
    ),
    data_root: Path = typer.Option(
        paths.get_data_root(),
        "--data-root",
        help="Root directory containing bronze and gold partitions (defaults to PROJECTIONS_DATA_ROOT or ./data).",
    ),
) -> None:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise typer.BadParameter("--end-date must be on/after --start-date.")

    partitions = _load_raw_partitions(data_root, start=start, end=end)
    missing = [day.date().isoformat() for day, path in partitions if not path.exists()]
    available = [(day, path) for day, path in partitions if path.exists()]
    if missing:
        typer.echo(
            f"[labels] warning: missing boxscore raw partitions for: {', '.join(missing)} (data_root={data_root})",
            err=True,
        )
    if not available:
        raise typer.BadParameter(f"No boxscore raw partitions found between {start.date()} and {end.date()}.")

    written_rows = 0
    skipped_dates: list[str] = []
    for day, raw_path in available:
        season_label = str(_season_start_from_day(day))
        raw_df = pd.read_parquet(raw_path)
        if "payload" in raw_df.columns:
            raw_objects = raw_df["payload"].tolist()
        else:
            raw_objects = raw_df.to_dict(orient="records")
        games = [_unwrap_game(obj) for obj in raw_objects]
        valid_games = [game for game in games if game]
        if not valid_games:
            skipped_dates.append(day.date().isoformat())
            continue
        records: list[dict] = []
        for payload in valid_games:
            records.extend(_boxscore_rows(payload, season_label=season_label))
        if not records:
            skipped_dates.append(day.date().isoformat())
            continue
        labels = pd.DataFrame.from_records(records)
        labels = _prepare_labels(labels)
        labels = labels.loc[labels["game_date"] == day.normalize()].copy()
        if labels.empty:
            skipped_dates.append(day.date().isoformat())
            continue
        out_dir = (
            data_root
            / "gold"
            / "labels_minutes_v1"
            / f"season={season_label}"
            / f"game_date={day.date().isoformat()}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / LABEL_FILENAME
        labels.to_parquet(out_path, index=False)
        written_rows += len(labels)
        typer.echo(f"[labels] {day.date()}: wrote {len(labels)} rows -> {out_path}")

    if written_rows == 0:
        raise RuntimeError("No labels were written; aborting.")
    if skipped_dates:
        typer.echo(
            f"[labels] warning: skipped dates with no usable payloads: {', '.join(skipped_dates)}",
            err=True,
        )
    typer.echo(f"[labels] completed: wrote {written_rows} rows across {len(partitions)} day(s).")


if __name__ == "__main__":
    app()
