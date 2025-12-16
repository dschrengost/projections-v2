"""Helpers for normalizing and persisting tracking payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

SUPPORTED_PT_MEASURE_TYPES: Sequence[str] = (
    "SpeedDistance",
    "Possessions",
    "Passing",
    "Defense",
    "Rebounding",
    "Drives",
    "CatchShoot",
    "PullUpShot",
    "Efficiency",
)

_ID_COLUMN_RENAMES: Mapping[str, str] = {
    "PLAYER_ID": "player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ID": "team_id",
    "TEAM_NAME": "team_name",
    "TEAM_ABBREVIATION": "team_abbreviation",
    "GP": "gp",
    "W": "w",
    "L": "l",
    "MIN": "min",
}

_DEDUP_COLUMNS = (
    "season",
    "season_type",
    "game_date",
    "pt_measure_type",
    "player_id",
    "team_id",
)

_DEFAULT_PART_FILENAME = "part-00000.parquet"


@dataclass(frozen=True)
class TrackingPartitionWrite:
    """Metadata for a bronze partition write."""

    path: Path
    rows_written: int
    total_rows: int


def _normalize_column_name(header: str) -> str:
    token = header.strip().lower()
    token = token.replace("%", "_pct")
    token = token.replace(".", "_")
    token = token.replace("/", "_")
    token = token.replace("-", "_")
    token = token.replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def _safe_partition_value(raw: str) -> str:
    return raw.strip().replace(" ", "+")


def tracking_bronze_root(data_root: Path) -> Path:
    """Return the default bronze root for tracking data."""

    return (data_root / "bronze" / "nba" / "tracking").resolve()


def tracking_partition_dir(
    *,
    data_root: Path,
    season: str,
    season_type: str,
    game_date: date,
    pt_measure_type: str,
    bronze_root: Path | None = None,
) -> Path:
    base = (bronze_root or tracking_bronze_root(data_root)).resolve()
    return (
        base
        / f"season={season}"
        / f"season_type={_safe_partition_value(season_type)}"
        / f"game_date={game_date.isoformat()}"
        / f"pt_measure_type={pt_measure_type}"
    )


def tracking_partition_path(
    *,
    data_root: Path,
    season: str,
    season_type: str,
    game_date: date,
    pt_measure_type: str,
    bronze_root: Path | None = None,
    filename: str = _DEFAULT_PART_FILENAME,
) -> Path:
    return tracking_partition_dir(
        data_root=data_root,
        season=season,
        season_type=season_type,
        game_date=game_date,
        pt_measure_type=pt_measure_type,
        bronze_root=bronze_root,
    ) / filename


def normalize_tracking_df(
    payload: Mapping[str, Any],
    *,
    season: str,
    season_type: str,
    game_date: date,
    pt_measure_type: str,
) -> pd.DataFrame:
    """Normalize the stats.nba.com payload into a bronze-friendly DataFrame."""

    result_sets = payload.get("resultSets") or []
    if not result_sets:
        return pd.DataFrame()
    dataset = result_sets[0]
    headers: Sequence[str] = dataset.get("headers") or []
    rows: Sequence[Sequence[Any]] = dataset.get("rowSet") or []
    if not headers or not rows:
        return pd.DataFrame(columns=headers)
    frame = pd.DataFrame(rows, columns=headers)
    rename_map = {
        column: _ID_COLUMN_RENAMES.get(column, _normalize_column_name(column))
        for column in headers
    }
    normalized = frame.rename(columns=rename_map).copy()
    normalized["season"] = season
    normalized["season_type"] = season_type
    normalized["game_date"] = pd.Timestamp(game_date)
    normalized["pt_measure_type"] = pt_measure_type
    ordered = _reorder_columns(normalized.columns)
    return normalized.loc[:, ordered]


def _reorder_columns(columns: Iterable[str]) -> list[str]:
    prefix = [
        "season",
        "season_type",
        "game_date",
        "pt_measure_type",
        "player_id",
        "player_name",
        "team_id",
        "team_abbreviation",
        "team_name",
        "gp",
        "w",
        "l",
        "min",
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for col in prefix:
        if col in columns and col not in seen:
            ordered.append(col)
            seen.add(col)
    for col in columns:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def _drop_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    subset = [col for col in _DEDUP_COLUMNS if col in frame.columns]
    if not subset:
        return frame
    return frame.drop_duplicates(subset=subset).reset_index(drop=True)


def write_tracking_partition(
    frame: pd.DataFrame,
    *,
    data_root: Path,
    season: str,
    season_type: str,
    game_date: date,
    pt_measure_type: str,
    bronze_root: Path | None = None,
) -> TrackingPartitionWrite:
    """Persist ``frame`` into the bronze tracking partition."""

    if frame.empty:
        raise ValueError("Cannot write empty tracking frame.")
    partition_path = tracking_partition_path(
        data_root=data_root,
        season=season,
        season_type=season_type,
        game_date=game_date,
        pt_measure_type=pt_measure_type,
        bronze_root=bronze_root,
    )
    partition_path.parent.mkdir(parents=True, exist_ok=True)
    frame_to_write = _drop_duplicates(frame)
    existing_rows = 0
    if partition_path.exists():
        existing = pd.read_parquet(partition_path)
        existing_rows = len(existing)
        combined = pd.concat([existing, frame_to_write], ignore_index=True)
        frame_to_write = _drop_duplicates(combined)
    frame_to_write.to_parquet(partition_path, index=False)
    new_rows = len(frame_to_write) - existing_rows
    return TrackingPartitionWrite(
        path=partition_path,
        rows_written=max(0, new_rows),
        total_rows=len(frame_to_write),
    )

