from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from projections.minutes_v1.logs import prediction_logs_base

TEST_MINUTES_RUN_ID = "minutes_test_run"


def _write_features(root: Path, game_date: str) -> None:
    day = pd.Timestamp(game_date)
    season = day.year
    features_dir = (
        root / "gold" / "features_minutes_v1" / f"season={season}" / f"month={day.month:02d}"
    )
    features_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "game_id": 1001,
                "player_id": 10,
                "team_id": 200,
                "game_date": day,
                "tip_ts": pd.Timestamp(f"{game_date}T23:00:00Z"),
                "feature_as_of_ts": pd.Timestamp(f"{game_date}T18:00:00Z"),
                "season": str(season),
                "minutes": 30.0,
                "starter_flag": 1,
                "pos_bucket": "G",
                "status": "available",
                "spread_home": -5.5,
                "total": 225.0,
                "home_flag": 1,
                "is_projected_starter": 1,
                "is_confirmed_starter": 0,
                "lineup_role": "starter",
            },
            {
                "game_id": 1001,
                "player_id": 11,
                "team_id": 200,
                "game_date": day,
                "tip_ts": pd.Timestamp(f"{game_date}T23:00:00Z"),
                "feature_as_of_ts": pd.Timestamp(f"{game_date}T18:05:00Z"),
                "season": str(season),
                "minutes": 18.0,
                "starter_flag": 0,
                "pos_bucket": "W",
                "status": "out - ankle",
                "spread_home": -5.5,
                "total": 225.0,
                "home_flag": 1,
                "is_projected_starter": 0,
                "is_confirmed_starter": 0,
                "lineup_role": "bench",
            },
        ]
    )
    df.to_parquet(features_dir / "features.parquet", index=False)


def _write_boxscores(root: Path, game_date: str) -> None:
    day = pd.Timestamp(game_date)
    season = day.year
    bronze_dir = root / "bronze" / "boxscores_raw" / f"season={season}" / f"date={game_date}"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    game_payload = {
        "game_id": "1001",
        "game_time_utc": f"{game_date}T23:00:00Z",
        "home": {
            "team_id": 200,
            "team_name": "Home",
            "players": [
                {
                    "person_id": 10,
                    "name": "Alpha Guard",
                    "statistics": {
                        "points": 25,
                        "reboundsTotal": 5,
                        "assists": 8,
                        "steals": 2,
                        "blocks": 0,
                        "turnovers": 3,
                        "threePointersMade": 4,
                        "minutes": "PT34M0S",
                        "fieldGoalsAttempted": 18,
                        "freeThrowsAttempted": 6,
                    },
                },
                {
                    "person_id": 11,
                    "name": "Wing Bench",
                    "statistics": {
                        "points": 12,
                        "reboundsTotal": 6,
                        "assists": 3,
                        "steals": 1,
                        "blocks": 1,
                        "turnovers": 1,
                        "threePointersMade": 1,
                        "minutes": "PT20M0S",
                        "fieldGoalsAttempted": 10,
                        "freeThrowsAttempted": 2,
                    },
                },
            ],
        },
        "away": {
            "team_id": 300,
            "team_name": "Away",
            "players": [],
        },
    }
    bronze_df = pd.DataFrame(
        [
            {
                "game_id": 1001,
                "payload": json.dumps(game_payload),
                "tip_ts": f"{game_date}T23:00:00Z",
            }
        ]
    )
    bronze_df.to_parquet(bronze_dir / "boxscores_raw.parquet", index=False)


def _write_prediction_logs(root: Path, game_date: str) -> None:
    day = pd.Timestamp(game_date)
    season = day.year
    logs_dir = (
        prediction_logs_base(root)
        / f"run={TEST_MINUTES_RUN_ID}"
        / f"season={season}"
        / f"month={day.month:02d}"
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "game_date": day,
                "game_id": 1001,
                "player_id": 10,
                "minutes_p10": 30.0,
                "minutes_p50": 33.0,
                "minutes_p90": 37.0,
                "play_prob": 0.95,
                "run_as_of_ts": pd.Timestamp(f"{game_date}T16:00:00Z"),
                "log_timestamp": pd.Timestamp(f"{game_date}T16:05:00Z"),
                "feature_as_of_ts": pd.Timestamp(f"{game_date}T15:45:00Z"),
            },
            {
                "game_date": day,
                "game_id": 1001,
                "player_id": 11,
                "minutes_p10": 15.0,
                "minutes_p50": 18.0,
                "minutes_p90": 22.0,
                "play_prob": 0.2,
                "run_as_of_ts": pd.Timestamp(f"{game_date}T16:00:00Z"),
                "log_timestamp": pd.Timestamp(f"{game_date}T16:05:00Z"),
                "feature_as_of_ts": pd.Timestamp(f"{game_date}T15:45:00Z"),
            },
        ]
    )
    df.to_parquet(logs_dir / f"{game_date}_test.parquet", index=False)


def _write_legacy_prediction_logs(root: Path, game_date: str) -> None:
    day = pd.Timestamp(game_date)
    season = day.year
    logs_dir = (
        root / "gold" / "prediction_logs_minutes_v1" / f"season={season}" / f"month={day.month:02d}"
    )
    # Legacy path: keep emitting files here so consumers can test fallback logic.
    logs_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "game_date": day,
                "game_id": 1001,
                "player_id": 10,
                "minutes_p10": 28.0,
                "minutes_p50": 31.0,
                "minutes_p90": 35.0,
                "play_prob": 0.9,
                "run_as_of_ts": pd.Timestamp(f"{game_date}T14:00:00Z"),
                "log_timestamp": pd.Timestamp(f"{game_date}T14:05:00Z"),
                "feature_as_of_ts": pd.Timestamp(f"{game_date}T13:45:00Z"),
            },
            {
                "game_date": day,
                "game_id": 1001,
                "player_id": 11,
                "minutes_p10": 12.0,
                "minutes_p50": 16.0,
                "minutes_p90": 21.0,
                "play_prob": 0.3,
                "run_as_of_ts": pd.Timestamp(f"{game_date}T14:00:00Z"),
                "log_timestamp": pd.Timestamp(f"{game_date}T14:05:00Z"),
                "feature_as_of_ts": pd.Timestamp(f"{game_date}T13:45:00Z"),
            },
        ]
    )
    df.to_parquet(logs_dir / f"{game_date}_legacy.parquet", index=False)


@pytest.fixture()
def sample_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    _write_features(data_root, "2024-10-01")
    _write_boxscores(data_root, "2024-10-01")
    _write_prediction_logs(data_root, "2024-10-01")
    return data_root
