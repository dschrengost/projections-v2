from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from projections.minutes_v1.eval_live import (
    MinutesLiveEvalDatasetBuilder,
    evaluate_minutes_live_run,
)


def _write_schedule(root: Path) -> None:
    schedule_dir = root / "silver" / "schedule" / "season=2024" / "month=10"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "game_id": 5001,
                "game_date": pd.Timestamp("2024-10-10"),
                "tip_ts": pd.Timestamp("2024-10-10T23:00:00Z"),
                "home_team_id": 100,
                "away_team_id": 101,
            },
            {
                "game_id": 5002,
                "game_date": pd.Timestamp("2024-10-11"),
                "tip_ts": pd.Timestamp("2024-10-11T00:00:00Z"),
                "home_team_id": 200,
                "away_team_id": 201,
            },
            {
                "game_id": 5003,
                "game_date": pd.Timestamp("2024-10-11"),
                "tip_ts": pd.Timestamp("2024-10-11T02:00:00Z"),
                "home_team_id": 300,
                "away_team_id": 301,
            },
        ]
    )
    df.to_parquet(schedule_dir / "schedule.parquet", index=False)


def _write_prediction_logs(root: Path) -> None:
    logs_dir = root / "gold" / "prediction_logs_minutes" / "season=2024" / "month=10"
    logs_dir.mkdir(parents=True, exist_ok=True)
    data = [
        # Game 5001 early run (should be ignored)
        {
            "game_date": pd.Timestamp("2024-10-10"),
            "game_id": 5001,
            "player_id": 1101,
            "team_id": 10,
            "status": "available",
            "starter_flag": 1,
            "spread_home": -2.0,
            "total": 224.5,
            "minutes_p10": 27.0,
            "minutes_p50": 30.0,
            "minutes_p90": 35.0,
            "play_prob": 0.95,
            "run_as_of_ts": pd.Timestamp("2024-10-10T18:00:00Z"),
            "log_timestamp": pd.Timestamp("2024-10-10T18:05:00Z"),
            "games_since_return": 5,
        },
        {
            "game_date": pd.Timestamp("2024-10-10"),
            "game_id": 5001,
            "player_id": 1102,
            "team_id": 10,
            "status": "questionable",
            "starter_flag": 0,
            "spread_home": -2.0,
            "total": 224.5,
            "minutes_p10": 18.0,
            "minutes_p50": 22.0,
            "minutes_p90": 28.0,
            "play_prob": 0.6,
            "run_as_of_ts": pd.Timestamp("2024-10-10T18:00:00Z"),
            "log_timestamp": pd.Timestamp("2024-10-10T18:05:00Z"),
            "games_since_return": 3,
        },
        {
            "game_date": pd.Timestamp("2024-10-10"),
            "game_id": 5001,
            "player_id": 1103,
            "team_id": 10,
            "status": "out - ankle",
            "starter_flag": 0,
            "spread_home": -2.0,
            "total": 224.5,
            "minutes_p10": 0.0,
            "minutes_p50": 5.0,
            "minutes_p90": 10.0,
            "play_prob": 0.2,
            "run_as_of_ts": pd.Timestamp("2024-10-10T18:00:00Z"),
            "log_timestamp": pd.Timestamp("2024-10-10T18:05:00Z"),
        },
        # Game 5001 later run (should be selected)
        {
            "game_date": pd.Timestamp("2024-10-10"),
            "game_id": 5001,
            "player_id": 1101,
            "team_id": 10,
            "status": "available",
            "starter_flag": 1,
            "spread_home": -2.0,
            "total": 224.5,
            "minutes_p10": 28.0,
            "minutes_p50": 32.0,
            "minutes_p90": 36.0,
            "play_prob": 0.97,
            "run_as_of_ts": pd.Timestamp("2024-10-10T21:00:00Z"),
            "log_timestamp": pd.Timestamp("2024-10-10T21:05:00Z"),
            "games_since_return": 5,
        },
        {
            "game_date": pd.Timestamp("2024-10-10"),
            "game_id": 5001,
            "player_id": 1102,
            "team_id": 10,
            "status": "questionable",
            "starter_flag": 0,
            "spread_home": -2.0,
            "total": 224.5,
            "minutes_p10": 18.0,
            "minutes_p50": 24.0,
            "minutes_p90": 30.0,
            "play_prob": 0.65,
            "run_as_of_ts": pd.Timestamp("2024-10-10T21:00:00Z"),
            "log_timestamp": pd.Timestamp("2024-10-10T21:05:00Z"),
            "games_since_return": 3,
        },
        {
            "game_date": pd.Timestamp("2024-10-10"),
            "game_id": 5001,
            "player_id": 1103,
            "team_id": 10,
            "status": "out - ankle",
            "starter_flag": 0,
            "spread_home": -2.0,
            "total": 224.5,
            "minutes_p10": 0.0,
            "minutes_p50": 4.0,
            "minutes_p90": 8.0,
            "play_prob": 0.2,
            "run_as_of_ts": pd.Timestamp("2024-10-10T21:00:00Z"),
            "log_timestamp": pd.Timestamp("2024-10-10T21:05:00Z"),
        },
        # Game 5002 valid pre-tip run
        {
            "game_date": pd.Timestamp("2024-10-11"),
            "game_id": 5002,
            "player_id": 1103,
            "team_id": 20,
            "status": "probable",
            "starter_flag": 1,
            "spread_home": 9.0,
            "total": 222.0,
            "minutes_p10": 20.0,
            "minutes_p50": 25.0,
            "minutes_p90": 28.0,
            "play_prob": 0.8,
            "run_as_of_ts": pd.Timestamp("2024-10-10T20:00:00Z"),
            "log_timestamp": pd.Timestamp("2024-10-10T20:05:00Z"),
            "games_since_return": 0,
        },
        # Game 5002 run after tip (should be ignored)
        {
            "game_date": pd.Timestamp("2024-10-11"),
            "game_id": 5002,
            "player_id": 1103,
            "team_id": 20,
            "status": "probable",
            "starter_flag": 1,
            "spread_home": 9.0,
            "total": 222.0,
            "minutes_p10": 21.0,
            "minutes_p50": 26.0,
            "minutes_p90": 30.0,
            "play_prob": 0.8,
            "run_as_of_ts": pd.Timestamp("2024-10-11T00:30:00Z"),
            "log_timestamp": pd.Timestamp("2024-10-11T00:35:00Z"),
            "games_since_return": 0,
        },
        # Game 5002 bench projection that ended up DNP
        {
            "game_date": pd.Timestamp("2024-10-11"),
            "game_id": 5002,
            "player_id": 1104,
            "team_id": 20,
            "status": "out - rest",
            "starter_flag": 0,
            "spread_home": 9.0,
            "total": 222.0,
            "minutes_p10": 16.0,
            "minutes_p50": 22.0,
            "minutes_p90": 28.0,
            "play_prob": 0.5,
            "run_as_of_ts": pd.Timestamp("2024-10-10T20:00:00Z"),
            "log_timestamp": pd.Timestamp("2024-10-10T20:05:00Z"),
            "games_since_return": 4,
        },
    ]
    pd.DataFrame(data).to_parquet(logs_dir / "logs.parquet", index=False)


def _write_labels(root: Path) -> None:
    for day_str, game_id, rows in [
        (
            "2024-10-10",
            5001,
            [
                {"player_id": 1101, "actual_minutes": 30.0, "team_id": 10},
                {"player_id": 1102, "actual_minutes": 20.0, "team_id": 10},
                {"player_id": 1103, "actual_minutes": 0.0, "team_id": 10},
            ],
        ),
        (
            "2024-10-11",
            5002,
            [
                {"player_id": 1103, "actual_minutes": 28.0, "team_id": 20},
                {"player_id": 1104, "actual_minutes": 0.0, "team_id": 20},
            ],
        ),
    ]:
        day = pd.Timestamp(day_str)
        season = 2024
        day_dir = root / "gold" / "labels_minutes_v1" / f"season={season}" / f"game_date={day.date().isoformat()}"
        day_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "game_id": game_id,
                    "player_id": entry["player_id"],
                    "game_date": day,
                    "actual_minutes": entry["actual_minutes"],
                    "team_id": entry["team_id"],
                }
                for entry in rows
            ]
        )
        df.to_parquet(day_dir / "labels.parquet", index=False)


def test_build_dataset_and_metrics(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_schedule(data_root)
    _write_prediction_logs(data_root)
    _write_labels(data_root)

    builder = MinutesLiveEvalDatasetBuilder(
        data_root=data_root,
        labels_root=data_root / "gold" / "labels_minutes_v1",
        schedule_root=data_root / "silver" / "schedule",
    )
    frame = builder.build(date(2024, 10, 10), date(2024, 10, 11))
    assert len(frame) == 5
    assert set(frame["player_id"]) == {1101, 1102, 1103, 1104}
    assert set(frame["snapshot_mode"]) == {"last_before_tip"}
    assert set(frame["status_bucket"].str.upper()) >= {"CLEAN", "QUESTIONABLE", "OUT"}
    player_1103 = frame.loc[frame["player_id"] == 1103]
    assert player_1103.loc[player_1103["actual_minutes"] > 0, "injury_return_flag"].iloc[0] == 1
    assert (player_1103.loc[player_1103["actual_minutes"] == 0, "injury_return_flag"].iloc[0]) == 0

    game1_rows = frame[frame["game_id"] == 5001]
    assert game1_rows["run_as_of_ts"].nunique() == 1
    assert game1_rows["run_as_of_ts"].iloc[0] == pd.Timestamp("2024-10-10T21:00:00Z")
    game2_rows = frame[frame["game_id"] == 5002]
    assert game2_rows["run_as_of_ts"].nunique() == 1
    assert game2_rows["run_as_of_ts"].iloc[0] == pd.Timestamp("2024-10-10T20:00:00Z")

    summary = builder.last_snapshot_summary or {}
    assert summary.get("total_games") == 3
    assert summary.get("games_with_snapshots") == 2
    assert summary.get("games_skipped") == 1
    skipped = summary.get("skipped_games") or []
    assert skipped and skipped[0]["reason"] == "no_logs"

    metrics = evaluate_minutes_live_run(frame)
    overall = metrics["overall"]
    assert overall["rows"] == len(frame)
    assert overall["coverage_p10_p90"] != overall["coverage_p10_p90_cond"]
    assert "mae_minutes" in overall and overall["mae_minutes"] >= 0

    starter_slice = {entry["bucket"]: entry for entry in metrics["slices"].get("starter_flag", [])}
    assert starter_slice["starter"]["rows"] == 2
    assert starter_slice["bench"]["rows"] == 3

    spread_slice = {entry["bucket"]: entry for entry in metrics["slices"].get("spread_home", [])}
    assert spread_slice["<=3"]["rows"] == 3
    assert spread_slice[">8"]["rows"] == 2

    rotation = {entry["bucket"]: entry for entry in metrics.get("rotation_slices", [])}
    assert {"rotation_all", "rotation_starters_30_plus", "rotation_mid_minutes"}.issubset(rotation.keys())
    assert rotation["rotation_all"]["rows"] == 3
    assert rotation["rotation_starters_30_plus"]["rows"] == 1
    assert rotation["rotation_mid_minutes"]["rows"] == 2
    for payload in rotation.values():
        assert "coverage_p10_p90_cond" in payload

    status_slices = {entry["bucket"]: entry for entry in metrics.get("status_slices", [])}
    assert {"CLEAN", "QUESTIONABLE", "OUT"}.issubset(status_slices.keys())
    assert status_slices["QUESTIONABLE"]["rows"] >= 1
    assert "coverage_p10_p90_cond" in status_slices["CLEAN"]

    injury_slices = {entry["bucket"]: entry for entry in metrics.get("injury_return_slices", [])}
    assert injury_slices["injury_return"]["rows"] == 1
    assert injury_slices["non_injury_return"]["rows"] == 2
    assert "coverage_p10_p90_cond" in injury_slices["injury_return"]
