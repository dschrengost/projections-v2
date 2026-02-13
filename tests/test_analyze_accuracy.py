from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analyze_accuracy import (
    _select_closest_pretip_runs,
    compute_metrics,
    load_predictions,
)


def test_select_closest_pretip_runs_picks_latest_run_before_tip() -> None:
    game_tips = {
        1001: pd.Timestamp("2026-01-01T00:00:00Z"),
        1002: pd.Timestamp("2026-01-01T01:00:00Z"),
    }
    run_index = {
        "20251231T230000Z": {
            "run_id": "20251231T230000Z",
            "run_ts": pd.Timestamp("2025-12-31T23:00:00Z"),
            "game_ids": {1001, 1002},
        },
        "20260101T003000Z": {
            "run_id": "20260101T003000Z",
            "run_ts": pd.Timestamp("2026-01-01T00:30:00Z"),
            "game_ids": {1002},
        },
        "20260101T013000Z": {
            "run_id": "20260101T013000Z",
            "run_ts": pd.Timestamp("2026-01-01T01:30:00Z"),
            "game_ids": {1001, 1002},
        },
    }

    selected, missing = _select_closest_pretip_runs(game_tips, run_index)

    assert selected == {
        1001: "20251231T230000Z",
        1002: "20260101T003000Z",
    }
    assert missing == []


def test_compute_metrics_prefers_world_p50_sources_for_mae() -> None:
    actuals = pd.DataFrame(
        [
            {
                "player_id": "42",
                "game_id": 1,
                "actual_minutes": 20.0,
                "actual_dk_fpts": 10.0,
                "boxscore_status": "ACTIVE",
                "team_id": 1,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "player_id": "42",
                "game_id": 1,
                "dk_fpts_mean": 100.0,
                "dk_fpts_p50": 12.0,
                "minutes_mean": 5.0,
                "minutes_sim_p50": 18.0,
            }
        ]
    )

    metrics = compute_metrics(actuals, predictions, min_minutes=5)

    assert metrics["fpts_point_source"] == "dk_fpts_p50"
    assert metrics["minutes_point_source"] == "minutes_sim_p50"
    assert metrics["fpts_mae"] == 2.0
    assert metrics["minutes_mae"] == 2.0


def test_load_predictions_selects_per_game_closest_pretip_snapshot(tmp_path: Path) -> None:
    data_root = tmp_path
    date_str = "2026-01-01"

    schedule_path = data_root / "silver" / "schedule" / "season=2025" / "month=01"
    schedule_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_id": 2001,
                "game_date": pd.Timestamp("2026-01-01"),
                "tip_ts": pd.Timestamp("2026-01-02T00:00:00Z"),
            },
            {
                "game_id": 2002,
                "game_date": pd.Timestamp("2026-01-01"),
                "tip_ts": pd.Timestamp("2026-01-02T01:00:00Z"),
            },
        ]
    ).to_parquet(schedule_path / "schedule.parquet", index=False)

    projections_root = data_root / "artifacts" / "projections" / date_str
    run_a = projections_root / "run=20260101T235000Z"
    run_b = projections_root / "run=20260102T003000Z"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"player_id": "a", "game_id": 2001, "dk_fpts_p50": 10.0, "minutes_sim_p50": 20.0},
            {"player_id": "b", "game_id": 2002, "dk_fpts_p50": 11.0, "minutes_sim_p50": 21.0},
        ]
    ).to_parquet(run_a / "projections.parquet", index=False)

    pd.DataFrame(
        [
            {"player_id": "b", "game_id": 2002, "dk_fpts_p50": 15.0, "minutes_sim_p50": 25.0},
        ]
    ).to_parquet(run_b / "projections.parquet", index=False)

    predictions, meta = load_predictions(date_str, data_root)

    assert meta["snapshot_selection_mode"] == "closest_pre_tip_per_game"
    assert meta["games_with_tip"] == 2
    assert meta["games_with_pre_tip_snapshot"] == 2
    assert meta["games_missing_pre_tip_snapshot"] == 0
    assert meta["selected_game_run_map"]["2001"] == "20260101T235000Z"
    assert meta["selected_game_run_map"]["2002"] == "20260102T003000Z"

    game_to_run = dict(zip(predictions["game_id"], predictions["snapshot_run_id"], strict=False))
    assert game_to_run[2001] == "20260101T235000Z"
    assert game_to_run[2002] == "20260102T003000Z"
