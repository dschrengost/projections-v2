from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from projections.api import flashback_api
from projections.api.minutes_api import create_app
from projections.post_contest.replay_analytics_service import ReplayAnalyticsBundle
from projections.post_contest.replay_calibration_service import ReplayCalibrationBundle


def test_list_flashback_contests(tmp_path: Path, monkeypatch) -> None:
    user_entries_path = tmp_path / "user_entries.parquet"
    pd.DataFrame(
        [
            {
                "date": "2099-01-01",
                "contest_id": "123",
                "draft_group_id": 999,
                "contest_name": "NBA And-One",
                "entry_fee": 3.0,
                "entry_name": "daniel (1/2)",
                "rank": 12,
                "prize_pool": 1000.0,
                "first_place_prize": 300.0,
            },
            {
                "date": "2099-01-01",
                "contest_id": "123",
                "draft_group_id": 999,
                "contest_name": "NBA And-One",
                "entry_fee": 3.0,
                "entry_name": "daniel (2/2)",
                "rank": 33,
                "prize_pool": 1000.0,
                "first_place_prize": 300.0,
            },
        ]
    ).to_parquet(user_entries_path, index=False)

    monkeypatch.setattr(flashback_api, "_user_entries_path", lambda: user_entries_path)
    monkeypatch.setattr(flashback_api, "find_latest_export_manifest", lambda **kwargs: None)

    client = TestClient(create_app())
    response = client.get("/api/flashback/contests", params={"date": "2099-01-01", "user_pattern": "daniel"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["contest_id"] == "123"
    assert payload[0]["entry_count"] == 2


def test_list_flashback_contests_falls_back_to_raw_results(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "bronze" / "dk_contests" / "nba_gpp_data" / "2099-01-01"
    results_dir = raw_dir / "results"
    results_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "contest_id": 123,
                "contest_name": "NBA And-One",
                "draft_group_id": 999,
                "entry_fee": 3.0,
                "current_entries": 1000,
            }
        ]
    ).to_csv(raw_dir / "nba_gpp_2099-01-01.csv", index=False)
    pd.DataFrame(
        [
            {"Rank": 12, "EntryId": 1, "EntryName": "daniel (1/2)", "Lineup": "PG A SG B SF C PF D C E G F F G UTIL H"},
            {"Rank": 20, "EntryId": 2, "EntryName": "other", "Lineup": "PG A SG B SF C PF D C E G F F G UTIL H"},
            {"Rank": 33, "EntryId": 3, "EntryName": "daniel (2/2)", "Lineup": "PG A SG B SF C PF D C E G F F G UTIL H"},
        ]
    ).to_csv(results_dir / "contest_123_results.csv", index=False)

    monkeypatch.setattr(flashback_api, "_user_entries_path", lambda: tmp_path / "missing.parquet")
    monkeypatch.setattr(flashback_api.paths, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    monkeypatch.setattr(flashback_api, "find_latest_export_manifest", lambda **kwargs: None)

    client = TestClient(create_app())
    response = client.get("/api/flashback/contests", params={"date": "2099-01-01", "user_pattern": "daniel"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["contest_id"] == "123"
    assert payload[0]["entry_count"] == 2
    assert payload[0]["contest_name"] == "NBA And-One"
    assert payload[0]["draft_group_id"] == 999


def test_run_flashback_returns_summary_and_previews(tmp_path: Path, monkeypatch) -> None:
    analytics_dir = tmp_path / "analytics"
    analytics_dir.mkdir(parents=True)
    player_path = analytics_dir / "player_calibration.parquet"
    lineup_path = analytics_dir / "lineup_calibration.parquet"
    field_path = analytics_dir / "field_calibration.parquet"
    regret_path = analytics_dir / "regret_summary.parquet"
    summary_path = analytics_dir / "summary.json"

    pd.DataFrame([{"player_id": "1", "sim_mean_fpts": 30.0}]).to_parquet(player_path, index=False)
    pd.DataFrame([{"lineup_key": "1|2|3", "sim_roi": 0.2}]).to_parquet(lineup_path, index=False)
    pd.DataFrame([{"contest_id": "123", "player_ownership_mae_pct": 5.0}]).to_parquet(field_path, index=False)
    pd.DataFrame([{"contest_id": "123", "selection_regret_roi": 0.1}]).to_parquet(regret_path, index=False)
    summary_path.write_text(json.dumps({"counts": {"lineup_rows": 1}, "artifacts": {"player_calibration_path": str(player_path)}}))

    monkeypatch.setattr(
        flashback_api,
        "build_post_contest_replay_analytics",
        lambda **kwargs: ReplayAnalyticsBundle(
            player_calibration_path=player_path,
            lineup_calibration_path=lineup_path,
            field_calibration_path=field_path,
            regret_summary_path=regret_path,
            summary_path=summary_path,
        ),
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/flashback/run",
        json={
            "game_date": "2099-01-01",
            "contest_id": "123",
            "user_pattern": "daniel",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["counts"]["lineup_rows"] == 1
    assert len(payload["previews"]["player_calibration"]) == 1


def test_run_flashback_calibration_returns_previews(tmp_path: Path, monkeypatch) -> None:
    out_dir = tmp_path / "gold" / "replay_calibration"
    out_dir.mkdir(parents=True)
    player_fpts_path = out_dir / "player_fpts_calibration.parquet"
    player_minutes_path = out_dir / "player_minutes_calibration.parquet"
    ownership_path = out_dir / "ownership_recalibration.parquet"
    field_model_path = out_dir / "field_model_calibration.parquet"
    regret_contest_path = out_dir / "optimizer_regret_by_contest.parquet"
    regret_bucket_path = out_dir / "optimizer_regret_by_bucket.parquet"
    regret_examples_path = out_dir / "optimizer_regret_examples.parquet"
    summary_path = out_dir / "summary.json"

    for path, rows in [
        (player_fpts_path, [{"bucket": "20_30", "recommended_mean_shift": 1.5}]),
        (player_minutes_path, [{"bucket": "20_30", "recommended_variance_scale": 1.2}]),
        (ownership_path, [{"projected_ownership_bucket": "20_35", "recommended_delta": 2.0}]),
        (field_model_path, [{"field_size_bucket": "medium", "mean_player_ownership_mae_pct": 5.0}]),
        (regret_contest_path, [{"contest_id": "123", "selection_regret_roi": 0.1}]),
        (regret_bucket_path, [{"field_size_bucket": "medium", "mean_selection_regret_roi": 0.1}]),
        (regret_examples_path, [{"contest_id": "123", "selection_regret_roi": 0.1}]),
    ]:
        pd.DataFrame(rows).to_parquet(path, index=False)
    summary_path.write_text(json.dumps({"artifact_counts": {"ownership_rows": 1}}))

    monkeypatch.setattr(
        flashback_api,
        "build_replay_calibration_artifacts",
        lambda **kwargs: ReplayCalibrationBundle(
            player_fpts_calibration_path=player_fpts_path,
            player_minutes_calibration_path=player_minutes_path,
            ownership_recalibration_path=ownership_path,
            field_model_calibration_path=field_model_path,
            optimizer_regret_by_contest_path=regret_contest_path,
            optimizer_regret_by_bucket_path=regret_bucket_path,
            optimizer_regret_examples_path=regret_examples_path,
            summary_path=summary_path,
        ),
    )

    client = TestClient(create_app())
    response = client.post("/api/flashback/calibration/run")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["artifact_counts"]["ownership_rows"] == 1
    assert len(payload["previews"]["ownership_recalibration"]) == 1
