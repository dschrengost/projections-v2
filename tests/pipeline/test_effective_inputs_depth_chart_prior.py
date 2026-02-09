from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from projections.pipeline.effective_inputs import (
    EFFECTIVE_INPUTS_SUMMARY,
    EFFECTIVE_MINUTES_FILENAME,
    write_effective_minutes_layer,
)


def _seed_depth_chart_inputs(data_root: Path) -> None:
    realgm_dir = data_root / "bronze" / "realgm"
    realgm_dir.mkdir(parents=True, exist_ok=True)

    snapshot = pd.DataFrame(
        [
            {
                "team_name": "New York Knicks",
                "player_name": "Jalen Brunson",
                "realgm_player_id": 1001,
                "position": "PG",
                "depth_role": "starter",
                "depth_order": 0,
                "recent_stats": "",
                "movement": "",
                "scraped_at": "2026-01-18T18:15:00Z",
            },
            {
                "team_name": "New York Knicks",
                "player_name": "Miles McBride",
                "realgm_player_id": 1002,
                "position": "PG",
                "depth_role": "limited",
                "depth_order": 0,
                "recent_stats": "",
                "movement": "",
                "scraped_at": "2026-01-18T18:15:00Z",
            },
        ]
    )
    snapshot.to_parquet(realgm_dir / "depth_charts.parquet", index=False)

    crosswalk = pd.DataFrame(
        [
            {"realgm_player_id": 1001, "player_id": 1, "updated_at": "2026-01-18T00:00:00Z"},
            {"realgm_player_id": 1002, "player_id": 2, "updated_at": "2026-01-18T00:00:00Z"},
        ]
    )
    crosswalk.to_parquet(realgm_dir / "player_id_crosswalk.parquet", index=False)


def test_write_effective_minutes_layer_applies_depth_chart_prior(
    tmp_path: Path, caplog
) -> None:
    data_root = tmp_path
    _seed_depth_chart_inputs(data_root)
    caplog.set_level("INFO")

    game_date = date(2026, 1, 18)
    run_dir = (
        data_root
        / "artifacts"
        / "minutes_v1"
        / "daily"
        / game_date.isoformat()
        / "run=20260118T181000Z"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    minutes = pd.DataFrame(
        [
            {
                "game_date": game_date.isoformat(),
                "game_id": 100,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "team_tricode": "NYK",
                "player_id": 1,
                "player_name": "Jalen Brunson",
                "status": "available",
                "play_prob": 0.90,
                "rotation_prob": 0.85,
                "minutes_p10": 20.0,
                "minutes_p50": 34.0,
                "minutes_p90": 42.0,
            },
            {
                "game_date": game_date.isoformat(),
                "game_id": 100,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "team_tricode": "NYK",
                "player_id": 2,
                "player_name": "Miles McBride",
                "status": "available",
                "play_prob": 0.60,
                "rotation_prob": 0.55,
                "minutes_p10": 1.0,
                "minutes_p50": 8.0,
                "minutes_p90": 40.0,
            },
            {
                "game_date": game_date.isoformat(),
                "game_id": 100,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "team_tricode": "NYK",
                "player_id": 3,
                "player_name": "Bench Three",
                "status": "available",
                "play_prob": 0.55,
                "rotation_prob": 0.50,
                "minutes_p10": 20.0,
                "minutes_p50": 34.0,
                "minutes_p90": 40.0,
            },
            {
                "game_date": game_date.isoformat(),
                "game_id": 100,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "team_tricode": "NYK",
                "player_id": 4,
                "player_name": "Bench Four",
                "status": "available",
                "play_prob": 0.45,
                "rotation_prob": 0.42,
                "minutes_p10": 20.0,
                "minutes_p50": 34.0,
                "minutes_p90": 39.0,
            },
            {
                "game_date": game_date.isoformat(),
                "game_id": 100,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "team_tricode": "NYK",
                "player_id": 5,
                "player_name": "Bench Five",
                "status": "available",
                "play_prob": 0.40,
                "rotation_prob": 0.38,
                "minutes_p10": 18.0,
                "minutes_p50": 33.0,
                "minutes_p90": 38.0,
            },
            {
                "game_date": game_date.isoformat(),
                "game_id": 100,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "team_tricode": "NYK",
                "player_id": 6,
                "player_name": "Bench Six",
                "status": "available",
                "play_prob": 0.35,
                "rotation_prob": 0.32,
                "minutes_p10": 18.0,
                "minutes_p50": 33.0,
                "minutes_p90": 37.0,
            },
            {
                "game_date": game_date.isoformat(),
                "game_id": 100,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "team_tricode": "NYK",
                "player_id": 7,
                "player_name": "Bench Seven",
                "status": "available",
                "play_prob": 0.30,
                "rotation_prob": 0.28,
                "minutes_p10": 16.0,
                "minutes_p50": 32.0,
                "minutes_p90": 36.0,
            },
            {
                "game_date": game_date.isoformat(),
                "game_id": 100,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "team_tricode": "NYK",
                "player_id": 8,
                "player_name": "Bench Eight",
                "status": "available",
                "play_prob": 0.28,
                "rotation_prob": 0.26,
                "minutes_p10": 16.0,
                "minutes_p50": 32.0,
                "minutes_p90": 35.0,
            },
        ]
    )
    minutes_path = run_dir / "minutes.parquet"
    minutes.to_parquet(minutes_path, index=False)

    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "20260118T181000Z", "as_of_ts": "2026-01-18T18:20:00Z"}),
        encoding="utf-8",
    )

    result = write_effective_minutes_layer(
        game_date=game_date,
        minutes_path=minutes_path,
        out_dir=run_dir,
        data_root=data_root,
    )

    eff = pd.read_parquet(result.effective_minutes_path)
    assert result.effective_minutes_path == run_dir / EFFECTIVE_MINUTES_FILENAME
    assert {"dc_present", "dc_role", "dc_role_priority", "dc_ahead_global", "dc_snapshot_ts"} <= set(eff.columns)

    mcbride = eff.loc[eff["player_id"] == 2].iloc[0]
    assert str(mcbride["dc_role"]) == "limited"
    assert float(mcbride["minutes_p90"]) <= 22.0 + 1e-9

    summary = json.loads((run_dir / EFFECTIVE_INPUTS_SUMMARY).read_text(encoding="utf-8"))
    assert summary["run_as_of_ts"] == "2026-01-18T18:20:00Z"
    assert isinstance(summary.get("depth_chart_crosswalk"), dict)
    assert summary["depth_chart_crosswalk"]["applied"] is True
    assert int(summary["depth_chart_crosswalk"]["matched_rows"]) >= 2
    assert isinstance(summary.get("depth_chart_prior"), dict)
    assert summary["depth_chart_prior"]["applied"] is True
    assert summary["depth_chart_prior"]["dc_snapshot_ts"] == "2026-01-18T18:15:00Z"
    assert isinstance(summary.get("depth_chart_alerts"), list)

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("[dc-prior]" in msg for msg in messages)
