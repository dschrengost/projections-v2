from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from projections.pipeline.dfs_readiness import run_dfs_readiness


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_dfs_readiness_smoke_passes_with_minimal_artifacts(tmp_path: Path) -> None:
    game_date = "2025-01-01"
    run_id = "20250101T000000Z"

    # Schedule (season_start=2024 for Jan 2025)
    schedule_path = tmp_path / "silver" / "schedule" / "season=2024" / "month=01" / "schedule.parquet"
    _write_parquet(
        schedule_path,
        pd.DataFrame(
            [
                {
                    "game_date": game_date,
                    "game_id": 1,
                    "tip_ts": f"{game_date}T23:00:00Z",
                    "home_team_id": 10,
                    "away_team_id": 20,
                }
            ]
        ),
    )

    # Fresh inputs (timestamps before as_of_ts)
    _write_parquet(
        tmp_path / "silver" / "injuries_snapshot" / "season=2024" / "part-0.parquet",
        pd.DataFrame([{"as_of_ts": f"{game_date}T10:00:00Z"}]),
    )
    _write_parquet(
        tmp_path / "silver" / "espn_injuries" / f"date={game_date}" / "injuries.parquet",
        pd.DataFrame([{"as_of_ts": f"{game_date}T10:05:00Z"}]),
    )
    _write_parquet(
        tmp_path / "silver" / "rotowire_lineups" / f"date={game_date}" / "lineups.parquet",
        pd.DataFrame([{"ingested_ts": f"{game_date}T10:10:00Z"}]),
    )
    _write_parquet(
        tmp_path / "silver" / "odds_snapshot" / "season=2024" / "part-0.parquet",
        pd.DataFrame([{"as_of_ts": f"{game_date}T10:15:00Z"}]),
    )

    # DK salaries
    _write_parquet(
        tmp_path
        / "gold"
        / "dk_salaries"
        / "site=dk"
        / f"game_date={game_date}"
        / "draft_group_id=1"
        / "salaries.parquet",
        pd.DataFrame([{"player_name": "A", "salary": 5000}]),
    )

    # Unified projections
    proj_path = (
        tmp_path / "artifacts" / "projections" / game_date / f"run={run_id}" / "projections.parquet"
    )
    _write_parquet(
        proj_path,
        pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "team_id": 10,
                    "player_id": pid,
                    "minutes_p50": 48.0,
                    "play_prob": 1.0,
                    "dk_fpts_mean": 50.0,
                    "dk_fpts_mean_uncond": 50.0,
                    "salary": 5000,
                    "pred_own_pct": 100.0,
                    "projections_run_id": run_id,
                    "minutes_run_id": run_id,
                    "sim_run_id": run_id,
                    "rates_run_id": run_id,
                }
                for pid in range(100, 105)
            ]
        ),
    )
    latest_pointer = tmp_path / "artifacts" / "projections" / game_date / "latest_run.json"
    latest_pointer.write_text(f'{{"run_id": "{run_id}"}}', encoding="utf-8")

    report = run_dfs_readiness(
        game_date=game_date,
        data_root=tmp_path,
        run_id=None,
        as_of_ts=datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc),
        strict=True,
    )
    assert report.passed, report.errors
