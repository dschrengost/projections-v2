from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from projections.cli.finalize_projections import _merge_locked_games_from_prior_runs


def test_locked_merge_skips_unhealthy_runs(tmp_path: Path) -> None:
    game_date = date(2025, 12, 27)
    season = 2025
    month = 12
    game_id = 22500433

    schedule_dir = tmp_path / "silver" / "schedule" / f"season={season}" / f"month={month:02d}"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    schedule = pd.DataFrame(
        [
            {
                "game_date": pd.Timestamp(game_date),
                "game_id": game_id,
                "tip_ts": pd.Timestamp("2025-12-28T00:00:00Z"),
            }
        ]
    )
    schedule.to_parquet(schedule_dir / "schedule.parquet", index=False)

    projections_day_dir = tmp_path / "artifacts" / "projections" / str(game_date)
    good_run = "20251227T235500Z"
    bad_run = "20251227T235959Z"

    for run_id, status_value in ((good_run, "Ava"), (bad_run, "UNK")):
        run_dir = projections_day_dir / f"run={run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "game_id": game_id,
                    "player_id": 1,
                    "player_name": "Test Player",
                    "minutes_p50": 30.0,
                    "status": status_value,
                }
            ]
        )
        df.to_parquet(run_dir / "projections.parquet", index=False)

    current = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "player_id": 1,
                "player_name": "Test Player",
                "minutes_p50": 30.0,
                "status": "Ava",
            }
        ]
    )

    merged, meta = _merge_locked_games_from_prior_runs(
        current,
        game_date=game_date,
        data_root=tmp_path,
        projections_run_id="20251228T001000Z",
        run_as_of_ts=pd.Timestamp("2025-12-28T00:10:00Z"),
    )
    assert not merged.empty
    assert meta["selected_runs_by_game_id"][str(game_id)] == good_run
    assert bad_run in meta.get("skipped_unhealthy_runs", [])

