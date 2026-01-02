from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from projections.cli.finalize_projections import _merge_locked_games_from_prior_runs


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_finalize_merges_locked_games_from_latest_pre_tip_run(tmp_path: Path) -> None:
    game_date = date(2025, 1, 1)

    # Schedule (Jan -> season start 2024).
    schedule_path = tmp_path / "silver" / "schedule" / "season=2024" / "month=01" / "schedule.parquet"
    _write_parquet(
        schedule_path,
        pd.DataFrame(
            [
                {"game_date": str(game_date), "game_id": 1, "tip_ts": "2025-01-01T01:30:00Z"},
                {"game_date": str(game_date), "game_id": 2, "tip_ts": "2025-01-01T03:00:00Z"},
            ]
        ),
    )

    # Prior run at 01:00Z includes both games.
    prior_run_id = "20250101T010000Z"
    prior_path = (
        tmp_path
        / "artifacts"
        / "projections"
        / str(game_date)
        / f"run={prior_run_id}"
        / "projections.parquet"
    )
    _write_parquet(
        prior_path,
        pd.DataFrame(
            [
                {"game_id": 1, "player_id": 101, "tip_ts": "2025-01-01T01:30:00Z"},
                {"game_id": 1, "player_id": 102, "tip_ts": "2025-01-01T01:30:00Z"},
                {"game_id": 2, "player_id": 201, "tip_ts": "2025-01-01T03:00:00Z"},
                {"game_id": 2, "player_id": 202, "tip_ts": "2025-01-01T03:00:00Z"},
            ]
        ),
    )

    # Current run at 02:00Z has only the late game (game_id=2).
    current_run_id = "20250101T020000Z"
    current = pd.DataFrame(
        [
            {
                "game_id": 2,
                "player_id": 201,
                "row_source_run_id": current_run_id,
                "row_source_reason": "current_run",
                "tip_ts": "2025-01-01T03:00:00Z",
            }
        ]
    )

    merged, meta = _merge_locked_games_from_prior_runs(
        current,
        game_date=game_date,
        data_root=tmp_path,
        projections_run_id=current_run_id,
        run_as_of_ts=pd.Timestamp("2025-01-01T02:00:00Z"),
    )

    assert meta["locked_games"] == 1
    assert meta["frozen_games"] == 1
    assert meta["selected_runs_by_game_id"]["1"] == prior_run_id
    assert set(pd.to_numeric(merged["game_id"], errors="coerce").dropna().astype(int).unique()) == {1, 2}

    frozen = merged[pd.to_numeric(merged["game_id"], errors="coerce") == 1].copy()
    assert not frozen.empty
    assert set(frozen["row_source_run_id"].astype(str).unique()) == {prior_run_id}
    assert set(frozen["row_source_reason"].astype(str).unique()) == {"locked_game_frozen"}

