from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.rotation.build_joint_rotation_rates_dataset_v1 import _apply_lineup_feature_contract
from scripts.rotation.build_rotation_train_dataset_v1 import _backfill_lineups_from_silver_daily_lineups


def test_apply_lineup_feature_contract_ignores_lineup_status_for_starter_signal() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1001, 1001, 1001, 1001, 1002, 1002],
            "team_id": [10, 10, 10, 10, 20, 20],
            "player_id": [1, 2, 3, 4, 5, 6],
            "lineup_timestamp": [
                "2026-01-18T17:00:00Z",
                "2026-01-18T17:00:00Z",
                "2026-01-18T17:00:00Z",
                "2026-01-18T17:00:00Z",
                pd.NaT,
                pd.NaT,
            ],
            "lineup_role": [
                "confirmed_starter",
                "bench",
                "bench",
                pd.NA,
                pd.NA,
                pd.NA,
            ],
            "lineup_status": [
                "confirmed",
                "confirmed",
                "expected",
                pd.NA,
                "confirmed",
                "expected",
            ],
            "is_projected_starter": [0, 0, 0, 1, 1, 0],
            "is_confirmed_starter": [0, 0, 0, 0, 0, 0],
        }
    )
    out = _apply_lineup_feature_contract(df)
    by_pid = out.set_index("player_id")

    # Team 1001/10 has lineup data available.
    assert int(by_pid.loc[1, "lineup_available"]) == 1
    assert int(by_pid.loc[2, "lineup_available"]) == 1
    assert int(by_pid.loc[3, "lineup_available"]) == 1
    assert int(by_pid.loc[4, "lineup_available"]) == 1

    # Starter signal should come from explicit starter role or projected starter flag.
    assert int(by_pid.loc[1, "lineup_starter_announced"]) == 1  # confirmed_starter role
    assert int(by_pid.loc[4, "lineup_starter_announced"]) == 1  # projected starter flag
    assert int(by_pid.loc[2, "lineup_starter_announced"]) == 0  # bench + confirmed status only
    assert int(by_pid.loc[3, "lineup_starter_announced"]) == 0  # bench + expected status only

    # Team 1002/20 has no lineup timestamp -> lineup unavailable suppresses starter flag.
    assert int(by_pid.loc[5, "lineup_available"]) == 0
    assert int(by_pid.loc[5, "lineup_starter_announced"]) == 0

    assert "is_projected_starter" not in out.columns
    assert "is_confirmed_starter" not in out.columns


def test_lineup_backfill_uses_role_not_status_for_projected_starter(tmp_path: Path) -> None:
    data_root = tmp_path
    lineup_dir = data_root / "silver" / "nba_daily_lineups" / "season=2025" / "date=2026-01-18"
    lineup_dir.mkdir(parents=True, exist_ok=True)

    lineups = pd.DataFrame(
        {
            "game_id": [2001, 2001],
            "team_id": [10, 10],
            "player_id": [101, 102],
            "lineup_role": ["bench", "confirmed_starter"],
            "lineup_status": ["confirmed", "confirmed"],
            "roster_status": ["active", "active"],
            "lineup_timestamp": ["2026-01-18T17:10:00Z", "2026-01-18T17:10:00Z"],
            "ingested_ts": ["2026-01-18T17:12:00Z", "2026-01-18T17:12:00Z"],
        }
    )
    lineups.to_parquet(lineup_dir / "lineups.parquet", index=False)

    base = pd.DataFrame(
        {
            "game_id": [2001, 2001],
            "team_id": [10, 10],
            "player_id": [101, 102],
            "game_date": ["2026-01-18", "2026-01-18"],
            "tip_ts": ["2026-01-18T19:00:00Z", "2026-01-18T19:00:00Z"],
            "lineup_timestamp": [pd.NaT, pd.NaT],
            "is_projected_starter": [False, False],
            "lineup_role": [pd.NA, pd.NA],
            "lineup_status": [pd.NA, pd.NA],
            "lineup_roster_status": [pd.NA, pd.NA],
        }
    )

    meta, out = _backfill_lineups_from_silver_daily_lineups(base, data_root=data_root)
    assert meta["rows_lineup_filled"] == 2

    by_pid = out.set_index("player_id")
    assert bool(by_pid.loc[101, "is_projected_starter"]) is False
    assert bool(by_pid.loc[102, "is_projected_starter"]) is True
    assert str(by_pid.loc[101, "lineup_role"]).lower() == "bench"
    assert str(by_pid.loc[102, "lineup_role"]).lower() == "confirmed_starter"
