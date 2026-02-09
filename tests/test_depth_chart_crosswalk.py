from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.minutes.depth_chart_crosswalk import refresh_realgm_player_crosswalk_from_minutes


def _seed_depth_snapshot(data_root: Path) -> None:
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
                "depth_role": "rotation",
                "depth_order": 0,
                "recent_stats": "",
                "movement": "",
                "scraped_at": "2026-01-18T18:15:00Z",
            },
        ]
    )
    snapshot.to_parquet(realgm_dir / "depth_charts.parquet", index=False)


def test_refresh_crosswalk_from_minutes_builds_mapping_and_applies_overrides(tmp_path: Path) -> None:
    data_root = tmp_path
    _seed_depth_snapshot(data_root)

    # Override remaps realgm_player_id=1002 to player_id=22.
    overrides = pd.DataFrame(
        [{"realgm_player_id": 1002, "player_id": 22, "note": "manual"}]
    )
    overrides_path = data_root / "bronze" / "realgm" / "player_id_crosswalk_overrides.csv"
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_csv(overrides_path, index=False)

    minutes = pd.DataFrame(
        [
            {
                "game_id": 10,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "player_id": 1,
                "player_name": "Jalen Brunson",
            },
            {
                "game_id": 10,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "player_id": 2,
                "player_name": "Miles McBride",
            },
        ]
    )

    diag = refresh_realgm_player_crosswalk_from_minutes(
        minutes,
        data_root=data_root,
        as_of_ts=pd.Timestamp("2026-01-18T18:30:00Z"),
    )
    assert diag["applied"] is True
    assert int(diag["matched_rows"]) == 2
    assert int(diag["override_rows"]) == 1
    assert int(diag["rows_written"]) == 2

    crosswalk_path = data_root / "bronze" / "realgm" / "player_id_crosswalk.parquet"
    assert crosswalk_path.exists()
    cw = pd.read_parquet(crosswalk_path)
    got = dict(zip(cw["realgm_player_id"].astype(int), cw["player_id"].astype(int)))
    assert got[1001] == 1
    assert got[1002] == 22  # override wins
