from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pandas as pd

from projections.minutes.depth_chart_prior import apply_depth_chart_prior_from_realgm


def _write_depth_inputs(data_root: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    snap_dir = data_root / "bronze" / "realgm"
    snap_dir.mkdir(parents=True, exist_ok=True)

    t0 = pd.Timestamp("2026-01-18T18:00:00Z")
    t1 = pd.Timestamp("2026-01-18T19:00:00Z")

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
                "scraped_at": t0,
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
                "scraped_at": t0,
            },
            {
                "team_name": "New York Knicks",
                "player_name": "Charlie Deep",
                "realgm_player_id": 1003,
                "position": "SG",
                "depth_role": "starter",
                "depth_order": 1,
                "recent_stats": "",
                "movement": "",
                "scraped_at": t0,
            },
            {
                "team_name": "New York Knicks",
                "player_name": "Jalen Brunson",
                "realgm_player_id": 1001,
                "position": "PG",
                "depth_role": "starter",
                "depth_order": 0,
                "recent_stats": "",
                "movement": "",
                "scraped_at": t1,
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
                "scraped_at": t1,
            },
            {
                "team_name": "New York Knicks",
                "player_name": "Charlie Deep",
                "realgm_player_id": 1003,
                "position": "SG",
                "depth_role": "starter",
                "depth_order": 1,
                "recent_stats": "",
                "movement": "",
                "scraped_at": t1,
            },
        ]
    )
    snapshot.to_parquet(snap_dir / "depth_charts.parquet", index=False)

    crosswalk = pd.DataFrame(
        [
            {"realgm_player_id": 1001, "player_id": 1, "updated_at": "2026-01-17T00:00:00Z"},
            {"realgm_player_id": 1002, "player_id": 2, "updated_at": "2026-01-17T00:00:00Z"},
            {"realgm_player_id": 1003, "player_id": 3, "updated_at": "2026-01-17T00:00:00Z"},
        ]
    )
    crosswalk.to_parquet(snap_dir / "player_id_crosswalk.parquet", index=False)
    return t0, t1


def _base_minutes_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": 10,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "player_id": 1,
                "player_name": "Jalen Brunson",
                "status": "active",
                "play_prob": 0.92,
                "rotation_prob": 0.88,
                "minutes_p10": 24.0,
                "minutes_p50": 34.0,
                "minutes_p90": 40.0,
            },
            {
                "game_id": 10,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "player_id": 2,
                "player_name": "Miles McBride",
                "status": "active",
                "play_prob": 0.60,
                "rotation_prob": 0.55,
                "minutes_p10": 2.0,
                "minutes_p50": 8.0,
                "minutes_p90": 40.0,
            },
            {
                "game_id": 10,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "player_id": 3,
                "player_name": "Charlie Deep",
                "status": "out",
                "play_prob": 0.0,
                "rotation_prob": 0.0,
                "minutes_p10": 1.0,
                "minutes_p50": 12.0,
                "minutes_p90": 28.0,
            },
            {
                "game_id": 10,
                "team_id": 1610612752,
                "team_name": "New York Knicks",
                "player_id": 4,
                "player_name": "Unlisted Guy",
                "status": "active",
                "play_prob": 0.25,
                "rotation_prob": 0.20,
                "minutes_p10": 0.0,
                "minutes_p50": 2.0,
                "minutes_p90": 12.0,
            },
        ]
    )


def test_depth_chart_prior_selects_latest_snapshot_le_as_of(tmp_path: Path) -> None:
    data_root = tmp_path
    t0, t1 = _write_depth_inputs(data_root)
    as_of = pd.Timestamp("2026-01-18T18:30:00Z")
    assert t0 < as_of < t1

    result = apply_depth_chart_prior_from_realgm(
        _base_minutes_frame(),
        data_root=data_root,
        as_of_ts=as_of,
    )
    out = result.frame
    diag = result.diagnostics

    assert diag["applied"] is True
    assert diag["dc_snapshot_ts"] == t0.isoformat().replace("+00:00", "Z")
    assert diag["matched_id"] == 3
    assert {"dc_present", "dc_role", "dc_role_priority", "dc_order_in_role", "dc_ahead_global", "dc_is_primary_backup", "dc_snapshot_ts"} <= set(
        out.columns
    )

    mcbride = out.loc[out["player_id"] == 2].iloc[0]
    assert str(mcbride["dc_role"]) == "rotation"
    assert bool(mcbride["dc_is_primary_backup"]) is True


def test_depth_chart_prior_caps_and_preserves_inactive_zero(tmp_path: Path) -> None:
    data_root = tmp_path
    _t0, t1 = _write_depth_inputs(data_root)

    result = apply_depth_chart_prior_from_realgm(
        _base_minutes_frame(),
        data_root=data_root,
        as_of_ts=pd.Timestamp("2026-01-18T20:00:00Z", tz=UTC),
    )
    out = result.frame

    mcbride = out.loc[out["player_id"] == 2].iloc[0]
    assert str(mcbride["dc_role"]) == "limited"
    assert float(mcbride["minutes_p90"]) <= 22.0 + 1e-9
    assert float(mcbride["minutes_p10"]) <= float(mcbride["minutes_p50"]) <= float(mcbride["minutes_p90"])

    deep = out.loc[out["player_id"] == 3].iloc[0]
    assert str(deep["dc_snapshot_ts"]).startswith("2026-01-18 19:00:00")
    assert float(deep["play_prob"]) == 0.0
    assert float(deep["rotation_prob"]) == 0.0
    assert float(deep["minutes_p10"]) == 0.0
    assert float(deep["minutes_p50"]) == 0.0
    assert float(deep["minutes_p90"]) == 0.0
    assert result.diagnostics["dc_snapshot_ts"] == t1.isoformat().replace("+00:00", "Z")
