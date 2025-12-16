from __future__ import annotations

import pandas as pd
import pytest

from projections.minutes_v1.horizons import TipTimeSplit, assign_game_splits, build_horizon_rows


def test_build_horizon_rows_selects_latest_snapshot_before_cutoff() -> None:
    tip_ts = pd.Timestamp("2025-01-01T20:00:00Z")
    df = pd.DataFrame(
        {
            "game_id": [1] * 5,
            "player_id": [10] * 5,
            "team_id": [100] * 5,
            "tip_ts": [tip_ts] * 5,
            "feature_as_of_ts": [
                "2025-01-01T14:00:00Z",
                "2025-01-01T18:40:00Z",
                "2025-01-01T19:10:00Z",
                "2025-01-01T19:40:00Z",
                "2025-01-01T19:59:00Z",
            ],
            "value": [1, 2, 3, 4, 5],
        }
    )

    out = build_horizon_rows(df, horizons_minutes=[60, 30, 0], max_snapshot_age_hours=None)
    # horizon 60 -> cutoff 19:00, selects 18:40 (value=2)
    # horizon 30 -> cutoff 19:30, selects 19:10 (value=3)
    # horizon 0  -> cutoff 20:00, selects 19:59 (value=5)
    got = {int(r.horizon_min): int(r.value) for r in out.itertuples(index=False)}
    assert got == {60: 2, 30: 3, 0: 5}


def test_build_horizon_rows_applies_snapshot_age_guard() -> None:
    tip_ts = pd.Timestamp("2025-01-01T20:00:00Z")
    df = pd.DataFrame(
        {
            "game_id": [1] * 3,
            "player_id": [10] * 3,
            "team_id": [100] * 3,
            "tip_ts": [tip_ts] * 3,
            "feature_as_of_ts": [
                "2025-01-01T18:40:00Z",  # 80m before tip
                "2025-01-01T19:10:00Z",  # 50m before tip
                "2025-01-01T19:59:00Z",  # 1m before tip
            ],
            "value": [2, 3, 5],
        }
    )

    out = build_horizon_rows(df, horizons_minutes=[60, 30, 0], max_snapshot_age_hours=1.0)
    got_horizons = sorted(int(x) for x in out["horizon_min"].tolist())
    # horizon 60 would select 18:40 but that violates the 1h freshness guard (tip-1h = 19:00)
    assert got_horizons == [0, 30]


def test_build_horizon_rows_breaks_ties_by_ingested_ts() -> None:
    tip_ts = pd.Timestamp("2025-01-01T20:00:00Z")
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "player_id": [10, 10],
            "team_id": [100, 100],
            "tip_ts": [tip_ts, tip_ts],
            "feature_as_of_ts": ["2025-01-01T19:10:00Z", "2025-01-01T19:10:00Z"],
            "ingested_ts": ["2025-01-01T19:11:00Z", "2025-01-01T19:12:00Z"],
            "value": [1, 2],
        }
    )

    out = build_horizon_rows(df, horizons_minutes=[30], max_snapshot_age_hours=None)
    assert out["value"].tolist() == [2]


def test_assign_game_splits_groups_all_rows_for_game() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "tip_ts": [
                "2025-01-01T20:00:00Z",
                "2025-01-01T20:00:00Z",
                "2025-01-02T20:00:00Z",
                "2025-01-02T20:00:00Z",
            ],
            "horizon_min": [60, 0, 60, 0],
        }
    )
    split = TipTimeSplit.from_bounds(
        train_end="2025-01-01T23:59:59Z",
        cal_end="2025-01-02T23:59:59Z",
        val_end="2025-01-03T23:59:59Z",
    )
    out = assign_game_splits(df, split)
    by_game = out.groupby("game_id")["split"].nunique().to_dict()
    assert by_game == {1: 1, 2: 1}
    assert out.loc[out["game_id"] == 1, "split"].unique().tolist() == ["train"]
    assert out.loc[out["game_id"] == 2, "split"].unique().tolist() == ["cal"]


def test_assign_game_splits_raises_on_mismatched_tip_ts() -> None:
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "tip_ts": ["2025-01-01T20:00:00Z", "2025-01-01T20:01:00Z"],
        }
    )
    split = TipTimeSplit.from_bounds(
        train_end="2025-01-01T23:59:59Z",
        cal_end="2025-01-02T23:59:59Z",
        val_end="2025-01-03T23:59:59Z",
    )
    with pytest.raises(ValueError, match="multiple tip_ts"):
        assign_game_splits(df, split)

