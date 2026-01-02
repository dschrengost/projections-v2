from __future__ import annotations

from pathlib import Path

import pandas as pd

from projections.rates_v1.training_base_schema import validate_rates_training_base
from scripts.rates import build_training_base as rtb


def test_attach_tip_and_asof_and_write_partition(tmp_path: Path) -> None:
    schedule = pd.DataFrame(
        {
            "game_id": [1, 2],
            "game_date": [pd.Timestamp("2024-10-22"), pd.Timestamp("2024-10-23")],
            "tip_ts": [
                pd.Timestamp("2024-10-22T23:30:00Z"),
                pd.Timestamp("2024-10-23T02:00:00Z"),
            ],
            "home_team_id": [10, 30],
            "away_team_id": [20, 40],
        }
    )

    # Minimal stats-like rows; tip_ts intentionally missing so schedule join is required.
    stats = pd.DataFrame(
        {
            "season": [2024, 2024],
            "game_id": [1, 2],
            "player_id": [100, 200],
            "team_id": [10, 30],
            "opponent_id": [20, 40],
            "home_flag": [1, 1],
            "game_date": [pd.Timestamp("2024-10-22"), pd.Timestamp("2024-10-23")],
            "tip_ts": [pd.NaT, pd.NaT],
        }
    )

    merged = rtb._attach_schedule_tip_ts(stats, schedule)
    out = rtb._attach_feature_as_of_ts(merged, asof_minutes_before_tip=30)

    assert "tip_ts" in out.columns
    assert "feature_as_of_ts" in out.columns
    assert (pd.to_datetime(out["feature_as_of_ts"], utc=True) < pd.to_datetime(out["tip_ts"], utc=True)).all()

    # feature_as_of_ts should be exactly tip_ts - 30 minutes.
    tip = pd.to_datetime(out["tip_ts"], utc=True)
    asof = pd.to_datetime(out["feature_as_of_ts"], utc=True)
    assert (asof == (tip - pd.Timedelta(minutes=30))).all()

    # Validate schema guard catches missing timestamps.
    missing = validate_rates_training_base(out.drop(columns=["feature_as_of_ts"]), strict=False)
    assert "feature_as_of_ts" in missing

    # Ensure parquet write keeps timestamp columns.
    # Create a minimal training-base-like frame with required columns to write.
    train_like = pd.DataFrame(
        {
            "season": [2024],
            "game_id": [1],
            "game_date": [pd.Timestamp("2024-10-22")],
            "tip_ts": [pd.Timestamp("2024-10-22T23:30:00Z")],
            "feature_as_of_ts": [pd.Timestamp("2024-10-22T23:00:00Z")],
            "team_id": [10],
            "opponent_id": [20],
            "home_flag": [1],
            "player_id": [100],
            "minutes_actual": [12.0],
            "fga2_per_min": [0.1],
            "fga3_per_min": [0.05],
            "fta_per_min": [0.02],
            "ast_per_min": [0.03],
            "tov_per_min": [0.01],
            "oreb_per_min": [0.02],
            "dreb_per_min": [0.05],
            "stl_per_min": [0.01],
            "blk_per_min": [0.01],
            "fg2_pct_label": [0.5],
            "fg3_pct_label": [0.35],
            "ft_pct_label": [0.8],
        }
    )
    out_root = tmp_path / "gold" / "rates_training_base"
    rtb._write_partitions(train_like, out_root)
    pq = out_root / "season=2024" / "game_date=2024-10-22" / "rates_training_base.parquet"
    reloaded = pd.read_parquet(pq)
    assert "tip_ts" in reloaded.columns
    assert "feature_as_of_ts" in reloaded.columns
