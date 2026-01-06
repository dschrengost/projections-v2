import pandas as pd

from projections.cli.score_minutes_rotation_set_v1 import _load_rotation_historical_features_for_dnp


def test_load_rotation_historical_features_for_dnp_uses_latest_run_and_filters(tmp_path) -> None:
    data_root = tmp_path
    season_dir = data_root / "gold" / "prediction_logs_minutes" / "season=2025" / "month=12"
    season_dir.mkdir(parents=True)

    # Two runs for the same day; the loader should pick the lexicographically-latest run id.
    early = pd.DataFrame(
        [
            {
                "game_date": "2025-12-10",
                "team_id": 1,
                "player_id": 10,
                "minutes": 5.0,
                "is_out": 0,
                "status": "AVAIL",
            },
            {
                "game_date": "2025-12-10",
                "team_id": 1,
                "player_id": 11,
                "minutes": 7.0,
                "is_out": 0,
                "status": "AVAIL",
            },
        ]
    )
    late = pd.DataFrame(
        [
            {
                "game_date": "2025-12-10",
                "team_id": 1,
                "player_id": 10,
                "minutes": 12.0,
                "is_out": 0,
                "status": "AVAIL",
            },
        ]
    )

    early.to_parquet(season_dir / "2025-12-10_20260101T000000Z.parquet", index=False)
    late.to_parquet(season_dir / "2025-12-10_20260101T010000Z.parquet", index=False)

    hist = _load_rotation_historical_features_for_dnp(
        data_root,
        season=2025,
        target_day=pd.Timestamp("2025-12-12"),
        team_ids={1},
        player_ids={10},
        lookback_days=10,
    )

    assert len(hist) == 1
    row = hist.iloc[0].to_dict()
    assert str(pd.Timestamp(row["game_date"]).date()) == "2025-12-10"
    assert int(row["team_id"]) == 1
    assert int(row["player_id"]) == 10
    assert float(row["minutes"]) == 12.0
    assert int(row["is_out"]) == 0

