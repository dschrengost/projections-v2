import pandas as pd

from projections.cli.score_minutes_rotation_set_v1 import _load_rotation_historical_features_for_dnp


def test_load_rotation_historical_features_for_dnp_uses_latest_run_and_filters(tmp_path) -> None:
    data_root = tmp_path
    injuries_dir = data_root / "bronze" / "injuries_raw" / "season=2025" / "date=2025-12-10"
    injuries_dir.mkdir(parents=True)
    labels_dir = data_root / "labels" / "season=2025"
    labels_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"team_id": 1, "player_id": 11, "status": "OUT"},
        ]
    ).to_parquet(injuries_dir / "injuries.parquet", index=False)

    pd.DataFrame(
        [
            {"game_date": "2025-12-10", "team_id": 1, "player_id": 10, "minutes": 12.0},
            {"game_date": "2025-12-10", "team_id": 1, "player_id": 11, "minutes": 0.0},
            {"game_date": "2025-12-10", "team_id": 2, "player_id": 10, "minutes": 30.0},
        ]
    ).to_parquet(labels_dir / "boxscore_labels.parquet", index=False)

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

    hist_out = _load_rotation_historical_features_for_dnp(
        data_root,
        season=2025,
        target_day=pd.Timestamp("2025-12-12"),
        team_ids={1},
        player_ids={11},
        lookback_days=10,
    )
    assert len(hist_out) == 1
    row_out = hist_out.iloc[0].to_dict()
    assert int(row_out["is_out"]) == 1
