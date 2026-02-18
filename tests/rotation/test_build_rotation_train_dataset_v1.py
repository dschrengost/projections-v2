from __future__ import annotations

import json

import pandas as pd

from projections.features.action_props import ACTION_MARKET_FEATURE_COLUMNS
from scripts.rotation.build_rotation_train_dataset_v1 import (
    PLAYER_MINUTES_FROM_STINTS_COL,
    TEAM_TOTAL_MINUTES_COL,
    _attach_action_props_training_features,
    _align_labels_to_features,
    _discover_action_props_days,
    _filter_invalid_team_games,
)


def test_builder_drops_incomplete_team_games_and_fills_dnp_labels() -> None:
    # Team-game A: complete coverage; one DNP row missing label -> should be kept and filled to 0.
    features = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 2, 2],
            "team_id": [10, 10, 10, 10, 20, 20],
            "player_id": [101, 102, 103, 104, 201, 202],
            TEAM_TOTAL_MINUTES_COL: [240.0, 240.0, 240.0, 240.0, 240.0, 240.0],
            PLAYER_MINUTES_FROM_STINTS_COL: [30.0, 20.0, 190.0, 0.0, 40.0, 60.0],
            "rotation_team_missing": [0, 0, 0, 0, 0, 0],
        }
    )
    labels = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2],
            "team_id": [10, 10, 10, 20, 20],
            "player_id": [101, 102, 103, 201, 202],
            "minutes": [30.0, 20.0, 190.0, 40.0, 60.0],
        }
    )

    aligned = _align_labels_to_features(features, labels)
    filtered_features, filtered_labels, meta = _filter_invalid_team_games(
        features,
        aligned,
        label_col="minutes",
        min_team_minutes_from_stints=200.0,
        max_team_minutes_gap=2.0,
    )

    # Team-game B should be dropped due to missing player coverage (only 100 minutes represented).
    assert meta["team_games_total"] == 2
    assert meta["team_games_kept"] == 1
    assert meta["team_games_dropped_by_reason"]["missing_player_coverage"] == 1

    assert filtered_features["team_id"].nunique() == 1
    assert int(filtered_features["team_id"].iloc[0]) == 10
    assert len(filtered_features) == 4

    # DNP row (player_id=104) should have minutes filled to 0.
    dnp_row = filtered_labels.loc[filtered_labels["player_id"] == 104, "minutes"]
    assert len(dnp_row) == 1
    assert float(dnp_row.iloc[0]) == 0.0

    # Kept team-game should have full team total minutes.
    assert abs(float(filtered_labels["minutes"].sum()) - 240.0) <= 1e-6


def test_discover_action_props_days(tmp_path) -> None:
    (tmp_path / "2025-01-01_123_NYK_BOS.json").write_text("{}", encoding="utf-8")
    (tmp_path / "not_a_day.json").write_text("{}", encoding="utf-8")
    days = _discover_action_props_days(tmp_path)
    assert days == {"2025-01-01"}


def test_attach_action_props_training_features(tmp_path) -> None:
    day = "2025-01-01"
    payload = {
        "game_id": 123456,
        "teams": ["NY", "BOS"],
        "away_team_id": 1,
        "home_team_id": 2,
        # Training join enforces strict as-of, so fetched_at must be pre-tip.
        "fetched_at": "2025-01-01T20:00:00Z",
        "props": {
            "players": {
                "10": {
                    "full_name": "Jalen Brunson",
                    "display_text": "NY - PG",
                    "team_id": 1,
                }
            },
            "player_props": {
                "points": [
                    {
                        "player_id": "10",
                        "custom_pick_type_name": "Points",
                        "lines": {
                            "15": [
                                {"period": "event", "side": "over", "odds": -110, "value": 25.5},
                                {"period": "event", "side": "under", "odds": -110, "value": 25.5},
                            ]
                        },
                    }
                ]
            },
        },
    }
    props_dir = tmp_path / "bronze" / "action_network" / "props"
    props_dir.mkdir(parents=True, exist_ok=True)
    (props_dir / f"{day}_123456_NY_BOS.json").write_text(json.dumps(payload), encoding="utf-8")

    base = pd.DataFrame(
        {
            "game_id": [123456, 123456],
            "team_id": [1, 2],
            "player_id": [10, 20],
            "game_date": [pd.Timestamp(day), pd.Timestamp(day)],
            "team_tricode": ["NYK", "BOS"],
            "player_name": ["Jalen Brunson", "Jaylen Brown"],
            "feature_as_of_ts": [pd.Timestamp("2025-01-01T21:00:00Z"), pd.Timestamp("2025-01-01T21:00:00Z")],
            "tip_ts": [pd.Timestamp("2025-01-01T22:00:00Z"), pd.Timestamp("2025-01-01T22:00:00Z")],
            "spread_home": [-2.5, -2.5],
            "total": [225.5, 225.5],
        }
    )

    meta, out = _attach_action_props_training_features(
        base,
        data_root=tmp_path,
        enabled=True,
        props_dir=props_dir,
    )

    assert meta["enabled"] is True
    assert meta["snapshot_rows"] >= 1
    assert meta["matched_rows"] == 1
    assert int(out["an_has_any_props"].sum()) == 1
    matched = out.loc[out["player_name"] == "Jalen Brunson"].iloc[0]
    assert float(matched["an_pts_line"]) == 25.5
    unmatched = out.loc[out["player_name"] == "Jaylen Brown"].iloc[0]
    assert int(unmatched["an_has_any_props"]) == 0
    for col in ACTION_MARKET_FEATURE_COLUMNS:
        assert col in out.columns
