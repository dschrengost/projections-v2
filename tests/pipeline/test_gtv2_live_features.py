from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from projections.rotation.game_transformer_v2 import GameTransformerV2Config
from projections.pipeline.gtv2_live_features import (
    GTV2FeatureSpec,
    build_gtv2_live_features,
    load_gtv2_feature_spec,
)
from projections.rotation.live_features_v1 import load_latest_rotation_priors_by_entity


def _minutes_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1001, 1001, 1001, 1001],
            "team_id": [10, 10, 20, 20],
            "player_id": [1, 2, 3, 4],
            "game_date": ["2026-01-18"] * 4,
            "lineup_timestamp": ["2026-01-18T17:00:00Z"] * 4,
            "lineup_role": ["projected_starter", "bench", "confirmed_starter", "bench"],
            # Bench rows can still carry lineup_status freshness labels; this must
            # not be treated as a starter signal by contract transforms.
            "lineup_status": ["expected", "confirmed", "confirmed", "confirmed"],
            "is_projected_starter": [1, 0, 1, 0],
            "spread_home": [-2.5, -2.5, -2.5, -2.5],
            "total": [228.5, 228.5, 228.5, 228.5],
            "team_pace_szn": [99.0, 99.0, 98.0, 98.0],
            "opp_pace_szn": [98.0, 98.0, 99.0, 99.0],
            "injury_snapshot_missing": [0, 0, 0, 0],
            "is_out": [0, 0, 0, 0],
            "home_team_id": [10, 10, 10, 10],
            "away_team_id": [20, 20, 20, 20],
            "home_flag": [1, 1, 0, 0],
            "opponent_team_id": [20, 20, 10, 10],
            "restriction_flag": [0, 0, 0, 0],
            "ramp_flag": [0, 0, 0, 0],
            "prior_play_prob": [0.95, 0.90, 0.94, 0.89],
            "an_has_any_props": [1, 0, 1, 0],
            "minutes_features_row_missing": [0, 0, 0, 0],
        }
    )


def test_load_gtv2_feature_spec_from_bundle_config(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    cfg = GameTransformerV2Config(
        feature_columns=["home_team_id", "lineup_available", "vegas_total"],
        feature_mean=[0.0, 0.0, 0.0],
        feature_std=[1.0, 1.0, 1.0],
        game_feature_columns=["vegas_total"],
        team_feature_columns=[],
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    spec = load_gtv2_feature_spec(bundle_dir)
    assert spec.feature_columns == ["home_team_id", "lineup_available", "vegas_total"]
    assert spec.game_feature_columns == ["vegas_total"]


def test_build_gtv2_live_features_applies_training_contract_and_orders_columns() -> None:
    minutes = _minutes_df()
    spec = GTV2FeatureSpec(
        bundle_dir=Path("/tmp/fake_bundle"),
        feature_columns=[
            "home_team_id",
            "away_team_id",
            "home_flag",
            "opponent_team_id",
            "lineup_available",
            "lineup_starter_announced",
            "vegas_total",
            "vegas_spread",
            "estimated_possessions",
            "vegas_total_missing",
            "vegas_spread_missing",
            "estimated_possessions_missing",
        ],
        game_feature_columns=[
            "vegas_total",
            "vegas_spread",
            "estimated_possessions",
            "vegas_total_missing",
            "vegas_spread_missing",
            "estimated_possessions_missing",
        ],
        team_feature_columns=[],
    )

    def fake_priors_loader(*args, **kwargs):  # noqa: ANN002, ANN003
        return SimpleNamespace(
            team_priors=pd.DataFrame({"game_id": ["0000001001"], "team_id": [10]}),
            player_priors=pd.DataFrame({"game_id": ["0000001001"], "team_id": [10], "person_id": [1]}),
            used_latest_fallback=False,
            warning_message=None,
        )

    def fake_dnp_loader(*args, **kwargs):  # noqa: ANN002, ANN003
        return pd.DataFrame(columns=["game_date", "team_id", "player_id", "minutes", "is_out"])

    def fake_rotation_builder(
        minutes_features: pd.DataFrame,
        *,
        team_priors: pd.DataFrame,
        player_priors: pd.DataFrame,
        feature_columns: list[str],
        historical_features: pd.DataFrame,
    ):
        assert not team_priors.empty
        assert not player_priors.empty
        assert isinstance(historical_features, pd.DataFrame)
        # Validate that contract columns are already present before projection.
        assert "lineup_available" in minutes_features.columns
        assert "lineup_starter_announced" in minutes_features.columns
        assert "vegas_total" in minutes_features.columns
        assert "vegas_spread" in minutes_features.columns
        assert "estimated_possessions" in minutes_features.columns

        out = minutes_features[["game_id", "team_id", "player_id"]].copy()
        for col in feature_columns:
            if col in minutes_features.columns:
                out[col] = minutes_features[col]
            else:
                out[col] = 0.0
        return SimpleNamespace(features=out, dropped_extra_columns=[])

    result = build_gtv2_live_features(
        minutes_features=minutes,
        spec=spec,
        data_root=Path("/tmp"),
        game_date="2026-01-18",
        priors_loader=fake_priors_loader,
        rotation_feature_builder=fake_rotation_builder,
        dnp_history_loader=fake_dnp_loader,
    )

    expected = ["game_id", "team_id", "player_id", *spec.feature_columns]
    assert list(result.features.columns) == expected
    assert len(result.features) == 4
    assert result.features["lineup_available"].isin([0, 1]).all()
    assert result.features["lineup_starter_announced"].isin([0, 1]).all()
    assert int(result.features["lineup_starter_announced"].sum()) == 2
    assert "game_context_contract" in result.transform_manifest
    assert result.transform_manifest["dnp_history"]["mode"] == "full_prior_history"
    assert result.transform_manifest["dnp_history"]["lookback_days"] is None


def test_load_latest_rotation_priors_by_entity_uses_concat_path_and_latest_rows(tmp_path: Path) -> None:
    data_root = tmp_path
    team_root = data_root / "silver" / "rotation_priors_v1" / "team_game_priors" / "season=2025"
    player_root = data_root / "silver" / "rotation_priors_v1" / "player_game_priors" / "season=2025"
    team_root.mkdir(parents=True, exist_ok=True)
    player_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"team_id": 10, "game_date": "2026-01-10", "team_metric": 1.0},
            {"team_id": 20, "game_date": "2026-01-10", "team_metric": 2.0},
        ]
    ).to_parquet(team_root / "game_id=0022500001.parquet", index=False)
    pd.DataFrame(
        [
            {"team_id": 10, "game_date": "2026-01-12", "team_metric": 3.0},
            {"team_id": 30, "game_date": "2026-01-12", "team_metric": 4.0},
        ]
    ).to_parquet(team_root / "game_id=0022500002.parquet", index=False)
    pd.DataFrame(
        [
            {"team_id": 10, "game_date": "2026-02-14", "team_metric": 999.0},
        ]
    ).to_parquet(team_root / "game_id=0062500001.parquet", index=False)

    pd.DataFrame(
        [
            {"person_id": 1, "game_date": "2026-01-10", "player_metric": 11.0},
            {"person_id": 2, "game_date": "2026-01-10", "player_metric": 12.0},
        ]
    ).to_parquet(player_root / "game_id=0022500001.parquet", index=False)
    pd.DataFrame(
        [
            {"person_id": 1, "game_date": "2026-01-12", "player_metric": 13.0},
            {"person_id": 3, "game_date": "2026-01-12", "player_metric": 14.0},
        ]
    ).to_parquet(player_root / "game_id=0022500002.parquet", index=False)

    team_priors, player_priors = load_latest_rotation_priors_by_entity(
        data_root,
        season=2025,
        team_ids=[10, 20],
        player_ids=[1, 2],
    )

    assert sorted(team_priors["team_id"].dropna().astype(int).tolist()) == [10, 20]
    assert float(team_priors.loc[team_priors["team_id"] == 10, "team_metric"].iloc[0]) == 3.0
    assert float(team_priors.loc[team_priors["team_id"] == 20, "team_metric"].iloc[0]) == 2.0

    assert sorted(player_priors["person_id"].dropna().astype(int).tolist()) == [1, 2]
    assert float(player_priors.loc[player_priors["person_id"] == 1, "player_metric"].iloc[0]) == 13.0
    assert float(player_priors.loc[player_priors["person_id"] == 2, "player_metric"].iloc[0]) == 12.0
