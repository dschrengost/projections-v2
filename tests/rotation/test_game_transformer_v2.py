from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projections.rotation.game_transformer_v2 import (
    MAX_PLAYERS_PER_TEAM,
    GameTransformerV2Config,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
)


def _toy_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    game_id = 1001
    game_date = "2026-01-18"
    home_team_id = 10
    away_team_id = 20

    # Home players
    for i, pid in enumerate([101, 102, 103, 104, 105, 106], start=1):
        rows.append(
            {
                "game_id": game_id,
                "team_id": home_team_id,
                "player_id": pid,
                "game_date": game_date,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_flag": 1,
                "lineup_starter_announced": 1 if i <= 5 else 0,
                "lineup_available": 1,
                "prior_play_prob": 0.9 - i * 0.1,
                "minutes_from_stints_prior_20": 24 - i,
                "vegas_total": 231.5,
                "vegas_spread": -3.5,
                "estimated_possessions": 101.0,
                "f1": float(i),
                "f2": float(i) * 0.1,
                "minutes_label": float(32 - i),
            }
        )

    # Away players
    for i, pid in enumerate([201, 202, 203, 204, 205, 206], start=1):
        rows.append(
            {
                "game_id": game_id,
                "team_id": away_team_id,
                "player_id": pid,
                "game_date": game_date,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_flag": 0,
                "lineup_starter_announced": 1 if i <= 5 else 0,
                "lineup_available": 0,
                "prior_play_prob": 0.85 - i * 0.1,
                "minutes_from_stints_prior_20": 22 - i,
                "vegas_total": 231.5,
                "vegas_spread": -3.5,
                "estimated_possessions": 101.0,
                "f1": float(i + 3),
                "f2": float(i) * 0.2,
                "minutes_label": float(30 - i),
            }
        )

    return pd.DataFrame(rows)


def test_build_game_level_examples_and_collate_shapes() -> None:
    df = _toy_frame()
    feature_columns = ["f1", "f2"]
    mean = np.array([0.0, 0.0], dtype=np.float32)
    std = np.array([1.0, 1.0], dtype=np.float32)

    examples = build_game_level_examples(
        df,
        feature_columns=feature_columns,
        feature_mean=mean,
        feature_std=std,
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        minutes_label_col="minutes_label",
    )
    assert len(examples) == 1

    ex = examples[0]
    assert ex.player_features.shape == (2, MAX_PLAYERS_PER_TEAM, len(feature_columns))
    assert ex.player_valid_mask.shape == (2, MAX_PLAYERS_PER_TEAM)
    assert int(ex.player_valid_mask.sum()) == 12

    batch = collate_game_level_examples([ex])
    assert batch["player_features"].shape == (1, 2, MAX_PLAYERS_PER_TEAM, len(feature_columns))
    assert batch["player_valid_mask"].shape == (1, 2, MAX_PLAYERS_PER_TEAM)


def test_game_transformer_v2_forward_shapes_and_team_minute_constraints() -> None:
    df = _toy_frame()
    feature_columns = ["f1", "f2"]
    config = GameTransformerV2Config(
        feature_columns=feature_columns,
        feature_mean=[0.0, 0.0],
        feature_std=[1.0, 1.0],
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        d_model=48,
        hidden_dim=64,
        num_layers=1,
        num_heads=6,
        dropout=0.0,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    examples = build_game_level_examples(
        df,
        feature_columns=feature_columns,
        feature_mean=np.array(config.feature_mean, dtype=np.float32),
        feature_std=np.array(config.feature_std, dtype=np.float32),
        game_feature_columns=config.game_feature_columns,
        team_feature_columns=config.team_feature_columns,
        minutes_label_col="minutes_label",
    )
    batch = collate_game_level_examples(examples)

    out = model(
        batch["player_features"],
        batch["player_valid_mask"],
        game_features=batch["game_features"],
        team_features=batch["team_features"],
        sample_active=False,
    )

    assert out.game_state.shape == (1, config.d_model)
    assert out.team_states.shape == (1, 2, config.d_model)
    assert out.player_states.shape == (1, 30, config.d_model)
    assert out.active.count_logits.shape == (1, 2, config.max_active_count - config.min_active_count + 1)
    assert out.minutes.minutes.shape == (1, 30)
    assert out.flow is None

    minutes = out.minutes.minutes[0]
    valid = out.player_valid_mask[0]
    team_idx = out.player_team_index[0]

    for team in (0, 1):
        mask = valid & (team_idx == team)
        total = float(minutes[mask].sum().item())
        assert total == pytest.approx(240.0, abs=1e-3)

    assert float(minutes[~valid].abs().max().item()) == pytest.approx(0.0, abs=1e-6)


def test_game_transformer_v2_config_defaults_match_locked_decisions() -> None:
    config = GameTransformerV2Config(
        feature_columns=["f1"],
        feature_mean=[0.0],
        feature_std=[1.0],
        game_feature_columns=["vegas_total"],
        team_feature_columns=[],
    )
    assert config.active_threshold_minutes == pytest.approx(4.0)
    assert config.flow_coupling_type == "affine"
    assert config.flow_num_blocks == 4
    assert config.flow_scale_clip == pytest.approx(2.0)
    assert config.include_pf_in_flow_targets is False


def test_game_transformer_v2_forward_with_flow_targets_returns_flow_outputs() -> None:
    df = _toy_frame()
    config = GameTransformerV2Config(
        feature_columns=["f1", "f2"],
        feature_mean=[0.0, 0.0],
        feature_std=[1.0, 1.0],
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        d_model=48,
        hidden_dim=64,
        num_layers=1,
        num_heads=6,
        dropout=0.0,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    flow_cols = ["fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov"]
    for idx, col in enumerate(flow_cols):
        df[col] = float(idx + 1)

    examples = build_game_level_examples(
        df,
        feature_columns=["f1", "f2"],
        feature_mean=np.array(config.feature_mean, dtype=np.float32),
        feature_std=np.array(config.feature_std, dtype=np.float32),
        game_feature_columns=config.game_feature_columns,
        team_feature_columns=config.team_feature_columns,
        flow_label_columns=flow_cols,
        minutes_label_col="minutes_label",
    )
    batch = collate_game_level_examples(examples)
    out = model(
        batch["player_features"],
        batch["player_valid_mask"],
        game_features=batch["game_features"],
        team_features=batch["team_features"],
        run_flow=True,
        flow_targets=batch["flow_targets"],
        flow_observed_mask=batch["flow_observed_mask"],
    )
    assert out.flow is not None
    assert out.flow.z.shape == (1, 30, len(flow_cols))
    assert out.flow.nll_mean.item() > 0.0
