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
    assert config.overflow_protected_prior_play_prob_floor == pytest.approx(0.938507)
    assert config.overflow_protected_prior_minutes_floor == pytest.approx(29.520922)
    assert config.overflow_risk_weight_consecutive_active_dnp == pytest.approx(0.579943)
    assert config.overflow_risk_weight_active_but_dnp_rate_last10 == pytest.approx(6.053079)
    assert config.overflow_risk_weight_inactive_streak_len == pytest.approx(0.117685)
    assert config.overflow_keep_weight_prior_play_prob == pytest.approx(2.202986)
    assert config.overflow_keep_weight_prior_minutes == pytest.approx(0.051353)


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


def test_build_game_level_examples_skips_malformed_single_side_games() -> None:
    df = _toy_frame()

    bad_rows: list[dict[str, object]] = []
    for i, pid in enumerate([301, 302, 303, 304, 305], start=1):
        bad_rows.append(
            {
                "game_id": 2002,
                "team_id": 30,
                "player_id": pid,
                "game_date": "2026-01-19",
                "home_team_id": 30,
                "away_team_id": 0,
                "home_flag": 1,
                "lineup_starter_announced": 1 if i <= 5 else 0,
                "lineup_available": 1,
                "prior_play_prob": 0.9 - i * 0.1,
                "minutes_from_stints_prior_20": 22 - i,
                "vegas_total": 225.0,
                "vegas_spread": -1.5,
                "estimated_possessions": 99.0,
                "f1": float(i),
                "f2": float(i) * 0.3,
                "minutes_label": float(28 - i),
            }
        )
    df2 = pd.concat([df, pd.DataFrame(bad_rows)], ignore_index=True)

    examples = build_game_level_examples(
        df2,
        feature_columns=["f1", "f2"],
        feature_mean=np.array([0.0, 0.0], dtype=np.float32),
        feature_std=np.array([1.0, 1.0], dtype=np.float32),
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        minutes_label_col="minutes_label",
    )
    assert len(examples) == 1
    assert examples[0].game_id_norm == "0000001001"


def test_build_game_level_examples_prefers_non_out_players_when_truncating() -> None:
    rows: list[dict[str, object]] = []
    game_id = 3003
    game_date = "2026-01-20"
    home_team_id = 10
    away_team_id = 20

    # 15 available home players with weak priors.
    for i in range(15):
        rows.append(
            {
                "game_id": game_id,
                "team_id": home_team_id,
                "player_id": 1000 + i,
                "game_date": game_date,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_flag": 1,
                "lineup_starter_announced": 0,
                "lineup_available": 1,
                "is_out": 0,
                "prior_play_prob": 0.10,
                "minutes_from_stints_prior_20": 2.0,
                "vegas_total": 228.0,
                "vegas_spread": -1.0,
                "estimated_possessions": 99.0,
                "f1": float(i),
                "f2": float(i) * 0.1,
                "minutes_label": 10.0,
            }
        )

    # 1 OUT home player with very strong priors; should still be truncated first.
    rows.append(
        {
            "game_id": game_id,
            "team_id": home_team_id,
            "player_id": 9999,
            "game_date": game_date,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_flag": 1,
            "lineup_starter_announced": 0,
            "lineup_available": 1,
            "is_out": 1,
            "prior_play_prob": 0.99,
            "minutes_from_stints_prior_20": 30.0,
            "vegas_total": 228.0,
            "vegas_spread": -1.0,
            "estimated_possessions": 99.0,
            "f1": 99.0,
            "f2": 9.9,
            "minutes_label": 0.0,
        }
    )

    # Minimal away side to satisfy feasible-game checks.
    for i in range(5):
        rows.append(
            {
                "game_id": game_id,
                "team_id": away_team_id,
                "player_id": 2000 + i,
                "game_date": game_date,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_flag": 0,
                "lineup_starter_announced": 1 if i < 5 else 0,
                "lineup_available": 1,
                "is_out": 0,
                "prior_play_prob": 0.9,
                "minutes_from_stints_prior_20": 25.0,
                "vegas_total": 228.0,
                "vegas_spread": -1.0,
                "estimated_possessions": 99.0,
                "f1": float(i + 30),
                "f2": float(i + 30) * 0.1,
                "minutes_label": 20.0,
            }
        )

    df = pd.DataFrame(rows)
    examples = build_game_level_examples(
        df,
        feature_columns=["f1", "f2"],
        feature_mean=np.array([0.0, 0.0], dtype=np.float32),
        feature_std=np.array([1.0, 1.0], dtype=np.float32),
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        minutes_label_col="minutes_label",
    )
    assert len(examples) == 1
    home_ids = set(int(v) for v in examples[0].player_ids[0][examples[0].player_valid_mask[0]].tolist())
    assert 9999 not in home_ids
    assert len(home_ids) == 15


def test_build_game_level_examples_truncation_protects_props_and_starters() -> None:
    rows: list[dict[str, object]] = []
    game_id = 3004
    game_date = "2026-01-20"
    home_team_id = 10
    away_team_id = 20

    # 15 ordinary non-out home players.
    for i in range(15):
        rows.append(
            {
                "game_id": game_id,
                "team_id": home_team_id,
                "player_id": 1100 + i,
                "game_date": game_date,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_flag": 1,
                "lineup_starter_announced": 0,
                "lineup_available": 1,
                "is_out": 0,
                "an_has_any_props": 0,
                "an_implied_minutes": 0.0,
                "prior_play_prob": 0.40,
                "minutes_from_stints_prior_20": 6.0,
                "consecutive_active_dnp": 0.0,
                "active_but_dnp_rate_last10": 0.0,
                "inactive_streak_len": 0.0,
                "vegas_total": 228.0,
                "vegas_spread": -1.0,
                "estimated_possessions": 99.0,
                "f1": float(i),
                "f2": float(i) * 0.1,
                "minutes_label": 10.0,
            }
        )

    # 1 extra player with strong props signal and weaker priors.
    rows.append(
        {
            "game_id": game_id,
            "team_id": home_team_id,
            "player_id": 1199,
            "game_date": game_date,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_flag": 1,
            "lineup_starter_announced": 0,
            "lineup_available": 1,
            "is_out": 0,
            "an_has_any_props": 1,
            "an_implied_minutes": 18.0,
            "prior_play_prob": 0.10,
            "minutes_from_stints_prior_20": 2.0,
            "consecutive_active_dnp": 4.0,
            "active_but_dnp_rate_last10": 0.7,
            "inactive_streak_len": 5.0,
            "vegas_total": 228.0,
            "vegas_spread": -1.0,
            "estimated_possessions": 99.0,
            "f1": 99.0,
            "f2": 9.9,
            "minutes_label": 0.0,
        }
    )

    # Minimal away side to satisfy feasibility.
    for i in range(5):
        rows.append(
            {
                "game_id": game_id,
                "team_id": away_team_id,
                "player_id": 2100 + i,
                "game_date": game_date,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_flag": 0,
                "lineup_starter_announced": 1,
                "lineup_available": 1,
                "is_out": 0,
                "an_has_any_props": 0,
                "an_implied_minutes": 0.0,
                "prior_play_prob": 0.9,
                "minutes_from_stints_prior_20": 25.0,
                "consecutive_active_dnp": 0.0,
                "active_but_dnp_rate_last10": 0.0,
                "inactive_streak_len": 0.0,
                "vegas_total": 228.0,
                "vegas_spread": -1.0,
                "estimated_possessions": 99.0,
                "f1": float(i + 30),
                "f2": float(i + 30) * 0.1,
                "minutes_label": 20.0,
            }
        )

    df = pd.DataFrame(rows)
    examples = build_game_level_examples(
        df,
        feature_columns=["f1", "f2"],
        feature_mean=np.array([0.0, 0.0], dtype=np.float32),
        feature_std=np.array([1.0, 1.0], dtype=np.float32),
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        minutes_label_col="minutes_label",
    )
    assert len(examples) == 1
    home_ids = set(int(v) for v in examples[0].player_ids[0][examples[0].player_valid_mask[0]].tolist())
    assert 1199 in home_ids
    assert len(home_ids) == 15


def test_build_game_level_examples_truncation_prefers_lower_dnp_risk() -> None:
    rows: list[dict[str, object]] = []
    game_id = 3005
    game_date = "2026-01-20"
    home_team_id = 10
    away_team_id = 20

    # 14 non-out baseline players.
    for i in range(14):
        rows.append(
            {
                "game_id": game_id,
                "team_id": home_team_id,
                "player_id": 1200 + i,
                "game_date": game_date,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_flag": 1,
                "lineup_starter_announced": 0,
                "lineup_available": 1,
                "is_out": 0,
                "an_has_any_props": 0,
                "an_implied_minutes": 0.0,
                "prior_play_prob": 0.50,
                "minutes_from_stints_prior_20": 8.0,
                "consecutive_active_dnp": 0.0,
                "active_but_dnp_rate_last10": 0.0,
                "inactive_streak_len": 0.0,
                "vegas_total": 228.0,
                "vegas_spread": -1.0,
                "estimated_possessions": 99.0,
                "f1": float(i),
                "f2": float(i) * 0.1,
                "minutes_label": 10.0,
            }
        )

    # Candidate A: low-risk fringe player.
    rows.append(
        {
            "game_id": game_id,
            "team_id": home_team_id,
            "player_id": 1301,
            "game_date": game_date,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_flag": 1,
            "lineup_starter_announced": 0,
            "lineup_available": 1,
            "is_out": 0,
            "an_has_any_props": 0,
            "an_implied_minutes": 0.0,
            "prior_play_prob": 0.30,
            "minutes_from_stints_prior_20": 4.0,
            "consecutive_active_dnp": 0.0,
            "active_but_dnp_rate_last10": 0.0,
            "inactive_streak_len": 0.0,
            "vegas_total": 228.0,
            "vegas_spread": -1.0,
            "estimated_possessions": 99.0,
            "f1": 31.0,
            "f2": 3.1,
            "minutes_label": 6.0,
        }
    )
    # Candidate B: high DNP-risk fringe player with same priors.
    rows.append(
        {
            "game_id": game_id,
            "team_id": home_team_id,
            "player_id": 1302,
            "game_date": game_date,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_flag": 1,
            "lineup_starter_announced": 0,
            "lineup_available": 1,
            "is_out": 0,
            "an_has_any_props": 0,
            "an_implied_minutes": 0.0,
            "prior_play_prob": 0.30,
            "minutes_from_stints_prior_20": 4.0,
            "consecutive_active_dnp": 6.0,
            "active_but_dnp_rate_last10": 0.9,
            "inactive_streak_len": 7.0,
            "vegas_total": 228.0,
            "vegas_spread": -1.0,
            "estimated_possessions": 99.0,
            "f1": 32.0,
            "f2": 3.2,
            "minutes_label": 0.0,
        }
    )

    for i in range(5):
        rows.append(
            {
                "game_id": game_id,
                "team_id": away_team_id,
                "player_id": 2200 + i,
                "game_date": game_date,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_flag": 0,
                "lineup_starter_announced": 1,
                "lineup_available": 1,
                "is_out": 0,
                "an_has_any_props": 0,
                "an_implied_minutes": 0.0,
                "prior_play_prob": 0.9,
                "minutes_from_stints_prior_20": 25.0,
                "consecutive_active_dnp": 0.0,
                "active_but_dnp_rate_last10": 0.0,
                "inactive_streak_len": 0.0,
                "vegas_total": 228.0,
                "vegas_spread": -1.0,
                "estimated_possessions": 99.0,
                "f1": float(i + 30),
                "f2": float(i + 30) * 0.1,
                "minutes_label": 20.0,
            }
        )

    df = pd.DataFrame(rows)
    examples = build_game_level_examples(
        df,
        feature_columns=["f1", "f2"],
        feature_mean=np.array([0.0, 0.0], dtype=np.float32),
        feature_std=np.array([1.0, 1.0], dtype=np.float32),
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        minutes_label_col="minutes_label",
    )
    assert len(examples) == 1
    home_ids = set(int(v) for v in examples[0].player_ids[0][examples[0].player_valid_mask[0]].tolist())
    assert 1301 in home_ids
    assert 1302 not in home_ids


def test_flow_head_set_scale_clip_overrides_all_blocks() -> None:
    """Smoke test: verify scale_clip override works and state_dict loads cleanly."""
    import torch

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
        flow_scale_clip=2.0,  # default
    )

    # Build model with default clip
    model_base = build_game_transformer_v2(config)
    state_dict = model_base.state_dict()

    # Build another model and load the same weights
    model_with_override = build_game_transformer_v2(config)
    model_with_override.load_state_dict(state_dict)  # Should succeed

    # Verify default clip
    for block in model_base.flow_head.blocks:
        assert block.scale_clip == pytest.approx(2.0)

    # Apply override
    model_with_override.flow_head.set_scale_clip(3.5)
    for block in model_with_override.flow_head.blocks:
        assert block.scale_clip == pytest.approx(3.5)

    # Verify forward pass works with overridden clip
    df = _toy_frame()
    flow_cols = ["fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov"]
    for col in flow_cols:
        df[col] = 5.0

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

    model_with_override.eval()
    with torch.no_grad():
        out = model_with_override(
            batch["player_features"],
            batch["player_valid_mask"],
            game_features=batch["game_features"],
            team_features=batch["team_features"],
            run_flow=True,
            flow_targets=batch["flow_targets"],
            flow_observed_mask=batch["flow_observed_mask"],
        )

    assert out.flow is not None
    assert out.flow.nll_mean.item() > 0.0  # Runs without error


def test_flow_head_set_scale_clip_with_different_values_produces_different_samples() -> None:
    """Verify different scale_clip values produce different inverse samples (smoke test)."""
    import torch

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

    model_clip2 = build_game_transformer_v2(config)
    model_clip4 = build_game_transformer_v2(config)

    # Share weights
    model_clip4.load_state_dict(model_clip2.state_dict())

    # Override one model's scale_clip
    model_clip4.flow_head.set_scale_clip(4.0)

    model_clip2.eval()
    model_clip4.eval()

    # Build batch
    df = _toy_frame()
    examples = build_game_level_examples(
        df,
        feature_columns=["f1", "f2"],
        feature_mean=np.array(config.feature_mean, dtype=np.float32),
        feature_std=np.array(config.feature_std, dtype=np.float32),
        game_feature_columns=config.game_feature_columns,
        team_feature_columns=config.team_feature_columns,
        minutes_label_col="minutes_label",
    )
    batch = collate_game_level_examples(examples)

    # Run forward to get player_states
    with torch.no_grad():
        out2 = model_clip2(
            batch["player_features"],
            batch["player_valid_mask"],
            game_features=batch["game_features"],
            team_features=batch["team_features"],
            sample_active=True,
        )
        out4 = model_clip4(
            batch["player_features"],
            batch["player_valid_mask"],
            game_features=batch["game_features"],
            team_features=batch["team_features"],
            sample_active=True,
        )

    # Sample from flow with same z
    torch.manual_seed(123)
    z = torch.randn(1, 30, 12)

    from projections.rotation.game_transformer_v2 import flow_target_columns

    ftc = flow_target_columns(include_pf=config.include_pf_in_flow_targets)
    assert len(ftc) == 12

    with torch.no_grad():
        y2 = model_clip2.flow_head.sample(
            z,
            player_states=out2.player_states,
            team_states=out2.team_states,
            game_state=out2.game_state,
            player_team_index=out2.player_team_index,
            valid_mask=out2.player_valid_mask,
        )
        y4 = model_clip4.flow_head.sample(
            z,
            player_states=out4.player_states,
            team_states=out4.team_states,
            game_state=out4.game_state,
            player_team_index=out4.player_team_index,
            valid_mask=out4.player_valid_mask,
        )

    # Different scale_clip should produce different samples (not identical)
    # Note: player_states are identical, z is identical, only scale_clip differs
    # The difference may be small but should be non-zero
    diff = (y2 - y4).abs().sum().item()
    assert diff > 0.0, "Different scale_clip should produce different samples"
