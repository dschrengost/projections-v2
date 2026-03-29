from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from projections.rotation.game_transformer_v2 import (
    FLOW_TARGET_COLUMNS_V1,
    FLOW_TARGET_COLUMNS_V2,
    MAX_PLAYERS_PER_TEAM,
    GameTransformerV2Config,
    _resolve_flow_conditioning_minutes,
    _resolve_minutes_active_mask,
    build_game_level_examples,
    build_game_transformer_v2,
    collate_game_level_examples,
    flow_contract_columns,
    flow_target_columns,
    reconstruct_flow_to_contract,
    select_flow_columns,
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


def test_build_game_level_examples_singleton_group_does_not_crash() -> None:
    # Singleton game groups can appear in inference when the input features are
    # malformed or heavily filtered. Pandas may represent `.groupby(...).indices`
    # values as scalars for singleton groups depending on version, so the
    # example builder must not assume an array-like index.
    df = pd.DataFrame(
        [
            {
                "game_id": 123,
                "team_id": 10,
                "player_id": 999,
                "game_date": "2026-03-22",
                "home_team_id": 10,
                "away_team_id": 20,
                "home_flag": 1,
                "lineup_available": 1,
                "f1": 1.0,
                "minutes_label": 0.0,
            }
        ]
    )
    # Not enough rows to build a valid (home+away) game example. The important
    # behavior is that we fail cleanly (ValueError) rather than crashing due to
    # singleton group indexing behavior.
    with pytest.raises(ValueError, match="No game-level examples were built"):
        build_game_level_examples(
            df,
            feature_columns=["f1"],
            feature_mean=np.array([0.0], dtype=np.float32),
            feature_std=np.array([1.0], dtype=np.float32),
            game_feature_columns=[],
            team_feature_columns=[],
            minutes_label_col="minutes_label",
        )


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
    assert ex.force_active_worlds.shape == (2, MAX_PLAYERS_PER_TEAM)
    assert int(ex.force_active_worlds.sum()) == 10  # five starters per team
    assert ex.starter_force_active_worlds.shape == (2, MAX_PLAYERS_PER_TEAM)
    assert int(ex.starter_force_active_worlds.sum()) == 10
    assert ex.force_active_minutes_anchor.shape == (2, MAX_PLAYERS_PER_TEAM)
    assert float(ex.force_active_minutes_anchor.sum()) == 0.0

    batch = collate_game_level_examples([ex])
    assert batch["player_features"].shape == (1, 2, MAX_PLAYERS_PER_TEAM, len(feature_columns))
    assert batch["player_valid_mask"].shape == (1, 2, MAX_PLAYERS_PER_TEAM)
    assert batch["force_active_worlds"].shape == (1, 2, MAX_PLAYERS_PER_TEAM)
    assert int(batch["force_active_worlds"].sum().item()) == 10
    assert batch["starter_force_active_worlds"].shape == (1, 2, MAX_PLAYERS_PER_TEAM)
    assert int(batch["starter_force_active_worlds"].sum().item()) == 10
    assert batch["force_active_minutes_anchor"].shape == (1, 2, MAX_PLAYERS_PER_TEAM)
    assert float(batch["force_active_minutes_anchor"].sum().item()) == 0.0


def test_build_game_level_examples_and_collate_sidecar_shapes() -> None:
    df = _toy_frame()
    df["fg2_pct_prior_5"] = np.linspace(0.45, 0.60, len(df))
    df["fg3_pct_prior_5"] = np.linspace(0.30, 0.42, len(df))
    df["opp_fg2_pct_allowed_prior_5"] = np.linspace(0.48, 0.55, len(df))
    feature_columns = ["f1", "f2"]
    sidecar_columns = ["fg2_pct_prior_5", "fg3_pct_prior_5", "opp_fg2_pct_allowed_prior_5"]

    examples = build_game_level_examples(
        df,
        feature_columns=feature_columns,
        feature_mean=np.array([0.0, 0.0], dtype=np.float32),
        feature_std=np.array([1.0, 1.0], dtype=np.float32),
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        efficiency_sidecar_feature_columns=sidecar_columns,
        efficiency_sidecar_feature_mean=np.array([0.5, 0.35, 0.5], dtype=np.float32),
        efficiency_sidecar_feature_std=np.array([0.05, 0.05, 0.03], dtype=np.float32),
        minutes_label_col="minutes_label",
    )

    batch = collate_game_level_examples(examples)
    assert batch["efficiency_sidecar_features"].shape == (1, 2, MAX_PLAYERS_PER_TEAM, len(sidecar_columns))
    assert torch.isfinite(batch["efficiency_sidecar_features"]).all()


def test_build_game_level_examples_with_team_feature_columns_preserves_side_alignment() -> None:
    df = _toy_frame()
    home_mask = df["team_id"] == 10
    away_mask = df["team_id"] == 20
    df.loc[home_mask, "is_b2b"] = 1.0
    df.loc[away_mask, "is_b2b"] = 0.0
    df.loc[home_mask, "team_pace_szn"] = 101.5
    df.loc[away_mask, "team_pace_szn"] = 97.25
    df.loc[home_mask, "team_off_rtg_szn"] = 118.0
    df.loc[away_mask, "team_off_rtg_szn"] = 109.5

    examples = build_game_level_examples(
        df,
        feature_columns=["f1", "f2"],
        feature_mean=np.array([0.0, 0.0], dtype=np.float32),
        feature_std=np.array([1.0, 1.0], dtype=np.float32),
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=["is_b2b", "team_pace_szn", "team_off_rtg_szn"],
        minutes_label_col="minutes_label",
    )

    batch = collate_game_level_examples(examples)
    team_features = batch["team_features"].numpy()
    assert team_features.shape == (1, 2, 3)
    np.testing.assert_allclose(team_features[0, 0], np.array([1.0, 101.5, 118.0], dtype=np.float32))
    np.testing.assert_allclose(team_features[0, 1], np.array([0.0, 97.25, 109.5], dtype=np.float32))


def test_game_transformer_v2_forward_supports_late_fused_backbone_env_features() -> None:
    df = _toy_frame()
    df["is_b2b"] = np.where(df["team_id"] == 10, 1.0, 0.0)
    df["team_pace_szn"] = np.where(df["team_id"] == 10, 101.5, 97.25)
    df["team_off_rtg_szn"] = np.where(df["team_id"] == 10, 118.0, 109.5)
    df["team_def_rtg_szn"] = np.where(df["team_id"] == 10, 111.0, 113.0)
    df["opp_pace_szn"] = np.where(df["team_id"] == 10, 97.25, 101.5)
    df["opp_def_rtg_szn"] = np.where(df["team_id"] == 10, 113.0, 111.0)

    feature_columns = [
        "f1",
        "f2",
        "is_b2b",
        "team_pace_szn",
        "team_off_rtg_szn",
        "team_def_rtg_szn",
        "opp_pace_szn",
        "opp_def_rtg_szn",
    ]
    config = GameTransformerV2Config(
        feature_columns=feature_columns,
        feature_mean=[0.0] * len(feature_columns),
        feature_std=[1.0] * len(feature_columns),
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        backbone_env_feature_columns=[
            "is_b2b",
            "team_pace_szn",
            "team_off_rtg_szn",
            "team_def_rtg_szn",
            "opp_pace_szn",
            "opp_def_rtg_szn",
        ],
        d_model=48,
        hidden_dim=64,
        num_layers=1,
        num_heads=6,
        dropout=0.0,
        enable_possession_backbone=True,
        enable_three_pa_share=True,
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
        sample_backbone=False,
    )

    assert out.possession is not None
    assert out.backbone is not None
    assert out.possession.mu.shape == (1,)
    assert out.backbone.fga.shape == (1, 2)


def test_game_transformer_v2_forward_supports_backbone_env_adapter() -> None:
    df = _toy_frame()
    df["is_b2b"] = np.where(df["team_id"] == 10, 1.0, 0.0)
    df["team_pace_szn"] = np.where(df["team_id"] == 10, 101.5, 97.25)
    df["team_off_rtg_szn"] = np.where(df["team_id"] == 10, 118.0, 109.5)
    df["team_def_rtg_szn"] = np.where(df["team_id"] == 10, 111.0, 113.0)
    df["opp_pace_szn"] = np.where(df["team_id"] == 10, 97.25, 101.5)
    df["opp_def_rtg_szn"] = np.where(df["team_id"] == 10, 113.0, 111.0)

    feature_columns = [
        "f1",
        "f2",
        "is_b2b",
        "team_pace_szn",
        "team_off_rtg_szn",
        "team_def_rtg_szn",
        "opp_pace_szn",
        "opp_def_rtg_szn",
    ]
    config = GameTransformerV2Config(
        feature_columns=feature_columns,
        feature_mean=[0.0] * len(feature_columns),
        feature_std=[1.0] * len(feature_columns),
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        backbone_env_feature_columns=[
            "is_b2b",
            "team_pace_szn",
            "team_off_rtg_szn",
            "team_def_rtg_szn",
            "opp_pace_szn",
            "opp_def_rtg_szn",
        ],
        backbone_env_adapter_dim=8,
        backbone_env_adapter_hidden=16,
        d_model=48,
        hidden_dim=64,
        num_layers=1,
        num_heads=6,
        dropout=0.0,
        enable_possession_backbone=True,
        enable_three_pa_share=True,
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
        sample_backbone=False,
    )

    assert model.backbone_env_adapter is not None
    assert out.possession is not None
    assert out.backbone is not None
    assert out.possession.mu.shape == (1,)
    assert out.backbone.fga.shape == (1, 2)


def test_game_transformer_v2_forward_supports_efficiency_sidecar() -> None:
    df = _toy_frame()
    df["fg2_pct_prior_5"] = np.linspace(0.45, 0.60, len(df))
    df["fg3_pct_prior_5"] = np.linspace(0.30, 0.42, len(df))
    df["ft_pct_prior_5"] = np.linspace(0.72, 0.88, len(df))
    df["opp_fg2_pct_allowed_prior_5"] = np.linspace(0.48, 0.55, len(df))
    df["opp_fg3_pct_allowed_prior_5"] = np.linspace(0.33, 0.39, len(df))

    feature_columns = ["f1", "f2"]
    sidecar_columns = [
        "fg2_pct_prior_5",
        "fg3_pct_prior_5",
        "ft_pct_prior_5",
        "opp_fg2_pct_allowed_prior_5",
        "opp_fg3_pct_allowed_prior_5",
    ]
    config = GameTransformerV2Config(
        feature_columns=feature_columns,
        feature_mean=[0.0] * len(feature_columns),
        feature_std=[1.0] * len(feature_columns),
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        efficiency_sidecar_feature_columns=sidecar_columns,
        efficiency_sidecar_feature_mean=[0.5, 0.35, 0.78, 0.5, 0.36],
        efficiency_sidecar_feature_std=[0.05, 0.05, 0.06, 0.04, 0.03],
        enable_efficiency_head=True,
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
        efficiency_sidecar_feature_columns=config.efficiency_sidecar_feature_columns,
        efficiency_sidecar_feature_mean=np.array(config.efficiency_sidecar_feature_mean, dtype=np.float32),
        efficiency_sidecar_feature_std=np.array(config.efficiency_sidecar_feature_std, dtype=np.float32),
        minutes_label_col="minutes_label",
    )
    batch = collate_game_level_examples(examples)
    out = model(
        batch["player_features"],
        batch["player_valid_mask"],
        game_features=batch["game_features"],
        team_features=batch["team_features"],
        efficiency_sidecar_features=batch["efficiency_sidecar_features"],
        sample_active=False,
        sample_backbone=False,
    )

    assert out.efficiency is not None
    assert out.efficiency.mean_fg2.shape == (1, 30)


def test_game_transformer_v2_forward_supports_env_side_channel() -> None:
    df = _toy_frame()
    df["is_b2b"] = np.where(df["team_id"] == 10, 1.0, 0.0)
    df["team_pace_szn"] = np.where(df["team_id"] == 10, 101.5, 97.25)
    df["team_off_rtg_szn"] = np.where(df["team_id"] == 10, 118.0, 109.5)
    df["team_def_rtg_szn"] = np.where(df["team_id"] == 10, 111.0, 113.0)
    df["opp_pace_szn"] = np.where(df["team_id"] == 10, 97.25, 101.5)
    df["opp_def_rtg_szn"] = np.where(df["team_id"] == 10, 113.0, 111.0)

    feature_columns = [
        "f1",
        "f2",
        "is_b2b",
        "team_pace_szn",
        "team_off_rtg_szn",
        "team_def_rtg_szn",
        "opp_pace_szn",
        "opp_def_rtg_szn",
    ]
    config = GameTransformerV2Config(
        feature_columns=feature_columns,
        feature_mean=[0.0] * len(feature_columns),
        feature_std=[1.0] * len(feature_columns),
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        backbone_env_feature_columns=[
            "is_b2b",
            "team_pace_szn",
            "team_off_rtg_szn",
            "team_def_rtg_szn",
            "opp_pace_szn",
            "opp_def_rtg_szn",
        ],
        backbone_env_enrich_features=True,
        enable_env_side_channel=True,
        env_side_channel_dim=12,
        env_side_channel_hidden=16,
        d_model=48,
        hidden_dim=64,
        num_layers=1,
        num_heads=6,
        dropout=0.0,
        enable_possession_backbone=True,
        enable_three_pa_share=True,
        flow_use_minutes_conditioning=True,
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
        flow_label_columns=["fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov"],
        minutes_label_col="minutes_label",
    )
    batch = collate_game_level_examples(examples)
    out = model(
        batch["player_features"],
        batch["player_valid_mask"],
        game_features=batch["game_features"],
        team_features=batch["team_features"],
        flow_targets=batch["flow_targets"],
        flow_observed_mask=batch["flow_observed_mask"],
        run_flow=True,
        flow_minutes_target=batch["y_minutes"],
        sample_active=False,
        sample_backbone=False,
    )

    assert model.env_side_channel_encoder is not None
    assert out.flow is not None
    assert out.possession is not None
    assert out.backbone is not None


def test_game_transformer_v2_builds_side_specific_backbone_market_context() -> None:
    config = GameTransformerV2Config(
        feature_columns=["f1"],
        feature_mean=[0.0],
        feature_std=[1.0],
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        enable_possession_backbone=True,
        backbone_side_market_context=True,
        d_model=48,
        hidden_dim=64,
        num_layers=1,
        num_heads=6,
        dropout=0.0,
    )
    model = build_game_transformer_v2(config)
    game_features = torch.tensor([[231.5, -16.5, 101.0]], dtype=torch.float32)

    ctx = model._build_backbone_team_market_context(game_features)

    assert ctx is not None
    assert ctx.shape == (1, 2, 6)
    home = ctx[0, 0].detach().cpu().numpy()
    away = ctx[0, 1].detach().cpu().numpy()
    np.testing.assert_allclose(home[:4], np.array([124.0, 107.5, 16.5, 16.5], dtype=np.float32))
    np.testing.assert_allclose(away[:4], np.array([107.5, 124.0, -16.5, 16.5], dtype=np.float32))
    np.testing.assert_allclose(home[4:], np.array([124.0 / 101.0, 107.5 / 101.0], dtype=np.float32))
    np.testing.assert_allclose(away[4:], np.array([107.5 / 101.0, 124.0 / 101.0], dtype=np.float32))


def test_game_transformer_v2_forward_supports_ast_factorization_heads() -> None:
    df = _toy_frame()
    df["an_ast_line"] = np.where(df["player_id"].isin([101, 201]), 7.5, 2.0)
    df["an_implied_minutes"] = np.where(df["player_id"].isin([101, 201]), 33.0, 18.0)
    df["started_proxy_rate_prior_20"] = np.where(df["player_id"].isin([101, 201]), 0.8, 0.2)
    config = GameTransformerV2Config(
        feature_columns=[
            "f1",
            "f2",
            "an_ast_line",
            "an_implied_minutes",
            "prior_play_prob",
            "started_proxy_rate_prior_20",
        ],
        feature_mean=[0.0] * 6,
        feature_std=[1.0] * 6,
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        d_model=48,
        hidden_dim=64,
        num_layers=1,
        num_heads=6,
        dropout=0.0,
        enable_team_ast_budget_head=True,
        team_ast_budget_head_hidden=32,
        enable_assist_share_head=True,
        assist_share_head_hidden=32,
        enable_ast_blend_gate=True,
        ast_blend_gate_hidden=24,
        ast_blend_gate_init_alpha=0.7,
        assist_share_condition_feature_columns=[
            "an_ast_line",
            "an_implied_minutes",
            "prior_play_prob",
            "started_proxy_rate_prior_20",
        ],
        assist_share_condition_hidden=16,
    )
    model = build_game_transformer_v2(config)
    model.eval()
    assert getattr(model, "gtv2_config", None) is not None

    examples = build_game_level_examples(
        df,
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.team_ast_budget is not None
    assert out.team_ast_budget.team_ast.shape == (1, 2)
    assert out.assist_share is not None
    assert out.assist_share.ast_logits.shape == (1, 30)
    assert out.ast_blend_gate is not None
    assert out.ast_blend_gate.gate.shape == (1, 30)


def test_game_transformer_v2_forward_supports_team_points_budget_head() -> None:
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
        enable_possession_backbone=True,
        enable_team_points_budget_head=True,
        team_points_budget_head_hidden=32,
        team_points_budget_to_backbone=True,
        team_points_budget_latent_hidden=16,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.team_points_budget is not None
    assert out.team_points_budget.team_points.shape == (1, 2)
    assert out.backbone is not None
    assert out.backbone.fga.shape == (1, 2)


def test_game_transformer_v2_forward_supports_team_ppp_head() -> None:
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
        enable_efficiency_head=True,
        enable_possession_backbone=True,
        enable_team_ppp_head=True,
        team_ppp_head_hidden=32,
        team_ppp_to_backbone=True,
        team_ppp_direct_backbone_context=True,
        team_ppp_to_efficiency=True,
        team_ppp_direct_efficiency_context=True,
        team_ppp_latent_hidden=16,
        team_ppp_backbone_alpha=0.5,
        team_ppp_efficiency_alpha=0.5,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.team_ppp is not None
    assert out.team_ppp.team_ppp.shape == (1, 2)
    assert out.efficiency is not None
    assert out.backbone is not None


def test_game_transformer_v2_forward_supports_team_advantage_head() -> None:
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
        enable_possession_backbone=True,
        enable_three_pa_share=True,
        enable_team_advantage_head=True,
        team_advantage_head_hidden=32,
        team_advantage_direct_backbone_context=True,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.team_advantage is not None
    assert out.team_advantage.mu.shape == (1,)
    assert out.backbone is not None


def test_game_transformer_v2_forward_supports_market_implied_team_points_context() -> None:
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
        enable_possession_backbone=True,
        enable_team_points_budget_head=False,
        team_points_budget_parameterization="market_implied",
        team_points_budget_to_backbone=True,
        team_points_budget_latent_hidden=16,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.team_points_budget is None
    assert out.backbone is not None
    assert out.backbone.fga.shape == (1, 2)


def test_game_transformer_v2_forward_supports_team_ppp_implied_team_points_context() -> None:
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
        enable_possession_backbone=True,
        enable_team_ppp_head=True,
        team_ppp_head_hidden=32,
        team_points_budget_parameterization="team_ppp_implied",
        team_points_budget_to_backbone=True,
        team_points_budget_latent_hidden=16,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.team_ppp is not None
    assert out.backbone is not None
    assert out.backbone.fga.shape == (1, 2)


def test_game_transformer_v2_forward_supports_market_implied_team_opportunity_context() -> None:
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
        enable_possession_backbone=True,
        team_opportunity_budget_parameterization="market_implied_share",
        team_opportunity_budget_to_backbone=True,
        team_opportunity_budget_latent_hidden=16,
        team_opportunity_budget_backbone_alpha=0.5,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.backbone is not None
    assert out.backbone.fga.shape == (1, 2)


def test_game_transformer_v2_forward_supports_team_possession_split_head() -> None:
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
        enable_possession_backbone=True,
        enable_team_possession_split_head=True,
        team_possession_max_delta=6.0,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.possession is not None
    assert out.possession.team_poss is not None
    assert out.possession.team_poss.shape == (1, 2)
    assert out.backbone is not None
    assert out.backbone.poss_used.shape == (1, 2)


def test_game_transformer_v2_forward_supports_efficiency_market_context() -> None:
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
        enable_efficiency_head=True,
        efficiency_market_context=True,
        efficiency_market_hidden=16,
        efficiency_market_alpha=0.5,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert model.efficiency_team_market_encoder is not None
    assert out.efficiency is not None
    assert out.efficiency.mean_fg2.shape == (1, 30)


def test_game_transformer_v2_forward_supports_rebound_factorization_heads() -> None:
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
        enable_team_rebound_budget_head=True,
        team_rebound_budget_head_hidden=32,
        enable_rebound_budget_blend_gate=True,
        rebound_budget_blend_gate_hidden=24,
        rebound_budget_blend_gate_init_alpha=0.2,
        enable_rebound_share_head=True,
        rebound_share_head_hidden=32,
    )
    model = build_game_transformer_v2(config)
    model.eval()

    examples = build_game_level_examples(
        df,
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.team_rebound_budget is not None
    assert out.team_rebound_budget.team_oreb.shape == (1, 2)
    assert out.team_rebound_budget.team_dreb.shape == (1, 2)
    assert out.rebound_budget_blend_gate is not None
    assert out.rebound_budget_blend_gate.oreb_gate.shape == (1, 2)
    assert out.rebound_budget_blend_gate.dreb_gate.shape == (1, 2)
    assert out.rebound_share is not None
    assert out.rebound_share.oreb_logits.shape == (1, 30)
    assert out.rebound_share.dreb_logits.shape == (1, 30)


def test_team_rebound_budget_head_supports_dreb_rate_parameterization() -> None:
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
        enable_team_rebound_budget_head=True,
        rebound_budget_parameterization="dreb_rate",
        rebound_dreb_rate_cap=0.8,
    )
    model = build_game_transformer_v2(config)
    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.team_rebound_budget is not None
    assert torch.all(out.team_rebound_budget.team_dreb >= 0.0)
    assert torch.all(out.team_rebound_budget.team_dreb <= 0.8 + 1e-6)


def test_team_rebound_budget_head_supports_dreb_rate_residual_parameterization() -> None:
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
        enable_team_rebound_budget_head=True,
        rebound_budget_parameterization="dreb_rate_residual",
        rebound_dreb_rate_cap=0.12,
    )
    model = build_game_transformer_v2(config)
    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.team_rebound_budget is not None
    assert torch.all(out.team_rebound_budget.team_dreb >= -0.12 - 1e-6)
    assert torch.all(out.team_rebound_budget.team_dreb <= 0.12 + 1e-6)


def test_rebound_share_head_supports_condition_features() -> None:
    config = GameTransformerV2Config(
        feature_columns=[
            "f1",
            "f2",
            "an_reb_line",
            "an_implied_minutes",
            "prior_play_prob",
            "started_proxy_rate_prior_20",
        ],
        feature_mean=[0.0, 0.0, 5.0, 24.0, 0.8, 0.4],
        feature_std=[1.0, 1.0, 2.0, 8.0, 0.2, 0.3],
        game_feature_columns=["vegas_total", "vegas_spread", "estimated_possessions"],
        team_feature_columns=[],
        d_model=48,
        hidden_dim=64,
        num_layers=1,
        num_heads=6,
        dropout=0.0,
        enable_rebound_share_head=True,
        rebound_share_head_hidden=32,
        rebound_share_condition_feature_columns=[
            "an_reb_line",
            "an_implied_minutes",
            "prior_play_prob",
            "started_proxy_rate_prior_20",
        ],
        rebound_share_condition_hidden=16,
    )
    model = build_game_transformer_v2(config)
    examples = build_game_level_examples(
        _toy_frame(),
        feature_columns=config.feature_columns,
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
        sample_backbone=False,
    )

    assert out.rebound_share is not None
    assert out.rebound_share.oreb_logits.shape == (1, 30)
    assert out.rebound_share.dreb_logits.shape == (1, 30)


def test_build_game_level_examples_force_active_worlds_includes_manual_force_in() -> None:
    df = _toy_frame()
    df["force_active_worlds"] = 0
    df.loc[df["player_id"] == 106, "force_active_worlds"] = 1  # bench force-in
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
    ex = examples[0]
    valid_flat = np.concatenate([ex.player_valid_mask[0], ex.player_valid_mask[1]], axis=0)
    force_flat = np.concatenate([ex.force_active_worlds[0], ex.force_active_worlds[1]], axis=0)
    player_flat = np.concatenate([ex.player_ids[0], ex.player_ids[1]], axis=0)
    by_pid = {
        int(pid): bool(force_flat[idx])
        for idx, pid in enumerate(player_flat)
        if bool(valid_flat[idx])
    }
    assert by_pid[106] is True
    assert int(np.sum(force_flat[valid_flat])) == 11


def test_build_game_level_examples_force_active_minutes_anchor_from_props() -> None:
    df = _toy_frame()
    df["an_implied_minutes"] = 0.0
    df["an_has_implied_minutes"] = 0
    df.loc[df["player_id"] == 101, "an_implied_minutes"] = 35.0
    df.loc[df["player_id"] == 101, "an_has_implied_minutes"] = 1
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
    ex = examples[0]
    valid_flat = np.concatenate([ex.player_valid_mask[0], ex.player_valid_mask[1]], axis=0)
    anchors_flat = np.concatenate([ex.force_active_minutes_anchor[0], ex.force_active_minutes_anchor[1]], axis=0)
    player_flat = np.concatenate([ex.player_ids[0], ex.player_ids[1]], axis=0)
    by_pid = {
        int(pid): float(anchors_flat[idx])
        for idx, pid in enumerate(player_flat)
        if bool(valid_flat[idx])
    }
    assert by_pid[101] == pytest.approx(35.0)
    assert by_pid[102] == pytest.approx(0.0)


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


def test_resolve_minutes_active_mask_returns_predicted_mask_when_prob_zero() -> None:
    predicted = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    target = torch.tensor([[False, True, False, True]], dtype=torch.bool)
    team_index = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)

    out = _resolve_minutes_active_mask(
        predicted,
        target_active_mask=target,
        player_team_index=team_index,
        minutes_teacher_forcing_prob=0.0,
        minutes_teacher_forcing_mode="team",
    )

    assert torch.equal(out, predicted)


def test_resolve_minutes_active_mask_returns_target_mask_when_prob_one() -> None:
    predicted = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    target = torch.tensor([[False, True, False, True]], dtype=torch.bool)
    team_index = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)

    out = _resolve_minutes_active_mask(
        predicted,
        target_active_mask=target,
        player_team_index=team_index,
        minutes_teacher_forcing_prob=1.0,
        minutes_teacher_forcing_mode="team",
    )

    assert torch.equal(out, target)


def test_resolve_minutes_active_mask_example_mode_mixes_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    predicted = torch.tensor(
        [
            [True, True, False, False],
            [False, False, True, True],
        ],
        dtype=torch.bool,
    )
    target = ~predicted
    team_index = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.long,
    )

    def fake_rand(shape: tuple[int, ...] | torch.Size, device: torch.device | None = None) -> torch.Tensor:
        assert tuple(shape) == (2, 1)
        return torch.tensor([[0.2], [0.8]], dtype=torch.float32, device=device)

    monkeypatch.setattr(torch, "rand", fake_rand)
    out = _resolve_minutes_active_mask(
        predicted,
        target_active_mask=target,
        player_team_index=team_index,
        minutes_teacher_forcing_prob=0.5,
        minutes_teacher_forcing_mode="example",
    )

    expected = torch.stack([target[0], predicted[1]], dim=0)
    assert torch.equal(out, expected)


def test_resolve_minutes_active_mask_team_mode_mixes_home_and_away_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted = torch.tensor(
        [
            [True, True, False, False],
            [False, False, True, True],
        ],
        dtype=torch.bool,
    )
    target = ~predicted
    team_index = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.long,
    )

    def fake_rand(shape: tuple[int, ...] | torch.Size, device: torch.device | None = None) -> torch.Tensor:
        assert tuple(shape) == (2, 2)
        return torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float32, device=device)

    monkeypatch.setattr(torch, "rand", fake_rand)
    out = _resolve_minutes_active_mask(
        predicted,
        target_active_mask=target,
        player_team_index=team_index,
        minutes_teacher_forcing_prob=0.5,
        minutes_teacher_forcing_mode="team",
    )

    expected = torch.tensor(
        [
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )
    expected[0, :2] = target[0, :2]
    expected[0, 2:] = predicted[0, 2:]
    expected[1, :2] = predicted[1, :2]
    expected[1, 2:] = target[1, 2:]
    assert torch.equal(out, expected)


def test_resolve_flow_conditioning_minutes_returns_predicted_when_prob_zero() -> None:
    predicted = torch.tensor([[30.0, 28.0, 12.0, 10.0]], dtype=torch.float32)
    target = torch.tensor([[31.0, 27.0, 11.0, 9.0]], dtype=torch.float32)
    team_index = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)

    out = _resolve_flow_conditioning_minutes(
        predicted,
        target_minutes=target,
        player_team_index=team_index,
        teacher_forcing_prob=0.0,
        teacher_forcing_mode="team",
    )

    assert torch.equal(out, predicted)


def test_resolve_flow_conditioning_minutes_team_mode_mixes_by_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted = torch.tensor(
        [
            [30.0, 28.0, 12.0, 10.0],
            [32.0, 26.0, 11.0, 9.0],
        ],
        dtype=torch.float32,
    )
    target = predicted + 1.0
    team_index = torch.tensor(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.long,
    )

    def fake_rand(shape: tuple[int, ...] | torch.Size, device: torch.device | None = None) -> torch.Tensor:
        assert tuple(shape) == (2, 2)
        return torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float32, device=device)

    monkeypatch.setattr(torch, "rand", fake_rand)
    out = _resolve_flow_conditioning_minutes(
        predicted,
        target_minutes=target,
        player_team_index=team_index,
        teacher_forcing_prob=0.5,
        teacher_forcing_mode="team",
    )

    expected = predicted.clone()
    expected[0, :2] = target[0, :2]
    expected[1, 2:] = target[1, 2:]
    assert torch.equal(out, expected)


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
    assert config.flow_scale_clip == pytest.approx(3.0)  # H1 fix: increased from 2.0
    assert config.flow_context_mode == "attention"  # H2 fix: gated attention instead of mean pooling
    assert config.flow_target_schema == "v1"
    assert config.flow_use_minutes_conditioning is False
    assert config.enable_minutes_hurdle_head is False
    assert config.minutes_hurdle_hidden == 64
    assert config.minutes_hurdle_sigma_floor == pytest.approx(0.5)
    assert config.include_pf_in_flow_targets is False
    assert config.overflow_protected_prior_play_prob_floor == pytest.approx(0.938507)
    assert config.overflow_protected_prior_minutes_floor == pytest.approx(29.520922)
    assert config.overflow_risk_weight_consecutive_active_dnp == pytest.approx(0.579943)
    assert config.overflow_risk_weight_active_but_dnp_rate_last10 == pytest.approx(6.053079)
    assert config.overflow_risk_weight_inactive_streak_len == pytest.approx(0.117685)
    assert config.overflow_keep_weight_prior_play_prob == pytest.approx(2.202986)
    assert config.overflow_keep_weight_prior_minutes == pytest.approx(0.051353)


def test_flow_target_schema_v2_columns_and_reconstruction() -> None:
    src_cols = list(FLOW_TARGET_COLUMNS_V1)
    tgt_cols = flow_target_columns(include_pf=False, schema="v2")
    assert tgt_cols == list(FLOW_TARGET_COLUMNS_V2)
    assert "fg2m" not in tgt_cols
    assert "fg3m" not in tgt_cols
    assert "ftm" not in tgt_cols

    full = torch.zeros((1, 2, len(src_cols)), dtype=torch.float32)
    full[..., src_cols.index("fga2")] = 8.0
    full[..., src_cols.index("fga3")] = 5.0
    full[..., src_cols.index("fta")] = 6.0
    full[..., src_cols.index("oreb")] = 2.0
    full[..., src_cols.index("dreb")] = 4.0
    full[..., src_cols.index("ast")] = 7.0
    full[..., src_cols.index("stl")] = 1.0
    full[..., src_cols.index("blk")] = 1.0
    full[..., src_cols.index("tov")] = 3.0

    flow_v2 = select_flow_columns(
        full,
        source_columns=src_cols,
        target_columns=tgt_cols,
        fill_value=0.0,
    )
    recon = reconstruct_flow_to_contract(
        flow_v2,
        flow_target_columns=tgt_cols,
        contract_columns=flow_contract_columns(include_pf=False),
        fg2_rate=0.5,
        fg3_rate=0.4,
        ft_rate=0.75,
    )
    assert recon.shape[-1] == len(FLOW_TARGET_COLUMNS_V1)
    assert torch.allclose(recon[..., src_cols.index("fg2m")], torch.full((1, 2), 4.0))
    assert torch.allclose(recon[..., src_cols.index("fg3m")], torch.full((1, 2), 2.0))
    assert torch.allclose(recon[..., src_cols.index("ftm")], torch.full((1, 2), 4.5))


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


def test_game_transformer_v2_forward_emits_minutes_hurdle_outputs_when_enabled() -> None:
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
        enable_minutes_hurdle_head=True,
        minutes_hurdle_hidden=32,
        minutes_hurdle_sigma_floor=0.7,
    )
    model = build_game_transformer_v2(config)
    model.eval()

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
    out = model(
        batch["player_features"],
        batch["player_valid_mask"],
        game_features=batch["game_features"],
        team_features=batch["team_features"],
        sample_active=False,
    )

    assert out.minutes.zero_logits is not None
    assert out.minutes.sigma is not None
    assert out.minutes.zero_logits.shape == (1, 30)
    assert out.minutes.sigma.shape == (1, 30)
    assert float(out.minutes.sigma.min().item()) >= 0.7 - 1e-6


def test_flow_head_sample_requires_minutes_context_when_enabled() -> None:
    df = _toy_frame()
    flow_cols = ["fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov"]
    for idx, col in enumerate(flow_cols):
        df[col] = float(idx + 1)

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
        flow_use_minutes_conditioning=True,
    )
    model = build_game_transformer_v2(config)
    model.eval()

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
    )

    z = torch.zeros((1, 30, len(flow_cols)), dtype=out.player_states.dtype)
    with pytest.raises(ValueError, match="minutes_context"):
        model.flow_head.sample(  # type: ignore[attr-defined]
            z,
            player_states=out.player_states,
            team_states=out.team_states,
            game_state=out.game_state,
            player_team_index=out.player_team_index,
            valid_mask=out.player_valid_mask,
        )

    sampled = model.flow_head.sample(  # type: ignore[attr-defined]
        z,
        player_states=out.player_states,
        team_states=out.team_states,
        game_state=out.game_state,
        player_team_index=out.player_team_index,
        valid_mask=out.player_valid_mask,
        minutes_context=out.minutes.minutes,
    )
    assert sampled.shape == (1, 30, len(flow_cols))


def test_game_transformer_v2_forward_emits_efficiency_outputs_when_enabled() -> None:
    df = _toy_frame()
    flow_cols = ["fga2", "fg2m", "fga3", "fg3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov"]
    for idx, col in enumerate(flow_cols):
        df[col] = float(idx + 1)

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
        enable_efficiency_head=True,
    )
    model = build_game_transformer_v2(config)
    model.eval()

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
    )
    assert out.efficiency is not None
    assert out.efficiency.alpha_ft.shape == (1, 30)
    assert out.efficiency.alpha_fg2.shape == (1, 30)
    assert out.efficiency.alpha_fg3.shape == (1, 30)
    assert float(out.efficiency.alpha_ft.min().item()) > 0.0
    assert float(out.efficiency.beta_ft.min().item()) > 0.0


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
