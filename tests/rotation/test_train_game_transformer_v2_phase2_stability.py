from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import torch
from torch import nn

from scripts.rotation.train_game_transformer_v2 import (
    BackboneEpochWeights,
    EarlyStopConfig,
    EarlyStopState,
    MinutesCheckpointCandidate,
    Phase2StabilityConfig,
    Phase2StabilityState,
    _build_bench_riser_example_sampling_weights,
    _build_sparse_candidate_example_sampling_weights,
    _count_backbone_coupled_epochs,
    _build_lineup_available_example_sampling_weights,
    _compute_named_direct_stat_losses,
    _compute_sparse_rerank_score,
    _freeze_parameter_prefixes,
    _mean_named_losses,
    _minutes_hurdle_nll,
    _matches_prefix,
    _mask_ast_from_flow_targets,
    parse_args,
    _parse_prefix_csv,
    _record_topk_minutes_checkpoint,
    _resolve_backbone_epoch_weights,
    _resolve_early_stop_metric_value,
    _resolve_minutes_teacher_forcing_prob,
    _resolve_phase2_epoch_weights,
    _select_sparse_rerank_candidate,
    _team_fixed_opportunity_rate_mse_loss,
    _team_ratio_mse_loss,
    _team_sum_by_side,
    _update_early_stop,
    _update_phase2_nll_guard,
    _weighted_masked_scaled_huber_loss,
)


def test_resolve_phase2_epoch_weights_applies_warmup_and_anchor_schedule() -> None:
    weights = _resolve_phase2_epoch_weights(
        epoch=1,
        enable_phase2_flow=True,
        enable_phase3_decision=False,
        w_minutes=1.0,
        w_minutes_nll=1.0,
        w_count=0.5,
        w_member=0.5,
        w_flow_nll=2.0,
        w_crps_fpts=1.0,
        w_team_energy=0.25,
        flow_warmup_epochs=4,
        anchor_start_weight=1.0,
        anchor_end_weight=0.5,
        a2_scale=1.0,
    )
    assert weights.flow_warmup == pytest.approx(0.25)
    assert weights.anchor_weight == pytest.approx(0.875)
    assert weights.w_minutes == pytest.approx(0.875)
    assert weights.w_count == pytest.approx(0.4375)
    assert weights.w_member == pytest.approx(0.4375)
    assert weights.w_minutes_nll == pytest.approx(1.0)
    assert weights.w_flow_nll == pytest.approx(0.5)
    assert weights.w_crps_fpts == pytest.approx(0.0)
    assert weights.w_team_energy == pytest.approx(0.0)


def test_resolve_minutes_teacher_forcing_prob_applies_linear_schedule() -> None:
    assert _resolve_minutes_teacher_forcing_prob(
        epoch=1,
        start_prob=1.0,
        end_prob=0.5,
        ramp_epochs=4,
    ) == pytest.approx(1.0)
    assert _resolve_minutes_teacher_forcing_prob(
        epoch=3,
        start_prob=1.0,
        end_prob=0.5,
        ramp_epochs=4,
    ) == pytest.approx(2.0 / 3.0)
    assert _resolve_minutes_teacher_forcing_prob(
        epoch=4,
        start_prob=1.0,
        end_prob=0.5,
        ramp_epochs=4,
    ) == pytest.approx(0.5)


def test_parse_prefix_csv_and_matches_prefix() -> None:
    prefixes = _parse_prefix_csv("active_head., minutes_head.,,encoder.")
    assert prefixes == ("active_head.", "minutes_head.", "encoder.")
    assert _matches_prefix("active_head.player_logits.weight", prefixes) is True
    assert _matches_prefix("flow_head.blocks.0.weight", prefixes) is False


def test_parse_args_accepts_props_aux_min_line_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_game_transformer_v2.py",
            "--props-pts-aux-min-line",
            "20",
            "--props-ast-aux-min-line",
            "6",
            "--props-reb-aux-min-line",
            "10",
        ],
    )
    args = parse_args()
    assert args.props_pts_aux_min_line == pytest.approx(20.0)
    assert args.props_ast_aux_min_line == pytest.approx(6.0)
    assert args.props_reb_aux_min_line == pytest.approx(10.0)


def test_parse_args_accepts_ast_factorization_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_game_transformer_v2.py",
            "--enable-team-ast-budget-head",
            "--team-ast-budget-head-hidden",
            "96",
            "--enable-assist-share-head",
            "--assist-share-head-hidden",
            "80",
            "--assist-share-condition-feature-cols",
            "an_ast_line,an_implied_minutes,prior_play_prob,started_proxy_rate_prior_20",
            "--assist-share-condition-hidden",
            "24",
            "--enable-ast-blend-gate",
            "--ast-blend-gate-hidden",
            "72",
            "--ast-blend-gate-init-alpha",
            "0.7",
            "--w-ast-blend-gate-aux",
            "0.06",
            "--ast-blend-gate-target-eps",
            "0.4",
            "--assist-share-replace-flow-ast",
            "--assist-share-factorized-ast",
            "--assist-share-reconcile-ast-budget",
            "--assist-share-reconcile-alpha",
            "0.65",
            "--assist-share-reconcile-temperature",
            "0.9",
            "--w-team-ast-budget-aux",
            "0.05",
            "--w-assist-share-aux",
            "0.08",
            "--w-assist-share-recon-aux",
            "0.03",
            "--team-ast-budget-target-scale",
            "9",
            "--assist-share-recon-target-scale",
            "4",
            "--assist-playmaker-line-center",
            "6.0",
            "--assist-playmaker-line-scale",
            "1.2",
            "--assist-playmaker-max-weight",
            "4.0",
            "--assist-underprediction-weight",
            "2.5",
        ],
    )
    args = parse_args()
    assert args.enable_team_ast_budget_head is True
    assert args.team_ast_budget_head_hidden == 96
    assert args.enable_assist_share_head is True
    assert args.assist_share_head_hidden == 80
    assert args.assist_share_condition_feature_cols == (
        "an_ast_line,an_implied_minutes,prior_play_prob,started_proxy_rate_prior_20"
    )
    assert args.assist_share_condition_hidden == 24
    assert args.enable_ast_blend_gate is True
    assert args.ast_blend_gate_hidden == 72
    assert args.ast_blend_gate_init_alpha == pytest.approx(0.7)
    assert args.w_ast_blend_gate_aux == pytest.approx(0.06)
    assert args.ast_blend_gate_target_eps == pytest.approx(0.4)
    assert args.assist_share_replace_flow_ast is True
    assert args.assist_share_factorized_ast is True
    assert args.assist_share_reconcile_ast_budget is True
    assert args.assist_share_reconcile_alpha == pytest.approx(0.65)
    assert args.assist_share_reconcile_temperature == pytest.approx(0.9)
    assert args.w_team_ast_budget_aux == pytest.approx(0.05)
    assert args.w_assist_share_aux == pytest.approx(0.08)
    assert args.w_assist_share_recon_aux == pytest.approx(0.03)
    assert args.team_ast_budget_target_scale == pytest.approx(9.0)
    assert args.assist_share_recon_target_scale == pytest.approx(4.0)
    assert args.assist_playmaker_line_center == pytest.approx(6.0)
    assert args.assist_playmaker_line_scale == pytest.approx(1.2)
    assert args.assist_playmaker_max_weight == pytest.approx(4.0)
    assert args.assist_underprediction_weight == pytest.approx(2.5)


def test_parse_args_accepts_rebound_factorization_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_game_transformer_v2.py",
            "--enable-team-rebound-budget-head",
            "--team-rebound-budget-head-hidden",
            "88",
            "--rebound-budget-parameterization",
            "dreb_rate",
            "--rebound-dreb-rate-cap",
            "0.8",
            "--enable-rebound-share-head",
            "--rebound-share-head-hidden",
            "72",
            "--rebound-factor-reconcile-oreb-dreb",
            "--rebound-factor-reconcile-mode",
            "dreb_only",
            "--rebound-factor-reconcile-alpha",
            "0.5",
            "--rebound-factor-reconcile-temperature",
            "0.85",
            "--rebound-dreb-budget-blend-alpha",
            "0.35",
            "--enable-rebound-budget-blend-gate",
            "--rebound-budget-blend-gate-hidden",
            "40",
            "--rebound-budget-blend-gate-init-alpha",
            "0.2",
            "--w-team-rebound-budget-rate-aux",
            "0.7",
            "--w-rebound-budget-blend-gate-aux",
            "0.6",
        ],
    )
    args = parse_args()
    assert args.enable_team_rebound_budget_head is True
    assert args.team_rebound_budget_head_hidden == 88
    assert args.rebound_budget_parameterization == "dreb_rate"
    assert args.rebound_dreb_rate_cap == pytest.approx(0.8)
    assert args.enable_rebound_share_head is True
    assert args.rebound_share_head_hidden == 72
    assert args.rebound_factor_reconcile_oreb_dreb is True
    assert args.rebound_factor_reconcile_mode == "dreb_only"
    assert args.rebound_factor_reconcile_alpha == pytest.approx(0.5)
    assert args.rebound_factor_reconcile_temperature == pytest.approx(0.85)
    assert args.rebound_dreb_budget_blend_alpha == pytest.approx(0.35)
    assert args.enable_rebound_budget_blend_gate is True
    assert args.rebound_budget_blend_gate_hidden == 40
    assert args.rebound_budget_blend_gate_init_alpha == pytest.approx(0.2)
    assert args.w_team_rebound_budget_rate_aux == pytest.approx(0.7)
    assert args.w_rebound_budget_blend_gate_aux == pytest.approx(0.6)


def test_parse_args_accepts_dreb_rate_residual_parameterization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_game_transformer_v2.py",
            "--enable-team-rebound-budget-head",
            "--rebound-budget-parameterization",
            "dreb_rate_residual",
            "--rebound-dreb-rate-cap",
            "0.12",
        ],
    )
    args = parse_args()
    assert args.enable_team_rebound_budget_head is True
    assert args.rebound_budget_parameterization == "dreb_rate_residual"
    assert args.rebound_dreb_rate_cap == pytest.approx(0.12)


def test_parse_args_accepts_dreb_deterministic_parameterization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_game_transformer_v2.py",
            "--rebound-budget-parameterization",
            "dreb_deterministic",
            "--rebound-dreb-deterministic-discount",
            "0.9054",
            "--enable-rebound-share-head",
            "--rebound-factor-reconcile-oreb-dreb",
            "--rebound-factor-reconcile-mode",
            "dreb_only",
        ],
    )
    args = parse_args()
    assert args.rebound_budget_parameterization == "dreb_deterministic"
    assert args.rebound_dreb_deterministic_discount == pytest.approx(0.9054)
    assert args.enable_team_rebound_budget_head is False
    assert args.enable_rebound_share_head is True
    assert args.rebound_factor_reconcile_oreb_dreb is True
    assert args.rebound_factor_reconcile_mode == "dreb_only"


def test_parse_args_accepts_rebound_share_condition_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_game_transformer_v2.py",
            "--enable-rebound-share-head",
            "--rebound-share-head-hidden",
            "96",
            "--rebound-share-condition-feature-cols",
            "an_reb_line,an_implied_minutes,prior_play_prob,started_proxy_rate_prior_20",
            "--rebound-share-condition-hidden",
            "24",
        ],
    )
    args = parse_args()
    assert args.enable_rebound_share_head is True
    assert args.rebound_share_head_hidden == 96
    assert args.rebound_share_condition_feature_cols == (
        "an_reb_line,an_implied_minutes,prior_play_prob,started_proxy_rate_prior_20"
    )
    assert args.rebound_share_condition_hidden == 24


def test_parse_args_accepts_rebound_oreb_flow_budget_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_game_transformer_v2.py",
            "--rebound-factor-reconcile-oreb-dreb",
            "--rebound-factor-reconcile-mode",
            "both",
            "--rebound-oreb-reconcile-use-flow-budget",
            "--enable-rebound-share-head",
        ],
    )
    args = parse_args()
    assert args.rebound_factor_reconcile_oreb_dreb is True
    assert args.rebound_factor_reconcile_mode == "both"
    assert args.rebound_oreb_reconcile_use_flow_budget is True


def test_parse_args_accepts_flow_anchor_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_game_transformer_v2.py",
            "--flow-anchor-teacher-run-dir",
            "/tmp/teacher",
            "--w-flow-anchor-nonast-aux",
            "0.02",
            "--flow-anchor-target-scale",
            "6.0",
        ],
    )
    args = parse_args()
    assert args.flow_anchor_teacher_run_dir == "/tmp/teacher"
    assert args.w_flow_anchor_nonast_aux == pytest.approx(0.02)
    assert args.flow_anchor_target_scale == pytest.approx(6.0)


def test_mask_ast_from_flow_targets_zeroes_ast_and_observed_flag() -> None:
    flow_targets = torch.tensor([[[1.0, 2.0, 3.0]]])
    flow_observed = torch.tensor([[[True, True, True]]])
    masked_targets, masked_observed = _mask_ast_from_flow_targets(
        flow_targets,
        flow_observed,
        flow_target_columns=["pts", "ast", "reb"],
    )
    assert torch.equal(masked_targets, torch.tensor([[[1.0, 0.0, 3.0]]]))
    assert torch.equal(masked_observed, torch.tensor([[[True, False, True]]]))


def test_build_lineup_available_example_sampling_weights_scales_with_coverage() -> None:
    class _Ex:
        def __init__(self, valid: np.ndarray, lineup: np.ndarray) -> None:
            self.player_valid_mask = valid
            self.lineup_available = lineup

    examples = [
        _Ex(
            valid=np.array([[1, 1], [1, 1]], dtype=bool),
            lineup=np.array([[0, 0], [0, 0]], dtype=bool),
        ),
        _Ex(
            valid=np.array([[1, 1], [1, 1]], dtype=bool),
            lineup=np.array([[1, 0], [1, 0]], dtype=bool),
        ),
        _Ex(
            valid=np.array([[1, 1], [1, 1]], dtype=bool),
            lineup=np.array([[1, 1], [1, 1]], dtype=bool),
        ),
    ]

    weights, meta = _build_lineup_available_example_sampling_weights(
        examples,
        lineup_available_weight=3.0,
    )

    assert weights.dtype == torch.double
    assert weights.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert meta["lineup_fraction_mean"] == pytest.approx(0.5)
    assert meta["sample_weight_min"] == pytest.approx(1.0)
    assert meta["sample_weight_max"] == pytest.approx(3.0)


def test_build_lineup_available_example_sampling_weights_handles_empty_examples() -> None:
    weights, meta = _build_lineup_available_example_sampling_weights(
        [],
        lineup_available_weight=2.5,
    )
    assert tuple(weights.shape) == (0,)
    assert weights.dtype == torch.double
    assert meta["sample_weight_mean"] == pytest.approx(1.0)
    assert meta["lineup_weight_target"] == pytest.approx(2.5)


def test_build_sparse_candidate_example_sampling_weights_flags_candidate_games() -> None:
    class _Ex:
        def __init__(self, valid: np.ndarray, starter: np.ndarray, feats: np.ndarray) -> None:
            self.player_valid_mask = valid
            self.starter_force_active_worlds = starter
            self.player_features = feats

    feature_columns = [
        "minutes_from_stints_prior_20",
        "recent_start_pct_10",
        "started_proxy_rate_prior_10",
        "started_proxy_rate_prior_20",
    ]
    feature_mean = np.zeros((len(feature_columns),), dtype=np.float32)
    feature_std = np.ones((len(feature_columns),), dtype=np.float32)
    valid = np.array([[1, 1], [1, 1]], dtype=bool)

    no_candidate_feats = np.array(
        [
            [[20.0, 0.5, 0.5, 0.5], [20.0, 0.5, 0.5, 0.5]],
            [[20.0, 0.5, 0.5, 0.5], [20.0, 0.5, 0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    candidate_feats = np.array(
        [
            [[10.0, 0.1, 0.1, 0.1], [20.0, 0.5, 0.5, 0.5]],
            [[20.0, 0.5, 0.5, 0.5], [20.0, 0.5, 0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    starter_none = np.zeros((2, 2), dtype=bool)
    starter_one = np.array([[1, 0], [0, 0]], dtype=bool)
    examples = [
        _Ex(valid, starter_none, no_candidate_feats),
        _Ex(valid, starter_one, candidate_feats),
    ]

    weights, meta = _build_sparse_candidate_example_sampling_weights(
        examples,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
        sparse_candidate_weight=4.0,
        prior_minutes_max=14.0,
        hist_start_rate_max=0.25,
    )

    assert weights.dtype == torch.double
    assert weights.tolist() == pytest.approx([1.0, 4.0])
    assert meta["candidate_game_rate"] == pytest.approx(0.5)
    assert meta["sample_weight_min"] == pytest.approx(1.0)
    assert meta["sample_weight_max"] == pytest.approx(4.0)


def test_build_bench_riser_example_sampling_weights_flags_candidate_games() -> None:
    class _Ex:
        def __init__(self, valid: np.ndarray, starter: np.ndarray, feats: np.ndarray) -> None:
            self.player_valid_mask = valid
            self.starter_force_active_worlds = starter
            self.player_features = feats

    feature_columns = [
        "minutes_from_stints_prior_20",
        "prior_play_prob",
        "recent_start_pct_10",
        "started_proxy_rate_prior_10",
        "started_proxy_rate_prior_20",
    ]
    feature_mean = np.zeros((len(feature_columns),), dtype=np.float32)
    feature_std = np.ones((len(feature_columns),), dtype=np.float32)
    valid = np.array([[1, 1], [1, 1]], dtype=bool)

    no_candidate_feats = np.array(
        [
            [[4.0, 0.3, 0.2, 0.2, 0.2], [4.0, 0.3, 0.2, 0.2, 0.2]],
            [[4.0, 0.3, 0.2, 0.2, 0.2], [4.0, 0.3, 0.2, 0.2, 0.2]],
        ],
        dtype=np.float32,
    )
    candidate_feats = np.array(
        [
            [[12.0, 0.9, 0.1, 0.1, 0.1], [4.0, 0.3, 0.2, 0.2, 0.2]],
            [[4.0, 0.3, 0.2, 0.2, 0.2], [4.0, 0.3, 0.2, 0.2, 0.2]],
        ],
        dtype=np.float32,
    )
    starter_none = np.zeros((2, 2), dtype=bool)
    starter_one = np.array([[0, 0], [1, 0]], dtype=bool)
    examples = [
        _Ex(valid, starter_none, no_candidate_feats),
        _Ex(valid, starter_one, candidate_feats),
    ]

    weights, meta = _build_bench_riser_example_sampling_weights(
        examples,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
        bench_candidate_weight=3.0,
        prior_minutes_min=8.0,
        hist_start_rate_max=0.35,
        prior_play_prob_min=0.80,
    )

    assert weights.dtype == torch.double
    assert weights.tolist() == pytest.approx([1.0, 3.0])
    assert meta["candidate_game_rate"] == pytest.approx(0.5)
    assert meta["sample_weight_min"] == pytest.approx(1.0)
    assert meta["sample_weight_max"] == pytest.approx(3.0)


def test_freeze_parameter_prefixes_freezes_only_matching_tensors() -> None:
    model = nn.Module()
    model.active_head = nn.Linear(3, 2)
    model.minutes_head = nn.Linear(3, 2)
    model.flow_head = nn.Linear(3, 2)

    _freeze_parameter_prefixes(
        model,
        prefixes=("active_head.", "minutes_head."),
        label="unit-test",
    )

    for name, param in model.named_parameters():
        if name.startswith(("active_head.", "minutes_head.")):
            assert param.requires_grad is False
        else:
            assert param.requires_grad is True


def test_update_phase2_nll_guard_halves_a2_after_two_consecutive_explosions() -> None:
    cfg = Phase2StabilityConfig(
        nll_explosion_ratio=2.0,
        nll_explosion_abs=10.0,
        nll_ema_alpha=0.1,
        nll_backoff_consecutive_batches=2,
        max_backoffs_before_rollback=3,
        min_a2_scale=0.125,
    )
    state = Phase2StabilityState(a2_scale=1.0, ema_gen_nll=4.0)

    exploded_1, backoff_1, _ = _update_phase2_nll_guard(
        epoch=1,
        batch_idx=1,
        gen_nll=20.0,
        config=cfg,
        state=state,
    )
    assert exploded_1 is True
    assert backoff_1 is False
    assert state.a2_scale == pytest.approx(1.0)

    exploded_2, backoff_2, _ = _update_phase2_nll_guard(
        epoch=1,
        batch_idx=2,
        gen_nll=21.0,
        config=cfg,
        state=state,
    )
    assert exploded_2 is True
    assert backoff_2 is True
    assert state.a2_scale == pytest.approx(0.5)
    assert state.backoff_count == 1
    assert state.rollback_requested is False


def test_update_phase2_nll_guard_requests_rollback_on_repeated_instability() -> None:
    cfg = Phase2StabilityConfig(
        nll_explosion_ratio=1.5,
        nll_explosion_abs=5.0,
        nll_ema_alpha=0.1,
        nll_backoff_consecutive_batches=1,
        max_backoffs_before_rollback=2,
        min_a2_scale=0.125,
    )
    state = Phase2StabilityState(a2_scale=1.0, ema_gen_nll=3.0)

    _update_phase2_nll_guard(
        epoch=1,
        batch_idx=1,
        gen_nll=50.0,
        config=cfg,
        state=state,
    )
    assert state.rollback_requested is False
    assert state.backoff_count == 1

    _update_phase2_nll_guard(
        epoch=1,
        batch_idx=2,
        gen_nll=60.0,
        config=cfg,
        state=state,
    )
    assert state.rollback_requested is True
    assert state.backoff_count == 2
    assert state.rollback_reason is not None


def test_resolve_phase2_epoch_weights_enables_phase3_decision_weights() -> None:
    weights = _resolve_phase2_epoch_weights(
        epoch=2,
        enable_phase2_flow=True,
        enable_phase3_decision=True,
        w_minutes=1.0,
        w_minutes_nll=1.0,
        w_count=0.5,
        w_member=0.5,
        w_flow_nll=1.0,
        w_crps_fpts=0.8,
        w_team_energy=0.2,
        flow_warmup_epochs=4,
        anchor_start_weight=1.0,
        anchor_end_weight=0.5,
        a2_scale=1.0,
    )
    assert weights.run_phase3_decision is True
    assert weights.w_crps_fpts == pytest.approx(0.8)
    assert weights.w_team_energy == pytest.approx(0.2)


def test_resolve_backbone_epoch_weights_applies_linear_ramp() -> None:
    weights = _resolve_backbone_epoch_weights(
        epoch=1,
        enable_possession_backbone=True,
        enable_three_pa_share=True,
        w_poss_nll=0.2,
        w_backbone_nll=0.1,
        w_three_pa_nll=0.05,
        w_poss_regression=5.0,
        loss_ramp_epochs=5,
        poss_loss_start_scale=0.1,
        backbone_loss_start_scale=0.2,
        three_pa_loss_start_scale=0.4,
        poss_regression_start_scale=0.2,
    )
    assert isinstance(weights, BackboneEpochWeights)
    assert weights.w_poss_nll == pytest.approx(0.02)
    assert weights.w_backbone_nll == pytest.approx(0.02)
    assert weights.w_three_pa_nll == pytest.approx(0.02)
    assert weights.w_poss_regression == pytest.approx(1.0)  # 5.0 * 0.2
    assert weights.ramp_scale_poss == pytest.approx(0.1)
    assert weights.ramp_scale_backbone == pytest.approx(0.2)
    assert weights.ramp_scale_three_pa == pytest.approx(0.4)
    assert weights.ramp_scale_poss_regression == pytest.approx(0.2)


def test_count_backbone_coupled_epochs_accounts_for_detach_schedule() -> None:
    assert _count_backbone_coupled_epochs(
        epoch=1,
        enable_possession_backbone=True,
        backbone_detach_until_epoch=0,
    ) == 1
    assert _count_backbone_coupled_epochs(
        epoch=3,
        enable_possession_backbone=True,
        backbone_detach_until_epoch=4,
    ) == 0
    assert _count_backbone_coupled_epochs(
        epoch=4,
        enable_possession_backbone=True,
        backbone_detach_until_epoch=4,
    ) == 1
    assert _count_backbone_coupled_epochs(
        epoch=6,
        enable_possession_backbone=True,
        backbone_detach_until_epoch=4,
    ) == 3


def test_update_early_stop_requests_stop_after_patience_exhausted() -> None:
    cfg = EarlyStopConfig(patience=2, min_delta=0.01, min_epochs=2, min_coupled_epochs=0)
    state = EarlyStopState()

    assert _update_early_stop(epoch=1, metric_value=5.0, coupled_epochs=0, config=cfg, state=state) is False
    assert state.best_epoch == 1
    assert state.best_metric == pytest.approx(5.0)

    assert _update_early_stop(epoch=2, metric_value=4.95, coupled_epochs=0, config=cfg, state=state) is False
    assert state.best_epoch == 2
    assert state.best_metric == pytest.approx(4.95)

    assert _update_early_stop(epoch=3, metric_value=4.955, coupled_epochs=0, config=cfg, state=state) is False
    assert state.bad_epochs == 1
    assert state.stop_requested is False

    assert _update_early_stop(epoch=4, metric_value=4.958, coupled_epochs=0, config=cfg, state=state) is True
    assert state.bad_epochs == 2
    assert state.stop_requested is True
    assert state.stop_epoch == 4
    assert state.best_epoch == 2
    assert state.stop_reason is not None


def test_update_early_stop_respects_min_epochs_gate() -> None:
    cfg = EarlyStopConfig(patience=1, min_delta=0.01, min_epochs=3, min_coupled_epochs=0)
    state = EarlyStopState(best_metric=5.0, best_epoch=1)

    assert _update_early_stop(epoch=2, metric_value=5.02, coupled_epochs=0, config=cfg, state=state) is False
    assert state.bad_epochs == 0
    assert state.stop_requested is False


def test_update_early_stop_respects_min_coupled_epochs_gate() -> None:
    cfg = EarlyStopConfig(patience=1, min_delta=0.01, min_epochs=0, min_coupled_epochs=2)
    state = EarlyStopState(best_metric=5.0, best_epoch=1)

    assert _update_early_stop(epoch=2, metric_value=5.02, coupled_epochs=1, config=cfg, state=state) is False
    assert state.bad_epochs == 0
    assert state.stop_requested is False

    assert _update_early_stop(epoch=3, metric_value=5.03, coupled_epochs=2, config=cfg, state=state) is True
    assert state.bad_epochs == 1
    assert state.stop_requested is True


def test_resolve_early_stop_metric_value_val_total_ex_possreg() -> None:
    v = _resolve_early_stop_metric_value(
        metric_name="val_total_ex_possreg",
        val_total=12.0,
        val_poss_regression=3.0,
        w_poss_regression=2.0,
        val_minutes_mae=4.5,
    )
    assert v == pytest.approx(6.0)


def test_resolve_early_stop_metric_value_val_minutes_mae() -> None:
    v = _resolve_early_stop_metric_value(
        metric_name="val_minutes_mae",
        val_total=12.0,
        val_poss_regression=3.0,
        w_poss_regression=2.0,
        val_minutes_mae=4.5,
    )
    assert v == pytest.approx(4.5)


def test_resolve_early_stop_metric_value_raises_on_unknown_metric() -> None:
    with pytest.raises(ValueError):
        _resolve_early_stop_metric_value(
            metric_name="unknown_metric",
            val_total=1.0,
            val_poss_regression=0.0,
            w_poss_regression=0.0,
            val_minutes_mae=1.0,
        )


def test_team_sum_by_side_respects_observed_mask() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    valid_mask = torch.tensor([[True, True, True, True]])
    team_index = torch.tensor([[0, 0, 1, 1]])
    observed_mask = torch.tensor([[True, False, True, True]])

    totals, seen = _team_sum_by_side(
        values=values,
        valid_mask=valid_mask,
        team_index=team_index,
        observed_mask=observed_mask,
    )

    assert totals.shape == (1, 2)
    assert seen.shape == (1, 2)
    assert totals.tolist() == [[1.0, 7.0]]
    assert seen.tolist() == [[True, True]]


def test_team_ratio_mse_loss_detaches_pred_denominator_and_masks_missing_rows() -> None:
    pred_num = torch.tensor([[8.0, 3.0], [5.0, 2.0]])
    pred_den = torch.tensor([[10.0, 6.0], [0.0, 0.0]], requires_grad=True)
    true_num = torch.tensor([[7.0, 4.0], [1.0, 1.0]])
    true_den = torch.tensor([[10.0, 8.0], [0.0, 0.0]])
    observed_mask = torch.tensor([[True, True], [True, True]])

    loss = _team_ratio_mse_loss(
        pred_numerator=pred_num,
        pred_denominator=pred_den,
        true_numerator=true_num,
        true_denominator=true_den,
        observed_mask=observed_mask,
    )

    # Row 2 should be ignored because the true denominator is zero.
    expected = (((0.8 - 0.7) ** 2) + ((0.5 - 0.5) ** 2)) / 2.0
    assert loss.item() == pytest.approx(expected)


def test_team_fixed_opportunity_rate_mse_loss_uses_true_budget_and_caps_overflow() -> None:
    pred_num = torch.tensor([[12.0, 3.0], [5.0, 1.0]])
    true_num = torch.tensor([[7.0, 4.0], [0.0, 0.0]])
    true_den = torch.tensor([[10.0, 8.0], [0.0, 0.0]])
    observed_mask = torch.tensor([[True, True], [True, True]])

    loss = _team_fixed_opportunity_rate_mse_loss(
        pred_numerator=pred_num,
        true_numerator=true_num,
        true_denominator=true_den,
        observed_mask=observed_mask,
    )

    # Row 1 side 0 clips from 1.2 to 1.0 against a true rate of 0.7.
    expected = (((1.0 - 0.7) ** 2) + ((3.0 / 8.0 - 4.0 / 8.0) ** 2)) / 2.0
    assert loss.item() == pytest.approx(expected)


def test_weighted_masked_scaled_huber_loss_matches_unweighted_with_unit_weights() -> None:
    pred = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float32)
    target = torch.tensor([2.0, 1.0, 2.0], dtype=torch.float32)
    mask = torch.tensor([True, True, False])
    weights = torch.ones_like(pred)
    loss = _weighted_masked_scaled_huber_loss(
        pred=pred,
        target=target,
        mask=mask,
        scale=1.0,
        delta=1.0,
        weights=weights,
    )
    # Huber(delta=1): errors [1,2] -> [0.5, 1.5], mean = 1.0
    assert loss.item() == pytest.approx(1.0)


def test_weighted_masked_scaled_huber_loss_returns_zero_when_mask_empty() -> None:
    pred = torch.tensor([1.0, 2.0], dtype=torch.float32)
    target = torch.tensor([1.5, 2.5], dtype=torch.float32)
    mask = torch.tensor([False, False])
    weights = torch.tensor([0.2, 0.8], dtype=torch.float32)
    loss = _weighted_masked_scaled_huber_loss(
        pred=pred,
        target=target,
        mask=mask,
        scale=2.0,
        delta=1.0,
        weights=weights,
    )
    assert loss.item() == pytest.approx(0.0)


def test_compute_named_direct_stat_losses_and_group_mean() -> None:
    losses = _compute_named_direct_stat_losses(
        stat_specs={
            "pts": (
                torch.tensor([10.0, 8.0]),
                torch.tensor([12.0, 8.0]),
                torch.tensor([True, True]),
                8.0,
            ),
            "ast": (
                torch.tensor([4.0, 3.0]),
                torch.tensor([4.0, 5.0]),
                torch.tensor([True, False]),
                3.0,
            ),
            "fga": (
                torch.tensor([11.0, 7.0]),
                torch.tensor([9.0, 7.0]),
                torch.tensor([True, True]),
                8.0,
            ),
        },
        delta=1.0,
    )
    assert set(losses) == {"pts", "ast", "fga"}
    assert float(losses["pts"].item()) > 0.0
    assert float(losses["ast"].item()) == pytest.approx(0.0)
    group_loss = _mean_named_losses(losses, ("pts", "ast"))
    expected = (float(losses["pts"].item()) + float(losses["ast"].item())) / 2.0
    assert float(group_loss.item()) == pytest.approx(expected)


def test_minutes_hurdle_nll_combines_zero_bce_and_positive_gaussian_terms() -> None:
    pred = torch.tensor([[0.2, 30.0, 10.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 32.0, 0.0]], dtype=torch.float32)
    zero_logits = torch.tensor([[3.0, -2.0, 1.0]], dtype=torch.float32)
    sigma = torch.tensor([[1.5, 2.0, 3.0]], dtype=torch.float32)
    valid = torch.tensor([[True, True, True]], dtype=torch.bool)

    loss = _minutes_hurdle_nll(
        pred_minutes=pred,
        target_minutes=target,
        zero_logits=zero_logits,
        sigma=sigma,
        valid_mask=valid,
        zero_threshold=0.5,
    )

    assert math.isfinite(float(loss.item()))
    assert float(loss.item()) > 0.0


def test_minutes_hurdle_nll_raises_on_shape_mismatch() -> None:
    pred = torch.zeros((1, 3), dtype=torch.float32)
    target = torch.zeros((1, 3), dtype=torch.float32)
    zero_logits = torch.zeros((1, 2), dtype=torch.float32)
    sigma = torch.ones((1, 3), dtype=torch.float32)
    valid = torch.ones((1, 3), dtype=torch.bool)

    with pytest.raises(ValueError):
        _minutes_hurdle_nll(
            pred_minutes=pred,
            target_minutes=target,
            zero_logits=zero_logits,
            sigma=sigma,
            valid_mask=valid,
            zero_threshold=0.5,
        )


def test_compute_sparse_rerank_score_penalizes_underprediction_and_shortfall() -> None:
    score = _compute_sparse_rerank_score(
        sparse_next_up_underpred_rate=0.6,
        active_count_mae=1.2,
        starter_sparse_pred_minutes_mean=4.0,
        target_starter_sparse_minutes=5.0,
        weight_sparse_underpred=6.0,
        weight_active_count_mae=1.0,
        weight_starter_sparse_shortfall=0.5,
    )
    assert score == pytest.approx(5.3)


def test_select_sparse_rerank_candidate_respects_minutes_tolerance_gate() -> None:
    selected = _select_sparse_rerank_candidate(
        candidates=[
            {
                "epoch": 10,
                "val_minutes_mae": 3.00,
                "sparse_rerank": {"sparse_score": 5.0},
            },
            {
                "epoch": 11,
                "val_minutes_mae": 3.20,
                "sparse_rerank": {"sparse_score": 1.0},
            },
            {
                "epoch": 12,
                "val_minutes_mae": 3.05,
                "sparse_rerank": {"sparse_score": 2.0},
            },
        ],
        minutes_mae_tolerance=0.08,
    )
    assert selected is not None
    assert selected["epoch"] == 12


def test_record_topk_minutes_checkpoint_keeps_lowest_minutes_mae(tmp_path) -> None:
    candidates: list[MinutesCheckpointCandidate] = []
    p1 = tmp_path / "epoch_001.pt"
    p2 = tmp_path / "epoch_002.pt"
    p3 = tmp_path / "epoch_003.pt"
    for p in (p1, p2, p3):
        p.write_text("x", encoding="utf-8")

    candidates = _record_topk_minutes_checkpoint(
        candidates=candidates,
        epoch=1,
        val_minutes_mae=3.2,
        checkpoint_metric_value=3.2,
        val_total=4.0,
        checkpoint_path=p1,
        top_k=2,
    )
    candidates = _record_topk_minutes_checkpoint(
        candidates=candidates,
        epoch=2,
        val_minutes_mae=3.0,
        checkpoint_metric_value=3.0,
        val_total=3.9,
        checkpoint_path=p2,
        top_k=2,
    )
    candidates = _record_topk_minutes_checkpoint(
        candidates=candidates,
        epoch=3,
        val_minutes_mae=3.4,
        checkpoint_metric_value=3.4,
        val_total=4.1,
        checkpoint_path=p3,
        top_k=2,
    )

    assert [item.epoch for item in candidates] == [2, 1]
    assert p1.exists() is True
    assert p2.exists() is True
    assert p3.exists() is False
