from __future__ import annotations

import pytest
import torch

from scripts.rotation.train_game_transformer_v2 import (
    BackboneEpochWeights,
    EarlyStopConfig,
    EarlyStopState,
    Phase2StabilityConfig,
    Phase2StabilityState,
    _count_backbone_coupled_epochs,
    _resolve_backbone_epoch_weights,
    _resolve_early_stop_metric_value,
    _resolve_phase2_epoch_weights,
    _team_fixed_opportunity_rate_mse_loss,
    _team_ratio_mse_loss,
    _team_sum_by_side,
    _update_early_stop,
    _update_phase2_nll_guard,
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
    )
    assert v == pytest.approx(6.0)


def test_resolve_early_stop_metric_value_raises_on_unknown_metric() -> None:
    with pytest.raises(ValueError):
        _resolve_early_stop_metric_value(
            metric_name="unknown_metric",
            val_total=1.0,
            val_poss_regression=0.0,
            w_poss_regression=0.0,
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
