from __future__ import annotations

import pytest

from scripts.rotation.train_game_transformer_v2 import (
    Phase2StabilityConfig,
    Phase2StabilityState,
    _resolve_phase2_epoch_weights,
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
