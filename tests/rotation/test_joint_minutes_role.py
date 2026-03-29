from __future__ import annotations

import pytest
import torch

from projections.rotation.joint_minutes import JointMinutesHead, build_minutes_role_targets
from scripts.rotation.train_game_transformer_v2 import _build_minutes_role_targets_contextual


def test_build_minutes_role_targets_assigns_team_relative_buckets() -> None:
    minutes = torch.tensor([[35.0, 33.0, 28.0, 25.0, 21.0, 18.0, 12.0, 9.0, 5.0, 0.0]], dtype=torch.float32)
    valid = torch.ones_like(minutes, dtype=torch.bool)
    team_index = torch.zeros_like(minutes, dtype=torch.long)

    role_targets = build_minutes_role_targets(
        minutes,
        valid,
        team_index,
        active_threshold=4.0,
    )

    expected = torch.tensor([[4, 4, 3, 3, 3, 2, 2, 2, 1, 0]], dtype=torch.long)
    assert torch.equal(role_targets, expected)


def test_joint_minutes_head_with_role_head_emits_role_outputs_and_team_totals() -> None:
    head = JointMinutesHead(
        d_model=16,
        hidden_dim=32,
        dropout=0.0,
        enable_role_head=True,
        role_hidden_dim=16,
        role_embedding_dim=8,
        num_role_classes=5,
    )
    player_states = torch.randn(1, 10, 16)
    team_states = torch.randn(1, 2, 16)
    player_team_index = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long)
    valid_mask = torch.ones(1, 10, dtype=torch.bool)
    active_mask = torch.ones(1, 10, dtype=torch.bool)

    out = head(
        player_states,
        team_states,
        player_team_index,
        valid_mask,
        active_mask,
    )

    assert out.role_logits is not None
    assert out.role_probs is not None
    assert out.role_logits.shape == (1, 10, 5)
    assert out.role_probs.shape == (1, 10, 5)
    assert out.minutes.shape == (1, 10)

    minutes = out.minutes[0]
    for team_idx in (0, 1):
        mask = player_team_index[0] == team_idx
        assert float(minutes[mask].sum().item()) == pytest.approx(240.0, abs=1e-3)


def test_joint_minutes_head_with_starter_promotion_emits_delta() -> None:
    head = JointMinutesHead(
        d_model=16,
        hidden_dim=32,
        dropout=0.0,
        enable_starter_promotion_head=True,
        starter_promotion_hidden_dim=16,
    )
    player_states = torch.randn(1, 10, 16)
    team_states = torch.randn(1, 2, 16)
    player_team_index = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long)
    valid_mask = torch.ones(1, 10, dtype=torch.bool)
    active_mask = torch.ones(1, 10, dtype=torch.bool)
    starter_hint = torch.tensor([[1, 1, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=torch.bool)

    out = head(
        player_states,
        team_states,
        player_team_index,
        valid_mask,
        active_mask,
        starter_hint_mask=starter_hint,
    )

    assert out.starter_promotion_delta is not None
    assert out.starter_promotion_delta.shape == (1, 10)
    assert torch.all(out.starter_promotion_delta[~starter_hint] == 0.0)


def test_contextual_role_targets_separate_starter_fillin_from_core() -> None:
    minutes = torch.tensor([[32.0, 31.0, 20.0, 11.0, 0.0]], dtype=torch.float32)
    valid = torch.ones_like(minutes, dtype=torch.bool)
    lineup_starter = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    hist_start = torch.tensor([[0.8, 0.1, 0.0, 0.0, 0.0]], dtype=torch.float32)
    prior_minutes = torch.tensor([[30.0, 8.0, 18.0, 6.0, 0.0]], dtype=torch.float32)

    role_targets = _build_minutes_role_targets_contextual(
        y_minutes=minutes,
        valid_mask=valid,
        active_threshold=4.0,
        lineup_starter_announced=lineup_starter,
        historical_start_rate=hist_start,
        prior_minutes=prior_minutes,
    )

    expected = torch.tensor([[4, 3, 2, 1, 0]], dtype=torch.long)
    assert torch.equal(role_targets, expected)
