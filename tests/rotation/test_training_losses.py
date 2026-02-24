"""Unit tests for rotation training loss helpers."""

from __future__ import annotations

import pytest
import torch

from projections.rotation.training_losses import (
    build_in_rotation_labels,
    compute_anti_smear_penalty,
    compute_crps_loss,
    compute_k_hat,
    compute_k_regularizer,
    compute_minutes_out_loss,
    compute_team_energy_score,
)


class TestBuildInRotationLabels:
    """Tests for build_in_rotation_labels function."""

    def test_basic_thresholding(self) -> None:
        """Labels should be 1 for minutes >= threshold, 0 otherwise."""
        y_minutes = torch.tensor([[5.0, 10.0, 2.0], [15.0, 6.0, 0.0]])
        mask = torch.ones(2, 3, dtype=torch.bool)
        threshold = 6.0

        result = build_in_rotation_labels(y_minutes, threshold, mask)

        expected = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        assert result.shape == y_minutes.shape
        torch.testing.assert_close(result, expected)

    def test_masked_slots_get_zero(self) -> None:
        """Masked-out slots should have label 0 regardless of minutes."""
        y_minutes = torch.tensor([[20.0, 30.0, 40.0]])
        mask = torch.tensor([[True, False, True]])
        threshold = 6.0

        result = build_in_rotation_labels(y_minutes, threshold, mask)

        expected = torch.tensor([[1.0, 0.0, 1.0]])
        torch.testing.assert_close(result, expected)

    def test_threshold_boundary(self) -> None:
        """Minutes exactly at threshold should be labeled as in rotation."""
        y_minutes = torch.tensor([[6.0, 5.99, 6.01]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        threshold = 6.0

        result = build_in_rotation_labels(y_minutes, threshold, mask)

        expected = torch.tensor([[1.0, 0.0, 1.0]])
        torch.testing.assert_close(result, expected)


class TestComputeKHat:
    """Tests for compute_k_hat function."""

    def test_shape_and_sum(self) -> None:
        """k_hat should be (B,) with sum of gate_prob per team."""
        gate_logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -10.0]])
        mask = torch.ones(2, 3, dtype=torch.bool)

        k_hat = compute_k_hat(gate_logits, mask)

        assert k_hat.shape == (2,)
        # sigmoid(0) = 0.5, so full mask row should give 3 * 0.5 = 1.5
        assert k_hat[0].item() == pytest.approx(1.5, abs=1e-5)
        # Third slot has sigmoid(-10) ≈ 0
        assert k_hat[1].item() == pytest.approx(1.0, abs=0.01)

    def test_mask_zeros_out_slots(self) -> None:
        """Masked-out slots should not contribute to k_hat."""
        gate_logits = torch.tensor([[10.0, 10.0, 10.0]])  # sigmoid ≈ 1
        mask = torch.tensor([[True, False, True]])

        k_hat = compute_k_hat(gate_logits, mask)

        # Only 2 slots contribute, each ≈ 1.0
        assert k_hat[0].item() == pytest.approx(2.0, abs=0.01)

    def test_gradient_flows(self) -> None:
        """k_hat should allow gradients to flow through."""
        gate_logits = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
        mask = torch.ones(1, 3, dtype=torch.bool)

        k_hat = compute_k_hat(gate_logits, mask)
        loss = k_hat.sum()
        loss.backward()

        assert gate_logits.grad is not None
        assert gate_logits.grad.shape == gate_logits.shape


class TestComputeKRegularizer:
    """Tests for compute_k_regularizer function."""

    def test_zero_when_at_target(self) -> None:
        """Loss should be zero when k_hat equals k_target."""
        # gate_logits = 0 gives sigmoid = 0.5
        # Need 19 players with sigmoid(0) = 0.5 to get k_hat = 9.5
        gate_logits = torch.zeros(1, 19)
        mask = torch.ones(1, 19, dtype=torch.bool)

        loss = compute_k_regularizer(gate_logits, mask, k_target=9.5)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_when_off_target(self) -> None:
        """Loss should be positive when k_hat differs from k_target."""
        gate_logits = torch.zeros(1, 10)  # k_hat = 5.0
        mask = torch.ones(1, 10, dtype=torch.bool)

        loss = compute_k_regularizer(gate_logits, mask, k_target=9.5)

        # (5.0 - 9.5)^2 = 20.25
        assert loss.item() == pytest.approx(20.25, abs=1e-4)

    def test_batch_averaging(self) -> None:
        """Loss should be mean over batch dimension."""
        gate_logits = torch.zeros(2, 10)  # Both teams get k_hat = 5.0
        mask = torch.ones(2, 10, dtype=torch.bool)

        loss = compute_k_regularizer(gate_logits, mask, k_target=9.5)

        # mean([(5-9.5)^2, (5-9.5)^2]) = 20.25
        assert loss.item() == pytest.approx(20.25, abs=1e-4)

    def test_tensor_target_per_team(self) -> None:
        """Regularizer should accept per-team tensor targets."""
        gate_logits = torch.zeros(2, 10)  # k_hat = [5.0, 5.0]
        mask = torch.ones(2, 10, dtype=torch.bool)
        targets = torch.tensor([5.0, 7.0], dtype=torch.float32)

        loss = compute_k_regularizer(gate_logits, mask, k_target=targets)

        # mean([(5-5)^2, (5-7)^2]) = 2.0
        assert loss.item() == pytest.approx(2.0, abs=1e-5)

    def test_tensor_target_shape_mismatch_raises(self) -> None:
        """Mismatched k_target tensor length should raise."""
        gate_logits = torch.zeros(2, 10)
        mask = torch.ones(2, 10, dtype=torch.bool)
        targets = torch.tensor([5.0], dtype=torch.float32)

        with pytest.raises(ValueError, match="batch size"):
            compute_k_regularizer(gate_logits, mask, k_target=targets)


class TestComputeAntiSmearPenalty:
    """Tests for compute_anti_smear_penalty function."""

    def test_zero_below_floor(self) -> None:
        """Penalty should be zero when pred_minutes <= floor."""
        pred_minutes = torch.tensor([[3.0, 4.0, 2.0]])
        gate_logits = torch.tensor([[-10.0, -10.0, -10.0]])  # low gate_prob
        mask = torch.ones(1, 3, dtype=torch.bool)

        penalty = compute_anti_smear_penalty(pred_minutes, gate_logits, mask, floor=4.0)

        assert penalty.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_above_floor_low_gate(self) -> None:
        """Penalty should be positive when pred > floor and gate_prob is low."""
        pred_minutes = torch.tensor([[8.0]])  # 4 above floor
        gate_logits = torch.tensor([[-100.0]])  # gate_prob ≈ 0
        mask = torch.ones(1, 1, dtype=torch.bool)

        penalty = compute_anti_smear_penalty(pred_minutes, gate_logits, mask, floor=4.0)

        # relu(8 - 4) * (1 - 0) = 4.0, mean over 1 element = 4.0
        assert penalty.item() == pytest.approx(4.0, abs=0.01)

    def test_attenuated_by_high_gate_prob(self) -> None:
        """Penalty should be reduced when gate_prob is high."""
        pred_minutes = torch.tensor([[8.0]])
        gate_logits = torch.tensor([[100.0]])  # gate_prob ≈ 1
        mask = torch.ones(1, 1, dtype=torch.bool)

        penalty = compute_anti_smear_penalty(pred_minutes, gate_logits, mask, floor=4.0)

        # relu(8 - 4) * (1 - 1) ≈ 0
        assert penalty.item() == pytest.approx(0.0, abs=0.01)

    def test_gate_logits_detached(self) -> None:
        """Gate logits should be detached (no gradient flow through gate)."""
        pred_minutes = torch.tensor([[8.0]], requires_grad=True)
        gate_logits = torch.tensor([[0.0]], requires_grad=True)
        mask = torch.ones(1, 1, dtype=torch.bool)

        penalty = compute_anti_smear_penalty(pred_minutes, gate_logits, mask, floor=4.0)
        penalty.backward()

        # pred_minutes should have gradient
        assert pred_minutes.grad is not None
        # gate_logits should NOT have gradient (detached)
        assert gate_logits.grad is None

    def test_mask_zeros_out_slots(self) -> None:
        """Masked-out slots should not contribute to penalty."""
        pred_minutes = torch.tensor([[8.0, 8.0, 8.0]])
        gate_logits = torch.tensor([[-100.0, -100.0, -100.0]])  # all low gate_prob
        mask = torch.tensor([[True, False, True]])

        penalty = compute_anti_smear_penalty(pred_minutes, gate_logits, mask, floor=4.0)

        # Only 2 valid slots, each contributes 4.0, mean = 4.0
        assert penalty.item() == pytest.approx(4.0, abs=0.01)


class TestComputeMinutesOutLoss:
    """Tests for compute_minutes_out_loss function."""

    def test_zero_for_rotation_players(self) -> None:
        """Loss should not penalize players marked as in rotation."""
        pred_minutes = torch.tensor([[20.0, 25.0, 30.0]])
        in_rotation = torch.tensor([[1.0, 1.0, 1.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        loss = compute_minutes_out_loss(pred_minutes, in_rotation, mask)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_penalizes_non_rotation_minutes(self) -> None:
        """Loss should penalize minutes for non-rotation players."""
        pred_minutes = torch.tensor([[5.0]])
        in_rotation = torch.tensor([[0.0]])
        mask = torch.ones(1, 1, dtype=torch.bool)

        loss = compute_minutes_out_loss(pred_minutes, in_rotation, mask)

        # abs(5.0) / 1 = 5.0
        assert loss.item() == pytest.approx(5.0, abs=1e-5)

    def test_mixed_rotation_status(self) -> None:
        """Loss should only penalize non-rotation players."""
        pred_minutes = torch.tensor([[20.0, 5.0, 3.0]])
        in_rotation = torch.tensor([[1.0, 0.0, 0.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        loss = compute_minutes_out_loss(pred_minutes, in_rotation, mask)

        # Only non-rotation: abs(5) + abs(3) = 8, divided by 2 = 4.0
        assert loss.item() == pytest.approx(4.0, abs=1e-5)

    def test_respects_mask(self) -> None:
        """Masked-out slots should not contribute to loss."""
        pred_minutes = torch.tensor([[5.0, 10.0, 15.0]])
        in_rotation = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.tensor([[True, False, True]])

        loss = compute_minutes_out_loss(pred_minutes, in_rotation, mask)

        # Valid non-rotation: abs(5) + abs(15) = 20, divided by 2 = 10.0
        assert loss.item() == pytest.approx(10.0, abs=1e-5)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through to pred_minutes."""
        pred_minutes = torch.tensor([[5.0, 3.0]], requires_grad=True)
        in_rotation = torch.tensor([[0.0, 0.0]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        loss = compute_minutes_out_loss(pred_minutes, in_rotation, mask)
        loss.backward()

        assert pred_minutes.grad is not None


class TestComputeCrpsLoss:
    """Tests for compute_crps_loss helper."""

    def test_crps_is_zero_for_perfect_degenerate_samples(self) -> None:
        samples = torch.tensor([[[10.0], [10.0], [10.0]]])  # (B=1,K=3,N=1)
        target = torch.tensor([[10.0]])
        mask = torch.tensor([[True]])
        loss = compute_crps_loss(samples, target, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_crps_matches_manual_value_single_player(self) -> None:
        samples = torch.tensor([[[1.0], [2.0], [3.0]]])  # K=3
        target = torch.tensor([[2.0]])
        mask = torch.tensor([[True]])
        loss = compute_crps_loss(samples, target, mask)
        # term1 = (1 + 0 + 1)/3 = 2/3
        # pairwise mean abs = 8/9 -> 0.5 * 8/9 = 4/9
        # crps = 2/3 - 4/9 = 2/9
        assert loss.item() == pytest.approx(2.0 / 9.0, abs=1e-6)

    def test_crps_respects_mask(self) -> None:
        samples = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])  # (1,3,2)
        target = torch.tensor([[2.0, 0.0]])
        mask = torch.tensor([[True, False]])
        loss = compute_crps_loss(samples, target, mask)
        assert loss.item() == pytest.approx(2.0 / 9.0, abs=1e-6)


class TestComputeTeamEnergyScore:
    """Tests for compute_team_energy_score helper."""

    def test_team_energy_zero_for_perfect_degenerate_samples(self) -> None:
        samples = torch.tensor([[[5.0, 7.0], [5.0, 7.0], [5.0, 7.0]]])  # (1,3,2)
        target = torch.tensor([[5.0, 7.0]])
        mask = torch.tensor([[True, True]])
        team_index = torch.tensor([[0, 0]])
        score = compute_team_energy_score(samples, target, mask, team_index)
        assert score.item() == pytest.approx(0.0, abs=1e-6)

    def test_team_energy_respects_team_partition(self) -> None:
        samples = torch.tensor(
            [
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            ]
        )  # (1,3,2)
        target = torch.tensor([[2.0, 3.0]])
        mask = torch.tensor([[True, True]])
        team_index = torch.tensor([[0, 1]])
        score = compute_team_energy_score(samples, target, mask, team_index)
        # With one player per team, team energy reduces to scalar CRPS per player.
        crps = compute_crps_loss(samples, target, mask)
        assert score.item() == pytest.approx(crps.item(), abs=1e-6)

    def test_team_energy_backward_is_finite_at_zero_distance(self) -> None:
        samples = torch.tensor(
            [[[5.0, 7.0], [5.0, 7.0], [5.0, 7.0]]],
            requires_grad=True,
        )
        target = torch.tensor([[5.0, 7.0]])
        mask = torch.tensor([[True, True]])
        team_index = torch.tensor([[0, 0]])

        score = compute_team_energy_score(samples, target, mask, team_index, eps=1e-6)
        score.backward()

        assert samples.grad is not None
        assert torch.isfinite(samples.grad).all()
