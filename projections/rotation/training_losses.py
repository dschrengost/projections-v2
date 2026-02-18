"""Training loss helpers for rotation-set minutes model.

This module provides:
- build_in_rotation_labels: Binary labels for rotation membership
- compute_k_hat: Expected rotation size per team-game from gate probabilities
- compute_anti_smear_penalty: Penalty for minutes predicted on low-gate-prob players
- compute_minutes_out_loss: Penalty encouraging non-rotation players toward zero minutes
"""

from __future__ import annotations

import torch


def build_in_rotation_labels(
    y_minutes: torch.Tensor,
    threshold: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Build binary in-rotation labels.

    Args:
        y_minutes: (B, N) ground-truth minutes per player
        threshold: minutes threshold for rotation membership (e.g., 6.0)
        mask: (B, N) boolean mask of valid roster slots

    Returns:
        (B, N) tensor with 1.0 for in-rotation, 0.0 otherwise (masked slots get 0)
    """
    in_rot = (y_minutes >= threshold).to(dtype=torch.float32)
    return in_rot * mask.to(dtype=torch.float32)


def compute_k_hat(
    gate_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute expected rotation size (k_hat) per team-game.

    Args:
        gate_logits: (B, N) gate logits from the model
        mask: (B, N) boolean mask of valid roster slots

    Returns:
        (B,) tensor with expected rotation size per team-game
    """
    gate_prob = torch.sigmoid(gate_logits) * mask.to(dtype=gate_logits.dtype)
    return gate_prob.sum(dim=1)


def compute_k_regularizer(
    gate_logits: torch.Tensor,
    mask: torch.Tensor,
    k_target: float | torch.Tensor,
) -> torch.Tensor:
    """Compute team expected-K regularizer loss.

    Encourages the sum of gate probabilities per team-game to be close to k_target.

    Args:
        gate_logits: (B, N) gate logits from the model
        mask: (B, N) boolean mask of valid roster slots
        k_target: target expected rotation size (scalar or per-team tensor)

    Returns:
        scalar loss: mean((k_hat - k_target)^2)
    """
    k_hat = compute_k_hat(gate_logits, mask)
    if isinstance(k_target, torch.Tensor):
        target = k_target.to(dtype=k_hat.dtype, device=k_hat.device)
        if target.ndim == 0:
            target = target.expand_as(k_hat)
        elif target.ndim == 1:
            if target.shape[0] != k_hat.shape[0]:
                raise ValueError("k_target tensor length must match batch size")
        else:
            raise ValueError("k_target tensor must be scalar or shape (B,)")
    else:
        target = torch.full_like(k_hat, float(k_target))
    return ((k_hat - target) ** 2).mean()


def compute_anti_smear_penalty(
    pred_minutes: torch.Tensor,
    gate_logits: torch.Tensor,
    mask: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    """Compute anti-smear penalty.

    Penalizes minutes predicted above `floor` for players with low gate probability.
    The penalty increases when the model predicts significant minutes for players
    the gate head doesn't believe are in the rotation.

    Args:
        pred_minutes: (B, N) predicted minutes
        gate_logits: (B, N) gate logits (detached internally to avoid double gradient)
        mask: (B, N) boolean mask of valid roster slots
        floor: minutes floor above which penalty applies (e.g., 4.0)

    Returns:
        scalar loss: mean(relu(pred - floor) * (1 - gate_prob) * mask)
    """
    gate_prob_detached = torch.sigmoid(gate_logits).detach()
    excess = torch.relu(pred_minutes - floor)
    smear = excess * (1.0 - gate_prob_detached)
    mask_f = mask.to(dtype=smear.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    return (smear * mask_f).sum() / denom


def compute_minutes_out_loss(
    pred_minutes: torch.Tensor,
    in_rotation: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute penalty for non-rotation players not approaching zero.

    Uses Huber-style loss (smooth L1) to encourage minutes -> 0 for
    players labeled as out of rotation.

    Args:
        pred_minutes: (B, N) predicted minutes
        in_rotation: (B, N) binary labels (1 = in rotation, 0 = out)
        mask: (B, N) boolean mask of valid roster slots

    Returns:
        scalar loss: mean absolute minutes for non-rotation players
    """
    out_rotation = (1.0 - in_rotation) * mask.to(dtype=pred_minutes.dtype)
    denom = out_rotation.sum().clamp(min=1.0)
    return (pred_minutes.abs() * out_rotation).sum() / denom
