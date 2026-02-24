"""Training loss helpers for rotation-set minutes model.

This module provides:
- build_in_rotation_labels: Binary labels for rotation membership
- compute_k_hat: Expected rotation size per team-game from gate probabilities
- compute_anti_smear_penalty: Penalty for minutes predicted on low-gate-prob players
- compute_minutes_out_loss: Penalty encouraging non-rotation players toward zero minutes
- compute_crps_loss: CRPS loss from sampled scalar outcomes
- compute_team_energy_score: Energy score over team vector samples
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
        scalar loss: mean smooth-L1 minutes for non-rotation players
    """
    out_rotation = (1.0 - in_rotation).to(dtype=pred_minutes.dtype) * mask.to(dtype=pred_minutes.dtype)
    denom = out_rotation.sum().clamp(min=1.0)
    smooth_l1 = torch.nn.functional.smooth_l1_loss(
        pred_minutes * out_rotation,
        torch.zeros_like(pred_minutes),
        reduction="none",
    )
    return smooth_l1.sum() / denom


def compute_crps_loss(
    samples: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute scalar CRPS from Monte Carlo samples.

    Uses the explicit pairwise form:
      CRPS = E|X - y| - 0.5 * E|X - X'|

    Args:
        samples: (B, K, N) sampled scalar outcomes
        target: (B, N) observed targets
        mask: (B, N) observed/eligible mask

    Returns:
        Scalar mean CRPS over masked entries.
    """
    if samples.ndim != 3:
        raise ValueError("samples must have shape (B,K,N)")
    if target.ndim != 2 or mask.ndim != 2:
        raise ValueError("target and mask must have shape (B,N)")
    if samples.shape[0] != target.shape[0] or samples.shape[2] != target.shape[1]:
        raise ValueError("samples shape must align with target shape")

    target_f = target.to(dtype=samples.dtype, device=samples.device)
    mask_f = mask.to(dtype=samples.dtype, device=samples.device)

    term_obs = (samples - target_f.unsqueeze(1)).abs().mean(dim=1)  # (B,N)
    pairwise = (samples.unsqueeze(2) - samples.unsqueeze(1)).abs().mean(dim=(1, 2))  # (B,N)
    crps = term_obs - 0.5 * pairwise

    denom = mask_f.sum().clamp(min=1.0)
    return (crps * mask_f).sum() / denom


def compute_team_energy_score(
    samples: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    team_index: torch.Tensor,
    *,
    eps: float = 0.0,
) -> torch.Tensor:
    """Compute team-level energy score for sampled FPTS vectors.

    Args:
        samples: (B, K, N) sampled scalar outcomes per player.
        target: (B, N) observed outcomes per player.
        mask: (B, N) observed/eligible mask.
        team_index: (B, N) team index (0 or 1) per player slot.
        eps: numeric epsilon for norm stability.

    Returns:
        Scalar mean energy score over valid team-games.
    """
    if samples.ndim != 3:
        raise ValueError("samples must have shape (B,K,N)")
    if target.ndim != 2 or mask.ndim != 2 or team_index.ndim != 2:
        raise ValueError("target, mask, and team_index must have shape (B,N)")
    if (
        samples.shape[0] != target.shape[0]
        or samples.shape[2] != target.shape[1]
        or mask.shape != target.shape
        or team_index.shape != target.shape
    ):
        raise ValueError("samples/target/mask/team_index shapes must align")

    if float(eps) < 0.0:
        raise ValueError("eps must be >= 0")

    target_f = target.to(dtype=samples.dtype, device=samples.device)
    mask_b = mask.to(dtype=torch.bool, device=samples.device)
    team_idx = team_index.to(dtype=torch.long, device=samples.device)

    total = samples.new_tensor(0.0)
    count = samples.new_tensor(0.0)
    for team_id in (0, 1):
        team_mask = mask_b & (team_idx == int(team_id))  # (B,N)
        if not bool(team_mask.any()):
            continue

        team_mask_f = team_mask.to(dtype=samples.dtype)
        x = samples * team_mask_f.unsqueeze(1)  # (B,K,N)
        y = target_f * team_mask_f  # (B,N)

        dist_xy = torch.sqrt(((x - y.unsqueeze(1)) ** 2).sum(dim=-1).clamp(min=float(eps)))  # (B,K)
        dist_xx = torch.sqrt(
            ((x.unsqueeze(2) - x.unsqueeze(1)) ** 2).sum(dim=-1).clamp(min=float(eps))
        )  # (B,K,K)
        es = dist_xy.mean(dim=1) - 0.5 * dist_xx.mean(dim=(1, 2))  # (B,)

        row_has_team = team_mask.any(dim=1)
        total = total + es[row_has_team].sum()
        count = count + row_has_team.to(dtype=samples.dtype).sum()

    return total / count.clamp(min=1.0)
