"""DeepSets-style team-allocation minutes model.

This module provides a CPU-friendly neural network that:
1. Treats each (game_id, team_id) as a set of players
2. Predicts gate logits (in-rotation probability) and share scores per player
3. Allocates exactly 240 minutes per team using a differentiable constraint layer

The model output layer IS the constraint, not a post-hoc rescaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TeamAllocConfig:
    """Configuration for the DeepSetsAllocator model."""

    n_features: int
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    team_total_minutes: float = 240.0
    # Allocation stability: baseline added to gate to prevent all-zero weights collapse.
    alloc_eps: float = 1e-4
    # Numerical stability: clamp to avoid division by zero.
    denom_eps: float = 1e-12

    def to_metadata(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "team_total_minutes": self.team_total_minutes,
            "alloc_eps": self.alloc_eps,
            "denom_eps": self.denom_eps,
        }


class DeepSetsAllocator(nn.Module):
    """DeepSets-style model for team minutes allocation.

    Architecture:
        1. Per-player encoder: MLP that processes each player's features
        2. Team aggregation: Mean pooling over valid (non-padded) players
        3. Per-player decoder: Combines player encoding with team context
        4. Dual-head output:
           - gate_logits: sigmoid → in-rotation probability
           - share_scores: used in softmax-style allocation

    Allocation layer (differentiable, not post-hoc):
        gate = sigmoid(gate_logits) * mask
        weights ∝ (sigmoid(gate_logits) + alloc_eps) * exp(share_scores)
        minutes_hat = 240 * weights / sum(weights)

    Input shapes:
        x: [B, P, F] - batch of team-games, P players, F features
        mask: [B, P] - 1.0 for real players, 0.0 for padding

    Output shapes:
        gate_logits: [B, P]
        share_scores: [B, P]
        minutes_hat: [B, P] - always sums to 240 per team (where mask.sum > 0)
    """

    def __init__(self, config: TeamAllocConfig) -> None:
        super().__init__()
        self.config = config

        # Per-player encoder
        encoder_layers: list[nn.Module] = []
        in_dim = config.n_features
        for _ in range(config.n_layers):
            encoder_layers.append(nn.Linear(in_dim, config.hidden_dim))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        self.player_encoder = nn.Sequential(*encoder_layers)

        # Decoder: takes player encoding + team context (mean pooled)
        decoder_input_dim = config.hidden_dim * 2  # player + team context
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
        )

        # Dual output heads
        head_input_dim = config.hidden_dim // 2
        self.gate_head = nn.Linear(head_input_dim, 1)  # gate logits
        self.share_head = nn.Linear(head_input_dim, 1)  # share scores

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        eligible_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with differentiable allocation constraint.

        Args:
            x: Player features [B, P, F]
            mask: Valid player mask [B, P], 1.0 for real, 0.0 for padding

        Returns:
            gate_logits: [B, P] raw gate logits (pre-sigmoid)
            share_scores: [B, P] raw share scores
            minutes_hat: [B, P] allocated minutes (sums to 240 per team)
        """
        B, P, F = x.shape

        # Encode each player
        # Reshape to [B*P, F] for efficient processing
        x_flat = x.view(B * P, F)
        player_enc = self.player_encoder(x_flat)  # [B*P, H]
        player_enc = player_enc.view(B, P, -1)  # [B, P, H]

        # Team context via masked mean pooling
        # mask: [B, P] -> [B, P, 1] for broadcasting
        mask_expanded = mask.unsqueeze(-1)  # [B, P, 1]
        masked_enc = player_enc * mask_expanded  # zero out padded players
        team_sum = masked_enc.sum(dim=1, keepdim=True)  # [B, 1, H]
        team_count = mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # [B, 1, 1]
        team_context = team_sum / team_count  # [B, 1, H]
        team_context = team_context.expand(-1, P, -1)  # [B, P, H]

        # Combine player encoding with team context
        combined = torch.cat([player_enc, team_context], dim=-1)  # [B, P, 2H]
        combined_flat = combined.view(B * P, -1)  # [B*P, 2H]
        decoded = self.decoder(combined_flat)  # [B*P, H/2]
        decoded = decoded.view(B, P, -1)  # [B, P, H/2]

        # Dual heads
        gate_logits = self.gate_head(decoded).squeeze(-1)  # [B, P]
        share_scores = self.share_head(decoded).squeeze(-1)  # [B, P]

        # Allocate minutes with differentiable constraint
        minutes_hat = self._allocate_minutes(gate_logits, share_scores, mask, eligible_mask=eligible_mask)

        return gate_logits, share_scores, minutes_hat

    def _allocate_minutes(
        self,
        gate_logits: torch.Tensor,
        share_scores: torch.Tensor,
        mask: torch.Tensor,
        *,
        eligible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Differentiable allocation: (gate + alloc_eps) * softmax(scores) * 240.

        Args:
            gate_logits: [B, P] raw gate logits
            share_scores: [B, P] raw share scores
            mask: [B, P] valid player mask

        Returns:
            minutes_hat: [B, P] allocated minutes summing to 240
        """
        alloc_eps = self.config.alloc_eps
        denom_eps = self.config.denom_eps
        total_minutes = self.config.team_total_minutes

        # Optionally restrict allocation to an eligible subset.
        # Final mask used for allocation = mask & eligible_mask, and alloc_eps applies only to eligible players.
        if eligible_mask is not None:
            if eligible_mask.shape != mask.shape:
                raise ValueError(
                    "eligible_mask must have same shape as mask. "
                    f"eligible_mask={tuple(eligible_mask.shape)} mask={tuple(mask.shape)}"
                )
            mask = mask * eligible_mask.to(dtype=mask.dtype)

        # Gate probability (soft in-rotation signal)
        gate_prob = torch.sigmoid(gate_logits)  # [B, P]

        # For numerical stability, shift share_scores before exp
        # mask out padded positions with large negative values
        masked_scores = share_scores.masked_fill(mask == 0, -1e9)
        shifted_scores = masked_scores - masked_scores.max(dim=1, keepdim=True).values
        exp_scores = torch.exp(shifted_scores) * mask  # [B, P]

        # Weighted shares: (gate + alloc_eps) * exp(score), masked to valid players.
        # This prevents the degenerate case where all gates go to ~0 early in training,
        # which would otherwise yield all-zero weights and minutes_hat ≈ 0.
        weights = (gate_prob + alloc_eps) * exp_scores  # [B, P]

        # Normalize to sum to 240.
        # Use float64 for the constraint layer to make sum-to-240 checks extremely tight
        # (important for downstream groupby/parquet sanity checks).
        weights64 = weights.to(dtype=torch.float64)
        weight_sum64 = weights64.sum(dim=1, keepdim=True).clamp(min=denom_eps)  # [B, 1]
        minutes_hat = (total_minutes * weights64 / weight_sum64) * mask.to(dtype=torch.float64)  # [B, P]

        # Sanity checks: minutes allocation should sum to 240 for any non-empty team mask.
        # If this fails, it indicates a mask bug or numerical collapse (NaN/Inf).
        with torch.no_grad():
            mask_sum = mask.sum(dim=1)  # [B]
            valid_teams = mask_sum > 0

            if not torch.isfinite(minutes_hat).all():
                bad = ~torch.isfinite(minutes_hat)
                bad_rows = torch.nonzero(bad.any(dim=1), as_tuple=False).squeeze(-1)[:5]
                raise RuntimeError(
                    "Non-finite minutes_hat detected in allocation. "
                    f"bad_rows={bad_rows.tolist()}"
                )

            if valid_teams.any():
                team_sums = minutes_hat.sum(dim=1)  # [B]
                deviations = (team_sums - total_minutes).abs()
                max_dev = deviations[valid_teams].max().item()
                if max_dev > 1e-3:
                    offending = torch.nonzero(
                        valid_teams & (deviations > 1e-3),
                        as_tuple=False,
                    ).squeeze(-1)[:5]
                    debug = []
                    for b in offending.tolist():
                        valid_mask = mask[b] == 1
                        gate_b = gate_prob[b][valid_mask]
                        scores_b = share_scores[b][valid_mask]
                        weights_b = weights[b][valid_mask]
                        debug.append(
                            {
                                "b": b,
                                "mask_sum": float(mask_sum[b].item()),
                                "team_sum": float(team_sums[b].item()),
                                "dev": float(deviations[b].item()),
                                "gate_prob_min": float(gate_b.min().item()) if gate_b.numel() else float("nan"),
                                "gate_prob_mean": float(gate_b.mean().item()) if gate_b.numel() else float("nan"),
                                "gate_prob_max": float(gate_b.max().item()) if gate_b.numel() else float("nan"),
                                "share_min": float(scores_b.min().item()) if scores_b.numel() else float("nan"),
                                "share_mean": float(scores_b.mean().item()) if scores_b.numel() else float("nan"),
                                "share_max": float(scores_b.max().item()) if scores_b.numel() else float("nan"),
                                "weights_sum": float(weights_b.sum().item()) if weights_b.numel() else float("nan"),
                                "weights_max": float(weights_b.max().item()) if weights_b.numel() else float("nan"),
                            }
                        )
                    raise RuntimeError(
                        "Sum-to-240 constraint violation in allocation. "
                        f"max_dev={max_dev:.6f} offending={debug}"
                    )

        return minutes_hat


def build_eligibility_mask(
    prior_minutes: torch.Tensor,
    mask: torch.Tensor,
    *,
    topk: int = 0,
    prior_threshold: float = float("-inf"),
) -> torch.Tensor:
    """Build an eligibility mask from a per-player prior minutes signal.

    Eligibility is computed only among non-padded players (mask == 1).

    Rules:
      - If topk > 0: keep at most topk players by prior_minutes per team.
      - If prior_threshold is finite: require prior_minutes >= prior_threshold.
      - If eligibility would be empty for a non-empty team, fall back to top-1 by prior.

    Returns:
        eligible_mask: [B, P] float mask (0/1), same dtype as `mask`
    """
    if prior_minutes.shape != mask.shape:
        raise ValueError(
            "prior_minutes must have same shape as mask. "
            f"prior_minutes={tuple(prior_minutes.shape)} mask={tuple(mask.shape)}"
        )
    if topk < 0:
        raise ValueError("topk must be >= 0")

    mask_bool = mask > 0
    if not mask_bool.any():
        return torch.zeros_like(mask)

    # Treat non-finite prior as very negative so it won't be selected unless forced by fallback.
    scores = prior_minutes.to(dtype=torch.float32)
    scores = torch.where(torch.isfinite(scores), scores, scores.new_full(scores.shape, -1e9))
    scores_masked = scores.masked_fill(~mask_bool, -1e9)

    eligible = mask_bool.clone()

    if topk > 0:
        k = min(int(topk), int(mask.shape[1]))
        _, idx = torch.topk(scores_masked, k=k, dim=1, largest=True)
        topk_mask = torch.zeros_like(mask_bool)
        topk_mask.scatter_(1, idx, True)
        topk_mask = topk_mask & mask_bool
        eligible = topk_mask

    if math.isfinite(prior_threshold):
        eligible = eligible & (scores >= float(prior_threshold))

    # Ensure at least one eligible per non-empty team.
    mask_sum = mask_bool.sum(dim=1)
    eligible_sum = eligible.sum(dim=1)
    needs_fallback = (mask_sum > 0) & (eligible_sum == 0)
    if needs_fallback.any():
        best_idx = scores_masked.argmax(dim=1)
        fallback = torch.zeros_like(mask_bool)
        fallback.scatter_(1, best_idx.unsqueeze(1), True)
        fallback = fallback & mask_bool
        eligible = torch.where(needs_fallback.unsqueeze(1), fallback, eligible)

    return eligible.to(dtype=mask.dtype)


@dataclass
class AllocationLoss:
    """Combined loss for team allocation model.

    Components:
        - gate_loss: BCE on gate logits vs actual played (minutes > 0)
        - minutes_loss: SmoothL1/Huber on predicted vs actual minutes
        - entropy_penalty: encourage sparsity (lower entropy on weights)
        - sparsity_penalty: L1 on gate activations
        - fp_dnp_penalty: punish false-positive minutes on DNP rows (minutes_actual == 0)

    All losses are computed only on valid (non-padded) positions.
    """

    alpha_gate: float = 0.5  # weight for gate BCE loss
    beta_min: float = 1.0  # weight for minutes loss
    gamma_entropy: float = 0.01  # weight for entropy penalty
    delta_sparsity: float = 0.0  # weight for L1 sparsity penalty
    gate_pos_weight: float | None = None  # optional pos_weight for BCE (handles class imbalance)
    lambda_fp_dnp: float = 0.0  # weight for false-positive DNP minutes penalty (off by default)
    fp_threshold: float = 0.25  # minutes above this count as false-positive on DNP rows
    fp_power: int = 2  # 1 (linear) or 2 (squared)

    def __call__(
        self,
        gate_logits: torch.Tensor,
        minutes_hat: torch.Tensor,
        minutes_actual: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            gate_logits: [B, P] raw gate logits
            minutes_hat: [B, P] predicted minutes
            minutes_actual: [B, P] actual minutes
            mask: [B, P] valid player mask

        Returns:
            total_loss: scalar tensor
            components: dict of individual loss values for logging
        """
        eps = 1e-8

        # Binary target: did player play?
        y_play = (minutes_actual > 0).float()  # [B, P]

        # Gate loss: BCE on valid positions
        # Use reduction='none' then mask
        pos_weight: torch.Tensor | None = None
        if self.gate_pos_weight is not None:
            pos_weight = torch.tensor(
                float(self.gate_pos_weight),
                device=gate_logits.device,
                dtype=gate_logits.dtype,
            )

        gate_bce = nn.functional.binary_cross_entropy_with_logits(
            gate_logits, y_play, reduction="none", pos_weight=pos_weight
        )
        gate_bce_masked = (gate_bce * mask).sum() / mask.sum().clamp(min=1)

        # Minutes loss: SmoothL1/Huber on valid positions
        minutes_loss = nn.functional.smooth_l1_loss(
            minutes_hat, minutes_actual, reduction="none"
        )
        minutes_loss_masked = (minutes_loss * mask).sum() / mask.sum().clamp(min=1)

        # Entropy penalty: encourage concentrated allocations
        # Compute entropy of normalized minutes distribution per team
        weight_sum = minutes_hat.sum(dim=1, keepdim=True).clamp(min=eps)
        probs = minutes_hat / weight_sum  # [B, P]
        # Entropy = -sum(p * log(p)), masked
        log_probs = torch.log(probs.clamp(min=eps))
        entropy = -(probs * log_probs * mask).sum(dim=1)  # [B]
        entropy_penalty = entropy.mean()

        # Sparsity penalty: L1 on gate activations
        gate_activated = torch.sigmoid(gate_logits) * mask
        sparsity_penalty = gate_activated.sum() / mask.sum().clamp(min=1)

        # False-positive DNP minutes penalty (one-sided).
        # Only applies where minutes_actual == 0 and mask == 1.
        minutes_hat32 = minutes_hat.to(dtype=torch.float32)
        minutes_actual32 = minutes_actual.to(dtype=torch.float32)
        mask32 = mask.to(dtype=torch.float32)
        dnp_mask = (minutes_actual32 == 0) & (mask32 == 1)
        fp = torch.relu(minutes_hat32 - float(self.fp_threshold))
        if self.fp_power == 1:
            fp_term = fp
        elif self.fp_power == 2:
            fp_term = fp * fp
        else:
            raise ValueError("fp_power must be 1 or 2")
        if dnp_mask.any():
            fp_dnp_penalty = fp_term[dnp_mask].mean().to(dtype=torch.float32)
        else:
            fp_dnp_penalty = minutes_hat32.new_tensor(0.0)

        # Combine losses
        total = (
            self.alpha_gate * gate_bce_masked
            + self.beta_min * minutes_loss_masked
            + self.gamma_entropy * entropy_penalty
            + self.delta_sparsity * sparsity_penalty
            + self.lambda_fp_dnp * fp_dnp_penalty
        )

        components = {
            "gate_bce": gate_bce_masked.item(),
            "minutes_loss": minutes_loss_masked.item(),
            "entropy": entropy_penalty.item(),
            "sparsity": sparsity_penalty.item(),
            "fp_dnp": fp_dnp_penalty.item(),
            "total": total.item(),
        }

        return total, components


def compute_metrics(
    gate_logits: torch.Tensor,
    minutes_hat: torch.Tensor,
    minutes_actual: torch.Tensor,
    mask: torch.Tensor,
    player_ids: torch.Tensor | None = None,
    top_k: int = 8,
    gate_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute evaluation metrics for team allocation model.

    Metrics:
        - mae: Mean Absolute Error on minutes (masked)
        - gate_precision: Precision for played classification
        - gate_recall: Recall for played classification
        - gate_f1: F1 score for played classification
        - top_k_overlap: Average overlap of top-k by predicted vs actual minutes
        - max_sum_deviation: Maximum deviation from 240 across teams

    Args:
        gate_logits: [B, P]
        minutes_hat: [B, P]
        minutes_actual: [B, P]
        mask: [B, P]
        player_ids: [B, P] optional, for debugging
        top_k: Number of top players to compare for overlap

    Returns:
        Dictionary of metric values
    """
    with torch.no_grad():
        # MAE on valid positions
        abs_err = torch.abs(minutes_hat - minutes_actual) * mask
        mae = abs_err.sum() / mask.sum().clamp(min=1)

        # Gate classification metrics
        valid = mask == 1
        y_play = (minutes_actual > 0) & valid
        y_pred = (torch.sigmoid(gate_logits) > gate_threshold) & valid

        tp = (y_pred & y_play).sum().float()
        fp = (y_pred & ~y_play).sum().float()
        fn = (~y_pred & y_play).sum().float()

        precision = tp / (tp + fp) if (tp + fp).item() > 0 else tp.new_tensor(0.0)
        recall = tp / (tp + fn) if (tp + fn).item() > 0 else tp.new_tensor(0.0)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall).item() > 0
            else tp.new_tensor(0.0)
        )

        # Top-k overlap: compare sets of top players by predicted vs actual
        B, P = minutes_hat.shape
        overlaps = []
        for b in range(B):
            valid_mask = mask[b] == 1
            if valid_mask.sum() == 0:
                continue

            # Get actual minutes for valid players
            actual_valid = minutes_actual[b][valid_mask]
            pred_valid = minutes_hat[b][valid_mask]

            # Top-k indices
            k = min(top_k, int(valid_mask.sum().item()))
            if k == 0:
                continue

            _, top_actual_idx = actual_valid.topk(k)
            _, top_pred_idx = pred_valid.topk(k)

            # Count overlap
            actual_set = set(top_actual_idx.tolist())
            pred_set = set(top_pred_idx.tolist())
            overlap = len(actual_set & pred_set) / k
            overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

        # Sum-to-240 sanity check
        team_sums = (minutes_hat * mask).sum(dim=1)  # [B]
        valid_teams = mask.sum(dim=1) > 0  # [B]
        if valid_teams.any():
            deviations = torch.abs(team_sums[valid_teams] - 240.0)
            max_deviation = deviations.max().item()
        else:
            max_deviation = 0.0

        return {
            "mae": mae.item(),
            "gate_precision": precision.item(),
            "gate_recall": recall.item(),
            "gate_f1": f1.item(),
            f"top_{top_k}_overlap": avg_overlap,
            "max_sum_deviation": max_deviation,
        }


__all__ = [
    "TeamAllocConfig",
    "DeepSetsAllocator",
    "build_eligibility_mask",
    "AllocationLoss",
    "compute_metrics",
]
