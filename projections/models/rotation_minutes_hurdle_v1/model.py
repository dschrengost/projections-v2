from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RMHOutputs:
    """Output bundle from RotationMinutesHurdleMLP.

    v1.1: Expanded to 7 quantiles for improved tail fidelity.
    """

    logits_play: torch.Tensor  # [batch]
    q05_cond: torch.Tensor  # [batch]
    q10_cond: torch.Tensor  # [batch]
    q25_cond: torch.Tensor  # [batch]
    q50_cond: torch.Tensor  # [batch]
    q75_cond: torch.Tensor  # [batch]
    q90_cond: torch.Tensor  # [batch]
    q95_cond: torch.Tensor  # [batch]


class RotationMinutesHurdleMLP(nn.Module):
    """Shared trunk + two heads for hurdle minutes modeling (RMH_v1/v1.1).

    v1.1: Expanded conditional quantiles from {q10, q50, q90} to {q05, q10, q25, q50, q75, q90, q95}.
    Non-crossing is enforced via cumulative softplus deltas from q50.
    """

    def __init__(
        self,
        *,
        n_continuous: int,
        cat_cardinalities: Sequence[int],
        emb_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, emb_dim) for cardinality in cat_cardinalities
        )
        input_dim = int(n_continuous) + int(emb_dim) * len(self.embeddings)
        if input_dim <= 0:
            raise ValueError("RotationMinutesHurdleMLP requires at least one feature column.")

        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden)))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(hidden)
        self.trunk = nn.Identity() if not layers else nn.Sequential(*layers)

        # Head A: play probability (logit)
        self.play_head = nn.Linear(prev, 1)

        # Head B: conditional minutes distribution (q50 + 6 cumulative softplus deltas)
        # v1.1: Predict q50 + deltas for {d25, d10, d05, d75, d90, d95}
        self.q50_head = nn.Linear(prev, 1)
        self.delta_head = nn.Linear(prev, 6)  # d25, d10, d05, d75, d90, d95

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> RMHOutputs:
        pieces: list[torch.Tensor] = []
        if x_cont.shape[1]:
            pieces.append(x_cont)
        if self.embeddings:
            embs = [emb(x_cat[:, idx]) for idx, emb in enumerate(self.embeddings)]
            pieces.append(torch.cat(embs, dim=1))
        if not pieces:
            raise ValueError("No features provided to RotationMinutesHurdleMLP.")

        hidden = torch.cat(pieces, dim=1)
        hidden = self.trunk(hidden)

        logits_play = self.play_head(hidden).squeeze(1)

        q50 = self.q50_head(hidden).squeeze(1)
        deltas = self.delta_head(hidden)

        # Non-crossing parameterization via cumulative softplus deltas from q50.
        # Below q50: q25 = q50 - sp(d25), q10 = q25 - sp(d10), q05 = q10 - sp(d05)
        # Above q50: q75 = q50 + sp(d75), q90 = q75 + sp(d90), q95 = q90 + sp(d95)
        d25 = F.softplus(deltas[:, 0])
        d10 = F.softplus(deltas[:, 1])
        d05 = F.softplus(deltas[:, 2])
        d75 = F.softplus(deltas[:, 3])
        d90 = F.softplus(deltas[:, 4])
        d95 = F.softplus(deltas[:, 5])

        q25 = q50 - d25
        q10 = q25 - d10
        q05 = q10 - d05

        q75 = q50 + d75
        q90 = q75 + d90
        q95 = q90 + d95

        return RMHOutputs(
            logits_play=logits_play,
            q05_cond=q05,
            q10_cond=q10,
            q25_cond=q25,
            q50_cond=q50,
            q75_cond=q75,
            q90_cond=q90,
            q95_cond=q95,
        )


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball loss for a single quantile (no reduction)."""

    diff = target - pred
    tau_t = float(tau)
    return torch.maximum(tau_t * diff, (tau_t - 1.0) * diff)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """BCE with per-row weights, normalized by total weight."""

    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted = losses * sample_weight
    denom = torch.clamp(sample_weight.sum(), min=1e-12)
    return weighted.sum() / denom


def conditional_minutes_pinball_loss_q10_q50_q90(
    *,
    q10: torch.Tensor,
    q50: torch.Tensor,
    q90: torch.Tensor,
    y_minutes: torch.Tensor,
    y_play: torch.Tensor,
    sample_weight: torch.Tensor,
    quantile_weights: Sequence[float],
) -> torch.Tensor:
    """Joint pinball loss on the played-only slice (y_play == 1).

    This is the key guardrail: DNP / OUT rows must NOT influence the minutes head.

    NOTE: This is the v1.0 loss function for backward compatibility. Use
    conditional_minutes_pinball_loss for v1.1 with 7 quantiles.
    """

    # y_play comes in as float32 (0/1).
    mask = y_play > 0.5
    if not torch.any(mask):
        return torch.zeros((), dtype=y_minutes.dtype, device=y_minutes.device)

    q10_m = q10[mask]
    q50_m = q50[mask]
    q90_m = q90[mask]
    y_m = y_minutes[mask]
    w_m = sample_weight[mask]

    qw0 = float(quantile_weights[0])
    qw1 = float(quantile_weights[1])
    qw2 = float(quantile_weights[2])
    qw_sum = max(qw0 + qw1 + qw2, 1e-12)
    # Weighted mean over quantiles (keeps overall scale stable while allowing relative emphasis).
    losses = (
        qw0 * pinball_loss(q10_m, y_m, 0.10)
        + qw1 * pinball_loss(q50_m, y_m, 0.50)
        + qw2 * pinball_loss(q90_m, y_m, 0.90)
    ) / qw_sum
    denom = torch.clamp(w_m.sum(), min=1e-12)
    return (losses * w_m).sum() / denom


def conditional_minutes_pinball_loss(
    *,
    quantile_preds: dict[float, torch.Tensor],
    y_minutes: torch.Tensor,
    y_play: torch.Tensor,
    sample_weight: torch.Tensor,
    quantile_weights: dict[float, float],
) -> torch.Tensor:
    """Joint pinball loss on the played-only slice (y_play == 1).

    v1.1: Generalized to support arbitrary quantile sets.

    Args:
        quantile_preds: Dict mapping tau -> predicted quantile tensor.
        y_minutes: Ground truth minutes.
        y_play: Binary indicator (1=played, 0=DNP/OUT).
        sample_weight: Per-row recency weights.
        quantile_weights: Dict mapping tau -> loss weight for that quantile.

    Returns:
        Weighted-average pinball loss over played rows.
    """
    mask = y_play > 0.5
    if not torch.any(mask):
        return torch.zeros((), dtype=y_minutes.dtype, device=y_minutes.device)

    y_m = y_minutes[mask]
    w_m = sample_weight[mask]

    total_loss = torch.zeros(y_m.shape, dtype=y_m.dtype, device=y_m.device)
    qw_sum = 0.0
    for tau, pred in quantile_preds.items():
        qw = float(quantile_weights.get(tau, 1.0))
        qw_sum += qw
        total_loss = total_loss + qw * pinball_loss(pred[mask], y_m, float(tau))

    qw_sum = max(qw_sum, 1e-12)
    losses = total_loss / qw_sum
    denom = torch.clamp(w_m.sum(), min=1e-12)
    return (losses * w_m).sum() / denom


__all__ = [
    "RMHOutputs",
    "RotationMinutesHurdleMLP",
    "conditional_minutes_pinball_loss",
    "conditional_minutes_pinball_loss_q10_q50_q90",
    "pinball_loss",
    "weighted_bce_with_logits",
]
