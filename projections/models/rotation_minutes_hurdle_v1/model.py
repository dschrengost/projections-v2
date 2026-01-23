from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RMHOutputs:
    logits_play: torch.Tensor  # [batch]
    q10_cond: torch.Tensor  # [batch]
    q50_cond: torch.Tensor  # [batch]
    q90_cond: torch.Tensor  # [batch]


class RotationMinutesHurdleMLP(nn.Module):
    """Shared trunk + two heads for hurdle minutes modeling (RMH_v1)."""

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

        # Head B: conditional minutes distribution (q50 + positive deltas)
        self.q50_head = nn.Linear(prev, 1)
        self.delta_head = nn.Linear(prev, 2)  # d10_raw, d90_raw

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
        d10 = F.softplus(deltas[:, 0])
        d90 = F.softplus(deltas[:, 1])

        # Non-crossing via q50 +/- softplus(deltas)
        q10 = q50 - d10
        q90 = q50 + d90

        return RMHOutputs(
            logits_play=logits_play,
            q10_cond=q10,
            q50_cond=q50,
            q90_cond=q90,
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


__all__ = [
    "RMHOutputs",
    "RotationMinutesHurdleMLP",
    "conditional_minutes_pinball_loss_q10_q50_q90",
    "pinball_loss",
    "weighted_bce_with_logits",
]
