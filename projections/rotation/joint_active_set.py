"""Joint active-set head for game-transformer rotation models.

This module provides:
- team active-count classification (`K_t in [min_active_count, max_active_count]`)
- without-replacement player subset selection per team
- supervision helpers for active labels and losses
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _team_context_for_players(team_states: torch.Tensor, player_team_index: torch.Tensor) -> torch.Tensor:
    """Broadcast team states to each player slot via team index (0/1)."""

    if team_states.ndim != 3 or team_states.shape[1] != 2:
        raise ValueError("team_states must have shape (B,2,D)")
    if player_team_index.ndim != 2:
        raise ValueError("player_team_index must have shape (B,P)")
    if team_states.shape[0] != player_team_index.shape[0]:
        raise ValueError("team_states and player_team_index batch sizes must match")

    bsz, num_players = player_team_index.shape
    d_model = team_states.shape[2]
    team_context = torch.zeros((bsz, num_players, d_model), dtype=team_states.dtype, device=team_states.device)
    for team_idx in (0, 1):
        mask = (player_team_index == team_idx).unsqueeze(-1)
        team_vec = team_states[:, team_idx, :].unsqueeze(1).expand(-1, num_players, -1)
        team_context = torch.where(mask, team_vec, team_context)
    return team_context


def _sample_gumbel(shape: torch.Size, *, device: torch.device, eps: float = 1e-8) -> torch.Tensor:
    u = torch.rand(shape, device=device)
    u = u.clamp(min=float(eps), max=1.0 - float(eps))
    return -torch.log(-torch.log(u))


def _select_topk_without_replacement(
    logits: torch.Tensor,
    eligible_mask: torch.Tensor,
    k: int,
    *,
    sample: bool,
    temperature: float,
) -> torch.Tensor:
    """Return boolean selection mask over a single team row."""

    if logits.ndim != 1 or eligible_mask.ndim != 1:
        raise ValueError("logits and eligible_mask must be rank-1")
    if logits.shape[0] != eligible_mask.shape[0]:
        raise ValueError("logits and eligible_mask lengths must match")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    num_eligible = int(eligible_mask.sum().item())
    if num_eligible <= 0 or k <= 0:
        return torch.zeros_like(eligible_mask, dtype=torch.bool)

    k_eff = int(min(max(k, 0), num_eligible))
    work = logits.to(dtype=torch.float32) / float(temperature)
    if sample:
        work = work + _sample_gumbel(work.shape, device=work.device)
    work = work.masked_fill(~eligible_mask, -1e9)

    topk_idx = torch.topk(work, k=k_eff, dim=0, largest=True).indices
    selected = torch.zeros_like(eligible_mask, dtype=torch.bool)
    selected[topk_idx] = True
    return selected & eligible_mask


@dataclass(frozen=True)
class JointActiveSetOutputs:
    count_logits: torch.Tensor
    player_logits: torch.Tensor
    sampled_counts: torch.Tensor
    active_mask: torch.Tensor


class JointActiveSetHead(nn.Module):
    """Joint team active-count + without-replacement player selection head."""

    def __init__(
        self,
        d_model: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        min_active_count: int = 5,
        max_active_count: int = 13,
    ) -> None:
        super().__init__()
        if min_active_count <= 0:
            raise ValueError("min_active_count must be > 0")
        if max_active_count < min_active_count:
            raise ValueError("max_active_count must be >= min_active_count")

        self.min_active_count = int(min_active_count)
        self.max_active_count = int(max_active_count)
        self.num_count_classes = int(max_active_count - min_active_count + 1)

        self.count_head = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_count_classes),
        )
        self.player_head = nn.Sequential(
            nn.LayerNorm(int(2 * d_model)),
            nn.Linear(int(2 * d_model), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def _resolve_counts(
        self,
        count_logits: torch.Tensor,
        *,
        sample: bool,
        temperature: float,
        target_counts: torch.Tensor | None,
        use_target_counts: bool,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        if use_target_counts:
            if target_counts is None:
                raise ValueError("target_counts is required when use_target_counts=True")
            out = target_counts.to(dtype=torch.long, device=count_logits.device)
        else:
            work = count_logits.to(dtype=torch.float32) / float(temperature)
            if sample:
                probs = torch.softmax(work, dim=-1)
                flat = probs.reshape(-1, probs.shape[-1])
                sampled = torch.multinomial(flat, num_samples=1, replacement=True).reshape(probs.shape[:-1])
                out = sampled + int(self.min_active_count)
            else:
                out = torch.argmax(work, dim=-1) + int(self.min_active_count)

        return out.clamp(min=int(self.min_active_count), max=int(self.max_active_count))

    def forward(
        self,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        player_team_index: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        sample: bool = False,
        temperature: float = 1.0,
        target_counts: torch.Tensor | None = None,
        use_target_counts: bool = False,
        target_active_mask: torch.Tensor | None = None,
        use_target_active_mask: bool = False,
    ) -> JointActiveSetOutputs:
        """Predict team active counts and sampled active-set membership."""

        if player_states.ndim != 3:
            raise ValueError("player_states must have shape (B,P,D)")
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if player_states.shape[0] != team_states.shape[0]:
            raise ValueError("player_states and team_states batch sizes must match")
        if valid_mask.shape != player_team_index.shape:
            raise ValueError("valid_mask and player_team_index must share shape (B,P)")
        if player_states.shape[:2] != player_team_index.shape:
            raise ValueError("player_states shape must align with player_team_index")

        valid = valid_mask.to(dtype=torch.bool)
        player_team_index = player_team_index.to(dtype=torch.long)

        count_logits = self.count_head(team_states)
        team_context = _team_context_for_players(team_states, player_team_index)
        player_logits = self.player_head(torch.cat([player_states, team_context], dim=-1)).squeeze(-1)
        eligible = valid & ((player_team_index == 0) | (player_team_index == 1))
        player_logits = player_logits.masked_fill(~eligible, -1e9)

        sampled_counts = self._resolve_counts(
            count_logits,
            sample=bool(sample),
            temperature=float(temperature),
            target_counts=target_counts,
            use_target_counts=bool(use_target_counts),
        )

        if use_target_active_mask:
            if target_active_mask is None:
                raise ValueError("target_active_mask is required when use_target_active_mask=True")
            active_mask = target_active_mask.to(dtype=torch.bool, device=player_states.device) & eligible
        else:
            bsz, num_players = eligible.shape
            active_mask = torch.zeros((bsz, num_players), dtype=torch.bool, device=player_states.device)
            for b_idx in range(bsz):
                for team_idx in (0, 1):
                    team_mask = eligible[b_idx] & (player_team_index[b_idx] == team_idx)
                    if not bool(team_mask.any()):
                        continue
                    k_team = int(sampled_counts[b_idx, team_idx].item())
                    selected = _select_topk_without_replacement(
                        player_logits[b_idx],
                        team_mask,
                        k_team,
                        sample=bool(sample),
                        temperature=float(temperature),
                    )
                    active_mask[b_idx] = active_mask[b_idx] | selected

        return JointActiveSetOutputs(
            count_logits=count_logits,
            player_logits=player_logits,
            sampled_counts=sampled_counts,
            active_mask=active_mask,
        )


@dataclass(frozen=True)
class ActiveSetLabelTargets:
    active_mask: torch.Tensor
    count_targets: torch.Tensor


def build_active_set_labels(
    y_minutes: torch.Tensor,
    valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    *,
    threshold: float,
    min_active_count: int,
    max_active_count: int,
) -> ActiveSetLabelTargets:
    """Build feasibility-aware active-set labels from minutes supervision."""

    if y_minutes.shape != valid_mask.shape or y_minutes.shape != player_team_index.shape:
        raise ValueError("y_minutes, valid_mask, and player_team_index must have shape (B,P)")
    if min_active_count <= 0:
        raise ValueError("min_active_count must be > 0")
    if max_active_count < min_active_count:
        raise ValueError("max_active_count must be >= min_active_count")

    minutes = y_minutes.to(dtype=torch.float32)
    valid = valid_mask.to(dtype=torch.bool)
    team_index = player_team_index.to(dtype=torch.long)

    initial = (minutes >= float(threshold)) & valid
    final = torch.zeros_like(initial)
    count_targets = torch.zeros((minutes.shape[0], 2), dtype=torch.long, device=minutes.device)

    for b_idx in range(minutes.shape[0]):
        for team_idx in (0, 1):
            team_mask = valid[b_idx] & (team_index[b_idx] == team_idx)
            num_team = int(team_mask.sum().item())
            if num_team <= 0:
                count_targets[b_idx, team_idx] = int(min_active_count)
                continue

            min_eff = int(min(int(min_active_count), num_team))
            max_eff = int(min(int(max_active_count), num_team))
            team_initial = initial[b_idx] & team_mask
            k_raw = int(team_initial.sum().item())
            k_target = int(min(max(k_raw, min_eff), max_eff))

            team_scores = minutes[b_idx].masked_fill(~team_mask, float("-inf"))
            topk_idx = torch.topk(team_scores, k=k_target, dim=0, largest=True).indices
            final[b_idx, topk_idx] = True
            count_targets[b_idx, team_idx] = k_target

    return ActiveSetLabelTargets(active_mask=final & valid, count_targets=count_targets)


def compute_active_set_losses(
    *,
    count_logits: torch.Tensor,
    player_logits: torch.Tensor,
    count_targets: torch.Tensor,
    active_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    min_active_count: int,
    positive_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute count CE + membership BCE losses for the active-set head."""

    if count_logits.ndim != 3 or count_logits.shape[1] != 2:
        raise ValueError("count_logits must have shape (B,2,K)")
    if player_logits.ndim != 2:
        raise ValueError("player_logits must have shape (B,P)")
    if count_targets.shape != (count_logits.shape[0], 2):
        raise ValueError("count_targets must have shape (B,2)")
    if active_targets.shape != player_logits.shape or valid_mask.shape != player_logits.shape:
        raise ValueError("active_targets and valid_mask must have shape (B,P)")

    count_idx = (count_targets.to(dtype=torch.long, device=count_logits.device) - int(min_active_count)).clamp(
        min=0,
        max=count_logits.shape[-1] - 1,
    )
    count_loss = F.cross_entropy(count_logits.reshape(-1, count_logits.shape[-1]), count_idx.reshape(-1), reduction="mean")

    valid = valid_mask.to(dtype=torch.bool, device=player_logits.device)
    active = active_targets.to(dtype=player_logits.dtype, device=player_logits.device)
    bce = F.binary_cross_entropy_with_logits(player_logits, active, reduction="none")
    if positive_weight > 1.0:
        pos_w = torch.full_like(bce, float(positive_weight))
        weight = torch.where(active > 0.5, pos_w, torch.ones_like(pos_w))
        bce = bce * weight

    denom = valid.to(dtype=player_logits.dtype).sum().clamp(min=1.0)
    member_loss = (bce * valid.to(dtype=player_logits.dtype)).sum() / denom

    return {
        "count_loss": count_loss,
        "member_loss": member_loss,
    }
