"""Player-level usage-share supervision head for GTv2.

Predicts per-player logits for within-team share allocation targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class UsageShareHeadOutputs:
    """Outputs for usage-share targets on flattened player axis (B, 30)."""

    fga_logits: torch.Tensor
    fta_logits: torch.Tensor
    tov_logits: torch.Tensor


class UsageShareHead(nn.Module):
    """Lightweight per-player head for FGA/FTA/TOV share logits."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = int(d_model) * 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 3),
        )

    def forward(
        self,
        *,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        player_team_index: torch.Tensor,
    ) -> UsageShareHeadOutputs:
        if player_states.ndim != 3:
            raise ValueError("player_states must have shape (B,P,D)")
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if game_state.ndim != 2:
            raise ValueError("game_state must have shape (B,D)")
        if player_team_index.ndim != 2:
            raise ValueError("player_team_index must have shape (B,P)")

        bsz, num_players, d_model = player_states.shape
        if team_states.shape[0] != bsz or team_states.shape[2] != d_model:
            raise ValueError("team_states must align with player_states batch and d_model")
        if game_state.shape[0] != bsz or game_state.shape[1] != d_model:
            raise ValueError("game_state must align with player_states batch and d_model")
        if player_team_index.shape != (bsz, num_players):
            raise ValueError("player_team_index must align with player axis")

        team_ctx = torch.zeros_like(player_states)
        for team_id in (0, 1):
            mask = (player_team_index == int(team_id)).unsqueeze(-1)
            team_vec = team_states[:, team_id, :].unsqueeze(1).expand(-1, num_players, -1)
            team_ctx = torch.where(mask, team_vec, team_ctx)

        game_ctx = game_state.unsqueeze(1).expand(-1, num_players, -1)
        h = torch.cat([player_states, team_ctx, game_ctx], dim=-1)
        logits = self.net(h)
        fga_logits = logits[..., 0]
        fta_logits = logits[..., 1]
        tov_logits = logits[..., 2]
        return UsageShareHeadOutputs(
            fga_logits=fga_logits,
            fta_logits=fta_logits,
            tov_logits=tov_logits,
        )


__all__ = ["UsageShareHead", "UsageShareHeadOutputs"]
