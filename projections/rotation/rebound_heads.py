"""Rebound-specific auxiliary heads for GTv2.

These heads factor rebounds into:
- team-level offensive and defensive rebound budgets per side
- player-level offensive and defensive rebound share logits within each team
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TeamReboundBudgetHeadOutputs:
    """Predicted team rebound budgets with shape (B, 2) per side."""

    team_oreb: torch.Tensor
    team_dreb: torch.Tensor


@dataclass(frozen=True)
class ReboundShareHeadOutputs:
    """Per-player rebound share logits with shape (B, 30)."""

    oreb_logits: torch.Tensor
    dreb_logits: torch.Tensor


@dataclass(frozen=True)
class ReboundBudgetBlendGateHeadOutputs:
    """Per-side rebound budget blend gates in [0, 1] with shape (B, 2)."""

    oreb_gate_logits: torch.Tensor
    oreb_gate: torch.Tensor
    dreb_gate_logits: torch.Tensor
    dreb_gate: torch.Tensor


class TeamReboundBudgetHead(nn.Module):
    """Predict team offensive and defensive rebound budgets or rate deltas."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        budget_parameterization: str = "absolute",
        oreb_rate_cap: float = 1.0,
        dreb_rate_cap: float = 1.0,
    ) -> None:
        super().__init__()
        self.budget_parameterization = str(budget_parameterization).strip().lower()
        self.oreb_rate_cap = float(max(0.0, min(1.0, oreb_rate_cap)))
        self.dreb_rate_cap = float(max(0.0, min(1.0, dreb_rate_cap)))
        in_dim = int(d_model) * 2
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.oreb_out = nn.Linear(int(hidden_dim), 1)
        self.dreb_out = nn.Linear(int(hidden_dim), 1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(
        self,
        *,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
    ) -> TeamReboundBudgetHeadOutputs:
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if game_state.ndim != 2:
            raise ValueError("game_state must have shape (B,D)")
        if team_states.shape[0] != game_state.shape[0] or team_states.shape[2] != game_state.shape[1]:
            raise ValueError("team_states and game_state must align on batch and d_model")

        game_ctx = game_state.unsqueeze(1).expand(-1, 2, -1)
        h = self.trunk(torch.cat([team_states, game_ctx], dim=-1))
        oreb_raw = self.oreb_out(h).squeeze(-1)
        dreb_raw = self.dreb_out(h).squeeze(-1)
        use_oreb_rate = self.budget_parameterization in {"oreb_rate", "both_rate"}
        use_dreb_rate = self.budget_parameterization in {"dreb_rate", "both_rate"}
        use_dreb_residual_rate = self.budget_parameterization in {"dreb_rate_residual"}
        team_oreb = (
            self.sigmoid(oreb_raw) * self.oreb_rate_cap if use_oreb_rate else self.softplus(oreb_raw)
        )
        team_dreb = (
            self.tanh(dreb_raw) * self.dreb_rate_cap
            if use_dreb_residual_rate
            else self.sigmoid(dreb_raw) * self.dreb_rate_cap
            if use_dreb_rate
            else self.softplus(dreb_raw)
        )
        return TeamReboundBudgetHeadOutputs(team_oreb=team_oreb, team_dreb=team_dreb)


class ReboundShareHead(nn.Module):
    """Predict within-team offensive and defensive rebound share logits."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int = 128,
        condition_dim: int = 0,
        condition_hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.condition_dim = max(0, int(condition_dim))
        self.condition_encoder: nn.Module | None = None
        encoded_condition_dim = 0
        if self.condition_dim > 0:
            encoded_condition_dim = max(4, int(condition_hidden_dim))
            self.condition_encoder = nn.Sequential(
                nn.LayerNorm(self.condition_dim),
                nn.Linear(self.condition_dim, encoded_condition_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
        in_dim = (int(d_model) * 3) + encoded_condition_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.oreb_out = nn.Linear(int(hidden_dim), 1)
        self.dreb_out = nn.Linear(int(hidden_dim), 1)

    def forward(
        self,
        *,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        player_team_index: torch.Tensor,
        condition_features: torch.Tensor | None = None,
    ) -> ReboundShareHeadOutputs:
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
        if game_state.shape != (bsz, d_model):
            raise ValueError("game_state must align with player_states batch and d_model")
        if player_team_index.shape != (bsz, num_players):
            raise ValueError("player_team_index must align with player axis")
        if self.condition_dim > 0:
            if condition_features is None:
                raise ValueError("condition_features is required when ReboundShareHead.condition_dim > 0")
            if condition_features.shape != (bsz, num_players, self.condition_dim):
                raise ValueError(
                    "condition_features must have shape "
                    f"(B,P,{self.condition_dim}) when rebound-share conditioning is enabled"
                )
        elif condition_features is not None:
            raise ValueError("condition_features provided but ReboundShareHead.condition_dim == 0")

        team_ctx = torch.zeros_like(player_states)
        for team_id in (0, 1):
            mask = (player_team_index == int(team_id)).unsqueeze(-1)
            team_vec = team_states[:, team_id, :].unsqueeze(1).expand(-1, num_players, -1)
            team_ctx = torch.where(mask, team_vec, team_ctx)
        game_ctx = game_state.unsqueeze(1).expand(-1, num_players, -1)
        parts = [player_states, team_ctx, game_ctx]
        if self.condition_encoder is not None:
            assert condition_features is not None
            parts.append(self.condition_encoder(condition_features))
        h = self.trunk(torch.cat(parts, dim=-1))
        oreb_logits = self.oreb_out(h).squeeze(-1)
        dreb_logits = self.dreb_out(h).squeeze(-1)
        return ReboundShareHeadOutputs(oreb_logits=oreb_logits, dreb_logits=dreb_logits)


class ReboundBudgetBlendGateHead(nn.Module):
    """Predict per-side blend gates between flow and factorized rebound budgets."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        init_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        in_dim = int(d_model) * 2
        self.input_norm = nn.LayerNorm(in_dim)
        self.hidden = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.oreb_out = nn.Linear(int(hidden_dim), 1)
        self.dreb_out = nn.Linear(int(hidden_dim), 1)
        init_alpha = min(max(float(init_alpha), 1e-4), 1.0 - 1e-4)
        init_bias = math.log(init_alpha / (1.0 - init_alpha))
        nn.init.zeros_(self.oreb_out.weight)
        nn.init.zeros_(self.dreb_out.weight)
        nn.init.constant_(self.oreb_out.bias, init_bias)
        nn.init.constant_(self.dreb_out.bias, init_bias)

    def forward(
        self,
        *,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
    ) -> ReboundBudgetBlendGateHeadOutputs:
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if game_state.ndim != 2:
            raise ValueError("game_state must have shape (B,D)")
        if team_states.shape[0] != game_state.shape[0] or team_states.shape[2] != game_state.shape[1]:
            raise ValueError("team_states and game_state must align on batch and d_model")

        game_ctx = game_state.unsqueeze(1).expand(-1, 2, -1)
        h = self.hidden(self.input_norm(torch.cat([team_states, game_ctx], dim=-1)))
        oreb_gate_logits = self.oreb_out(h).squeeze(-1)
        dreb_gate_logits = self.dreb_out(h).squeeze(-1)
        return ReboundBudgetBlendGateHeadOutputs(
            oreb_gate_logits=oreb_gate_logits,
            oreb_gate=torch.sigmoid(oreb_gate_logits),
            dreb_gate_logits=dreb_gate_logits,
            dreb_gate=torch.sigmoid(dreb_gate_logits),
        )


__all__ = [
    "ReboundBudgetBlendGateHead",
    "ReboundBudgetBlendGateHeadOutputs",
    "ReboundShareHead",
    "ReboundShareHeadOutputs",
    "TeamReboundBudgetHead",
    "TeamReboundBudgetHeadOutputs",
]
