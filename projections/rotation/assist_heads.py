"""Assist-specific auxiliary heads for GTv2.

These heads factor assists into:
- a team-level AST budget per side
- a player-level passer share within each team
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TeamAstBudgetHeadOutputs:
    """Predicted team AST budgets with shape (B, 2)."""

    team_ast: torch.Tensor


@dataclass(frozen=True)
class AssistShareHeadOutputs:
    """Per-player AST share logits with shape (B, 30)."""

    ast_logits: torch.Tensor


@dataclass(frozen=True)
class AstBlendGateHeadOutputs:
    """Per-player factorized-AST blend gate in [0, 1] with shape (B, 30)."""

    gate_logits: torch.Tensor
    gate: torch.Tensor


class TeamAstBudgetHead(nn.Module):
    """Predict positive team AST budgets from team/game state."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = int(d_model) * 2
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.softplus = nn.Softplus()

    def forward(
        self,
        *,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
    ) -> TeamAstBudgetHeadOutputs:
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if game_state.ndim != 2:
            raise ValueError("game_state must have shape (B,D)")
        if team_states.shape[0] != game_state.shape[0] or team_states.shape[2] != game_state.shape[1]:
            raise ValueError("team_states and game_state must align on batch and d_model")

        game_ctx = game_state.unsqueeze(1).expand(-1, 2, -1)
        h = torch.cat([team_states, game_ctx], dim=-1)
        raw = self.net(h).squeeze(-1)
        team_ast = self.softplus(raw)
        return TeamAstBudgetHeadOutputs(team_ast=team_ast)


class AssistShareHead(nn.Module):
    """Predict within-team AST share logits for players."""

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
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        *,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        player_team_index: torch.Tensor,
        condition_features: torch.Tensor | None = None,
    ) -> AssistShareHeadOutputs:
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
                raise ValueError("condition_features is required when AssistShareHead.condition_dim > 0")
            if condition_features.shape != (bsz, num_players, self.condition_dim):
                raise ValueError(
                    "condition_features must have shape "
                    f"(B,P,{self.condition_dim}) when assist-share conditioning is enabled"
                )
        elif condition_features is not None:
            raise ValueError("condition_features provided but AssistShareHead.condition_dim == 0")

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
        h = torch.cat(parts, dim=-1)
        ast_logits = self.net(h).squeeze(-1)
        return AssistShareHeadOutputs(ast_logits=ast_logits)


class AstBlendGateHead(nn.Module):
    """Predict per-player blend gates between flow AST and factorized AST."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int = 128,
        condition_dim: int = 0,
        condition_hidden_dim: int = 32,
        dropout: float = 0.1,
        init_alpha: float = 0.75,
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
        self.input_norm = nn.LayerNorm(in_dim)
        self.hidden = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.out = nn.Linear(int(hidden_dim), 1)
        init_alpha = min(max(float(init_alpha), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, math.log(init_alpha / (1.0 - init_alpha)))

    def forward(
        self,
        *,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        player_team_index: torch.Tensor,
        condition_features: torch.Tensor | None = None,
    ) -> AstBlendGateHeadOutputs:
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
                raise ValueError("condition_features is required when AstBlendGateHead.condition_dim > 0")
            if condition_features.shape != (bsz, num_players, self.condition_dim):
                raise ValueError(
                    "condition_features must have shape "
                    f"(B,P,{self.condition_dim}) when ast-blend-gate conditioning is enabled"
                )
        elif condition_features is not None:
            raise ValueError("condition_features provided but AstBlendGateHead.condition_dim == 0")

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
        h = torch.cat(parts, dim=-1)
        gate_logits = self.out(self.hidden(self.input_norm(h))).squeeze(-1)
        gate = torch.sigmoid(gate_logits)
        return AstBlendGateHeadOutputs(gate_logits=gate_logits, gate=gate)


__all__ = [
    "AstBlendGateHead",
    "AstBlendGateHeadOutputs",
    "AssistShareHead",
    "AssistShareHeadOutputs",
    "TeamAstBudgetHead",
    "TeamAstBudgetHeadOutputs",
]
