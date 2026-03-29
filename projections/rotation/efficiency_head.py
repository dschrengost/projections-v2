from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EfficiencyHeadOutputs:
    alpha_ft: torch.Tensor
    beta_ft: torch.Tensor
    alpha_fg2: torch.Tensor
    beta_fg2: torch.Tensor
    alpha_fg3: torch.Tensor
    beta_fg3: torch.Tensor
    mean_ft: torch.Tensor
    mean_fg2: torch.Tensor
    mean_fg3: torch.Tensor


class EfficiencyHead(nn.Module):
    """Predict per-player Beta-Binomial parameters for FT/FG2/FG3 make rates."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_team_context_features: int = 0,
        ft_prior_mean: float = 0.77,
        ft_prior_strength: float = 6.0,
        fg2_prior_mean: float = 0.54,
        fg2_prior_strength: float = 8.0,
        fg3_prior_mean: float = 0.36,
        fg3_prior_strength: float = 8.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_team_context_features < 0:
            raise ValueError("num_team_context_features must be >= 0")

        self.num_team_context_features = int(num_team_context_features)

        self.net = nn.Sequential(
            nn.Linear(3 * int(d_model) + self.num_team_context_features, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 6),
        )
        self.softplus = nn.Softplus()

        self.ft_prior_alpha = float(max(1e-4, ft_prior_mean * ft_prior_strength))
        self.ft_prior_beta = float(max(1e-4, (1.0 - ft_prior_mean) * ft_prior_strength))
        self.fg2_prior_alpha = float(max(1e-4, fg2_prior_mean * fg2_prior_strength))
        self.fg2_prior_beta = float(max(1e-4, (1.0 - fg2_prior_mean) * fg2_prior_strength))
        self.fg3_prior_alpha = float(max(1e-4, fg3_prior_mean * fg3_prior_strength))
        self.fg3_prior_beta = float(max(1e-4, (1.0 - fg3_prior_mean) * fg3_prior_strength))

    def forward(
        self,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        player_team_index: torch.Tensor,
        valid_mask: torch.Tensor,
        team_context: torch.Tensor | None = None,
    ) -> EfficiencyHeadOutputs:
        if player_states.ndim != 3:
            raise ValueError("player_states must have shape (B,P,D)")
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if game_state.ndim != 2:
            raise ValueError("game_state must have shape (B,D)")
        if player_team_index.shape != player_states.shape[:2]:
            raise ValueError("player_team_index must align with player_states first two dims")
        if valid_mask.shape != player_states.shape[:2]:
            raise ValueError("valid_mask must align with player_states first two dims")

        bsz, n_players, _ = player_states.shape
        gather_idx = player_team_index.to(dtype=torch.long).unsqueeze(-1).expand(-1, -1, team_states.shape[-1])
        team_ctx = torch.gather(team_states, dim=1, index=gather_idx)
        game_ctx = game_state.unsqueeze(1).expand(bsz, n_players, -1)

        parts = [player_states, team_ctx, game_ctx]
        if self.num_team_context_features > 0:
            if team_context is None:
                raise ValueError("team_context required when num_team_context_features > 0")
            if team_context.ndim != 3 or team_context.shape[:2] != (bsz, 2):
                raise ValueError("team_context must have shape (B,2,C)")
            if int(team_context.shape[-1]) != self.num_team_context_features:
                raise ValueError("team_context feature dim does not match configured num_team_context_features")
            gather_team_ctx = player_team_index.to(dtype=torch.long).unsqueeze(-1).expand(
                -1, -1, team_context.shape[-1]
            )
            player_team_context = torch.gather(team_context, dim=1, index=gather_team_ctx)
            parts.append(player_team_context.to(dtype=player_states.dtype))

        x = torch.cat(parts, dim=-1)
        logits = self.net(x)
        valid_f = valid_mask.to(dtype=logits.dtype).unsqueeze(-1)
        logits = logits * valid_f

        raw_a_ft, raw_b_ft, raw_a_fg2, raw_b_fg2, raw_a_fg3, raw_b_fg3 = logits.unbind(dim=-1)
        alpha_ft = self.softplus(raw_a_ft) + float(self.ft_prior_alpha)
        beta_ft = self.softplus(raw_b_ft) + float(self.ft_prior_beta)
        alpha_fg2 = self.softplus(raw_a_fg2) + float(self.fg2_prior_alpha)
        beta_fg2 = self.softplus(raw_b_fg2) + float(self.fg2_prior_beta)
        alpha_fg3 = self.softplus(raw_a_fg3) + float(self.fg3_prior_alpha)
        beta_fg3 = self.softplus(raw_b_fg3) + float(self.fg3_prior_beta)

        mean_ft = alpha_ft / (alpha_ft + beta_ft).clamp(min=1e-8)
        mean_fg2 = alpha_fg2 / (alpha_fg2 + beta_fg2).clamp(min=1e-8)
        mean_fg3 = alpha_fg3 / (alpha_fg3 + beta_fg3).clamp(min=1e-8)
        return EfficiencyHeadOutputs(
            alpha_ft=alpha_ft,
            beta_ft=beta_ft,
            alpha_fg2=alpha_fg2,
            beta_fg2=beta_fg2,
            alpha_fg3=alpha_fg3,
            beta_fg3=beta_fg3,
            mean_ft=mean_ft,
            mean_fg2=mean_fg2,
            mean_fg3=mean_fg3,
        )
