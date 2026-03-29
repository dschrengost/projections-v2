"""Team-budget auxiliary heads for GTv2."""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import nn


@dataclass(frozen=True)
class TeamPointsBudgetHeadOutputs:
    """Predicted team points budgets with shape (B, 2)."""

    team_points: torch.Tensor


@dataclass(frozen=True)
class TeamPPPHeadOutputs:
    """Predicted team PPP values with shape (B, 2)."""

    team_ppp: torch.Tensor


@dataclass(frozen=True)
class TeamAdvantageHeadOutputs:
    """Predicted game-level team advantage latent.

    The scalar `mu` is interpreted as the signed side-0 minus side-1 latent
    advantage for the game. `sampled_advantage` is populated only when sampling.
    """

    mu: torch.Tensor
    sigma: torch.Tensor
    df: torch.Tensor
    sampled_advantage: torch.Tensor | None


class TeamPointsBudgetHead(nn.Module):
    """Predict positive home/away team points budgets from team/game state."""

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
    ) -> TeamPointsBudgetHeadOutputs:
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if game_state.ndim != 2:
            raise ValueError("game_state must have shape (B,D)")
        if team_states.shape[0] != game_state.shape[0] or team_states.shape[2] != game_state.shape[1]:
            raise ValueError("team_states and game_state must align on batch and d_model")

        game_ctx = game_state.unsqueeze(1).expand(-1, 2, -1)
        h = torch.cat([team_states, game_ctx], dim=-1)
        raw = self.net(h).squeeze(-1)
        return TeamPointsBudgetHeadOutputs(team_points=self.softplus(raw))


class TeamPPPHead(nn.Module):
    """Predict positive home/away team PPP values from team/game state."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        baseline_ppp: float = 1.12,
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
        self.baseline_ppp = float(baseline_ppp)
        with torch.no_grad():
            bias = torch.full((1,), float(self.baseline_ppp))
            self.net[-1].bias.copy_(bias)

    def forward(
        self,
        *,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
    ) -> TeamPPPHeadOutputs:
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if game_state.ndim != 2:
            raise ValueError("game_state must have shape (B,D)")
        if team_states.shape[0] != game_state.shape[0] or team_states.shape[2] != game_state.shape[1]:
            raise ValueError("team_states and game_state must align on batch and d_model")

        game_ctx = game_state.unsqueeze(1).expand(-1, 2, -1)
        h = torch.cat([team_states, game_ctx], dim=-1)
        raw = self.net(h).squeeze(-1)
        return TeamPPPHeadOutputs(team_ppp=self.softplus(raw))


class TeamAdvantageHead(nn.Module):
    """Predict a signed game-level team advantage latent with Student-t residuals."""

    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        min_df: float = 2.5,
        max_df: float = 30.0,
        min_sigma: float = 0.05,
        max_sigma: float = 15.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        self.min_df = float(min_df)
        self.max_df = float(max_df)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        in_dim = int(d_model) * 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 3),
        )
        with torch.no_grad():
            self.net[-1].bias.zero_()

    def forward(
        self,
        *,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        sample: bool = False,
    ) -> TeamAdvantageHeadOutputs:
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if game_state.ndim != 2:
            raise ValueError("game_state must have shape (B,D)")
        if team_states.shape[0] != game_state.shape[0] or team_states.shape[2] != game_state.shape[1]:
            raise ValueError("team_states and game_state must align on batch and d_model")

        h = torch.cat([team_states[:, 0, :], team_states[:, 1, :], game_state], dim=-1)
        raw = self.net(h)
        mu = raw[:, 0]
        sigma = torch.clamp(nn.functional.softplus(raw[:, 1]) + self.min_sigma, max=self.max_sigma)
        df = self.min_df + (self.max_df - self.min_df) * torch.sigmoid(raw[:, 2])
        sampled_advantage: torch.Tensor | None = None
        if sample:
            sampled_advantage = self._sample_student_t(mu, sigma, df)
        return TeamAdvantageHeadOutputs(mu=mu, sigma=sigma, df=df, sampled_advantage=sampled_advantage)

    @staticmethod
    def _sample_student_t(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        df: torch.Tensor,
    ) -> torch.Tensor:
        half_df = df * 0.5
        chi2 = torch._standard_gamma(half_df) / half_df  # type: ignore[attr-defined]
        chi2 = torch.clamp(chi2, min=1e-6)
        z = torch.randn_like(mu)
        return mu + sigma * z / torch.sqrt(chi2)

    @staticmethod
    def nll_student_t(
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        df: torch.Tensor,
    ) -> torch.Tensor:
        half_df = df * 0.5
        half_dfp1 = (df + 1.0) * 0.5
        z = (x - mu) / sigma
        return (
            -torch.lgamma(half_dfp1)
            + torch.lgamma(half_df)
            + 0.5 * torch.log(df * math.pi)
            + torch.log(sigma)
            + half_dfp1 * torch.log1p(z * z / df)
        )


__all__ = [
    "TeamAdvantageHead",
    "TeamAdvantageHeadOutputs",
    "TeamPointsBudgetHead",
    "TeamPointsBudgetHeadOutputs",
    "TeamPPPHead",
    "TeamPPPHeadOutputs",
]
