"""Possession-coupled team event backbone for game-level stat generation.

This module implements:
  - PossessionHead: Shared game-possessions latent (P_game) with Student-t residuals.
  - TeamEventBackbone: Joint team-event vector per world enforcing possession identity
    by construction (FGA = P + OREB - TOV - 0.44*FTA).
  - ThreePAShareHead: Optional team-level shot-mix latent (3PA share).

See GAME_TRANSFORMER_SPEC.md section 15 for design rationale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

# Possession identity coefficient for FTA term
FTA_POSS_COEFF = 0.44

# Backbone event names (per team, order matters for indexing)
BACKBONE_EVENT_NAMES = ("fga", "fta", "tov", "oreb")


@dataclass(frozen=True)
class PossessionHeadOutputs:
    """Outputs from the possession head.

    Attributes:
        mu: Predicted mean possessions per game (B,).
        sigma: Predicted std dev of possessions per game (B,).
        df: Predicted degrees of freedom for Student-t (B,).
        sampled_poss: Sampled possession count per world (B,). None during training.
    """
    mu: torch.Tensor
    sigma: torch.Tensor
    df: torch.Tensor
    sampled_poss: torch.Tensor | None


@dataclass(frozen=True)
class TeamEventBackboneOutputs:
    """Outputs from the joint team event backbone.

    Attributes:
        fga: FGA per team (B, 2). Deterministic from possession identity.
        fta: FTA per team (B, 2).
        tov: TOV per team (B, 2).
        oreb: OREB per team (B, 2).
        three_pa_share: 3PA share per team (B, 2). None if shot-mix head disabled.
        poss_used: The shared possession value used for construction (B,).
    """
    fga: torch.Tensor
    fta: torch.Tensor
    tov: torch.Tensor
    oreb: torch.Tensor
    three_pa_share: torch.Tensor | None
    poss_used: torch.Tensor


class PossessionHead(nn.Module):
    """Predicts a shared game-possessions distribution from the game-level state.

    Uses Student-t residuals: heavier tails than Gaussian, simpler than flows.

    Outputs mu_P, sigma_P, df_P such that:
        P_game ~ StudentT(df=df_P, loc=mu_P, scale=sigma_P)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        mu_mode: str = "absolute",
        mu_baseline: float = 100.0,
        min_df: float = 2.5,
        max_df: float = 30.0,
        min_sigma: float = 0.5,
        max_sigma: float = 15.0,
        num_game_features: int = 0,
    ) -> None:
        super().__init__()
        if str(mu_mode) not in {"absolute", "baseline_delta"}:
            raise ValueError("mu_mode must be one of: absolute, baseline_delta")
        self.mu_mode = str(mu_mode)
        self.mu_baseline = float(mu_baseline)
        self.min_df = float(min_df)
        self.max_df = float(max_df)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.num_game_features = int(num_game_features)

        input_dim = d_model + int(num_game_features)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # mu, raw_sigma, raw_df
        )
        # Initialize output bias for sensible defaults.
        # absolute mode: mu ~ 97 (legacy behavior)
        # baseline_delta mode: delta_mu ~ 0, so mu starts near configured baseline
        with torch.no_grad():
            mu_bias = 97.0 if self.mu_mode == "absolute" else 0.0
            self.net[-1].bias.copy_(torch.tensor([mu_bias, 0.0, 0.0]))

    def forward(
        self,
        game_state: torch.Tensor,
        *,
        sample: bool = False,
        game_features: torch.Tensor | None = None,
    ) -> PossessionHeadOutputs:
        """Predict possession distribution parameters.

        Args:
            game_state: (B, D) game-level hidden state.
            sample: If True, sample P_game from the predicted distribution.
            game_features: (B, G) raw game features for direct injection.
                Required when num_game_features > 0.
        """
        if game_state.ndim != 2:
            raise ValueError("game_state must have shape (B, D)")

        if self.num_game_features > 0:
            if game_features is None:
                raise ValueError("game_features required when num_game_features > 0")
            inp = torch.cat([game_state, game_features], dim=-1)
        else:
            inp = game_state
        raw = self.net(inp)  # (B, 3)
        if self.mu_mode == "baseline_delta":
            mu = self.mu_baseline + raw[:, 0]
        else:
            mu = raw[:, 0]
        # sigma: softplus then clamp to [min_sigma, max_sigma]
        sigma = torch.clamp(
            nn.functional.softplus(raw[:, 1]) + self.min_sigma,
            max=self.max_sigma,
        )
        # df: sigmoid scaled to [min_df, max_df]
        df = self.min_df + (self.max_df - self.min_df) * torch.sigmoid(raw[:, 2])

        sampled: torch.Tensor | None = None
        if sample:
            sampled = self._sample_student_t(mu, sigma, df)

        return PossessionHeadOutputs(mu=mu, sigma=sigma, df=df, sampled_poss=sampled)

    @staticmethod
    def _sample_student_t(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        df: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from Student-t(df, mu, sigma) via reparameterization."""
        # chi2 sample: gamma(df/2, df/2) -> chi2(df) / df
        half_df = df * 0.5
        chi2 = torch._standard_gamma(half_df) / half_df  # type: ignore[attr-defined]
        chi2 = torch.clamp(chi2, min=1e-6)  # numerical safety
        z = torch.randn_like(mu)
        return mu + sigma * z / torch.sqrt(chi2)

    @staticmethod
    def nll_student_t(
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        df: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood of Student-t distribution.

        Returns per-sample NLL (B,).
        """
        half_df = df * 0.5
        half_dfp1 = (df + 1.0) * 0.5
        z = (x - mu) / sigma
        nll = (
            -torch.lgamma(half_dfp1)
            + torch.lgamma(half_df)
            + 0.5 * torch.log(df * math.pi)
            + torch.log(sigma)
            + half_dfp1 * torch.log1p(z * z / df)
        )
        return nll


class TeamEventBackbone(nn.Module):
    """Joint team event generator enforcing possession identity by construction.

    For each team, predicts rates for FTA, TOV, OREB in bounded spaces,
    then deterministically computes FGA to satisfy:

        FGA_team = P_game + OREB_team - TOV_team - 0.44 * FTA_team

    Uses Student-t residuals on the rate predictions for heavy-tailed sampling.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        min_df: float = 2.5,
        max_df: float = 30.0,
        num_game_features: int = 0,
    ) -> None:
        super().__init__()
        self.min_df = float(min_df)
        self.max_df = float(max_df)
        self.num_game_features = int(num_game_features)

        # Per-team rate predictor: from [team_state, game_state, poss_embedding, game_features]
        # Outputs for each team: 3 rates * 3 params (mu, raw_sigma, raw_df) = 9
        # Rates: fta_rate, tov_rate, oreb_rate (all as fractions of possessions)
        input_dim = d_model * 2 + 1 + int(num_game_features)  # team_state + game_state + poss_scalar + game_features
        self.rate_net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 9),  # 3 rates * (mu, raw_sigma, raw_df)
        )

        # Initialize biases for reasonable NBA rate defaults:
        # FTA rate ~ 0.23 (22 FTA per ~97 poss), logit(0.23) ~ -1.2
        # TOV rate ~ 0.14 (14 TOV per ~97 poss), logit(0.14) ~ -1.8
        # OREB rate ~ 0.10 (10 OREB per ~97 poss), logit(0.10) ~ -2.2
        with torch.no_grad():
            bias = torch.zeros(9)
            bias[0] = -1.2   # fta_rate mu (logit space)
            bias[3] = -1.8   # tov_rate mu (logit space)
            bias[6] = -2.2   # oreb_rate mu (logit space)
            self.rate_net[-1].bias.copy_(bias)

    def forward(
        self,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        poss: torch.Tensor,
        *,
        sample: bool = False,
        game_features: torch.Tensor | None = None,
    ) -> TeamEventBackboneOutputs:
        """Generate team event vectors.

        Args:
            team_states: (B, 2, D) team hidden states.
            game_state: (B, D) game hidden state.
            poss: (B,) shared possession count for this world.
            sample: If True, sample rates from Student-t; else use mean (mu).
            game_features: (B, G) raw game features for direct injection.
                Required when num_game_features > 0.
        """
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B, 2, D)")
        if game_state.ndim != 2:
            raise ValueError("game_state must have shape (B, D)")
        if poss.ndim != 1:
            raise ValueError("poss must have shape (B,)")

        bsz = team_states.shape[0]
        device = team_states.device
        dtype = team_states.dtype

        # Expand poss for concatenation
        poss_feat = poss.unsqueeze(-1).to(dtype=dtype)  # (B, 1)
        game_exp = game_state.unsqueeze(1).expand(-1, 2, -1)  # (B, 2, D)
        poss_exp = poss_feat.unsqueeze(1).expand(-1, 2, -1)  # (B, 2, 1)

        # Concatenate: [team_state, game_state, poss, game_features]
        parts = [team_states, game_exp, poss_exp]
        if self.num_game_features > 0:
            if game_features is None:
                raise ValueError("game_features required when num_game_features > 0")
            gf_exp = game_features.unsqueeze(1).expand(-1, 2, -1)  # (B, 2, G)
            parts.append(gf_exp)
        inp = torch.cat(parts, dim=-1)  # (B, 2, 2D+1+G)
        raw = self.rate_net(inp)  # (B, 2, 9)

        # Parse into 3 rates x 3 params
        # raw[:, :, 0:3] = fta_rate (mu, raw_sigma, raw_df)
        # raw[:, :, 3:6] = tov_rate (mu, raw_sigma, raw_df)
        # raw[:, :, 6:9] = oreb_rate (mu, raw_sigma, raw_df)
        rate_mus_logit = raw[:, :, [0, 3, 6]]      # (B, 2, 3)
        rate_raw_sigmas = raw[:, :, [1, 4, 7]]      # (B, 2, 3)
        rate_raw_dfs = raw[:, :, [2, 5, 8]]         # (B, 2, 3)

        # Convert to distribution params
        rate_sigmas = torch.clamp(nn.functional.softplus(rate_raw_sigmas) + 0.01, max=2.0)
        rate_dfs = self.min_df + (self.max_df - self.min_df) * torch.sigmoid(rate_raw_dfs)

        if sample:
            # Sample in logit space using Student-t
            logit_samples = _sample_student_t_batch(rate_mus_logit, rate_sigmas, rate_dfs)
            rates = torch.sigmoid(logit_samples)  # (B, 2, 3)
        else:
            # Use mean (sigmoid of mu in logit space)
            rates = torch.sigmoid(rate_mus_logit)  # (B, 2, 3)

        fta_rate = rates[:, :, 0]   # (B, 2)
        tov_rate = rates[:, :, 1]   # (B, 2)
        oreb_rate = rates[:, :, 2]  # (B, 2)

        # Convert rates to counts using shared possession count
        poss_2d = poss.unsqueeze(-1).expand(-1, 2)  # (B, 2)
        fta = fta_rate * poss_2d
        tov = tov_rate * poss_2d
        oreb = oreb_rate * poss_2d

        # FGA deterministic from possession identity:
        # P = FGA - OREB + TOV + 0.44 * FTA
        # => FGA = P + OREB - TOV - 0.44 * FTA
        fga = poss_2d + oreb - tov - FTA_POSS_COEFF * fta

        # Lightweight feasibility: FGA must be non-negative
        fga = torch.clamp(fga, min=0.0)

        return TeamEventBackboneOutputs(
            fga=fga,
            fta=fta,
            tov=tov,
            oreb=oreb,
            three_pa_share=None,  # Populated by ThreePAShareHead if wired
            poss_used=poss,
        )

    def nll_rates(
        self,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        poss: torch.Tensor,
        *,
        fta_true: torch.Tensor,
        tov_true: torch.Tensor,
        oreb_true: torch.Tensor,
        game_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute NLL of observed rates under the predicted Student-t in logit space.

        All truth tensors have shape (B, 2).
        Returns scalar mean NLL.
        """
        bsz = team_states.shape[0]
        dtype = team_states.dtype
        device = team_states.device

        poss_2d = poss.unsqueeze(-1).expand(-1, 2).clamp(min=1.0)
        # Convert counts to rates
        fta_rate_true = (fta_true / poss_2d).clamp(1e-6, 1.0 - 1e-6)
        tov_rate_true = (tov_true / poss_2d).clamp(1e-6, 1.0 - 1e-6)
        oreb_rate_true = (oreb_true / poss_2d).clamp(1e-6, 1.0 - 1e-6)

        # Convert to logit space
        true_logits = torch.stack([
            torch.logit(fta_rate_true),
            torch.logit(tov_rate_true),
            torch.logit(oreb_rate_true),
        ], dim=-1)  # (B, 2, 3)

        # Get predicted distribution params
        poss_feat = poss.unsqueeze(-1).to(dtype=dtype)
        game_exp = game_state.unsqueeze(1).expand(-1, 2, -1)
        poss_exp = poss_feat.unsqueeze(1).expand(-1, 2, -1)
        parts = [team_states, game_exp, poss_exp]
        if self.num_game_features > 0:
            if game_features is None:
                raise ValueError("game_features required when num_game_features > 0")
            gf_exp = game_features.unsqueeze(1).expand(-1, 2, -1)
            parts.append(gf_exp)
        inp = torch.cat(parts, dim=-1)
        raw = self.rate_net(inp)  # (B, 2, 9)

        rate_mus_logit = raw[:, :, [0, 3, 6]]
        rate_raw_sigmas = raw[:, :, [1, 4, 7]]
        rate_raw_dfs = raw[:, :, [2, 5, 8]]
        rate_sigmas = torch.clamp(nn.functional.softplus(rate_raw_sigmas) + 0.01, max=2.0)
        rate_dfs = self.min_df + (self.max_df - self.min_df) * torch.sigmoid(rate_raw_dfs)

        # NLL in logit space (per-rate, per-team)
        nll = _nll_student_t_batch(true_logits, rate_mus_logit, rate_sigmas, rate_dfs)
        return nll.mean()


class ThreePAShareHead(nn.Module):
    """Optional team-level 3PA share latent.

    Predicts three_pa_share_team ∈ (0,1) per team per world:
        3PA_team = three_pa_share_team * FGA_team

    Uses logit-normal distribution (Student-t in logit space).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        min_df: float = 2.5,
        max_df: float = 30.0,
        num_game_features: int = 0,
    ) -> None:
        super().__init__()
        self.min_df = float(min_df)
        self.max_df = float(max_df)
        self.num_game_features = int(num_game_features)

        # Input: team_state + game_state + FGA_team scalar + game_features
        input_dim = d_model * 2 + 1 + int(num_game_features)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # mu_logit, raw_sigma, raw_df
        )
        # Default 3PA share ~ 0.38 -> logit(0.38) ~ -0.49
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([-0.49, 0.0, 0.0]))

    def forward(
        self,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        fga: torch.Tensor,
        *,
        sample: bool = False,
        game_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict 3PA share per team.

        Args:
            team_states: (B, 2, D).
            game_state: (B, D).
            fga: (B, 2) FGA per team from backbone.
            sample: If True, sample from Student-t in logit space.
            game_features: (B, G) raw game features for direct injection.

        Returns:
            three_pa_share: (B, 2) in (0, 1).
        """
        game_exp = game_state.unsqueeze(1).expand(-1, 2, -1)
        fga_feat = fga.unsqueeze(-1).to(dtype=team_states.dtype)
        parts = [team_states, game_exp, fga_feat]
        if self.num_game_features > 0:
            if game_features is None:
                raise ValueError("game_features required when num_game_features > 0")
            gf_exp = game_features.unsqueeze(1).expand(-1, 2, -1)
            parts.append(gf_exp)
        inp = torch.cat(parts, dim=-1)
        raw = self.net(inp)  # (B, 2, 3)

        mu_logit = raw[:, :, 0]
        sigma = torch.clamp(nn.functional.softplus(raw[:, :, 1]) + 0.01, max=2.0)
        df = self.min_df + (self.max_df - self.min_df) * torch.sigmoid(raw[:, :, 2])

        if sample:
            logit_sample = _sample_student_t_2d(mu_logit, sigma, df)
            return torch.sigmoid(logit_sample)
        else:
            return torch.sigmoid(mu_logit)

    def nll(
        self,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        fga: torch.Tensor,
        *,
        three_pa_share_true: torch.Tensor,
        game_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """NLL of observed 3PA share under logit-normal Student-t.

        Args:
            three_pa_share_true: (B, 2) ground truth 3PA shares.
            game_features: (B, G) raw game features for direct injection.

        Returns:
            Scalar mean NLL.
        """
        game_exp = game_state.unsqueeze(1).expand(-1, 2, -1)
        fga_feat = fga.unsqueeze(-1).to(dtype=team_states.dtype)
        parts = [team_states, game_exp, fga_feat]
        if self.num_game_features > 0:
            if game_features is None:
                raise ValueError("game_features required when num_game_features > 0")
            gf_exp = game_features.unsqueeze(1).expand(-1, 2, -1)
            parts.append(gf_exp)
        inp = torch.cat(parts, dim=-1)
        raw = self.net(inp)

        mu_logit = raw[:, :, 0]
        sigma = torch.clamp(nn.functional.softplus(raw[:, :, 1]) + 0.01, max=2.0)
        df = self.min_df + (self.max_df - self.min_df) * torch.sigmoid(raw[:, :, 2])

        true_clamped = three_pa_share_true.clamp(1e-6, 1.0 - 1e-6)
        true_logit = torch.logit(true_clamped)
        nll = _nll_student_t_2d(true_logit, mu_logit, sigma, df)
        return nll.mean()


# ---------------------------------------------------------------------------
# Utility: possession truth label computation
# ---------------------------------------------------------------------------

def compute_possession_truth(
    fga: torch.Tensor,
    oreb: torch.Tensor,
    tov: torch.Tensor,
    fta: torch.Tensor,
) -> torch.Tensor:
    """Compute per-team possession estimate from box score counts.

    P_team = FGA - OREB + TOV + 0.44 * FTA

    Args:
        fga, oreb, tov, fta: Each (B, 2) for home/away teams.

    Returns:
        P_game: (B,) = mean(P_home, P_away) for stability.
    """
    poss_per_team = fga - oreb + tov + FTA_POSS_COEFF * fta  # (B, 2)
    return poss_per_team.mean(dim=1)  # (B,)


def compute_possession_truth_per_team(
    fga: torch.Tensor,
    oreb: torch.Tensor,
    tov: torch.Tensor,
    fta: torch.Tensor,
) -> torch.Tensor:
    """Compute per-team possession estimates.

    Returns:
        poss_per_team: (B, 2).
    """
    return fga - oreb + tov + FTA_POSS_COEFF * fta


# ---------------------------------------------------------------------------
# Student-t sampling/NLL helpers (batched)
# ---------------------------------------------------------------------------

def _sample_student_t_batch(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    df: torch.Tensor,
) -> torch.Tensor:
    """Sample from Student-t for arbitrary-shape tensors."""
    half_df = df * 0.5
    chi2 = torch._standard_gamma(half_df) / half_df.clamp(min=1e-6)  # type: ignore[attr-defined]
    chi2 = chi2.clamp(min=1e-6)
    z = torch.randn_like(mu)
    return mu + sigma * z / torch.sqrt(chi2)


def _sample_student_t_2d(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    df: torch.Tensor,
) -> torch.Tensor:
    """Sample from Student-t for (B, 2) tensors."""
    return _sample_student_t_batch(mu, sigma, df)


def _nll_student_t_batch(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    df: torch.Tensor,
) -> torch.Tensor:
    """Student-t NLL for arbitrary-shape tensors. Returns element-wise NLL."""
    half_df = df * 0.5
    half_dfp1 = (df + 1.0) * 0.5
    z = (x - mu) / sigma
    nll = (
        -torch.lgamma(half_dfp1)
        + torch.lgamma(half_df)
        + 0.5 * torch.log(df * math.pi)
        + torch.log(sigma)
        + half_dfp1 * torch.log1p(z * z / df)
    )
    return nll


def _nll_student_t_2d(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    df: torch.Tensor,
) -> torch.Tensor:
    """Student-t NLL for (B, 2) tensors."""
    return _nll_student_t_batch(x, mu, sigma, df)
