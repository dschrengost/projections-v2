"""Joint team-constrained minutes head with capped-simplex projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _project_single_team_capped_simplex(
    preferences: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    total_minutes: float,
    max_minutes_per_player: float,
    eps: float,
    max_iter: int,
    allow_scale_down_infeasible: bool,
) -> torch.Tensor:
    """Project one (B,N) tensor row-wise onto {m: sum m = total, 0<=m<=cap} with active mask."""

    if preferences.ndim != 2 or active_mask.ndim != 2:
        raise ValueError("preferences and active_mask must have shape (B,N)")
    if preferences.shape != active_mask.shape:
        raise ValueError("preferences and active_mask must have the same shape")
    if total_minutes <= 0:
        raise ValueError("total_minutes must be > 0")
    if max_minutes_per_player <= 0:
        raise ValueError("max_minutes_per_player must be > 0")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")

    bsz, num_players = preferences.shape
    out = torch.zeros((bsz, num_players), dtype=preferences.dtype, device=preferences.device)
    active = active_mask.to(dtype=torch.bool)

    total = float(total_minutes)
    cap = float(max_minutes_per_player)
    tol = float(eps)

    for b_idx in range(bsz):
        mask = active[b_idx]
        num_active = int(mask.sum().item())
        if num_active <= 0:
            continue
        target_total = total
        if float(num_active) * cap + tol < total:
            if not bool(allow_scale_down_infeasible):
                raise ValueError(
                    f"Infeasible capped simplex projection: active_count={num_active}, cap={cap}, total={total}"
                )
            target_total = float(num_active) * cap

        u = preferences[b_idx, mask].to(dtype=torch.float32)
        lower = torch.min(u - cap)
        upper = torch.max(u)

        for _ in range(int(max_iter)):
            lam = 0.5 * (lower + upper)
            m = torch.clamp(u - lam, min=0.0, max=cap)
            m_sum = float(m.sum().item())
            if m_sum > target_total:
                lower = lam
            else:
                upper = lam

        lam = 0.5 * (lower + upper)
        m = torch.clamp(u - lam, min=0.0, max=cap)

        # Small numeric cleanup to hit exact team totals after finite-iter bisection.
        diff = target_total - float(m.sum().item())
        if abs(diff) > tol:
            free = (m > tol) & (m < cap - tol)
            if bool(free.any()):
                m = m.clone()
                m[free] = m[free] + diff / float(free.sum().item())
                m = torch.clamp(m, min=0.0, max=cap)
            else:
                m = m.clone()
                if diff > 0:
                    slack = cap - m
                    idx = torch.argmax(slack)
                    m[idx] = torch.clamp(m[idx] + diff, min=0.0, max=cap)
                else:
                    idx = torch.argmax(m)
                    m[idx] = torch.clamp(m[idx] + diff, min=0.0, max=cap)

        out[b_idx, mask] = m.to(dtype=preferences.dtype)

    return out


def project_minutes_capped_simplex(
    preferences: torch.Tensor,
    active_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    player_team_index: torch.Tensor,
    *,
    total_minutes_per_team: float = 240.0,
    max_minutes_per_player: float = 48.0,
    eps: float = 1e-6,
    max_iter: int = 64,
    fallback_to_valid_on_infeasible: bool = True,
    allow_scale_down_infeasible: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project player preferences into exact-feasible team minutes for both teams."""

    if preferences.ndim != 2:
        raise ValueError("preferences must have shape (B,P)")
    if active_mask.shape != preferences.shape:
        raise ValueError("active_mask must match preferences shape")
    if valid_mask.shape != preferences.shape:
        raise ValueError("valid_mask must match preferences shape")
    if player_team_index.shape != preferences.shape:
        raise ValueError("player_team_index must match preferences shape")

    valid = valid_mask.to(dtype=torch.bool)
    team_index = player_team_index.to(dtype=torch.long)
    active = active_mask.to(dtype=torch.bool) & valid

    out = torch.zeros_like(preferences)
    used_active = active.clone()

    for team_idx in (0, 1):
        team_valid = valid & (team_index == team_idx)
        team_active = active & (team_index == team_idx)

        if bool(fallback_to_valid_on_infeasible):
            per_row_count = team_active.sum(dim=1)
            infeasible = per_row_count.to(dtype=torch.float32) * float(max_minutes_per_player) < float(total_minutes_per_team)
            if bool(infeasible.any()):
                team_active = torch.where(infeasible.unsqueeze(-1), team_valid, team_active)
                used_active = torch.where((infeasible.unsqueeze(-1) & (team_index == team_idx)), team_active, used_active)

        proj = _project_single_team_capped_simplex(
            preferences,
            team_active,
            total_minutes=float(total_minutes_per_team),
            max_minutes_per_player=float(max_minutes_per_player),
            eps=float(eps),
            max_iter=int(max_iter),
            allow_scale_down_infeasible=bool(allow_scale_down_infeasible),
        )
        out = out + proj

    return out * valid.to(dtype=out.dtype), used_active


def _team_context_for_players(team_states: torch.Tensor, player_team_index: torch.Tensor) -> torch.Tensor:
    if team_states.ndim != 3 or team_states.shape[1] != 2:
        raise ValueError("team_states must have shape (B,2,D)")
    if player_team_index.ndim != 2:
        raise ValueError("player_team_index must have shape (B,P)")
    if team_states.shape[0] != player_team_index.shape[0]:
        raise ValueError("team_states and player_team_index batch sizes must match")

    bsz, num_players = player_team_index.shape
    d_model = team_states.shape[2]
    out = torch.zeros((bsz, num_players, d_model), dtype=team_states.dtype, device=team_states.device)
    for team_idx in (0, 1):
        mask = (player_team_index == team_idx).unsqueeze(-1)
        vec = team_states[:, team_idx, :].unsqueeze(1).expand(-1, num_players, -1)
        out = torch.where(mask, vec, out)
    return out


@dataclass(frozen=True)
class JointMinutesOutputs:
    minutes: torch.Tensor
    preferences: torch.Tensor
    used_active_mask: torch.Tensor


class JointMinutesHead(nn.Module):
    """Minutes preference head + capped-simplex projection."""

    def __init__(
        self,
        d_model: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        total_minutes_per_team: float = 240.0,
        max_minutes_per_player: float = 48.0,
        eps: float = 1e-6,
        max_iter: int = 64,
        fallback_to_valid_on_infeasible: bool = True,
    ) -> None:
        super().__init__()
        self.total_minutes_per_team = float(total_minutes_per_team)
        self.max_minutes_per_player = float(max_minutes_per_player)
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.fallback_to_valid_on_infeasible = bool(fallback_to_valid_on_infeasible)

        self.preference_head = nn.Sequential(
            nn.LayerNorm(int(2 * d_model)),
            nn.Linear(int(2 * d_model), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        player_team_index: torch.Tensor,
        valid_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> JointMinutesOutputs:
        if player_states.ndim != 3:
            raise ValueError("player_states must have shape (B,P,D)")
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if player_states.shape[:2] != valid_mask.shape:
            raise ValueError("valid_mask must match player_states first two dims")
        if active_mask.shape != valid_mask.shape or player_team_index.shape != valid_mask.shape:
            raise ValueError("active_mask and player_team_index must match valid_mask shape")

        team_context = _team_context_for_players(team_states, player_team_index)
        logits = self.preference_head(torch.cat([player_states, team_context], dim=-1)).squeeze(-1)
        preferences = logits.masked_fill(~valid_mask.to(dtype=torch.bool), 0.0)

        minutes, used_active = project_minutes_capped_simplex(
            preferences,
            active_mask,
            valid_mask,
            player_team_index,
            total_minutes_per_team=float(self.total_minutes_per_team),
            max_minutes_per_player=float(self.max_minutes_per_player),
            eps=float(self.eps),
            max_iter=int(self.max_iter),
            fallback_to_valid_on_infeasible=bool(self.fallback_to_valid_on_infeasible),
        )

        return JointMinutesOutputs(minutes=minutes, preferences=preferences, used_active_mask=used_active)
