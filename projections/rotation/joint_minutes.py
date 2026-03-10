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
    active = active_mask.to(dtype=torch.bool)
    active_f = active.to(dtype=torch.float32)

    total = float(total_minutes)
    cap = float(max_minutes_per_player)
    tol = float(eps)

    u = preferences.to(dtype=torch.float32)
    active_count = active.sum(dim=1).to(dtype=torch.float32)
    valid_rows = active_count > 0
    feasible_totals = active_count * cap

    if not bool(allow_scale_down_infeasible):
        infeasible = valid_rows & (feasible_totals + tol < total)
        if bool(infeasible.any()):
            first_idx = int(torch.nonzero(infeasible, as_tuple=False)[0].item())
            raise ValueError(
                f"Infeasible capped simplex projection: active_count={int(active_count[first_idx].item())}, "
                f"cap={cap}, total={total}"
            )

    target_total = torch.full((bsz,), float(total), dtype=torch.float32, device=u.device)
    if bool(allow_scale_down_infeasible):
        target_total = torch.minimum(target_total, feasible_totals)
    target_total = torch.where(valid_rows, target_total, torch.zeros_like(target_total))

    lower = torch.where(active, u - cap, torch.full_like(u, torch.inf)).amin(dim=1)
    upper = torch.where(active, u, torch.full_like(u, -torch.inf)).amax(dim=1)
    lower = torch.where(valid_rows, lower, torch.zeros_like(lower))
    upper = torch.where(valid_rows, upper, torch.zeros_like(upper))

    for _ in range(int(max_iter)):
        lam = 0.5 * (lower + upper)
        m = torch.clamp(u - lam.unsqueeze(1), min=0.0, max=cap) * active_f
        m_sum = m.sum(dim=1)
        too_high = valid_rows & (m_sum > target_total)
        lower = torch.where(too_high, lam, lower)
        upper = torch.where(too_high, upper, lam)

    lam = 0.5 * (lower + upper)
    out_f32 = torch.clamp(u - lam.unsqueeze(1), min=0.0, max=cap) * active_f

    # Small numeric cleanup to hit exact team totals after finite-iter bisection.
    diff = target_total - out_f32.sum(dim=1)
    adjust_rows = valid_rows & diff.abs().gt(tol)
    if bool(adjust_rows.any()):
        free = ((out_f32 > tol) & (out_f32 < (cap - tol))) & active
        free_count = free.sum(dim=1).to(dtype=torch.float32)
        free_rows = adjust_rows & free_count.gt(0)
        if bool(free_rows.any()):
            delta = diff / free_count.clamp(min=1.0)
            out_f32 = torch.where(
                free_rows.unsqueeze(1) & free,
                out_f32 + delta.unsqueeze(1),
                out_f32,
            )
            out_f32 = torch.clamp(out_f32, min=0.0, max=cap)

        # Final residual correction for rows with no free coordinates.
        residual = target_total - out_f32.sum(dim=1)
        residual_rows = valid_rows & residual.abs().gt(tol)
        if bool(residual_rows.any()):
            add_rows = residual_rows & residual.gt(0)
            if bool(add_rows.any()):
                slack = (cap - out_f32) * active_f
                idx_add = torch.argmax(slack, dim=1)
                row_idx = torch.nonzero(add_rows, as_tuple=False).squeeze(1)
                out_f32[row_idx, idx_add[row_idx]] = torch.clamp(
                    out_f32[row_idx, idx_add[row_idx]] + residual[row_idx],
                    min=0.0,
                    max=cap,
                )

            sub_rows = residual_rows & residual.lt(0)
            if bool(sub_rows.any()):
                avail = out_f32 * active_f
                idx_sub = torch.argmax(avail, dim=1)
                row_idx = torch.nonzero(sub_rows, as_tuple=False).squeeze(1)
                out_f32[row_idx, idx_sub[row_idx]] = torch.clamp(
                    out_f32[row_idx, idx_sub[row_idx]] + residual[row_idx],
                    min=0.0,
                    max=cap,
                )

    out_f32 = out_f32 * active_f
    return out_f32.to(dtype=preferences.dtype)


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
