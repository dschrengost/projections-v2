"""Joint game-level affine coupling flow over (players, stats) tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


LOG_2PI = float(math.log(2.0 * math.pi))


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


def _masked_player_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = weight.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (values * weight).sum(dim=1, keepdim=True) / denom


def _masked_team_mean(values: torch.Tensor, mask: torch.Tensor, team_index: torch.Tensor, team_id: int) -> torch.Tensor:
    team_mask = mask & (team_index == int(team_id))
    return _masked_player_mean(values, team_mask)


@dataclass(frozen=True)
class JointGameFlowOutputs:
    z: torch.Tensor
    log_det: torch.Tensor
    nll: torch.Tensor
    nll_per_dim: torch.Tensor
    observed_dims: torch.Tensor

    @property
    def nll_mean(self) -> torch.Tensor:
        return self.nll_per_dim.mean()


class _AffineCouplingConditioner(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_stats: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        cond_dim = int(3 * num_stats)
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.player_proj = nn.Linear(int(d_model), int(hidden_dim))
        self.team_proj = nn.Linear(int(d_model), int(hidden_dim))
        self.game_proj = nn.Linear(int(d_model), int(hidden_dim))
        self.out = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(2 * num_stats)),
        )

    def forward(
        self,
        y_cond: torch.Tensor,
        *,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        player_team_index: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = valid_mask.to(dtype=torch.bool)
        team_index = player_team_index.to(dtype=torch.long)

        game_ctx = _masked_player_mean(y_cond, valid).expand(-1, y_cond.shape[1], -1)
        team0_ctx = _masked_team_mean(y_cond, valid, team_index, team_id=0)
        team1_ctx = _masked_team_mean(y_cond, valid, team_index, team_id=1)
        team_ctx = torch.where(
            (team_index == 0).unsqueeze(-1),
            team0_ctx.expand(-1, y_cond.shape[1], -1),
            team1_ctx.expand(-1, y_cond.shape[1], -1),
        )

        cond_in = torch.cat([y_cond, team_ctx, game_ctx], dim=-1)
        cond_h = self.cond_proj(cond_in)
        player_h = self.player_proj(player_states)
        team_h = self.team_proj(_team_context_for_players(team_states, team_index))
        game_h = self.game_proj(game_state).unsqueeze(1).expand(-1, y_cond.shape[1], -1)

        fused = cond_h + player_h + team_h + game_h
        out = self.out(fused)
        shift, log_scale = torch.chunk(out, chunks=2, dim=-1)
        return shift, log_scale


class _AffineCouplingBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_stats: int,
        hidden_dim: int,
        dropout: float,
        stat_mask: torch.Tensor,
        scale_clip: float,
    ) -> None:
        super().__init__()
        if stat_mask.ndim != 1 or stat_mask.shape[0] != int(num_stats):
            raise ValueError("stat_mask must be rank-1 with length=num_stats")
        self.conditioner = _AffineCouplingConditioner(
            d_model=int(d_model),
            num_stats=int(num_stats),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self.scale_clip = float(scale_clip)
        self.register_buffer("stat_mask", stat_mask.to(dtype=torch.bool), persistent=False)

    def _transform_masks(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond_mask = self.stat_mask.view(1, 1, -1).to(device=y.device)
        xform_mask = (~self.stat_mask).view(1, 1, -1).to(device=y.device)
        return cond_mask, xform_mask

    def forward(
        self,
        y: torch.Tensor,
        *,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        player_team_index: torch.Tensor,
        valid_mask: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_mask, xform_mask = self._transform_masks(y)
        y_cond = y * cond_mask.to(dtype=y.dtype)
        shift, log_scale = self.conditioner(
            y_cond,
            player_states=player_states,
            team_states=team_states,
            game_state=game_state,
            player_team_index=player_team_index,
            valid_mask=valid_mask,
        )
        log_scale = torch.tanh(log_scale) * float(self.scale_clip)
        xform_float = xform_mask.to(dtype=y.dtype)

        y_out = y_cond + xform_float * (y * torch.exp(log_scale) + shift)
        log_det = (log_scale * xform_float * observed_mask.to(dtype=y.dtype)).sum(dim=(1, 2))
        return y_out, log_det

    def inverse(
        self,
        y: torch.Tensor,
        *,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        player_team_index: torch.Tensor,
        valid_mask: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_mask, xform_mask = self._transform_masks(y)
        y_cond = y * cond_mask.to(dtype=y.dtype)
        shift, log_scale = self.conditioner(
            y_cond,
            player_states=player_states,
            team_states=team_states,
            game_state=game_state,
            player_team_index=player_team_index,
            valid_mask=valid_mask,
        )
        log_scale = torch.tanh(log_scale) * float(self.scale_clip)
        xform_float = xform_mask.to(dtype=y.dtype)

        y_inv = y_cond + xform_float * ((y - shift) * torch.exp(-log_scale))
        log_det = -(log_scale * xform_float * observed_mask.to(dtype=y.dtype)).sum(dim=(1, 2))
        return y_inv, log_det


class JointGameFlow(nn.Module):
    """Set-equivariant affine coupling flow for full-game stat tensors."""

    def __init__(
        self,
        *,
        d_model: int,
        num_stats: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_blocks: int = 4,
        coupling_type: str = "affine",
        scale_clip: float = 2.0,
    ) -> None:
        super().__init__()
        if int(num_stats) <= 0:
            raise ValueError("num_stats must be > 0")
        if int(num_blocks) <= 0:
            raise ValueError("num_blocks must be > 0")
        if str(coupling_type).lower() != "affine":
            raise ValueError(f"Unsupported coupling_type={coupling_type!r}; only affine is implemented")

        self.num_stats = int(num_stats)
        self.num_blocks = int(num_blocks)
        self.coupling_type = str(coupling_type).lower()

        blocks: list[_AffineCouplingBlock] = []
        for block_idx in range(self.num_blocks):
            mask = torch.tensor(
                [((j + block_idx) % 2 == 0) for j in range(self.num_stats)],
                dtype=torch.bool,
            )
            blocks.append(
                _AffineCouplingBlock(
                    d_model=int(d_model),
                    num_stats=self.num_stats,
                    hidden_dim=int(hidden_dim),
                    dropout=float(dropout),
                    stat_mask=mask,
                    scale_clip=float(scale_clip),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def _resolve_observed_mask(
        self,
        y: torch.Tensor,
        valid_mask: torch.Tensor,
        observed_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = valid_mask.to(dtype=torch.bool)
        if observed_mask is None:
            observed = valid.unsqueeze(-1).expand_as(y)
        else:
            if observed_mask.shape != y.shape:
                raise ValueError("observed_mask must have shape (B,P,S)")
            observed = observed_mask.to(dtype=torch.bool)
        return observed & valid.unsqueeze(-1)

    def forward(
        self,
        y: torch.Tensor,
        *,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        player_team_index: torch.Tensor,
        valid_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> JointGameFlowOutputs:
        if y.ndim != 3:
            raise ValueError("y must have shape (B,P,S)")
        if y.shape[-1] != self.num_stats:
            raise ValueError(f"expected y.shape[-1]=={self.num_stats}, got {y.shape[-1]}")
        if player_states.shape[:2] != y.shape[:2]:
            raise ValueError("player_states must align with y shape (B,P,*)")
        if team_states.ndim != 3 or team_states.shape[1] != 2:
            raise ValueError("team_states must have shape (B,2,D)")
        if game_state.ndim != 2 or game_state.shape[0] != y.shape[0]:
            raise ValueError("game_state must have shape (B,D)")
        if player_team_index.shape != y.shape[:2] or valid_mask.shape != y.shape[:2]:
            raise ValueError("player_team_index and valid_mask must have shape (B,P)")

        observed = self._resolve_observed_mask(y, valid_mask, observed_mask)
        z = y
        log_det = torch.zeros((y.shape[0],), dtype=y.dtype, device=y.device)
        for block in self.blocks:
            z, ld = block(
                z,
                player_states=player_states,
                team_states=team_states,
                game_state=game_state,
                player_team_index=player_team_index,
                valid_mask=valid_mask,
                observed_mask=observed,
            )
            log_det = log_det + ld

        observed_float = observed.to(dtype=y.dtype)
        observed_dims = observed_float.sum(dim=(1, 2)).clamp(min=1.0)
        base_nll = 0.5 * (z * z + float(LOG_2PI))
        nll = (base_nll * observed_float).sum(dim=(1, 2)) - log_det
        nll_per_dim = nll / observed_dims

        return JointGameFlowOutputs(
            z=z,
            log_det=log_det,
            nll=nll,
            nll_per_dim=nll_per_dim,
            observed_dims=observed_dims,
        )

    def sample(
        self,
        z: torch.Tensor,
        *,
        player_states: torch.Tensor,
        team_states: torch.Tensor,
        game_state: torch.Tensor,
        player_team_index: torch.Tensor,
        valid_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError("z must have shape (B,P,S)")
        if z.shape[-1] != self.num_stats:
            raise ValueError(f"expected z.shape[-1]=={self.num_stats}, got {z.shape[-1]}")

        observed = self._resolve_observed_mask(z, valid_mask, observed_mask)
        y = z
        for block in reversed(self.blocks):
            y, _ = block.inverse(
                y,
                player_states=player_states,
                team_states=team_states,
                game_state=game_state,
                player_team_index=player_team_index,
                valid_mask=valid_mask,
                observed_mask=observed,
            )
        return y
