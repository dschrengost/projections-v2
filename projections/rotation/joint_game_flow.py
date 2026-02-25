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


class _GatedTeamAttention(nn.Module):
    """Cross-attention from each player to their teammates for context aggregation.

    Unlike mean pooling which equally weights all teammates, this module learns
    attention weights that can selectively focus on relevant players (e.g., the
    starting lineup, key scorers) and ignore less informative ones (bench players).

    This addresses the H2 star under-projection issue: mean pooling dilutes star
    information with bench averages, but attention can preserve star-specific signal.
    """

    def __init__(self, *, num_stats: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_stats = int(num_stats)
        self.num_heads = int(num_heads)
        # Each head operates over a subset of stats
        head_dim = max(1, self.num_stats // self.num_heads)
        self.head_dim = head_dim
        # Project y_cond to Q, K, V
        self.q_proj = nn.Linear(int(num_stats), self.num_heads * head_dim)
        self.k_proj = nn.Linear(int(num_stats), self.num_heads * head_dim)
        self.v_proj = nn.Linear(int(num_stats), self.num_heads * head_dim)
        self.out_proj = nn.Linear(self.num_heads * head_dim, int(num_stats))
        self.dropout = nn.Dropout(float(dropout))
        # Learnable gate to interpolate between attention context and original
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        y_cond: torch.Tensor,
        valid_mask: torch.Tensor,
        team_index: torch.Tensor,
        team_id: int,
    ) -> torch.Tensor:
        """Compute attended team context for each player.

        Args:
            y_cond: (B, P, S) partially decoded stat values
            valid_mask: (B, P) boolean mask of valid players
            team_index: (B, P) team assignment (0 or 1)
            team_id: which team to compute context for (0 or 1)

        Returns:
            (B, 1, S) attended context for this team, broadcasted across players
        """
        B, P, S = y_cond.shape
        team_mask = valid_mask.to(dtype=torch.bool) & (team_index == int(team_id))

        # Build attention mask: each player attends to same-team players
        # Shape: (B, P, P) where [b, i, j] = True if player j is valid teammate of player i
        attn_mask = team_mask.unsqueeze(1).expand(B, P, P)  # (B, P, P)

        # Project to Q, K, V
        q = self.q_proj(y_cond)  # (B, P, num_heads * head_dim)
        k = self.k_proj(y_cond)
        v = self.v_proj(y_cond)

        # Reshape for multi-head attention
        q = q.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, P, D)
        k = k.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, P, P)

        # Apply attention mask: -inf for non-teammates
        mask_expanded = attn_mask.unsqueeze(1).expand(B, self.num_heads, P, P)
        scores = scores.masked_fill(~mask_expanded, float("-inf"))

        # Softmax (with -inf → 0 attention weight)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle all-masked rows
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)  # (B, H, P, D)
        attended = attended.transpose(1, 2).contiguous().view(B, P, -1)  # (B, P, H*D)
        attended = self.out_proj(attended)  # (B, P, S)

        # Pool attended representations for this team (take mean of team players)
        # This gives a single context vector for the team
        team_ctx = _masked_player_mean(attended, team_mask)  # (B, 1, S)

        # Also compute mean-pooled context as baseline for gating
        mean_ctx = _masked_player_mean(y_cond, team_mask)

        # Gate: sigmoid(gate) * attended + (1 - sigmoid(gate)) * mean
        # Starts at 0.5 interpolation, learns optimal mix
        g = torch.sigmoid(self.gate)
        return g * team_ctx + (1 - g) * mean_ctx


class _GatedGameAttention(nn.Module):
    """Cross-attention from each player to all players in the game for global context."""

    def __init__(self, *, num_stats: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_stats = int(num_stats)
        self.num_heads = int(num_heads)
        head_dim = max(1, self.num_stats // self.num_heads)
        self.head_dim = head_dim
        self.q_proj = nn.Linear(int(num_stats), self.num_heads * head_dim)
        self.k_proj = nn.Linear(int(num_stats), self.num_heads * head_dim)
        self.v_proj = nn.Linear(int(num_stats), self.num_heads * head_dim)
        self.out_proj = nn.Linear(self.num_heads * head_dim, int(num_stats))
        self.dropout = nn.Dropout(float(dropout))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, y_cond: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Compute attended game-level context.

        Args:
            y_cond: (B, P, S) partially decoded stat values
            valid_mask: (B, P) boolean mask of valid players

        Returns:
            (B, 1, S) attended game context
        """
        B, P, S = y_cond.shape
        valid = valid_mask.to(dtype=torch.bool)

        # Attention mask: attend to all valid players
        attn_mask = valid.unsqueeze(1).expand(B, P, P)

        q = self.q_proj(y_cond)
        k = self.k_proj(y_cond)
        v = self.v_proj(y_cond)

        q = q.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, P, self.num_heads, self.head_dim).transpose(1, 2)

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask_expanded = attn_mask.unsqueeze(1).expand(B, self.num_heads, P, P)
        scores = scores.masked_fill(~mask_expanded, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(B, P, -1)
        attended = self.out_proj(attended)

        # Pool to single game context
        game_ctx = _masked_player_mean(attended, valid)
        mean_ctx = _masked_player_mean(y_cond, valid)

        g = torch.sigmoid(self.gate)
        return g * game_ctx + (1 - g) * mean_ctx


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
    """Conditioner network that produces shift/scale for affine coupling.

    Supports two context modes:
    - "mean": Fixed mean-pooling over teammates/game (original, causes H2 issue)
    - "attention": Learned gated attention over teammates/game (H2 fix)
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_stats: int,
        hidden_dim: int,
        dropout: float,
        mean_ctx_weight: float,
        context_mode: str = "mean",
    ) -> None:
        super().__init__()
        cond_dim = int(3 * num_stats)
        self.mean_ctx_weight = float(mean_ctx_weight)
        self.context_mode = str(context_mode).lower()

        if self.context_mode not in ("mean", "attention"):
            raise ValueError(f"context_mode must be 'mean' or 'attention', got {context_mode!r}")

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

        # Attention modules for H2 fix (only created if context_mode == "attention")
        if self.context_mode == "attention":
            self.team_attn = _GatedTeamAttention(
                num_stats=int(num_stats),
                num_heads=4,
                dropout=float(dropout),
            )
            self.game_attn = _GatedGameAttention(
                num_stats=int(num_stats),
                num_heads=4,
                dropout=float(dropout),
            )
        else:
            self.team_attn = None
            self.game_attn = None

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

        if self.context_mode == "attention" and self.team_attn is not None and self.game_attn is not None:
            # H2 fix: use learned attention-based context
            game_ctx = self.game_attn(y_cond, valid).expand(-1, y_cond.shape[1], -1)
            team0_ctx = self.team_attn(y_cond, valid, team_index, team_id=0)
            team1_ctx = self.team_attn(y_cond, valid, team_index, team_id=1)
            team_ctx = torch.where(
                (team_index == 0).unsqueeze(-1),
                team0_ctx.expand(-1, y_cond.shape[1], -1),
                team1_ctx.expand(-1, y_cond.shape[1], -1),
            )
        else:
            # Original mean-pooling (causes H2 star under-projection)
            game_ctx = _masked_player_mean(y_cond, valid).expand(-1, y_cond.shape[1], -1)
            team0_ctx = _masked_team_mean(y_cond, valid, team_index, team_id=0)
            team1_ctx = _masked_team_mean(y_cond, valid, team_index, team_id=1)
            team_ctx = torch.where(
                (team_index == 0).unsqueeze(-1),
                team0_ctx.expand(-1, y_cond.shape[1], -1),
                team1_ctx.expand(-1, y_cond.shape[1], -1),
            )

        mean_ctx_weight = float(self.mean_ctx_weight)
        team_ctx_scaled = team_ctx * mean_ctx_weight
        game_ctx_scaled = game_ctx * mean_ctx_weight
        cond_in = torch.cat([y_cond, team_ctx_scaled, game_ctx_scaled], dim=-1)
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
        mean_ctx_weight: float,
        context_mode: str = "mean",
    ) -> None:
        super().__init__()
        if stat_mask.ndim != 1 or stat_mask.shape[0] != int(num_stats):
            raise ValueError("stat_mask must be rank-1 with length=num_stats")
        self.conditioner = _AffineCouplingConditioner(
            d_model=int(d_model),
            num_stats=int(num_stats),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            mean_ctx_weight=float(mean_ctx_weight),
            context_mode=str(context_mode),
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
    """Set-equivariant affine coupling flow for full-game stat tensors.

    Args:
        d_model: Dimension of player/team/game state embeddings.
        num_stats: Number of stats per player (y dimension).
        hidden_dim: Hidden dimension in conditioner MLPs.
        dropout: Dropout rate.
        num_blocks: Number of affine coupling blocks.
        coupling_type: Type of coupling ("affine" only for now).
        scale_clip: Maximum absolute value for log_scale in coupling (H1 fix: use 3.0).
        mean_ctx_weight: Weight for mean-pooled context (only used in "mean" mode).
        context_mode: "mean" (original) or "attention" (H2 fix for star under-projection).
    """

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
        mean_ctx_weight: float = 1.0,
        context_mode: str = "mean",
    ) -> None:
        super().__init__()
        if int(num_stats) <= 0:
            raise ValueError("num_stats must be > 0")
        if int(num_blocks) <= 0:
            raise ValueError("num_blocks must be > 0")
        if str(coupling_type).lower() != "affine":
            raise ValueError(f"Unsupported coupling_type={coupling_type!r}; only affine is implemented")
        if str(context_mode).lower() not in ("mean", "attention"):
            raise ValueError(f"context_mode must be 'mean' or 'attention', got {context_mode!r}")

        self.num_stats = int(num_stats)
        self.num_blocks = int(num_blocks)
        self.coupling_type = str(coupling_type).lower()
        self.mean_ctx_weight = float(mean_ctx_weight)
        self.context_mode = str(context_mode).lower()

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
                    mean_ctx_weight=float(mean_ctx_weight),
                    context_mode=self.context_mode,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def set_mean_ctx_weight(self, value: float) -> None:
        self.mean_ctx_weight = float(value)
        for block in self.blocks:
            block.conditioner.mean_ctx_weight = float(value)

    def set_scale_clip(self, value: float) -> None:
        """Override scale_clip on all coupling blocks at inference time.

        This is safe because scale_clip only affects the hard-clamp range of
        tanh(log_scale) * scale_clip and does not change parameter shapes.
        Useful for H1 experiments testing whether the 2.0 ceiling suppresses
        star stat projections.
        """
        for block in self.blocks:
            block.scale_clip = float(value)

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
