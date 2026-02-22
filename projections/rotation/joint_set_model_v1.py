"""Joint rotation+minutes+rates set model (v1).

This module extends the existing rotation-set minutes architecture with
per-player rates and efficiency heads while preserving minute allocation
semantics (team 240-minute constraint).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn

from projections.rotation.set_model import (
    MLP,
    SetTransformerMinutesModel,
    minutes_from_gate_and_share_logits,
    minutes_from_logits,
)


@dataclass(frozen=True)
class JointRotationRatesModelConfig:
    """Serialized config for joint rotation+minutes+rates model artifacts."""

    feature_columns: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    # Minutes backbone config (mirrors SetTransformerMinutesModel controls).
    use_prior_head: bool = False
    prior_weight_col: str = "minutes_from_stints_prior_20"
    prior_weight_floor: float = 0.0
    use_team_embeddings: bool = False
    team_id_vocab: list[int] | None = None
    team_embedding_dim: int = 8
    use_player_embeddings: bool = False
    player_id_vocab: list[int] | None = None
    player_embedding_dim: int = 16
    use_player_team_embeddings: bool = False
    player_team_hash_buckets: int = 16384
    player_team_embedding_dim: int = 8
    embed_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    use_gate_head: bool = True
    num_experts: int = 1
    router_feature_dim: int = 0
    router_temperature: float = 1.0
    router_mode: str = "soft"
    router_gumbel: bool = False
    router_gumbel_tau: float = 1.0
    alloc_activation: str = "entmax"
    entmax_alpha: float = 1.5
    share_temperature: float = 1.0
    total_minutes: float = 240.0
    eps: float = 1e-6
    # Joint heads.
    rates_hidden_dim: int = 128
    rate_target_names: list[str] | None = None
    efficiency_target_names: list[str] | None = None
    # Efficiency output bounds (inclusive clipping range via sigmoid affine map).
    efficiency_lows: list[float] | None = None
    efficiency_highs: list[float] | None = None
    version: str = "joint_rotation_rates_v1"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "JointRotationRatesModelConfig":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in dict(payload).items() if k in known}
        return cls(**filtered)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "JointRotationRatesModelConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class JointSetTransformerRatesModel(nn.Module):
    """Set-transformer minutes backbone + rates/efficiency heads."""

    def __init__(
        self,
        num_features: int,
        *,
        rate_target_names: list[str],
        efficiency_target_names: list[str],
        efficiency_lows: list[float],
        efficiency_highs: list[float],
        use_prior_head: bool = False,
        prior_weight_floor: float = 0.0,
        use_team_embeddings: bool = False,
        num_teams: int | None = None,
        team_embedding_dim: int = 8,
        use_player_embeddings: bool = False,
        num_players: int | None = None,
        player_embedding_dim: int = 16,
        use_player_team_embeddings: bool = False,
        player_team_hash_buckets: int = 16384,
        player_team_embedding_dim: int = 8,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        use_gate_head: bool = True,
        num_experts: int = 1,
        router_feature_dim: int = 0,
        router_temperature: float = 1.0,
        router_mode: str = "soft",
        router_gumbel: bool = False,
        router_gumbel_tau: float = 1.0,
        alloc_activation: str = "entmax",
        entmax_alpha: float = 1.5,
        share_temperature: float = 1.0,
        total_minutes: float = 240.0,
        eps: float = 1e-6,
        rates_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if len(rate_target_names) <= 0:
            raise ValueError("rate_target_names must be non-empty")
        if len(efficiency_target_names) <= 0:
            raise ValueError("efficiency_target_names must be non-empty")
        if not (len(efficiency_target_names) == len(efficiency_lows) == len(efficiency_highs)):
            raise ValueError("Efficiency names/lows/highs must have same length")

        self.rate_target_names = list(rate_target_names)
        self.efficiency_target_names = list(efficiency_target_names)
        low = torch.tensor(efficiency_lows, dtype=torch.float32)
        high = torch.tensor(efficiency_highs, dtype=torch.float32)
        if torch.any(high <= low):
            raise ValueError("Each efficiency_high must be > efficiency_low")
        self.register_buffer("eff_lows", low, persistent=True)
        self.register_buffer("eff_highs", high, persistent=True)

        self.minutes_model = SetTransformerMinutesModel(
            num_features=num_features,
            use_prior_head=bool(use_prior_head),
            prior_weight_floor=float(prior_weight_floor),
            use_team_embeddings=bool(use_team_embeddings),
            num_teams=num_teams,
            team_embedding_dim=int(team_embedding_dim),
            use_player_embeddings=bool(use_player_embeddings),
            num_players=num_players,
            player_embedding_dim=int(player_embedding_dim),
            use_player_team_embeddings=bool(use_player_team_embeddings),
            player_team_hash_buckets=int(player_team_hash_buckets),
            player_team_embedding_dim=int(player_team_embedding_dim),
            embed_dim=int(embed_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            num_transformer_layers=int(num_transformer_layers),
            num_attention_heads=int(num_attention_heads),
            use_gate_head=bool(use_gate_head),
            num_experts=int(num_experts),
            router_feature_dim=int(router_feature_dim),
            router_temperature=float(router_temperature),
            router_mode=str(router_mode),
            router_gumbel=bool(router_gumbel),
            router_gumbel_tau=float(router_gumbel_tau),
            alloc_activation=str(alloc_activation),
            entmax_alpha=float(entmax_alpha),
            share_temperature=float(share_temperature),
            total_minutes=float(total_minutes),
            eps=float(eps),
        )
        self.rates_trunk = MLP(
            in_dim=int(embed_dim),
            out_dim=int(rates_hidden_dim),
            hidden_dim=int(rates_hidden_dim),
            num_layers=2,
            dropout=float(dropout),
        )
        self.rate_head = nn.Linear(int(rates_hidden_dim), len(self.rate_target_names))
        self.eff_head = nn.Linear(int(rates_hidden_dim), len(self.efficiency_target_names))

    def _efficiency_from_raw(self, eff_raw: torch.Tensor) -> torch.Tensor:
        lows = self.eff_lows.to(dtype=eff_raw.dtype, device=eff_raw.device)
        highs = self.eff_highs.to(dtype=eff_raw.dtype, device=eff_raw.device)
        return torch.sigmoid(eff_raw) * (highs - lows) + lows

    def forward(
        self,
        x: torch.Tensor,
        team_idx: torch.Tensor,
        opp_idx: torch.Tensor,
        mask: torch.Tensor,
        *,
        player_idx: torch.Tensor | None = None,
        player_team_idx: torch.Tensor | None = None,
        alloc_mask: torch.Tensor | None = None,
        prior_weights: torch.Tensor | None = None,
        router_features: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Return minutes, auxiliary minutes logits, and rates/efficiency preds."""

        h = self.minutes_model.forward_embeddings(
            x,
            team_idx,
            opp_idx,
            mask,
            player_idx=player_idx,
            player_team_idx=player_team_idx,
        )
        final_mask = alloc_mask if alloc_mask is not None else mask
        base = prior_weights if self.minutes_model.use_prior_head else None

        gate_logits: torch.Tensor | None = None
        share_logits: torch.Tensor
        router_pi_soft: torch.Tensor | None = None
        router_pi_used: torch.Tensor | None = None
        if self.minutes_model.use_gate_head:
            if self.minutes_model.num_experts > 1:
                minutes, gate_logits, share_logits, router_pi_soft, router_pi_used = self.minutes_model.forward_with_aux(
                    x,
                    team_idx,
                    opp_idx,
                    mask,
                    player_idx=player_idx,
                    player_team_idx=player_team_idx,
                    alloc_mask=alloc_mask,
                    prior_weights=prior_weights,
                    router_features=router_features,
                )
            else:
                if self.minutes_model.gate_head is None or self.minutes_model.share_head is None:
                    raise RuntimeError("minutes_model gate/share heads missing")
                gate_logits = self.minutes_model.gate_head(h).squeeze(-1)
                share_logits = self.minutes_model.share_head(h).squeeze(-1)
                minutes = minutes_from_gate_and_share_logits(
                    gate_logits,
                    share_logits,
                    final_mask,
                    total_minutes=self.minutes_model.total_minutes,
                    eps=self.minutes_model.eps,
                    base_weights=base,
                    base_floor=self.minutes_model.prior_weight_floor,
                    alloc_activation=self.minutes_model.alloc_activation,
                    entmax_alpha=self.minutes_model.entmax_alpha,
                    share_temperature=self.minutes_model.share_temperature,
                )
        else:
            if self.minutes_model.head is None:
                raise RuntimeError("minutes_model head missing")
            share_logits = self.minutes_model.head(h).squeeze(-1)
            minutes = minutes_from_logits(
                share_logits,
                final_mask,
                total_minutes=self.minutes_model.total_minutes,
                eps=self.minutes_model.eps,
                base_weights=base,
                base_floor=self.minutes_model.prior_weight_floor,
                alloc_activation=self.minutes_model.alloc_activation,
                entmax_alpha=self.minutes_model.entmax_alpha,
                temperature=self.minutes_model.share_temperature,
            )

        rates_h = self.rates_trunk(h)
        pred_rates = self.rate_head(rates_h)
        pred_eff_raw = self.eff_head(rates_h)
        pred_eff = self._efficiency_from_raw(pred_eff_raw)

        mask_f = mask.unsqueeze(-1).to(dtype=pred_rates.dtype)
        pred_rates = pred_rates * mask_f
        pred_eff = pred_eff * mask_f
        return minutes, gate_logits, share_logits, pred_rates, pred_eff, router_pi_soft, router_pi_used


def build_joint_model(config: JointRotationRatesModelConfig) -> JointSetTransformerRatesModel:
    num_features = len(config.feature_columns)
    use_team_embeddings = bool(config.use_team_embeddings)
    num_teams = (len(config.team_id_vocab) + 1) if (use_team_embeddings and config.team_id_vocab) else None
    use_player_embeddings = bool(config.use_player_embeddings)
    num_players = (len(config.player_id_vocab) + 1) if (use_player_embeddings and config.player_id_vocab) else None
    return JointSetTransformerRatesModel(
        num_features=num_features,
        rate_target_names=list(config.rate_target_names or []),
        efficiency_target_names=list(config.efficiency_target_names or []),
        efficiency_lows=list(config.efficiency_lows or []),
        efficiency_highs=list(config.efficiency_highs or []),
        use_prior_head=bool(config.use_prior_head),
        prior_weight_floor=float(config.prior_weight_floor),
        use_team_embeddings=bool(config.use_team_embeddings),
        num_teams=num_teams,
        team_embedding_dim=int(config.team_embedding_dim),
        use_player_embeddings=bool(config.use_player_embeddings),
        num_players=num_players,
        player_embedding_dim=int(config.player_embedding_dim),
        use_player_team_embeddings=bool(config.use_player_team_embeddings),
        player_team_hash_buckets=int(config.player_team_hash_buckets),
        player_team_embedding_dim=int(config.player_team_embedding_dim),
        embed_dim=int(config.embed_dim),
        hidden_dim=int(config.hidden_dim),
        dropout=float(config.dropout),
        num_transformer_layers=int(config.num_transformer_layers),
        num_attention_heads=int(config.num_attention_heads),
        use_gate_head=bool(config.use_gate_head),
        num_experts=int(config.num_experts),
        router_feature_dim=int(config.router_feature_dim),
        router_temperature=float(config.router_temperature),
        router_mode=str(config.router_mode),
        router_gumbel=bool(config.router_gumbel),
        router_gumbel_tau=float(config.router_gumbel_tau),
        alloc_activation=str(config.alloc_activation),
        entmax_alpha=float(config.entmax_alpha),
        share_temperature=float(config.share_temperature),
        total_minutes=float(config.total_minutes),
        eps=float(config.eps),
        rates_hidden_dim=int(config.rates_hidden_dim),
    )


__all__ = [
    "JointRotationRatesModelConfig",
    "JointSetTransformerRatesModel",
    "build_joint_model",
]
