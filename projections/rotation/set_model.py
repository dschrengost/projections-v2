"""Permutation-invariant TEAM-SET minutes model (rotation_set_minutes v1).

This module provides:
- DeepSets and SetTransformer variants that operate on a variable-size set of players per (game_id, team_id)
- A minutes normalization head that enforces per-team totals of 240 minutes
- A minimal `predict_minutes` helper that accepts a flat dataframe and returns aligned per-row predictions
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

# Re-export utility functions for backward compatibility
from projections.rotation.utils import (
    GAME_ID_NORM_COL,
    KEY_COLS,
    OPPONENT_TEAM_ID_COL,
    _coerce_int_series,
    normalize_key_columns,
    zfill_game_id_series,
)

MODEL_DIR_ENV = "ROTATION_SET_MODEL_DIR"
MODEL_WEIGHTS_FILENAME = "model.pt"
MODEL_CONFIG_FILENAME = "config.json"


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean across dim=1 respecting a (B, N) boolean mask."""

    if x.ndim != 3 or mask.ndim != 2:
        raise ValueError("masked_mean expects x=(B,N,D) and mask=(B,N)")
    mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
    summed = (x * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / denom


def minutes_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    total_minutes: float = 240.0,
    eps: float = 1e-6,
    base_weights: torch.Tensor | None = None,
    base_floor: float = 0.0,
    alloc_activation: str = "softplus",
    entmax_alpha: float = 1.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert per-player logits to minutes that sum to `total_minutes` per team-game."""

    if logits.ndim != 2 or mask.ndim != 2:
        raise ValueError("minutes_from_logits expects logits=(B,N) and mask=(B,N)")
    act = _normalize_alloc_activation(alloc_activation)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if act == "entmax" and not (1.0 < float(entmax_alpha) <= 2.0):
        raise ValueError("entmax_alpha must be in (1, 2]")
    if act not in {"softplus", "softmax", "sparsemax", "entmax"}:
        raise ValueError(
            "alloc_activation must be one of: softplus, softmax, sparsemax, entmax "
            f"(got {alloc_activation!r})"
        )

    if act == "softplus":
        positive = F.softplus(logits / float(temperature))
        positive = positive * mask.to(dtype=positive.dtype)

        if base_weights is not None:
            if base_weights.shape != logits.shape:
                raise ValueError("base_weights must have the same shape as logits")
            if base_floor < 0:
                raise ValueError("base_floor must be >= 0")
            base = base_weights.to(dtype=positive.dtype).clamp(min=0)
            if base_floor:
                base = base + float(base_floor)
            base = base * mask.to(dtype=positive.dtype)
            positive = positive * base

        denom = positive.sum(dim=1, keepdim=True).clamp(min=eps)
        minutes = positive / denom * float(total_minutes)
        return minutes * mask.to(dtype=minutes.dtype)

    mask_bool = mask.to(dtype=torch.bool)
    if base_floor < 0:
        raise ValueError("base_floor must be >= 0")

    scaled = logits / float(temperature)
    if act == "softmax":
        shares = _masked_softmax(scaled, mask_bool)
    elif act == "sparsemax":
        shares = _sparsemax(scaled, mask_bool, eps=float(eps))
    else:
        shares = _entmax(scaled, mask_bool, alpha=float(entmax_alpha), eps=float(eps))

    if base_weights is not None:
        if base_weights.shape != logits.shape:
            raise ValueError("base_weights must have the same shape as logits")
        base = base_weights.to(dtype=shares.dtype).clamp(min=0)
        if base_floor:
            base = base + float(base_floor)
        base = base * mask.to(dtype=shares.dtype)
        weighted = shares * base
        denom = weighted.sum(dim=1, keepdim=True)
        shares = torch.where(denom > float(eps), weighted / denom.clamp(min=float(eps)), shares)

    minutes = shares * float(total_minutes)
    return minutes * mask.to(dtype=minutes.dtype)


def minutes_from_gate_and_share_logits(
    gate_logits: torch.Tensor,
    share_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    total_minutes: float = 240.0,
    eps: float = 1e-6,
    base_weights: torch.Tensor | None = None,
    base_floor: float = 0.0,
    alloc_activation: str = "entmax",
    entmax_alpha: float = 1.5,
    share_temperature: float = 1.0,
) -> torch.Tensor:
    """Convert (gate, share) logits to minutes that sum to `total_minutes` per team-game.

    Mapping (mask-aware):
      g_i = sigmoid(gate_logits_i)
      p_i = alloc_activation(share_logits_i / T)  (simplex over eligible players)
      w_i = g_i * p_i
      minutes_i = total_minutes * w_i / sum_j w_j
    """

    if gate_logits.shape != share_logits.shape:
        raise ValueError("gate_logits and share_logits must have the same shape")
    if gate_logits.ndim != 2 or mask.ndim != 2:
        raise ValueError("minutes_from_gate_and_share_logits expects logits=(B,N) and mask=(B,N)")
    if share_temperature <= 0:
        raise ValueError("share_temperature must be > 0")
    if base_floor < 0:
        raise ValueError("base_floor must be >= 0")

    act = _normalize_alloc_activation(alloc_activation)
    mask_bool = mask.to(dtype=torch.bool)
    n_valid = mask_bool.sum(dim=1, keepdim=True)
    if (n_valid == 0).all():
        return gate_logits.new_zeros(gate_logits.shape)

    g = torch.sigmoid(gate_logits)
    g = g * mask.to(dtype=g.dtype)

    scaled = share_logits / float(share_temperature)
    if act == "softmax":
        p = _masked_softmax(scaled, mask_bool)
    elif act == "sparsemax":
        p = _sparsemax(scaled, mask_bool, eps=float(eps))
    elif act == "softplus":
        # Useful for ablations: non-sparse positive weights normalized to a simplex.
        positive = F.softplus(scaled) * mask.to(dtype=scaled.dtype)
        denom = positive.sum(dim=1, keepdim=True).clamp(min=float(eps))
        p = positive / denom
    else:
        if not (1.0 < float(entmax_alpha) <= 2.0):
            raise ValueError("entmax_alpha must be in (1, 2]")
        p = _entmax(scaled, mask_bool, alpha=float(entmax_alpha), eps=float(eps))

    w = (g * p).to(dtype=torch.float32)
    if base_weights is not None:
        if base_weights.shape != share_logits.shape:
            raise ValueError("base_weights must have the same shape as logits")
        base = base_weights.to(dtype=w.dtype).clamp(min=0)
        if base_floor:
            base = base + float(base_floor)
        base = base * mask.to(dtype=w.dtype)
        w = w * base

    w_sum = w.sum(dim=1, keepdim=True)
    # If the gate collapses everything (numerical underflow), fall back to p-only.
    w = torch.where(w_sum > float(eps), w, p.to(dtype=w.dtype))
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=float(eps))

    minutes = w / w_sum * float(total_minutes)
    minutes = minutes * mask.to(dtype=minutes.dtype)
    minutes = torch.where(n_valid > 0, minutes, minutes.new_zeros((1, 1)).expand_as(minutes))
    return minutes.to(dtype=share_logits.dtype)


def minutes_from_moe_gate_and_share_logits(
    gate_logits: torch.Tensor,
    share_logits: torch.Tensor,
    router_pi: torch.Tensor,
    mask: torch.Tensor,
    *,
    total_minutes: float = 240.0,
    eps: float = 1e-6,
    base_weights: torch.Tensor | None = None,
    base_floor: float = 0.0,
    alloc_activation: str = "entmax",
    entmax_alpha: float = 1.5,
    share_temperature: float = 1.0,
) -> torch.Tensor:
    """Convert MoE (gate, share) logits + router weights to minutes summing to `total_minutes`.

    Shapes:
      gate_logits: (B, E, N)
      share_logits: (B, E, N)
      router_pi: (B, E)
      mask: (B, N)
    """

    if gate_logits.shape != share_logits.shape:
        raise ValueError("gate_logits and share_logits must have the same shape")
    if gate_logits.ndim != 3:
        raise ValueError("minutes_from_moe_gate_and_share_logits expects gate_logits=(B,E,N)")
    if share_temperature <= 0:
        raise ValueError("share_temperature must be > 0")
    if base_floor < 0:
        raise ValueError("base_floor must be >= 0")

    B, E, N = gate_logits.shape
    if router_pi.shape != (B, E):
        raise ValueError("router_pi must have shape (B,E)")
    if mask.shape != (B, N):
        raise ValueError("mask must have shape (B,N)")

    act = _normalize_alloc_activation(alloc_activation)
    mask_bool = mask.to(dtype=torch.bool)
    n_valid = mask_bool.sum(dim=1, keepdim=True)
    if (n_valid == 0).all():
        return gate_logits.new_zeros((B, N))

    # Normalize router weights in case of numerical drift.
    pi = router_pi.to(dtype=torch.float32).clamp(min=0.0)
    pi = pi / pi.sum(dim=1, keepdim=True).clamp(min=float(eps))

    # Flatten expert dimension for efficient masked simplex activation.
    scaled = (share_logits / float(share_temperature)).reshape(B * E, N)
    mask_flat = mask_bool.unsqueeze(1).expand(B, E, N).reshape(B * E, N)

    if act == "softmax":
        p_flat = _masked_softmax(scaled, mask_flat)
    elif act == "sparsemax":
        p_flat = _sparsemax(scaled, mask_flat, eps=float(eps))
    elif act == "softplus":
        positive = F.softplus(scaled) * mask_flat.to(dtype=scaled.dtype)
        denom = positive.sum(dim=1, keepdim=True).clamp(min=float(eps))
        p_flat = positive / denom
    else:
        if not (1.0 < float(entmax_alpha) <= 2.0):
            raise ValueError("entmax_alpha must be in (1, 2]")
        p_flat = _entmax(scaled, mask_flat, alpha=float(entmax_alpha), eps=float(eps))

    p = p_flat.reshape(B, E, N).to(dtype=torch.float32)
    g = torch.sigmoid(gate_logits.to(dtype=torch.float32)) * mask_bool.unsqueeze(1).to(dtype=torch.float32)
    w = g * p

    # If an expert's gate collapses, fall back to p-only for that expert.
    w_sum = w.sum(dim=2, keepdim=True)
    w = torch.where(w_sum > float(eps), w, p)

    w_mix = (pi.unsqueeze(-1) * w).sum(dim=1)
    p_mix = (pi.unsqueeze(-1) * p).sum(dim=1)

    if base_weights is not None:
        if base_weights.shape != (B, N):
            raise ValueError("base_weights must have shape (B,N)")
        base = base_weights.to(dtype=w_mix.dtype).clamp(min=0)
        if base_floor:
            base = base + float(base_floor)
        base = base * mask.to(dtype=w_mix.dtype)
        w_mix = w_mix * base

    w_sum = w_mix.sum(dim=1, keepdim=True)
    w_mix = torch.where(w_sum > float(eps), w_mix, p_mix.to(dtype=w_mix.dtype))
    w_sum = w_mix.sum(dim=1, keepdim=True).clamp(min=float(eps))

    minutes = w_mix / w_sum * float(total_minutes)
    minutes = minutes * mask.to(dtype=minutes.dtype)
    minutes = torch.where(n_valid > 0, minutes, minutes.new_zeros((1, 1)).expand_as(minutes))
    return minutes.to(dtype=share_logits.dtype)


def _normalize_alloc_activation(value: str) -> str:
    act = str(value).strip().lower()
    if act == "entmax15":
        return "entmax"
    return act


def _normalize_router_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"soft", "top1_st", "top2_st"}:
        return mode
    raise ValueError(f"router_mode must be one of: soft, top1_st, top2_st (got {value!r})")


def _sample_gumbel(shape: torch.Size, *, device: torch.device, eps: float) -> torch.Tensor:
    u = torch.rand(shape, device=device)
    u = u.clamp(min=float(eps), max=1.0 - float(eps))
    return -torch.log(-torch.log(u))


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or mask.ndim != 2:
        raise ValueError("_masked_softmax expects logits=(B,N) and mask=(B,N)")
    if logits.shape != mask.shape:
        raise ValueError("_masked_softmax expects logits and mask to have the same shape")

    mask_bool = mask.to(dtype=torch.bool)
    n_valid = mask_bool.sum(dim=1, keepdim=True)
    if (n_valid == 0).all():
        return logits.new_zeros(logits.shape)

    x = logits.to(dtype=torch.float32)
    x = x.masked_fill(~mask_bool, -1e9)
    x = x - x.max(dim=1, keepdim=True).values
    p = torch.softmax(x, dim=1)
    p = p * mask_bool.to(dtype=p.dtype)
    p_sum = p.sum(dim=1, keepdim=True)
    p = torch.where(p_sum > 0, p / p_sum, p)
    p = torch.where(n_valid > 0, p, p.new_zeros((1, 1)).expand_as(p))
    return p.to(dtype=logits.dtype)


def _sparsemax(logits: torch.Tensor, mask: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Sparsemax activation (Martins & Astudillo 2016), mask-aware."""

    if logits.ndim != 2 or mask.ndim != 2:
        raise ValueError("_sparsemax expects logits=(B,N) and mask=(B,N)")
    if logits.shape != mask.shape:
        raise ValueError("_sparsemax expects logits and mask to have the same shape")

    B, N = logits.shape
    if N == 0:
        return logits.new_zeros((B, N))

    mask_bool = mask.to(dtype=torch.bool)
    n_valid = mask_bool.sum(dim=1, keepdim=True)
    if (n_valid == 0).all():
        return logits.new_zeros((B, N))

    x = logits.to(dtype=torch.float32)
    x = x.masked_fill(~mask_bool, -1e9)
    x = x - x.max(dim=1, keepdim=True).values

    x_sorted, _ = torch.sort(x, dim=1, descending=True)
    k = torch.arange(1, N + 1, device=logits.device, dtype=torch.float32).view(1, -1)
    x_cumsum = x_sorted.cumsum(dim=1)
    tau = (x_cumsum - 1.0) / k
    support = x_sorted > tau
    support = support & (k <= n_valid.to(dtype=torch.float32))
    k_z = support.sum(dim=1, keepdim=True).clamp(min=1)
    tau_star = tau.gather(1, (k_z - 1).to(dtype=torch.int64))

    p = torch.relu(x - tau_star)
    p = p * mask_bool.to(dtype=p.dtype)
    p_sum = p.sum(dim=1, keepdim=True)
    p = torch.where(p_sum > 0, p / p_sum.clamp(min=eps), p)
    p = torch.where(n_valid > 0, p, p.new_zeros((1, 1)).expand_as(p))
    return p.to(dtype=logits.dtype)


def _entmax(logits: torch.Tensor, mask: torch.Tensor, *, alpha: float, eps: float) -> torch.Tensor:
    """entmax with general alpha in (1,2], mask-aware."""

    if not (1.0 < float(alpha) <= 2.0):
        raise ValueError("alpha must be in (1, 2]")
    if abs(float(alpha) - 1.5) < 1e-8:
        return _entmax15(logits, mask, eps=eps)
    if abs(float(alpha) - 2.0) < 1e-8:
        return _sparsemax(logits, mask, eps=eps)
    return _entmax_bisect(logits, mask, alpha=float(alpha), eps=eps)


def _entmax_bisect(logits: torch.Tensor, mask: torch.Tensor, *, alpha: float, eps: float, iters: int = 50) -> torch.Tensor:
    """Bisection-based entmax projection for alpha in (1,2), mask-aware."""

    if logits.ndim != 2 or mask.ndim != 2:
        raise ValueError("_entmax_bisect expects logits=(B,N) and mask=(B,N)")
    if logits.shape != mask.shape:
        raise ValueError("_entmax_bisect expects logits and mask to have the same shape")

    B, N = logits.shape
    if N == 0:
        return logits.new_zeros((B, N))

    mask_bool = mask.to(dtype=torch.bool)
    n_valid = mask_bool.sum(dim=1, keepdim=True)
    if (n_valid == 0).all():
        return logits.new_zeros((B, N))

    x = logits.to(dtype=torch.float32)
    x = x.masked_fill(~mask_bool, -1e9)
    x = x - x.max(dim=1, keepdim=True).values

    # Lower bound: push tau low enough that sum(p) > 1 even in near-uniform cases.
    x_valid_min = x.masked_fill(~mask_bool, float("inf")).min(dim=1, keepdim=True).values
    c = float(alpha) - 1.0
    beta = 1.0 / c
    n_valid_f = n_valid.to(dtype=torch.float32).clamp(min=1.0)
    tau_lo = x_valid_min - (n_valid_f ** c) / c
    tau_hi = torch.zeros_like(tau_lo)

    lo = tau_lo
    hi = tau_hi
    for _ in range(int(iters)):
        mid = (lo + hi) / 2.0
        p_mid = torch.relu(c * (x - mid)) ** beta
        p_mid = p_mid * mask_bool.to(dtype=p_mid.dtype)
        s = p_mid.sum(dim=1, keepdim=True)
        cond = s > 1.0
        lo = torch.where(cond, mid, lo)
        hi = torch.where(cond, hi, mid)

    tau = hi
    p = torch.relu(c * (x - tau)) ** beta
    p = p * mask_bool.to(dtype=p.dtype)
    p_sum = p.sum(dim=1, keepdim=True)
    p = torch.where(p_sum > 0, p / p_sum.clamp(min=eps), p)
    p = torch.where(n_valid > 0, p, p.new_zeros((1, 1)).expand_as(p))
    return p.to(dtype=logits.dtype)


def _entmax15(logits: torch.Tensor, mask: torch.Tensor, *, eps: float) -> torch.Tensor:
    """entmax15 activation (alpha=1.5), mask-aware."""

    if logits.ndim != 2 or mask.ndim != 2:
        raise ValueError("_entmax15 expects logits=(B,N) and mask=(B,N)")
    if logits.shape != mask.shape:
        raise ValueError("_entmax15 expects logits and mask to have the same shape")

    B, N = logits.shape
    if N == 0:
        return logits.new_zeros((B, N))

    mask_bool = mask.to(dtype=torch.bool)
    n_valid = mask_bool.sum(dim=1, keepdim=True)
    if (n_valid == 0).all():
        return logits.new_zeros((B, N))

    x = logits.to(dtype=torch.float32)
    x = x.masked_fill(~mask_bool, -1e9)
    x = x - x.max(dim=1, keepdim=True).values
    # entmax15 (alpha=1.5) corresponds to squared sparsemax on logits / 2.
    x = x / 2.0

    x_sorted, _ = torch.sort(x, dim=1, descending=True)
    x_cumsum = x_sorted.cumsum(dim=1)
    x2_cumsum = (x_sorted * x_sorted).cumsum(dim=1)
    k = torch.arange(1, N + 1, device=logits.device, dtype=torch.float32).view(1, -1)

    disc = x_cumsum * x_cumsum - k * (x2_cumsum - 1.0)
    disc = torch.clamp(disc, min=0.0)
    tau = (x_cumsum - torch.sqrt(disc)) / k

    support = x_sorted > tau
    support = support & (k <= n_valid.to(dtype=torch.float32))
    k_z = support.sum(dim=1, keepdim=True).clamp(min=1)
    tau_star = tau.gather(1, (k_z - 1).to(dtype=torch.int64))

    p = torch.relu(x - tau_star) ** 2
    p = p * mask_bool.to(dtype=p.dtype)
    p_sum = p.sum(dim=1, keepdim=True)
    p = torch.where(p_sum > 0, p / p_sum.clamp(min=eps), p)
    p = torch.where(n_valid > 0, p, p.new_zeros((1, 1)).expand_as(p))
    return p.to(dtype=logits.dtype)


ModelType = Literal["deepsets", "settransformer"]


@dataclass(frozen=True)
class RotationSetModelConfig:
    model: ModelType
    feature_columns: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    use_prior_head: bool = False
    prior_weight_col: str = "minutes_from_stints_prior_20"
    prior_weight_floor: float = 0.0
    use_team_embeddings: bool = False
    team_id_vocab: list[int] | None = None
    team_embedding_dim: int = 8
    embed_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.1
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    use_gate_head: bool = False
    num_experts: int = 1
    router_feature_dim: int = 0
    router_temperature: float = 1.0
    router_mode: str = "soft"
    router_gumbel: bool = False
    router_gumbel_tau: float = 1.0
    alloc_activation: str = "softplus"
    entmax_alpha: float = 1.5
    share_temperature: float = 1.0
    lambda_rot: float = 0.0
    lambda_fp: float = 0.0
    lambda_router: float = 0.0
    lambda_router_load_balance: float = 0.0
    lambda_router_entropy: float = 0.0
    router_reg_type: str = "load_balance"
    inj_weight_scale: float = 0.0
    rot_target_mode: str = "hard"
    rot_target_threshold: float = 8.0
    rot_target_slope: float = 2.0
    fp_minutes_threshold: float = 6.0
    rot_loss_type: str = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    total_minutes: float = 240.0
    eps: float = 1e-6
    version: str = "rotation_set_minutes_v1"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RotationSetModelConfig":
        data = dict(payload)
        # Backward compatibility: historical configs used rotation_set_alloc_activation.
        if "alloc_activation" not in data and "rotation_set_alloc_activation" in data:
            data["alloc_activation"] = data.pop("rotation_set_alloc_activation")
        data.pop("rotation_set_alloc_activation", None)

        alloc = _normalize_alloc_activation(data.get("alloc_activation", "softplus"))
        if alloc == "entmax15":
            alloc = "entmax"
        data["alloc_activation"] = alloc

        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RotationSetModelConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TeamOpponentEmbedding(nn.Module):
    def __init__(self, *, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.team_emb = nn.Embedding(num_embeddings, embedding_dim)
        self.opp_emb = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, team_idx: torch.Tensor, opp_idx: torch.Tensor) -> torch.Tensor:
        team_e = self.team_emb(team_idx)
        opp_e = self.opp_emb(opp_idx)
        return torch.cat([team_e, opp_e], dim=-1)


class DeepSetsMinutesModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        *,
        use_prior_head: bool = False,
        prior_weight_floor: float = 0.0,
        use_team_embeddings: bool = False,
        num_teams: int | None = None,
        team_embedding_dim: int = 8,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_gate_head: bool = False,
        alloc_activation: str = "softplus",
        entmax_alpha: float = 1.5,
        share_temperature: float = 1.0,
        total_minutes: float = 240.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.total_minutes = float(total_minutes)
        self.eps = float(eps)
        self.use_gate_head = bool(use_gate_head)
        self.alloc_activation = _normalize_alloc_activation(alloc_activation)
        self.entmax_alpha = float(entmax_alpha)
        self.share_temperature = float(share_temperature)
        self.use_prior_head = bool(use_prior_head)
        self.prior_weight_floor = float(prior_weight_floor)
        self.use_team_embeddings = bool(use_team_embeddings)
        if self.use_team_embeddings:
            if not num_teams or num_teams <= 1:
                raise ValueError("num_teams must be provided when use_team_embeddings=True")
            self.embeddings = TeamOpponentEmbedding(num_embeddings=num_teams, embedding_dim=team_embedding_dim)
            in_dim = num_features + 2 * team_embedding_dim
        else:
            self.embeddings = None
            in_dim = num_features

        self.phi = MLP(in_dim, embed_dim, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
        if self.use_gate_head:
            self.gate_head = MLP(embed_dim * 2, 1, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
            self.share_head = MLP(embed_dim * 2, 1, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
            self.rho = None
        else:
            self.rho = MLP(embed_dim * 2, 1, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
            self.gate_head = None
            self.share_head = None

    def _build_inputs(self, x: torch.Tensor, team_idx: torch.Tensor, opp_idx: torch.Tensor) -> torch.Tensor:
        if not self.use_team_embeddings:
            return x
        if self.embeddings is None:
            raise RuntimeError("embeddings module missing")
        emb = self.embeddings(team_idx, opp_idx)
        return torch.cat([x, emb], dim=-1)

    def _pooled_repr(
        self, x: torch.Tensor, team_idx: torch.Tensor, opp_idx: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        inputs = self._build_inputs(x, team_idx, opp_idx)
        h = self.phi(inputs)
        pooled = masked_mean(h, mask)
        pooled_rep = pooled.unsqueeze(1).expand(-1, h.shape[1], -1)
        return torch.cat([h, pooled_rep], dim=-1)

    def forward_with_aux(
        self,
        x: torch.Tensor,
        team_idx: torch.Tensor,
        opp_idx: torch.Tensor,
        mask: torch.Tensor,
        *,
        alloc_mask: torch.Tensor | None = None,
        prior_weights: torch.Tensor | None = None,
        router_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        pooled = self._pooled_repr(x, team_idx, opp_idx, mask)
        final_mask = alloc_mask if alloc_mask is not None else mask
        base = prior_weights if self.use_prior_head else None

        if not self.use_gate_head:
            if self.rho is None:
                raise RuntimeError("rho head missing")
            share_logits = self.rho(pooled).squeeze(-1)
            minutes = minutes_from_logits(
                share_logits,
                final_mask,
                total_minutes=self.total_minutes,
                eps=self.eps,
                base_weights=base,
                base_floor=self.prior_weight_floor,
                alloc_activation=self.alloc_activation,
                entmax_alpha=self.entmax_alpha,
                temperature=self.share_temperature,
            )
            return minutes, None, share_logits, None, None

        if self.gate_head is None or self.share_head is None:
            raise RuntimeError("gate/share heads missing")
        gate_logits = self.gate_head(pooled).squeeze(-1)
        share_logits = self.share_head(pooled).squeeze(-1)
        minutes = minutes_from_gate_and_share_logits(
            gate_logits,
            share_logits,
            final_mask,
            total_minutes=self.total_minutes,
            eps=self.eps,
            base_weights=base,
            base_floor=self.prior_weight_floor,
            alloc_activation=self.alloc_activation,
            entmax_alpha=self.entmax_alpha,
            share_temperature=self.share_temperature,
        )
        return minutes, gate_logits, share_logits, None, None

    def forward(
        self,
        x: torch.Tensor,
        team_idx: torch.Tensor,
        opp_idx: torch.Tensor,
        mask: torch.Tensor,
        alloc_mask: torch.Tensor | None = None,
        prior_weights: torch.Tensor | None = None,
        router_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        minutes, _, _, _, _ = self.forward_with_aux(
            x,
            team_idx,
            opp_idx,
            mask,
            alloc_mask=alloc_mask,
            prior_weights=prior_weights,
            router_features=router_features,
        )
        return minutes


class SetTransformerMinutesModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        *,
        use_prior_head: bool = False,
        prior_weight_floor: float = 0.0,
        use_team_embeddings: bool = False,
        num_teams: int | None = None,
        team_embedding_dim: int = 8,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        use_gate_head: bool = False,
        num_experts: int = 1,
        router_feature_dim: int = 0,
        router_temperature: float = 1.0,
        router_mode: str = "soft",
        router_gumbel: bool = False,
        router_gumbel_tau: float = 1.0,
        alloc_activation: str = "softplus",
        entmax_alpha: float = 1.5,
        share_temperature: float = 1.0,
        total_minutes: float = 240.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.total_minutes = float(total_minutes)
        self.eps = float(eps)
        self.use_gate_head = bool(use_gate_head)
        self.num_experts = int(num_experts)
        self.router_feature_dim = int(router_feature_dim)
        self.router_temperature = float(router_temperature)
        self.router_mode = _normalize_router_mode(router_mode)
        self.router_gumbel = bool(router_gumbel)
        self.router_gumbel_tau = float(router_gumbel_tau)
        self.alloc_activation = _normalize_alloc_activation(alloc_activation)
        self.entmax_alpha = float(entmax_alpha)
        self.share_temperature = float(share_temperature)
        self.use_prior_head = bool(use_prior_head)
        self.prior_weight_floor = float(prior_weight_floor)
        self.use_team_embeddings = bool(use_team_embeddings)
        if self.num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if self.num_experts > 1 and not self.use_gate_head:
            raise ValueError("num_experts > 1 requires use_gate_head=True")
        if self.router_feature_dim < 0:
            raise ValueError("router_feature_dim must be >= 0")
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be > 0")
        if self.router_gumbel_tau <= 0:
            raise ValueError("router_gumbel_tau must be > 0")
        if self.use_team_embeddings:
            if not num_teams or num_teams <= 1:
                raise ValueError("num_teams must be provided when use_team_embeddings=True")
            self.embeddings = TeamOpponentEmbedding(num_embeddings=num_teams, embedding_dim=team_embedding_dim)
            in_dim = num_features + 2 * team_embedding_dim
        else:
            self.embeddings = None
            in_dim = num_features

        self.input_proj = nn.Linear(in_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        if self.use_gate_head:
            if self.num_experts > 1:
                router_in_dim = embed_dim + self.router_feature_dim
                self.router = MLP(router_in_dim, self.num_experts, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
                self.expert_gate_heads = nn.ModuleList(
                    [
                        MLP(embed_dim, 1, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
                        for _ in range(self.num_experts)
                    ]
                )
                self.expert_share_heads = nn.ModuleList(
                    [
                        MLP(embed_dim, 1, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
                        for _ in range(self.num_experts)
                    ]
                )
                self.gate_head = None
                self.share_head = None
            else:
                self.router = None
                self.expert_gate_heads = None
                self.expert_share_heads = None
                self.gate_head = MLP(embed_dim, 1, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
                self.share_head = MLP(embed_dim, 1, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
            self.head = None
        else:
            self.head = MLP(embed_dim, 1, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
            self.gate_head = None
            self.share_head = None
            self.router = None
            self.expert_gate_heads = None
            self.expert_share_heads = None

    def _build_inputs(self, x: torch.Tensor, team_idx: torch.Tensor, opp_idx: torch.Tensor) -> torch.Tensor:
        if not self.use_team_embeddings:
            return x
        if self.embeddings is None:
            raise RuntimeError("embeddings module missing")
        emb = self.embeddings(team_idx, opp_idx)
        return torch.cat([x, emb], dim=-1)

    def forward_embeddings(self, x: torch.Tensor, team_idx: torch.Tensor, opp_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = self._build_inputs(x, team_idx, opp_idx)
        h = self.input_proj(inputs)
        padding_mask = ~mask
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return h

    def forward_with_aux(
        self,
        x: torch.Tensor,
        team_idx: torch.Tensor,
        opp_idx: torch.Tensor,
        mask: torch.Tensor,
        *,
        alloc_mask: torch.Tensor | None = None,
        prior_weights: torch.Tensor | None = None,
        router_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        h = self.forward_embeddings(x, team_idx, opp_idx, mask)
        final_mask = alloc_mask if alloc_mask is not None else mask
        base = prior_weights if self.use_prior_head else None

        if not self.use_gate_head:
            if self.head is None:
                raise RuntimeError("head missing")
            share_logits = self.head(h).squeeze(-1)
            minutes = minutes_from_logits(
                share_logits,
                final_mask,
                total_minutes=self.total_minutes,
                eps=self.eps,
                base_weights=base,
                base_floor=self.prior_weight_floor,
                alloc_activation=self.alloc_activation,
                entmax_alpha=self.entmax_alpha,
                temperature=self.share_temperature,
            )
            return minutes, None, share_logits, None, None

        if self.num_experts <= 1:
            if self.gate_head is None or self.share_head is None:
                raise RuntimeError("gate/share heads missing")
            gate_logits = self.gate_head(h).squeeze(-1)
            share_logits = self.share_head(h).squeeze(-1)
            minutes = minutes_from_gate_and_share_logits(
                gate_logits,
                share_logits,
                final_mask,
                total_minutes=self.total_minutes,
                eps=self.eps,
                base_weights=base,
                base_floor=self.prior_weight_floor,
                alloc_activation=self.alloc_activation,
                entmax_alpha=self.entmax_alpha,
                share_temperature=self.share_temperature,
            )
            return minutes, gate_logits, share_logits, None, None

        if self.router is None or self.expert_gate_heads is None or self.expert_share_heads is None:
            raise RuntimeError("moe heads missing")
        pooled = masked_mean(h, mask)
        if self.router_feature_dim > 0:
            if router_features is None:
                router_features = pooled.new_zeros((pooled.shape[0], self.router_feature_dim))
            elif router_features.shape != (pooled.shape[0], self.router_feature_dim):
                raise ValueError("router_features must have shape (B, router_feature_dim)")
            router_in = torch.cat([pooled, router_features.to(dtype=pooled.dtype)], dim=1)
        else:
            router_in = pooled
        router_logits = self.router(router_in)
        router_pi_soft = torch.softmax(router_logits / float(self.router_temperature), dim=1)
        router_pi_used = router_pi_soft
        if self.router_mode != "soft":
            router_logits_sel = router_logits
            if self.training and self.router_gumbel:
                noise = _sample_gumbel(router_logits.shape, device=router_logits.device, eps=self.eps).to(
                    dtype=router_logits.dtype
                )
                router_logits_sel = router_logits_sel + noise * float(self.router_gumbel_tau)
            if self.router_mode == "top1_st":
                chosen = router_logits_sel.argmax(dim=1)
                z_hard = F.one_hot(chosen, num_classes=self.num_experts).to(dtype=router_pi_soft.dtype)
            elif self.router_mode == "top2_st":
                k = 2 if self.num_experts >= 2 else 1
                topk = torch.topk(router_logits_sel, k=k, dim=1).indices
                topk_soft = router_pi_soft.gather(1, topk)
                denom = topk_soft.sum(dim=1, keepdim=True).clamp(min=float(self.eps))
                weights = topk_soft / denom
                z_hard = torch.zeros_like(router_pi_soft)
                z_hard.scatter_(1, topk, weights)
            else:
                raise RuntimeError(f"Unhandled router_mode={self.router_mode!r}")
            router_pi_used = z_hard - router_pi_soft.detach() + router_pi_soft

        gate_stack: list[torch.Tensor] = []
        share_stack: list[torch.Tensor] = []
        for gate_head, share_head in zip(self.expert_gate_heads, self.expert_share_heads, strict=True):
            gate_stack.append(gate_head(h).squeeze(-1))
            share_stack.append(share_head(h).squeeze(-1))
        gate_logits_e = torch.stack(gate_stack, dim=1)
        share_logits_e = torch.stack(share_stack, dim=1)

        minutes = minutes_from_moe_gate_and_share_logits(
            gate_logits_e,
            share_logits_e,
            router_pi_used,
            final_mask,
            total_minutes=self.total_minutes,
            eps=self.eps,
            base_weights=base,
            base_floor=self.prior_weight_floor,
            alloc_activation=self.alloc_activation,
            entmax_alpha=self.entmax_alpha,
            share_temperature=self.share_temperature,
        )

        # Mix gate probabilities for membership supervision (BCEWithLogits expects logits).
        gate_prob = torch.sigmoid(gate_logits_e.to(dtype=torch.float32))
        gate_prob = (router_pi_used.to(dtype=gate_prob.dtype).unsqueeze(-1) * gate_prob).sum(dim=1)
        gate_prob = gate_prob * final_mask.to(dtype=gate_prob.dtype)
        gate_prob = gate_prob.clamp(min=float(self.eps), max=1.0 - float(self.eps))
        gate_logits_mix = torch.log(gate_prob / (1.0 - gate_prob))
        gate_logits_mix = torch.where(final_mask.to(dtype=torch.bool), gate_logits_mix, gate_logits_mix.new_zeros(()))

        share_logits_out = share_logits_e[:, 0, :]
        return (
            minutes,
            gate_logits_mix.to(dtype=share_logits_out.dtype),
            share_logits_out,
            router_pi_soft.to(dtype=share_logits_out.dtype),
            router_pi_used.to(dtype=share_logits_out.dtype),
        )

    def forward(
        self,
        x: torch.Tensor,
        team_idx: torch.Tensor,
        opp_idx: torch.Tensor,
        mask: torch.Tensor,
        alloc_mask: torch.Tensor | None = None,
        prior_weights: torch.Tensor | None = None,
        router_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        minutes, _, _, _, _ = self.forward_with_aux(
            x,
            team_idx,
            opp_idx,
            mask,
            alloc_mask=alloc_mask,
            prior_weights=prior_weights,
            router_features=router_features,
        )
        return minutes


def build_model(config: RotationSetModelConfig) -> nn.Module:
    num_features = len(config.feature_columns)
    use_team_embeddings = bool(config.use_team_embeddings)
    num_teams = (len(config.team_id_vocab) + 1) if (use_team_embeddings and config.team_id_vocab) else None
    if config.model == "deepsets":
        return DeepSetsMinutesModel(
            num_features,
            use_prior_head=bool(config.use_prior_head),
            prior_weight_floor=float(config.prior_weight_floor),
            use_team_embeddings=use_team_embeddings,
            num_teams=num_teams,
            team_embedding_dim=config.team_embedding_dim,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            use_gate_head=bool(config.use_gate_head),
            alloc_activation=str(config.alloc_activation),
            entmax_alpha=float(config.entmax_alpha),
            share_temperature=float(config.share_temperature),
            total_minutes=config.total_minutes,
            eps=config.eps,
        )
    if config.model == "settransformer":
        return SetTransformerMinutesModel(
            num_features,
            use_prior_head=bool(config.use_prior_head),
            prior_weight_floor=float(config.prior_weight_floor),
            use_team_embeddings=use_team_embeddings,
            num_teams=num_teams,
            team_embedding_dim=config.team_embedding_dim,
            embed_dim=config.embed_dim,
            hidden_dim=max(config.hidden_dim, 2 * config.embed_dim),
            dropout=config.dropout,
            num_transformer_layers=config.num_transformer_layers,
            num_attention_heads=config.num_attention_heads,
            use_gate_head=bool(config.use_gate_head),
            num_experts=int(getattr(config, "num_experts", 1)),
            router_feature_dim=int(getattr(config, "router_feature_dim", 0)),
            router_temperature=float(getattr(config, "router_temperature", 1.0)),
            router_mode=str(getattr(config, "router_mode", "soft")),
            router_gumbel=bool(getattr(config, "router_gumbel", False)),
            router_gumbel_tau=float(getattr(config, "router_gumbel_tau", 1.0)),
            alloc_activation=str(config.alloc_activation),
            entmax_alpha=float(config.entmax_alpha),
            share_temperature=float(config.share_temperature),
            total_minutes=config.total_minutes,
            eps=config.eps,
        )
    raise ValueError(f"Unknown model type: {config.model}")


def _build_feature_matrix(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input df is missing required feature columns: {missing}")
    frame = df.loc[:, list(feature_columns)].copy()
    for col in feature_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.fillna(0.0)
    arr = frame.to_numpy(dtype="float32", copy=False)
    return (arr - feature_mean) / feature_std


@dataclass(frozen=True)
class TeamGameSet:
    game_id: str
    team_id: int
    row_indices: np.ndarray
    x: np.ndarray
    team_idx: np.ndarray
    opp_idx: np.ndarray


def _build_team_id_mapping(vocab: list[int] | None) -> dict[int, int]:
    if not vocab:
        return {}
    return {int(team_id): i + 1 for i, team_id in enumerate(vocab)}


def _map_ids_to_index(series: pd.Series, mapping: dict[int, int]) -> np.ndarray:
    if series.empty:
        return np.zeros(0, dtype=np.int64)
    coerced = pd.to_numeric(series, errors="coerce").astype("Int64")
    mapped = coerced.map(lambda x: mapping.get(int(x), 0) if pd.notna(x) else 0)
    return mapped.fillna(0).to_numpy(dtype=np.int64, copy=False)


class RotationSetMinutesPredictor:
    def __init__(self, *, config: RotationSetModelConfig, model: nn.Module, device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self._feature_mean = np.asarray(config.feature_mean, dtype="float32")
        self._feature_std = np.asarray(config.feature_std, dtype="float32")

        if self._feature_mean.shape != (len(config.feature_columns),):
            raise ValueError("feature_mean length does not match feature_columns")
        if self._feature_std.shape != (len(config.feature_columns),):
            raise ValueError("feature_std length does not match feature_columns")

        self._team_id_mapping = _build_team_id_mapping(config.team_id_vocab)

    @classmethod
    def load(cls, model_dir: Path, *, device: str = "cpu") -> "RotationSetMinutesPredictor":
        model_dir = Path(model_dir)
        config_path = model_dir / MODEL_CONFIG_FILENAME
        weights_path = model_dir / MODEL_WEIGHTS_FILENAME
        config = RotationSetModelConfig.load(config_path)
        model = build_model(config)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        return cls(config=config, model=model, device=device)

    def predict(self, df_features: pd.DataFrame, *, batch_size: int = 64) -> pd.DataFrame:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        df = normalize_key_columns(df_features)
        df = df.reset_index(drop=True)
        feature_matrix = _build_feature_matrix(
            df,
            feature_columns=self.config.feature_columns,
            feature_mean=self._feature_mean,
            feature_std=self._feature_std,
        )

        if self.config.use_team_embeddings and not self._team_id_mapping:
            raise ValueError("Model config requires team embeddings, but team_id_vocab is empty/missing")
        team_idx_all = (
            _map_ids_to_index(df["team_id"], self._team_id_mapping)
            if self.config.use_team_embeddings
            else np.zeros(len(df), dtype=np.int64)
        )
        if self.config.use_team_embeddings:
            if OPPONENT_TEAM_ID_COL in df.columns:
                opp_idx_all = _map_ids_to_index(df[OPPONENT_TEAM_ID_COL], self._team_id_mapping)
            else:
                opp_idx_all = np.zeros(len(df), dtype=np.int64)
        else:
            opp_idx_all = np.zeros(len(df), dtype=np.int64)

        groups: list[TeamGameSet] = []
        for (game_id, team_id), idx in df.groupby([GAME_ID_NORM_COL, "team_id"], sort=False).indices.items():
            idx_arr = np.asarray(idx, dtype=np.int64)
            groups.append(
                TeamGameSet(
                    game_id=str(game_id),
                    team_id=int(team_id),
                    row_indices=idx_arr,
                    x=feature_matrix[idx_arr],
                    team_idx=team_idx_all[idx_arr],
                    opp_idx=opp_idx_all[idx_arr],
                )
            )

        preds = np.zeros(len(df), dtype="float32")
        num_features = len(self.config.feature_columns)

        alloc_flag_all: np.ndarray | None = None
        is_out_all: np.ndarray | None = None
        if "is_out" in df.columns:
            is_out_all = pd.to_numeric(df["is_out"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
            alloc_flag_all = is_out_all == 0

        router_feat_dim = int(getattr(self.config, "router_feature_dim", 0))
        prior20_all: np.ndarray | None = None
        if router_feat_dim > 0 and "minutes_from_stints_prior_20" in df.columns and is_out_all is not None:
            prior20_all = (
                pd.to_numeric(df["minutes_from_stints_prior_20"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                .to_numpy(dtype="float32")
            )

        prior_weight_all: np.ndarray | None = None
        if bool(self.config.use_prior_head):
            col = str(self.config.prior_weight_col)
            if col not in df_features.columns:
                raise ValueError(f"use_prior_head=True requires prior_weight_col={col!r} in df_features")
            prior_weight_all = (
                pd.to_numeric(df_features[col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype="float32")
            )

        with torch.no_grad():
            for start in range(0, len(groups), batch_size):
                batch = groups[start : start + batch_size]
                max_n = max(g.x.shape[0] for g in batch)
                x = torch.zeros((len(batch), max_n, num_features), dtype=torch.float32, device=self.device)
                mask = torch.zeros((len(batch), max_n), dtype=torch.bool, device=self.device)
                alloc_mask = torch.zeros((len(batch), max_n), dtype=torch.bool, device=self.device)
                prior_w = torch.zeros((len(batch), max_n), dtype=torch.float32, device=self.device)
                team_idx = torch.zeros((len(batch), max_n), dtype=torch.long, device=self.device)
                opp_idx = torch.zeros((len(batch), max_n), dtype=torch.long, device=self.device)
                router_features = (
                    torch.zeros((len(batch), router_feat_dim), dtype=torch.float32, device=self.device)
                    if router_feat_dim > 0
                    else None
                )

                for i, group in enumerate(batch):
                    n = group.x.shape[0]
                    x[i, :n] = torch.from_numpy(group.x).to(self.device)
                    mask[i, :n] = True
                    if alloc_flag_all is not None:
                        alloc = alloc_flag_all[group.row_indices]
                        if not bool(np.any(alloc)):
                            alloc_mask[i, :n] = True
                        else:
                            alloc_mask[i, :n] = torch.as_tensor(alloc.astype(bool), device=self.device)
                    else:
                        alloc_mask[i, :n] = True
                    if prior_weight_all is not None:
                        prior_w[i, :n] = torch.as_tensor(prior_weight_all[group.row_indices], device=self.device)
                    team_idx[i, :n] = torch.from_numpy(group.team_idx).to(self.device)
                    opp_idx[i, :n] = torch.from_numpy(group.opp_idx).to(self.device)
                    if (
                        router_features is not None
                        and prior20_all is not None
                        and is_out_all is not None
                        and router_feat_dim >= 1
                    ):
                        idx = group.row_indices
                        group_prior20 = prior20_all[idx].astype("float64", copy=False)
                        group_out = is_out_all[idx].astype("int64", copy=False)
                        vac = float(np.sum(group_prior20[group_out == 1]))
                        router_features[i, 0] = float(np.clip(vac / 60.0, 0.0, 1.0))
                        if router_feat_dim >= 2:
                            out_count = float(np.sum(group_out))
                            router_features[i, 1] = float(np.clip(out_count / 10.0, 0.0, 1.0))
                        if router_feat_dim >= 3:
                            k = int(min(5, group_prior20.shape[0]))
                            if k > 0:
                                top_idx = np.argsort(group_prior20)[::-1][:k]
                                starters_out_proxy = float(np.sum(group_out[top_idx]))
                            else:
                                starters_out_proxy = 0.0
                            router_features[i, 2] = float(np.clip(starters_out_proxy / 5.0, 0.0, 1.0))

                minutes = (
                    self.model(
                        x,
                        team_idx,
                        opp_idx,
                        mask,
                        alloc_mask=alloc_mask,
                        prior_weights=prior_w if prior_weight_all is not None else None,
                        router_features=router_features,
                    )
                    .cpu()
                    .numpy()
                )
                for i, group in enumerate(batch):
                    n = group.x.shape[0]
                    preds[group.row_indices] = minutes[i, :n]

        out = df_features.copy()
        out["pred_minutes"] = preds
        return out


def predict_minutes(
    df_features: pd.DataFrame,
    *,
    model_dir: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> pd.DataFrame:
    """Predict per-player minutes from a flat feature dataframe.

    The dataframe is internally grouped by (game_id, team_id) and padded/masked per batch.
    """

    resolved_dir = model_dir or os.environ.get(MODEL_DIR_ENV)
    if not resolved_dir:
        raise ValueError(f"model_dir must be provided or {MODEL_DIR_ENV} must be set")
    predictor = RotationSetMinutesPredictor.load(Path(resolved_dir), device=device)
    return predictor.predict(df_features, batch_size=batch_size)
