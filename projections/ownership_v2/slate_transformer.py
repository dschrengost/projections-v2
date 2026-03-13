from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


DEFAULT_OWNERSHIP_TRANSFORMER_FEATURES: tuple[str, ...] = (
    "proj_fpts",
    "salary",
    "minutes_mean",
    "dk_fpts_p90",
    "dk_fpts_p50",
    "minutes_sim_mean",
    "sim_p_active",
    "play_prob_eff",
    "value_per_k",
    "salary_rank",
    "proj_fpts_rank",
    "proj_fpts_zscore",
    "is_value_tier",
    "is_mid_tier",
    "is_high_tier",
    "pos_PG",
    "pos_SG",
    "pos_SF",
    "pos_PF",
    "pos_C",
    "player_is_questionable",
    "team_outs_count",
    "player_own_avg_10",
    "slate_size",
    "salary_pct_of_max",
    "is_min_salary",
    "slate_near_min_count",
    "value_vs_slate_avg",
    "salary_vs_median",
    "is_min_priced_by_pos",
    "game_count_on_slate",
    "player_own_median",
    "player_own_variance",
    "player_chalk_rate",
    "value_x_value_tier",
    "outs_x_salary_rank",
)


@dataclass(frozen=True)
class OwnershipSlateTransformerConfig:
    feature_columns: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 3
    hidden_dim: int = 256
    dropout: float = 0.1
    target_sum_pct: float = 800.0
    max_players: int = 256
    version: str = "ownership_slate_transformer_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OwnershipSlateTransformerConfig":
        return cls(**payload)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OwnershipSlateTransformerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class OwnershipSlateDataset(Dataset[dict[str, Any]]):
    """Groups player rows into slate-level sequences."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        target_sum_pct: float = 800.0,
        sort_columns: tuple[str, ...] = ("salary", "proj_fpts", "player_id"),
    ) -> None:
        if frame.empty:
            raise ValueError("frame must be non-empty")
        if "slate_id" not in frame.columns:
            raise ValueError("frame missing slate_id")
        missing = sorted(set(feature_columns) - set(frame.columns))
        if missing:
            raise KeyError(f"frame missing required feature columns: {missing}")
        if "actual_own_pct" not in frame.columns:
            raise ValueError("frame missing actual_own_pct")

        self.feature_columns = list(feature_columns)
        self.target_sum_pct = float(target_sum_pct)
        self.examples: list[dict[str, Any]] = []

        work = frame.copy()
        work["game_date"] = work["game_date"].astype(str)
        work["player_name"] = work.get("player_name", "").astype(str)
        work["team"] = work.get("team", "").astype(str)
        work["pos"] = work.get("pos", "").astype(str)
        work["player_id"] = pd.to_numeric(work.get("player_id", 0), errors="coerce").fillna(0).astype(np.int64)
        work["actual_own_pct"] = pd.to_numeric(work["actual_own_pct"], errors="coerce").fillna(0.0).clip(lower=0.0)

        ascending = [False, False, True]
        grouped = work.groupby("slate_id", sort=False)
        for slate_id, group in grouped:
            ordered = group.sort_values(list(sort_columns), ascending=ascending[: len(sort_columns)]).reset_index(drop=True)
            feats = (
                ordered[self.feature_columns]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=False)
            )
            target_pct = ordered["actual_own_pct"].to_numpy(dtype=np.float32, copy=False)
            actual_sum = float(target_pct.sum())
            target_share = np.zeros_like(target_pct, dtype=np.float32)
            if actual_sum > 0.0:
                target_share = target_pct / actual_sum

            self.examples.append(
                {
                    "slate_id": str(slate_id),
                    "game_date": str(ordered["game_date"].iloc[0]),
                    "features": feats,
                    "target_pct": target_pct.astype(np.float32, copy=False),
                    "target_share": target_share.astype(np.float32, copy=False),
                    "player_name": ordered["player_name"].tolist(),
                    "player_id": ordered["player_id"].tolist(),
                    "team": ordered["team"].tolist(),
                    "pos": ordered["pos"].tolist(),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_ownership_slates(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("batch must be non-empty")

    batch_size = len(batch)
    max_len = max(int(item["features"].shape[0]) for item in batch)
    feat_dim = int(batch[0]["features"].shape[1])

    features = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    target_pct = torch.zeros((batch_size, max_len), dtype=torch.float32)
    target_share = torch.zeros((batch_size, max_len), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    slate_ids: list[str] = []
    game_dates: list[str] = []
    player_names: list[list[str]] = []
    player_ids: list[list[int]] = []

    for row_idx, item in enumerate(batch):
        n = int(item["features"].shape[0])
        features[row_idx, :n] = torch.from_numpy(item["features"])
        target_pct[row_idx, :n] = torch.from_numpy(item["target_pct"])
        target_share[row_idx, :n] = torch.from_numpy(item["target_share"])
        valid_mask[row_idx, :n] = True
        slate_ids.append(str(item["slate_id"]))
        game_dates.append(str(item["game_date"]))
        player_names.append(list(item["player_name"]))
        player_ids.append(list(item["player_id"]))

    return {
        "features": features,
        "target_pct": target_pct,
        "target_share": target_share,
        "valid_mask": valid_mask,
        "slate_id": slate_ids,
        "game_date": game_dates,
        "player_name": player_names,
        "player_id": player_ids,
    }


class OwnershipSlateTransformer(nn.Module):
    """Full-slate transformer with masked softmax allocation."""

    def __init__(self, config: OwnershipSlateTransformerConfig) -> None:
        super().__init__()
        if not config.feature_columns:
            raise ValueError("feature_columns must be non-empty")
        if config.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if config.num_heads <= 0 or config.d_model % config.num_heads != 0:
            raise ValueError("num_heads must divide d_model")

        self.config = config
        feat_dim = len(config.feature_columns)
        self.input_proj = nn.Linear(feat_dim, config.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_players + 1, config.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=config.num_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.logit_head = nn.Sequential(
            nn.Linear(config.d_model * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape (B,N,F)")
        if valid_mask.ndim != 2 or valid_mask.shape[:2] != features.shape[:2]:
            raise ValueError("valid_mask must have shape (B,N) aligned to features")

        batch_size, seq_len, _ = features.shape
        if seq_len > self.config.max_players:
            raise ValueError(f"seq_len {seq_len} exceeds configured max_players={self.config.max_players}")

        token_inputs = self.input_proj(features)
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, token_inputs], dim=1)
        seq = seq + self.pos_embedding[:, : seq.shape[1], :]

        full_mask = torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.bool, device=valid_mask.device), valid_mask],
            dim=1,
        )
        encoded = self.encoder(seq, src_key_padding_mask=~full_mask)
        encoded = self.final_norm(encoded)

        global_state = encoded[:, :1, :].expand(-1, seq_len, -1)
        player_states = encoded[:, 1:, :]
        logits = self.logit_head(torch.cat([player_states, global_state], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, float("-inf"))

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        probs = probs * valid_mask.to(dtype=probs.dtype)
        pred_pct = probs * float(self.config.target_sum_pct)

        return {
            "logits": logits,
            "log_probs": log_probs,
            "probs": probs,
            "pred_pct": pred_pct,
        }


def standardize_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Standardize numeric feature columns in-place and return train stats."""
    result = frame.copy()
    raw = (
        result[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )
    if feature_mean is None:
        feature_mean = raw.mean(axis=0, dtype=np.float64)
    if feature_std is None:
        feature_std = raw.std(axis=0, dtype=np.float64)
    safe_std = np.where(np.asarray(feature_std) <= 1e-6, 1.0, np.asarray(feature_std))
    scaled = (raw - np.asarray(feature_mean, dtype=np.float32)[None, :]) / np.asarray(safe_std, dtype=np.float32)[None, :]
    scaled_df = pd.DataFrame(scaled, index=result.index, columns=feature_columns, dtype=np.float32)
    for col in feature_columns:
        result[col] = scaled_df[col]
    return result, np.asarray(feature_mean, dtype=np.float32), np.asarray(safe_std, dtype=np.float32)
