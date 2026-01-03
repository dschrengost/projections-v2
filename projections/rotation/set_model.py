"""Permutation-invariant TEAM-SET minutes model (rotation_set_minutes v1).

This module provides:
- DeepSets and SetTransformer variants that operate on a variable-size set of players per (game_id, team_id)
- A minutes normalization head that enforces per-team totals of 240 minutes
- A minimal `predict_minutes` helper that accepts a flat dataframe and returns aligned per-row predictions
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

MODEL_DIR_ENV = "ROTATION_SET_MODEL_DIR"
MODEL_WEIGHTS_FILENAME = "model.pt"
MODEL_CONFIG_FILENAME = "config.json"

GAME_ID_NORM_COL = "game_id_norm"
KEY_COLS = ("game_id", "team_id", "player_id")


def zfill_game_id_series(series: pd.Series) -> pd.Series:
    """Normalize NBA game ids to a zero-filled 10-character string."""

    coerced = pd.to_numeric(series, errors="coerce").astype("Int64")
    return coerced.astype("string").str.zfill(10)


def _coerce_int_series(series: pd.Series, *, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype("Int64")
    if out.isna().any():
        raise ValueError(f"{name} contains missing/invalid values after coercion")
    return out


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized key columns used for grouping."""

    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required key columns: {missing}")
    out = df.copy()
    out[GAME_ID_NORM_COL] = zfill_game_id_series(out["game_id"])
    if out[GAME_ID_NORM_COL].isna().any():
        raise ValueError("game_id contains missing/invalid values after normalization")
    out["team_id"] = _coerce_int_series(out["team_id"], name="team_id")
    out["player_id"] = _coerce_int_series(out["player_id"], name="player_id")
    return out


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
) -> torch.Tensor:
    """Convert per-player logits to minutes that sum to `total_minutes` per team-game."""

    if logits.ndim != 2 or mask.ndim != 2:
        raise ValueError("minutes_from_logits expects logits=(B,N) and mask=(B,N)")
    positive = F.softplus(logits)
    positive = positive * mask.to(dtype=positive.dtype)
    denom = positive.sum(dim=1, keepdim=True).clamp(min=eps)
    minutes = positive / denom * float(total_minutes)
    return minutes * mask.to(dtype=minutes.dtype)


ModelType = Literal["deepsets", "settransformer"]


@dataclass(frozen=True)
class RotationSetModelConfig:
    model: ModelType
    feature_columns: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    embed_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.1
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    total_minutes: float = 240.0
    eps: float = 1e-6
    version: str = "rotation_set_minutes_v1"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RotationSetModelConfig":
        return cls(**payload)

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


class DeepSetsMinutesModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        *,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        total_minutes: float = 240.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.total_minutes = float(total_minutes)
        self.eps = float(eps)
        self.phi = MLP(num_features, embed_dim, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)
        self.rho = MLP(embed_dim * 2, 1, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)

    def forward_logits(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.phi(x)
        pooled = masked_mean(h, mask)
        pooled_rep = pooled.unsqueeze(1).expand(-1, h.shape[1], -1)
        logits = self.rho(torch.cat([h, pooled_rep], dim=-1)).squeeze(-1)
        return logits

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x, mask)
        return minutes_from_logits(logits, mask, total_minutes=self.total_minutes, eps=self.eps)


class SetTransformerMinutesModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        *,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        total_minutes: float = 240.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.total_minutes = float(total_minutes)
        self.eps = float(eps)
        self.input_proj = nn.Linear(num_features, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.head = MLP(embed_dim, 1, hidden_dim=hidden_dim, num_layers=2, dropout=dropout)

    def forward_logits(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        padding_mask = ~mask
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.head(h).squeeze(-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(x, mask)
        return minutes_from_logits(logits, mask, total_minutes=self.total_minutes, eps=self.eps)


def build_model(config: RotationSetModelConfig) -> nn.Module:
    num_features = len(config.feature_columns)
    if config.model == "deepsets":
        return DeepSetsMinutesModel(
            num_features,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            total_minutes=config.total_minutes,
            eps=config.eps,
        )
    if config.model == "settransformer":
        return SetTransformerMinutesModel(
            num_features,
            embed_dim=config.embed_dim,
            hidden_dim=max(config.hidden_dim, 2 * config.embed_dim),
            dropout=config.dropout,
            num_transformer_layers=config.num_transformer_layers,
            num_attention_heads=config.num_attention_heads,
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

        groups: list[TeamGameSet] = []
        for (game_id, team_id), idx in df.groupby([GAME_ID_NORM_COL, "team_id"], sort=False).indices.items():
            idx_arr = np.asarray(idx, dtype=np.int64)
            groups.append(
                TeamGameSet(
                    game_id=str(game_id),
                    team_id=int(team_id),
                    row_indices=idx_arr,
                    x=feature_matrix[idx_arr],
                )
            )

        preds = np.zeros(len(df), dtype="float32")
        num_features = len(self.config.feature_columns)

        with torch.no_grad():
            for start in range(0, len(groups), batch_size):
                batch = groups[start : start + batch_size]
                max_n = max(g.x.shape[0] for g in batch)
                x = torch.zeros((len(batch), max_n, num_features), dtype=torch.float32, device=self.device)
                mask = torch.zeros((len(batch), max_n), dtype=torch.bool, device=self.device)

                for i, group in enumerate(batch):
                    n = group.x.shape[0]
                    x[i, :n] = torch.from_numpy(group.x).to(self.device)
                    mask[i, :n] = True

                minutes = self.model(x, mask).cpu().numpy()
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

