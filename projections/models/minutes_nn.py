"""PyTorch utilities for the experimental multi-quantile minutes model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from projections.models.minutes_features import (
    DEFAULT_MINUTES_ALPHAS,
    MinutesFeatureSpec,
    build_feature_spec,
)


@dataclass
class MinutesNNConfig:
    """Training hyper-parameters for the quantile MLP."""

    alphas: list[float] = field(default_factory=lambda: list(DEFAULT_MINUTES_ALPHAS))
    emb_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 50
    early_stop_patience: int = 5
    weight_decay: float = 1e-5
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["device"] = self.device
        return payload


@dataclass
class MinutesPreprocessorState:
    """Serialization wrapper for feature scaling + vocab metadata."""

    continuous: dict[str, dict[str, float]]
    categorical: dict[str, list[str]]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "continuous": self.continuous,
            "categorical": self.categorical,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "MinutesPreprocessorState":
        return cls(
            continuous={col: {"mean": float(stats["mean"]), "std": float(stats["std"])} for col, stats in payload["continuous"].items()},
            categorical={col: list(values) for col, values in payload["categorical"].items()},
        )


class TabularMinutesDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Minimal dataset wrapper for (cont, cat, target) tensors."""

    def __init__(self, x_cont: np.ndarray, x_cat: np.ndarray, y: np.ndarray) -> None:
        if x_cont.shape[0] != y.shape[0] or x_cat.shape[0] != y.shape[0]:
            raise ValueError("Feature and target arrays must share the first dimension.")
        self.x_cont = torch.from_numpy(x_cont.astype(np.float32, copy=False))
        if self.x_cont.ndim == 1:
            self.x_cont = self.x_cont[:, None]
        self.x_cat = torch.from_numpy(x_cat.astype(np.int64, copy=False))
        if self.x_cat.ndim == 1:
            self.x_cat = self.x_cat[:, None]
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_cont[idx], self.x_cat[idx], self.y[idx]


class MinutesQuantileMLP(nn.Module):
    """Simple MLP with categorical embeddings for quantile regression."""

    def __init__(
        self,
        *,
        n_continuous: int,
        cat_cardinalities: Sequence[int],
        alphas: Sequence[float],
        hidden_dims: Sequence[int],
        emb_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.alphas = list(alphas)
        self.embeddings = nn.ModuleList(nn.Embedding(cardinality, emb_dim) for cardinality in cat_cardinalities)
        input_dim = n_continuous + emb_dim * len(self.embeddings)
        if input_dim == 0:
            raise ValueError("Model requires at least one feature column.")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden
        self.mlp = nn.Identity() if not layers else nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, len(self.alphas))

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        if x_cont.shape[1]:
            pieces.append(x_cont)
        if self.embeddings:
            cat_embs = []
            for idx, embedding in enumerate(self.embeddings):
                cat_embs.append(embedding(x_cat[:, idx]))
            cat_concat = torch.cat(cat_embs, dim=1) if cat_embs else None
            if cat_concat is not None:
                pieces.append(cat_concat)
        if not pieces:
            raise ValueError("No features provided to MinutesQuantileMLP.")
        hidden = torch.cat(pieces, dim=1)
        hidden = self.mlp(hidden)
        return self.output(hidden)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Standard pinball loss for a single quantile."""

    diff = target - pred
    return torch.maximum(alpha * diff, (alpha - 1.0) * diff)


def multi_quantile_loss(preds: torch.Tensor, target: torch.Tensor, alphas: Sequence[float]) -> torch.Tensor:
    """Aggregate pinball loss across multiple quantiles."""

    if preds.shape[1] != len(alphas):
        raise ValueError(f"Predictions have {preds.shape[1]} columns but {len(alphas)} alphas were provided.")
    losses = []
    for idx, alpha in enumerate(alphas):
        losses.append(pinball_loss(preds[:, idx], target, alpha).mean())
    return torch.stack(losses).mean()


def fit_preprocessor(train_df: pd.DataFrame, spec: MinutesFeatureSpec) -> MinutesPreprocessorState:
    """Fit scaling statistics + categorical vocabularies on the training frame."""

    cont_stats: dict[str, dict[str, float]] = {}
    for col in spec.continuous:
        values = pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=float)
        if not len(values):
            mean = 0.0
            std = 1.0
        else:
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values))
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
        cont_stats[col] = {"mean": mean, "std": std}

    cat_vocab: dict[str, list[str]] = {}
    for col in spec.categorical:
        series = train_df[col]
        tokens = [_canonicalize_category(value) for value in series]
        # Preserve encounter order to make vocab deterministic.
        ordered_unique: list[str] = list(dict.fromkeys(tokens))
        if not ordered_unique:
            ordered_unique = ["__nan__"]
        cat_vocab[col] = ordered_unique

    return MinutesPreprocessorState(continuous=cont_stats, categorical=cat_vocab)


def transform_frame(
    df: pd.DataFrame,
    spec: MinutesFeatureSpec,
    state: MinutesPreprocessorState,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply scaling + categorical encoding to a dataframe."""

    cont_arrays: list[np.ndarray] = []
    for col in spec.continuous:
        stats = state.continuous[col]
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        standardized = (values - stats["mean"]) / stats["std"]
        cont_arrays.append(np.nan_to_num(standardized, copy=False).astype(np.float32, copy=False))
    if cont_arrays:
        x_cont = np.column_stack(cont_arrays).astype(np.float32, copy=False)
    else:
        x_cont = np.zeros((len(df), 0), dtype=np.float32)

    cat_arrays: list[np.ndarray] = []
    for col in spec.categorical:
        vocab = state.categorical[col]
        index_map = {token: idx for idx, token in enumerate(vocab)}
        encoded = np.empty(len(df), dtype=np.int64)
        for idx, value in enumerate(df[col]):
            token = _canonicalize_category(value)
            encoded[idx] = index_map.get(token, len(vocab))
        cat_arrays.append(encoded)
    if cat_arrays:
        x_cat = np.column_stack(cat_arrays).astype(np.int64, copy=False)
    else:
        x_cat = np.zeros((len(df), 0), dtype=np.int64)
    return x_cont, x_cat


def _canonicalize_category(value: Any) -> str:
    if pd.isna(value):
        return "__nan__"
    return str(value)


__all__ = [
    "MinutesFeatureSpec",
    "MinutesNNConfig",
    "MinutesPreprocessorState",
    "MinutesQuantileMLP",
    "TabularMinutesDataset",
    "build_feature_spec",
    "fit_preprocessor",
    "multi_quantile_loss",
    "pinball_loss",
    "transform_frame",
]
