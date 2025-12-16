"""
Production loader for usage shares v1 models.

Provides a unified interface for loading and predicting with both LightGBM and NN backends.

Usage:
    from projections.usage_shares_v1.production import load_bundle, predict_log_weights
    
    bundle = load_bundle(data_root, run_id="20251215_215957", backend="lgbm")
    log_weights = predict_log_weights(bundle, df, target="fga")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from projections.paths import data_path
from projections.usage_shares_v1.features import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    add_derived_features,
)


# =============================================================================
# Bundle Dataclass
# =============================================================================


@dataclass
class UsageSharesBundle:
    """Loaded usage shares model bundle."""
    backend: str  # "lgbm" or "nn"
    run_id: str
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    targets: list[str]
    meta: dict[str, Any]
    
    # LGBM-specific
    lgbm_models: dict[str, lgb.Booster] | None = None
    
    # NN-specific
    nn_model: torch.nn.Module | None = None
    nn_config: dict[str, Any] | None = None
    category_maps: dict[str, dict[Any, int]] | None = None
    scaler_params: dict[str, tuple[float, float]] | None = None


# =============================================================================
# NN Model Definition (must match train_nn.py)
# =============================================================================


class UsageSharesNN(torch.nn.Module):
    """Neural network for usage shares prediction."""
    
    def __init__(
        self,
        n_numeric: int,
        n_cat: int,
        vocab_sizes: list[int],
        embed_dim: int = 8,
        hidden_sizes: list[int] = [128, 64],
        n_targets: int = 3,
    ):
        super().__init__()
        
        # Embedding layers
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(vocab_size + 1, min(embed_dim, (vocab_size + 1) // 2 + 1))
            for vocab_size in vocab_sizes
        ])
        embed_total = sum(e.embedding_dim for e in self.embeddings)
        
        # Input dimension
        input_dim = n_numeric + embed_total
        
        # Shared trunk
        layers: list[torch.nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        self.trunk = torch.nn.Sequential(*layers)
        
        # Target heads
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(prev_dim, 1) for _ in range(n_targets)
        ])
        
        self.n_targets = n_targets
    
    def forward(self, X_num: torch.Tensor, X_cat: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits [batch, n_targets]."""
        # Embed categoricals
        embeds = []
        for i, embed_layer in enumerate(self.embeddings):
            if X_cat.shape[-1] > i:
                embeds.append(embed_layer(X_cat[:, i]))
        
        # Concatenate
        if embeds:
            x = torch.cat([X_num] + embeds, dim=-1)
        else:
            x = X_num
        
        # Through trunk
        x = self.trunk(x)
        
        # Through heads - each produces one logit
        logits = torch.cat([head(x) for head in self.heads], dim=-1)
        return logits


# =============================================================================
# Loading Functions
# =============================================================================


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(
    data_root: Path | str | None = None,
    run_id: str | None = None,
    backend: str = "lgbm",
    config_path: Path | None = None,
) -> UsageSharesBundle:
    """
    Load a usage shares model bundle.
    
    Args:
        data_root: Root data directory (defaults to PROJECTIONS_DATA_ROOT)
        run_id: Run ID to load (defaults to config file)
        backend: "lgbm" or "nn"
        config_path: Path to config file (defaults to config/usage_shares_current_run.json)
        
    Returns:
        UsageSharesBundle with loaded model and metadata
    """
    root = Path(data_root) if data_root else data_path()
    
    # Get run_id from config if not provided
    if run_id is None:
        config_file = config_path or (root.parent / "projections-v2" / "config" / "usage_shares_current_run.json")
        if config_file.exists():
            config = _load_json(config_file)
            run_id = config.get("run_id")
            if "backend" not in config:
                backend = config.get("backend", backend)
        if run_id is None:
            raise ValueError("run_id not specified and no config file found")
    
    run_dir = root / "artifacts" / "usage_shares_v1" / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load feature columns
    feature_cols_path = run_dir / "feature_columns.json"
    if feature_cols_path.exists():
        fc = _load_json(feature_cols_path)
        feature_cols = fc.get("feature_cols", [])
        numeric_cols = fc.get("numeric_cols", [c for c in feature_cols if c in NUMERIC_COLS])
        categorical_cols = fc.get("categorical_cols", [c for c in feature_cols if c in CATEGORICAL_COLS])
    else:
        feature_cols = list(NUMERIC_COLS) + list(CATEGORICAL_COLS)
        numeric_cols = list(NUMERIC_COLS)
        categorical_cols = list(CATEGORICAL_COLS)
    
    # Load meta
    meta_path = run_dir / "meta.json"
    lgbm_meta_path = run_dir / "lgbm" / "meta.json" if backend == "lgbm" else None
    nn_meta_path = run_dir / "nn" / "meta.json" if backend == "nn" else None
    
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = _load_json(meta_path)
    if lgbm_meta_path and lgbm_meta_path.exists():
        meta.update(_load_json(lgbm_meta_path))
    if nn_meta_path and nn_meta_path.exists():
        meta.update(_load_json(nn_meta_path))
    
    targets = meta.get("targets", ["fga", "fta", "tov"])
    
    bundle = UsageSharesBundle(
        backend=backend,
        run_id=run_id,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        targets=targets,
        meta=meta,
    )
    
    if backend == "lgbm":
        _load_lgbm_models(bundle, run_dir)
    elif backend == "nn":
        _load_nn_model(bundle, run_dir)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    return bundle


def _load_lgbm_models(bundle: UsageSharesBundle, run_dir: Path) -> None:
    """Load LightGBM models into bundle."""
    lgbm_dir = run_dir / "lgbm"
    if not lgbm_dir.exists():
        raise FileNotFoundError(f"LGBM directory not found: {lgbm_dir}")
    
    bundle.lgbm_models = {}
    for target in bundle.targets:
        model_path = lgbm_dir / f"model_{target}.txt"
        if model_path.exists():
            bundle.lgbm_models[target] = lgb.Booster(model_file=str(model_path))


def _load_nn_model(bundle: UsageSharesBundle, run_dir: Path) -> None:
    """Load NN model into bundle."""
    nn_dir = run_dir / "nn"
    if not nn_dir.exists():
        raise FileNotFoundError(f"NN directory not found: {nn_dir}")
    
    # Load config
    config_path = nn_dir / "nn_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"NN config not found: {config_path}")
    bundle.nn_config = _load_json(config_path)
    
    # Load scaler
    scaler_path = nn_dir / "scaler.json"
    if scaler_path.exists():
        scaler_raw = _load_json(scaler_path)
        bundle.scaler_params = {k: tuple(v) for k, v in scaler_raw.items()}
    
    # Load category maps
    raw_maps = bundle.nn_config.get("category_maps", {})
    bundle.category_maps = {}
    for col, col_map in raw_maps.items():
        bundle.category_maps[col] = {
            (int(k) if k.lstrip("-").isdigit() else k): v 
            for k, v in col_map.items()
        }
    
    # Create model
    model = UsageSharesNN(
        n_numeric=bundle.nn_config["n_numeric"],
        n_cat=bundle.nn_config["n_cat"],
        vocab_sizes=bundle.nn_config["vocab_sizes"],
        embed_dim=bundle.nn_config.get("embed_dim", 8),
        hidden_sizes=bundle.nn_config.get("hidden_sizes", [128, 64]),
        n_targets=bundle.nn_config["n_targets"],
    )
    
    # Load weights
    model_path = nn_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"NN model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    bundle.nn_model = model


# =============================================================================
# Prediction Functions
# =============================================================================


def prepare_features(
    bundle: UsageSharesBundle,
    df: pd.DataFrame,
    add_missing_indicators: bool = True,
    strict_categoricals: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare feature arrays from dataframe.
    
    Args:
        bundle: Loaded model bundle
        df: Input dataframe
        add_missing_indicators: If True, add <col>_missing features (default: True)
        strict_categoricals: If True, raise ValueError on missing categorical columns
        
    Returns:
        X_num: Numeric features array (n_rows, n_numeric + n_missing_indicators)
        X_cat: Categorical features array (n_rows, n_cat)
        warnings: List of warning messages about missing features
    """
    import warnings as warn_module
    
    # Add derived features
    df = add_derived_features(df)
    
    warnings_list: list[str] = []
    
    # Check for missing categorical columns first
    missing_cat_cols = [c for c in bundle.categorical_cols if c not in df.columns]
    if missing_cat_cols and strict_categoricals:
        raise ValueError(f"Missing required categorical columns: {missing_cat_cols}")
    
    # Numeric features
    X_num_list: list[np.ndarray] = []
    missing_indicator_names: list[str] = []
    
    for col in bundle.numeric_cols:
        if col in df.columns:
            raw_vals = pd.to_numeric(df[col], errors="coerce")
            is_missing = raw_vals.isna()
            vals = raw_vals.fillna(0.0).values
            
            if is_missing.any():
                missing_count = is_missing.sum()
                warnings_list.append(f"Feature '{col}' has {missing_count}/{len(df)} missing values, filled with 0")
                warn_module.warn(warnings_list[-1])
                
                if add_missing_indicators:
                    missing_indicator_names.append(f"{col}_missing")
        else:
            vals = np.zeros(len(df))
            is_missing = np.ones(len(df), dtype=bool)
            warnings_list.append(f"Feature '{col}' missing from input, filled with 0")
            warn_module.warn(warnings_list[-1])
            
            if add_missing_indicators:
                missing_indicator_names.append(f"{col}_missing")
        
        # Scale for NN
        if bundle.backend == "nn" and bundle.scaler_params:
            mean, std = bundle.scaler_params.get(col, (0.0, 1.0))
            vals = (vals - mean) / std
        
        X_num_list.append(vals)
        
        # Add missing indicator if applicable
        if add_missing_indicators and (col not in df.columns or is_missing.any()):
            X_num_list.append(is_missing.astype(np.float32))
    
    X_num = np.column_stack(X_num_list) if X_num_list else np.zeros((len(df), 0), dtype=np.float32)
    
    # Categorical features
    X_cat = np.zeros((len(df), len(bundle.categorical_cols)), dtype=np.int64)
    for i, col in enumerate(bundle.categorical_cols):
        if col in df.columns:
            if bundle.backend == "nn" and bundle.category_maps:
                col_map = bundle.category_maps.get(col, {})
                # Map values, with UNK (index 0) for unknown values
                def map_with_unk(x):
                    idx = col_map.get(x, 0)  # 0 = UNK
                    return idx
                X_cat[:, i] = df[col].map(map_with_unk).fillna(0).astype(int).values
            else:
                # LGBM: use raw values as codes
                X_cat[:, i] = df[col].fillna(-1).astype(int).values
        else:
            # Missing categorical should have been caught above if strict
            X_cat[:, i] = 0 if bundle.backend == "nn" else -1
    
    return X_num, X_cat, warnings_list


def predict_log_weights(
    bundle: UsageSharesBundle,
    df: pd.DataFrame,
    target: str,
    add_missing_indicators: bool = False,  # Disabled by default for inference
    strict_categoricals: bool = True,
) -> np.ndarray:
    """
    Predict log-weights for a target.
    
    Args:
        bundle: Loaded model bundle
        df: Input dataframe with feature columns
        target: Target to predict (fga/fta/tov)
        add_missing_indicators: If True, add missing indicators (disabled by default for inference)
        strict_categoricals: If True, raise on missing categorical columns
        
    Returns:
        Log-weights array of shape (n_rows,)
    """
    if target not in bundle.targets:
        raise ValueError(f"Target {target} not in bundle targets: {bundle.targets}")
    
    X_num, X_cat, _warnings = prepare_features(
        bundle, df, 
        add_missing_indicators=add_missing_indicators,
        strict_categoricals=strict_categoricals,
    )
    
    if bundle.backend == "lgbm":
        return _predict_lgbm(bundle, X_num, X_cat, target)
    elif bundle.backend == "nn":
        return _predict_nn(bundle, X_num, X_cat, target)
    else:
        raise ValueError(f"Unknown backend: {bundle.backend}")


def _predict_lgbm(
    bundle: UsageSharesBundle,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    target: str,
) -> np.ndarray:
    """Predict with LightGBM model."""
    if bundle.lgbm_models is None or target not in bundle.lgbm_models:
        raise ValueError(f"LGBM model for target {target} not loaded")
    
    # Combine features
    X = np.hstack([X_num, X_cat.astype(np.float64)])
    
    # Create DataFrame with proper column names
    feature_cols = bundle.numeric_cols + bundle.categorical_cols
    X_df = pd.DataFrame(X, columns=feature_cols)
    
    return bundle.lgbm_models[target].predict(X_df)


def _predict_nn(
    bundle: UsageSharesBundle,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    target: str,
) -> np.ndarray:
    """Predict with NN model."""
    if bundle.nn_model is None:
        raise ValueError("NN model not loaded")
    
    target_idx = bundle.targets.index(target)
    
    with torch.no_grad():
        X_num_t = torch.from_numpy(X_num).float()
        X_cat_t = torch.from_numpy(X_cat).long()
        logits = bundle.nn_model(X_num_t, X_cat_t)
        return logits[:, target_idx].numpy()


# =============================================================================
# Config Management
# =============================================================================


def get_current_run_id(config_path: Path | None = None) -> str | None:
    """Get current run ID from config file."""
    from projections.paths import get_project_root
    
    config_file = config_path or (get_project_root() / "config" / "usage_shares_current_run.json")
    if config_file.exists():
        config = _load_json(config_file)
        return config.get("run_id")
    return None


def load_production_bundle(backend: str = "lgbm") -> UsageSharesBundle:
    """Load the production bundle from config."""
    run_id = get_current_run_id()
    if run_id is None:
        raise ValueError("No production run configured in usage_shares_current_run.json")
    return load_bundle(run_id=run_id, backend=backend)


__all__ = [
    "UsageSharesBundle",
    "load_bundle",
    "load_production_bundle",
    "predict_log_weights",
    "prepare_features",
    "get_current_run_id",
]
