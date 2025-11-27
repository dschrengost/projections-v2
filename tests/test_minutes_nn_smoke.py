"""Smoke tests for the Minutes NN helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from projections.models.minutes_features import build_feature_spec
from projections.models.minutes_nn import (
    MinutesNNConfig,
    MinutesQuantileMLP,
    TabularMinutesDataset,
    fit_preprocessor,
    multi_quantile_loss,
    transform_frame,
)


def test_minutes_quantile_mlp_forward_and_loss() -> None:
    """Ensure the NN forward + loss wiring produces finite values."""

    df = pd.DataFrame(
        {
            "minutes": [10.0, 20.0, 5.0, 15.0],
            "recent_usage": [0.2, 0.3, 0.15, 0.25],
            "injury_flag": [0.0, 1.0, 0.0, 1.0],
            "role_bucket": ["starter", "bench", "bench", "starter"],
        }
    )
    feature_columns = ["recent_usage", "injury_flag", "role_bucket"]
    spec = build_feature_spec(feature_columns, categorical_columns=["role_bucket"])
    preproc = fit_preprocessor(df, spec)
    x_cont, x_cat = transform_frame(df, spec, preproc)
    dataset = TabularMinutesDataset(x_cont, x_cat, df["minutes"].to_numpy(dtype=np.float32))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    x_cont_batch, x_cat_batch, y_batch = next(iter(loader))
    config = MinutesNNConfig(hidden_dims=[16], emb_dim=4, alphas=[0.1, 0.5, 0.9], dropout=0.0, batch_size=2, max_epochs=1)
    model = MinutesQuantileMLP(
        n_continuous=len(spec.continuous),
        cat_cardinalities=[len(preproc.categorical["role_bucket"]) + 1],
        alphas=config.alphas,
        hidden_dims=config.hidden_dims,
        emb_dim=config.emb_dim,
        dropout=config.dropout,
    )
    preds = model(x_cont_batch, x_cat_batch)
    assert preds.shape == (2, len(config.alphas))
    loss = multi_quantile_loss(preds, y_batch, config.alphas)
    assert torch.isfinite(loss)


def test_minutes_nn_one_step_train_loop() -> None:
    """Ensure a tiny training step runs without runtime errors."""

    df = pd.DataFrame(
        {
            "minutes": [12.0, 24.0, 6.0, 18.0, 30.0, 8.0],
            "recent_usage": [0.25, 0.35, 0.18, 0.28, 0.4, 0.2],
            "starter_flag": [1, 1, 0, 1, 1, 0],
            "home_flag": [1, 0, 1, 0, 1, 0],
            "injury_snapshot_missing": [0, 0, 1, 0, 0, 1],
        }
    )
    feature_columns = ["recent_usage", "starter_flag", "home_flag", "injury_snapshot_missing"]
    categorical = ["starter_flag", "home_flag", "injury_snapshot_missing"]
    spec = build_feature_spec(feature_columns, categorical_columns=categorical)
    preproc = fit_preprocessor(df, spec)
    x_cont, x_cat = transform_frame(df, spec, preproc)
    dataset = TabularMinutesDataset(x_cont, x_cat, df["minutes"].to_numpy(dtype=np.float32))
    loader = DataLoader(dataset, batch_size=3, shuffle=False)

    config = MinutesNNConfig(hidden_dims=[8], emb_dim=2, dropout=0.0, batch_size=3, max_epochs=1)
    model = MinutesQuantileMLP(
        n_continuous=len(spec.continuous),
        cat_cardinalities=[len(preproc.categorical[col]) + 1 for col in spec.categorical],
        alphas=config.alphas,
        hidden_dims=config.hidden_dims,
        emb_dim=config.emb_dim,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    x_cont_batch, x_cat_batch, y_batch = next(iter(loader))
    preds = model(x_cont_batch, x_cat_batch)
    loss = multi_quantile_loss(preds, y_batch, config.alphas)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
