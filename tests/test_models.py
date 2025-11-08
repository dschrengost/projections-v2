"""Tests for classical and deep learning model helpers."""

import numpy as np
import pytest
import torch

pytest.importorskip("xgboost")
from xgboost import XGBRegressor  # noqa: E402

from projections.models import classical, deep


def test_train_xgboost_model_returns_metrics():
    X = np.random.rand(50, 4)
    y = np.random.rand(50)
    result = classical.train_xgboost_model(
        X, y, X_valid=X[:10], y_valid=y[:10], params={"n_estimators": 10}
    )
    assert isinstance(result.model, XGBRegressor)
    assert "mae" in result.metrics


def test_train_lstm_model_records_losses():
    input_size = 3
    model = deep.LSTMMinutesPredictor(input_size=input_size)
    train_loader = [
        (torch.randn(4, 5, input_size), torch.randn(4)) for _ in range(2)
    ]
    artifacts = deep.train_lstm_model(
        model, train_loader, epochs=1, learning_rate=1e-2
    )
    assert len(artifacts.train_losses) == 1
