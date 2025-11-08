"""Minimal PyTorch-based LSTM scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import nn


class LSTMMinutesPredictor(nn.Module):
    """Simple sequence model predicting minutes from player histories."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        sequence_output, _ = self.lstm(x)
        final_state = sequence_output[:, -1, :]
        return self.regressor(final_state)


@dataclass
class DeepTrainingArtifacts:
    """Artifacts for deep learning experiments."""

    model: LSTMMinutesPredictor
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def train_lstm_model(
    model: LSTMMinutesPredictor,
    train_loader: Iterable,
    val_loader: Iterable | None = None,
    *,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> DeepTrainingArtifacts:
    """Train an LSTM model using mean squared error loss."""

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        train_steps = 0
        for features_batch, targets in train_loader:
            features_batch = features_batch.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(features_batch).squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_steps += 1
        train_losses.append(epoch_loss / max(1, train_steps))

        if val_loader:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for features_batch, targets in val_loader:
                    features_batch = features_batch.to(device)
                    targets = targets.to(device)
                    outputs = model(features_batch).squeeze(-1)
                    val_loss += criterion(outputs, targets).item()
                    val_steps += 1
            val_losses.append(val_loss / max(1, val_steps))

    return DeepTrainingArtifacts(model=model, train_losses=train_losses, val_losses=val_losses)
