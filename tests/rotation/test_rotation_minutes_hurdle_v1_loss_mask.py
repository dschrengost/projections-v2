from __future__ import annotations

import torch

from projections.models.rotation_minutes_hurdle_v1.model import (
    conditional_minutes_pinball_loss,
    conditional_minutes_pinball_loss_q10_q50_q90,
)


def test_conditional_minutes_loss_uses_only_played_rows():
    """Test v1.0 loss function: DNP rows should not influence minutes head."""
    # Two played rows with perfect preds, one unplayed row with absurd target.
    q10 = torch.tensor([10.0, 0.0, 20.0])
    q50 = torch.tensor([10.0, 0.0, 20.0])
    q90 = torch.tensor([10.0, 0.0, 20.0])
    y_minutes = torch.tensor([10.0, 999.0, 20.0])
    y_play = torch.tensor([1.0, 0.0, 1.0])
    w = torch.ones_like(y_minutes)

    loss = conditional_minutes_pinball_loss_q10_q50_q90(
        q10=q10,
        q50=q50,
        q90=q90,
        y_minutes=y_minutes,
        y_play=y_play,
        sample_weight=w,
        quantile_weights=[1.0, 1.0, 1.0],
    )

    assert float(loss.item()) == 0.0

    # If we flip the unplayed row to played, the loss must become non-zero.
    loss_including_bad_row = conditional_minutes_pinball_loss_q10_q50_q90(
        q10=q10,
        q50=q50,
        q90=q90,
        y_minutes=y_minutes,
        y_play=torch.tensor([1.0, 1.0, 1.0]),
        sample_weight=w,
        quantile_weights=[1.0, 1.0, 1.0],
    )
    assert float(loss_including_bad_row.item()) > 0.0


def test_conditional_minutes_loss_v11_uses_only_played_rows():
    """Test v1.1 loss function: DNP rows should not influence minutes head."""
    # Two played rows with perfect preds, one unplayed row with absurd target.
    # v1.1 uses dict-based quantile interface.
    quantile_preds = {
        0.05: torch.tensor([5.0, 0.0, 15.0]),
        0.10: torch.tensor([10.0, 0.0, 20.0]),
        0.25: torch.tensor([12.0, 0.0, 22.0]),
        0.50: torch.tensor([15.0, 0.0, 25.0]),
        0.75: torch.tensor([18.0, 0.0, 28.0]),
        0.90: torch.tensor([20.0, 0.0, 30.0]),
        0.95: torch.tensor([22.0, 0.0, 32.0]),
    }
    # Perfect predictions for played rows at their respective quantiles
    y_minutes = torch.tensor([15.0, 999.0, 25.0])  # matches q50 for played rows
    y_play = torch.tensor([1.0, 0.0, 1.0])
    w = torch.ones_like(y_minutes)
    quantile_weights = {0.05: 1.0, 0.10: 1.0, 0.25: 1.0, 0.50: 1.0, 0.75: 1.0, 0.90: 1.0, 0.95: 1.0}

    loss = conditional_minutes_pinball_loss(
        quantile_preds=quantile_preds,
        y_minutes=y_minutes,
        y_play=y_play,
        sample_weight=w,
        quantile_weights=quantile_weights,
    )

    # Loss should be small but not exactly zero due to pinball loss asymmetry
    # (only q50 has zero loss when y=pred; other quantiles contribute)
    assert float(loss.item()) < 5.0  # Reasonably small loss

    # If we flip the unplayed row to played, the loss must become much larger.
    loss_including_bad_row = conditional_minutes_pinball_loss(
        quantile_preds=quantile_preds,
        y_minutes=y_minutes,
        y_play=torch.tensor([1.0, 1.0, 1.0]),
        sample_weight=w,
        quantile_weights=quantile_weights,
    )
    assert float(loss_including_bad_row.item()) > float(loss.item()) * 10  # Much larger due to 999.0 target

