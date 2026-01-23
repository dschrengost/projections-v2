from __future__ import annotations

import torch

from projections.models.rotation_minutes_hurdle_v1.model import (
    conditional_minutes_pinball_loss_q10_q50_q90,
)


def test_conditional_minutes_loss_uses_only_played_rows():
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

