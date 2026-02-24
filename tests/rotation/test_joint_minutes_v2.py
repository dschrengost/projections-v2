from __future__ import annotations

import torch
import pytest

from projections.rotation.joint_minutes import project_minutes_capped_simplex


def _team_index(batch_size: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros((batch_size, 15), dtype=torch.long),
            torch.ones((batch_size, 15), dtype=torch.long),
        ],
        dim=1,
    )


def test_project_minutes_capped_simplex_team_totals_and_bounds() -> None:
    torch.manual_seed(7)
    bsz = 3
    num_players = 30

    preferences = torch.randn((bsz, num_players), dtype=torch.float32)
    valid_mask = torch.ones((bsz, num_players), dtype=torch.bool)
    team_index = _team_index(bsz)
    active_mask = torch.zeros((bsz, num_players), dtype=torch.bool)
    active_mask[:, :8] = True
    active_mask[:, 15:24] = True

    minutes, used_active = project_minutes_capped_simplex(
        preferences,
        active_mask,
        valid_mask,
        team_index,
        total_minutes_per_team=240.0,
        max_minutes_per_player=48.0,
        fallback_to_valid_on_infeasible=False,
    )

    assert torch.all(minutes >= -1e-6)
    assert torch.all(minutes <= 48.0 + 1e-5)

    for b_idx in range(bsz):
        for team in (0, 1):
            mask = (team_index[b_idx] == team) & valid_mask[b_idx]
            total = float(minutes[b_idx, mask].sum().item())
            assert total == pytest.approx(240.0, abs=1e-3)

    inactive = (~used_active) & valid_mask
    assert float(minutes[inactive].abs().max().item()) == pytest.approx(0.0, abs=1e-6)


def test_project_minutes_capped_simplex_fallback_to_valid_on_infeasible() -> None:
    bsz = 1
    num_players = 30

    preferences = torch.linspace(-1.0, 1.0, num_players, dtype=torch.float32).unsqueeze(0)
    valid_mask = torch.ones((bsz, num_players), dtype=torch.bool)
    team_index = _team_index(bsz)

    # Infeasible on team 0: only 4 active players => max 4*48 < 240.
    active_mask = torch.zeros((bsz, num_players), dtype=torch.bool)
    active_mask[:, :4] = True
    active_mask[:, 15:22] = True

    with pytest.raises(ValueError, match="Infeasible capped simplex projection"):
        project_minutes_capped_simplex(
            preferences,
            active_mask,
            valid_mask,
            team_index,
            total_minutes_per_team=240.0,
            max_minutes_per_player=48.0,
            fallback_to_valid_on_infeasible=False,
            allow_scale_down_infeasible=False,
        )

    minutes, used_active = project_minutes_capped_simplex(
        preferences,
        active_mask,
        valid_mask,
        team_index,
        total_minutes_per_team=240.0,
        max_minutes_per_player=48.0,
        fallback_to_valid_on_infeasible=True,
    )

    mask0 = (team_index[0] == 0) & valid_mask[0]
    mask1 = (team_index[0] == 1) & valid_mask[0]
    assert float(minutes[0, mask0].sum().item()) == pytest.approx(240.0, abs=1e-3)
    assert float(minutes[0, mask1].sum().item()) == pytest.approx(240.0, abs=1e-3)

    # Fallback should activate more than the original 4 players for team 0.
    assert int((used_active[0] & mask0).sum().item()) > 4
