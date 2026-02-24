from __future__ import annotations

import torch

from projections.rotation.joint_game_flow import JointGameFlow


def test_joint_game_flow_forward_shapes_and_nll() -> None:
    torch.manual_seed(7)
    bsz, num_players, num_stats, d_model = 2, 30, 12, 24
    flow = JointGameFlow(
        d_model=d_model,
        num_stats=num_stats,
        hidden_dim=32,
        dropout=0.0,
        num_blocks=3,
        coupling_type="affine",
        scale_clip=1.5,
    )
    y = torch.randn((bsz, num_players, num_stats), dtype=torch.float32)
    player_states = torch.randn((bsz, num_players, d_model), dtype=torch.float32)
    team_states = torch.randn((bsz, 2, d_model), dtype=torch.float32)
    game_state = torch.randn((bsz, d_model), dtype=torch.float32)
    player_team_index = torch.cat(
        [torch.zeros((bsz, 15), dtype=torch.long), torch.ones((bsz, 15), dtype=torch.long)],
        dim=1,
    )
    valid_mask = torch.ones((bsz, num_players), dtype=torch.bool)
    observed_mask = torch.ones((bsz, num_players, num_stats), dtype=torch.bool)
    observed_mask[:, :2, :] = False

    out = flow(
        y,
        player_states=player_states,
        team_states=team_states,
        game_state=game_state,
        player_team_index=player_team_index,
        valid_mask=valid_mask,
        observed_mask=observed_mask,
    )
    assert out.z.shape == y.shape
    assert out.log_det.shape == (bsz,)
    assert out.nll.shape == (bsz,)
    assert out.nll_per_dim.shape == (bsz,)
    assert out.nll_mean.item() > 0.0


def test_joint_game_flow_inverse_sampling_round_trip() -> None:
    torch.manual_seed(11)
    bsz, num_players, num_stats, d_model = 1, 30, 12, 16
    flow = JointGameFlow(
        d_model=d_model,
        num_stats=num_stats,
        hidden_dim=32,
        dropout=0.0,
        num_blocks=4,
        coupling_type="affine",
        scale_clip=1.2,
    )
    y = torch.randn((bsz, num_players, num_stats), dtype=torch.float32)
    player_states = torch.randn((bsz, num_players, d_model), dtype=torch.float32)
    team_states = torch.randn((bsz, 2, d_model), dtype=torch.float32)
    game_state = torch.randn((bsz, d_model), dtype=torch.float32)
    player_team_index = torch.cat(
        [torch.zeros((bsz, 15), dtype=torch.long), torch.ones((bsz, 15), dtype=torch.long)],
        dim=1,
    )
    valid_mask = torch.ones((bsz, num_players), dtype=torch.bool)

    out = flow(
        y,
        player_states=player_states,
        team_states=team_states,
        game_state=game_state,
        player_team_index=player_team_index,
        valid_mask=valid_mask,
    )
    y_recon = flow.sample(
        out.z,
        player_states=player_states,
        team_states=team_states,
        game_state=game_state,
        player_team_index=player_team_index,
        valid_mask=valid_mask,
    )
    assert torch.max(torch.abs(y_recon - y)).item() < 1e-4
