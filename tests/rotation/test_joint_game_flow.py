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


def test_joint_game_flow_rqs_round_trip_and_finite_nll() -> None:
    torch.manual_seed(21)
    bsz, num_players, num_stats, d_model = 2, 30, 12, 20
    flow = JointGameFlow(
        d_model=d_model,
        num_stats=num_stats,
        hidden_dim=32,
        dropout=0.0,
        num_blocks=2,
        coupling_type="rqs",
        rqs_num_bins=8,
        rqs_tail_bound=20.0,
        rqs_min_bin_width=1e-3,
        rqs_min_bin_height=1e-3,
        rqs_min_derivative=1e-3,
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
    assert torch.isfinite(out.nll).all()
    assert torch.isfinite(out.log_det).all()

    y_recon = flow.sample(
        out.z,
        player_states=player_states,
        team_states=team_states,
        game_state=game_state,
        player_team_index=player_team_index,
        valid_mask=valid_mask,
    )
    assert torch.max(torch.abs(y_recon - y)).item() < 2e-3


def test_joint_game_flow_mean_ctx_weight_runtime_override() -> None:
    torch.manual_seed(17)
    bsz, num_players, num_stats, d_model = 1, 30, 12, 16
    flow = JointGameFlow(
        d_model=d_model,
        num_stats=num_stats,
        hidden_dim=32,
        dropout=0.0,
        num_blocks=2,
        coupling_type="affine",
        scale_clip=1.2,
        mean_ctx_weight=1.0,
    )
    flow.set_mean_ctx_weight(0.7)
    assert flow.mean_ctx_weight == 0.7
    assert all(block.conditioner.mean_ctx_weight == 0.7 for block in flow.blocks)

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


def test_joint_game_flow_attention_context_mode_forward_and_inverse() -> None:
    """Test that attention context mode works end-to-end with forward/inverse passes."""
    torch.manual_seed(42)
    bsz, num_players, num_stats, d_model = 2, 30, 12, 24
    flow = JointGameFlow(
        d_model=d_model,
        num_stats=num_stats,
        hidden_dim=32,
        dropout=0.0,
        num_blocks=3,
        coupling_type="affine",
        scale_clip=3.0,
        context_mode="attention",  # H2 fix
    )
    assert flow.context_mode == "attention"
    assert all(block.conditioner.context_mode == "attention" for block in flow.blocks)
    assert all(block.conditioner.team_attn is not None for block in flow.blocks)
    assert all(block.conditioner.game_attn is not None for block in flow.blocks)

    y = torch.randn((bsz, num_players, num_stats), dtype=torch.float32)
    player_states = torch.randn((bsz, num_players, d_model), dtype=torch.float32)
    team_states = torch.randn((bsz, 2, d_model), dtype=torch.float32)
    game_state = torch.randn((bsz, d_model), dtype=torch.float32)
    player_team_index = torch.cat(
        [torch.zeros((bsz, 15), dtype=torch.long), torch.ones((bsz, 15), dtype=torch.long)],
        dim=1,
    )
    valid_mask = torch.ones((bsz, num_players), dtype=torch.bool)

    # Forward pass
    out = flow(
        y,
        player_states=player_states,
        team_states=team_states,
        game_state=game_state,
        player_team_index=player_team_index,
        valid_mask=valid_mask,
    )
    assert out.z.shape == y.shape
    assert out.nll_mean.item() > 0.0

    # Inverse pass (sampling)
    y_recon = flow.sample(
        out.z,
        player_states=player_states,
        team_states=team_states,
        game_state=game_state,
        player_team_index=player_team_index,
        valid_mask=valid_mask,
    )
    # Round-trip should be exact (up to float precision)
    assert torch.max(torch.abs(y_recon - y)).item() < 1e-4


def test_joint_game_flow_attention_vs_mean_produces_different_samples() -> None:
    """Verify attention and mean context modes produce different samples from same noise."""
    torch.manual_seed(123)
    bsz, num_players, num_stats, d_model = 1, 30, 12, 24

    flow_mean = JointGameFlow(
        d_model=d_model,
        num_stats=num_stats,
        hidden_dim=32,
        dropout=0.0,
        num_blocks=3,
        scale_clip=3.0,
        context_mode="mean",
    )
    flow_attn = JointGameFlow(
        d_model=d_model,
        num_stats=num_stats,
        hidden_dim=32,
        dropout=0.0,
        num_blocks=3,
        scale_clip=3.0,
        context_mode="attention",
    )

    z = torch.randn((bsz, num_players, num_stats), dtype=torch.float32)
    player_states = torch.randn((bsz, num_players, d_model), dtype=torch.float32)
    team_states = torch.randn((bsz, 2, d_model), dtype=torch.float32)
    game_state = torch.randn((bsz, d_model), dtype=torch.float32)
    player_team_index = torch.cat(
        [torch.zeros((bsz, 15), dtype=torch.long), torch.ones((bsz, 15), dtype=torch.long)],
        dim=1,
    )
    valid_mask = torch.ones((bsz, num_players), dtype=torch.bool)

    with torch.no_grad():
        y_mean = flow_mean.sample(
            z,
            player_states=player_states,
            team_states=team_states,
            game_state=game_state,
            player_team_index=player_team_index,
            valid_mask=valid_mask,
        )
        y_attn = flow_attn.sample(
            z,
            player_states=player_states,
            team_states=team_states,
            game_state=game_state,
            player_team_index=player_team_index,
            valid_mask=valid_mask,
        )

    # Different context modes should produce different samples
    assert not torch.allclose(y_mean, y_attn, atol=1e-4)
