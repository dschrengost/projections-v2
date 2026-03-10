from __future__ import annotations

import torch

from projections.rotation.joint_active_set import (
    JointActiveSetHead,
    _select_topk_without_replacement,
    _select_topk_without_replacement_batched,
)


def test_batched_topk_matches_rowwise_no_sampling() -> None:
    torch.manual_seed(11)
    bsz = 8
    num_players = 30

    logits = torch.randn((bsz, num_players), dtype=torch.float32)
    eligible = torch.rand((bsz, num_players)) > 0.35
    # Ensure each row has at least one eligible player.
    eligible[:, 0] = True
    k = torch.randint(low=0, high=16, size=(bsz,), dtype=torch.long)

    expected = []
    for b_idx in range(bsz):
        expected.append(
            _select_topk_without_replacement(
                logits[b_idx],
                eligible[b_idx],
                int(k[b_idx].item()),
                sample=False,
                temperature=1.0,
            )
        )
    expected_mask = torch.stack(expected, dim=0)

    got_mask = _select_topk_without_replacement_batched(
        logits,
        eligible,
        k,
        sample=False,
        temperature=1.0,
    )

    assert torch.equal(got_mask, expected_mask)
    got_counts = got_mask.sum(dim=1).to(dtype=torch.long)
    exp_counts = torch.minimum(k, eligible.sum(dim=1).to(dtype=torch.long))
    assert torch.equal(got_counts, exp_counts)


def test_joint_active_set_forward_respects_team_caps_and_valid_mask() -> None:
    torch.manual_seed(23)
    bsz = 4
    num_players = 30
    d_model = 8

    head = JointActiveSetHead(
        d_model=d_model,
        hidden_dim=16,
        min_active_count=5,
        max_active_count=13,
    )

    player_states = torch.randn((bsz, num_players, d_model), dtype=torch.float32)
    team_states = torch.randn((bsz, 2, d_model), dtype=torch.float32)
    player_team_index = torch.cat(
        [
            torch.zeros((bsz, 15), dtype=torch.long),
            torch.ones((bsz, 15), dtype=torch.long),
        ],
        dim=1,
    )
    valid_mask = torch.ones((bsz, num_players), dtype=torch.bool)
    # Force a few invalid slots so selected count may be < requested.
    valid_mask[0, :3] = False
    valid_mask[1, 15:20] = False
    valid_mask[2, 10:15] = False
    valid_mask[3, 25:] = False

    target_counts = torch.tensor(
        [
            [5, 7],
            [8, 9],
            [12, 10],
            [13, 6],
        ],
        dtype=torch.long,
    )

    out = head(
        player_states,
        team_states,
        player_team_index,
        valid_mask,
        sample=False,
        temperature=1.0,
        target_counts=target_counts,
        use_target_counts=True,
    )

    assert out.active_mask.shape == (bsz, num_players)
    assert bool((out.active_mask & ~valid_mask).any()) is False

    for b_idx in range(bsz):
        for team_idx in (0, 1):
            team_valid = valid_mask[b_idx] & (player_team_index[b_idx] == team_idx)
            selected = out.active_mask[b_idx] & team_valid
            expected_count = min(
                int(target_counts[b_idx, team_idx].item()),
                int(team_valid.sum().item()),
            )
            assert int(selected.sum().item()) == expected_count
